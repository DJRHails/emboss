"""The per-key in-flight latch and first-write-wins settle on the `@cached` miss path.

A redundant recompute of an already-cached (or concurrently-computing) key must never
replace the value a first caller returned: for a non-deterministic function (an LLM draw
with no API seed) the replacement is a different sample, served under every later reader.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import pytest

from emboss import CacheMiss, LogCache, cache_key, cache_only, cached
from emboss._cached import _ASYNC_LATCH, _LATCH


@pytest.fixture
def cache(tmp_path):
    return LogCache(tmp_path / "cache", writer_id="test")


def test_concurrent_same_key_callers_compute_once(cache):
    """Eight threads racing one key: one compute, eight identical results."""
    calls = {"n": 0}
    entered = threading.Barrier(8)

    @cached(cache)
    def draw(x: int) -> float:
        calls["n"] += 1
        time.sleep(0.05)  # widen the window the latch has to close
        return calls["n"] * 1000.0 + x

    def race(_: int) -> float:
        entered.wait()  # all eight arrive before any looks up the cache
        return draw(7)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(race, range(8)))

    assert calls["n"] == 1, "the latch must let exactly one caller compute"
    assert results == [1007.0] * 8
    assert _LATCH.in_flight() == 0, "latches are released and dropped after the miss settles"


def test_distinct_keys_do_not_serialise(cache):
    """The latch is per key: two different arguments compute concurrently."""
    started = threading.Barrier(2, timeout=5)

    @cached(cache)
    def slow(x: int) -> int:
        started.wait()  # deadlocks (times out) if the two calls were serialised
        return x

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(slow, [1, 2])) == [1, 2]


def test_first_write_wins_when_another_writer_lands_mid_compute(cache, caplog):
    """A value stored by a peer while this call computed is served; the fresh
    result is discarded with a warning naming the function."""

    @cached(cache)
    def draw(x: int) -> float:
        # Simulate a peer process (or node) storing the cell under our own key
        # while we are still computing — outside the in-process latch.
        cache.set(cache_key(draw, x), 15.0)
        return 65.0

    with caplog.at_level(logging.WARNING, logger="emboss._cached"):
        served = draw(1)

    assert served == 15.0, "the first-written value wins, not the fresh draw"
    assert draw(1) == 15.0, "later readers see the same value the racing caller returned"
    assert any(
        "draw() key" in rec.message and "first-written" in rec.message for rec in caplog.records
    ), caplog.text


def test_identical_concurrent_write_is_not_a_warning(cache, caplog):
    """A deterministic function's benign race logs at info, never warning."""

    @cached(cache)
    def det(x: int) -> int:
        cache.set(cache_key(det, x), x * 2)
        return x * 2

    with caplog.at_level(logging.INFO, logger="emboss._cached"):
        assert det(4) == 8
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], caplog.text
    assert any("identical value" in r.message for r in caplog.records), caplog.text


def test_failed_compute_releases_the_latch(cache):
    """A raising holder must not wedge the key: the next caller computes."""
    calls = {"n": 0}

    @cached(cache)
    def flaky(x: int) -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return x

    with pytest.raises(RuntimeError, match="transient"):
        flaky(3)
    assert flaky(3) == 3
    assert calls["n"] == 2
    assert _LATCH.in_flight() == 0


def test_cache_only_raises_under_the_latch(cache):
    calls = {"n": 0}

    @cached(cache)
    def f(x: int) -> int:
        calls["n"] += 1
        return x

    with cache_only(), pytest.raises(CacheMiss):
        f(1)
    assert calls["n"] == 0
    assert _LATCH.in_flight() == 0


def test_self_recursion_is_a_recursion_error_not_a_deadlock(cache):
    """The per-key lock is re-entrant: a function recursing into its own key on
    the same thread fails the way it always did instead of hanging forever."""

    @cached(cache)
    def loop(x: int) -> int:
        return loop(x)

    with pytest.raises(RecursionError):
        loop(1)
    assert _LATCH.in_flight() == 0


def test_async_concurrent_same_key_awaits_compute_once(cache):
    calls = {"n": 0}

    @cached(cache)
    async def draw(x: int) -> float:
        calls["n"] += 1
        await asyncio.sleep(0.02)
        return calls["n"] * 1000.0 + x

    async def run():
        results = await asyncio.gather(*(draw(5) for _ in range(6)))
        return results, _ASYNC_LATCH.in_flight()

    results, in_flight = asyncio.run(run())
    assert calls["n"] == 1
    assert results == [1005.0] * 6
    assert in_flight == 0


def test_async_first_write_wins(cache):
    @cached(cache)
    async def draw(x: int) -> float:
        cache.set(cache_key(draw, x), 15.0)
        await asyncio.sleep(0)
        return 65.0

    assert asyncio.run(draw(2)) == 15.0
    assert asyncio.run(draw(2)) == 15.0


def test_async_same_task_reentry_passes_through(cache):
    """A cached coroutine awaiting its own key from the same task must not
    deadlock on the non-reentrant asyncio.Lock — it recurses like before."""

    @cached(cache)
    async def loop(x: int) -> int:
        return await loop(x)

    with pytest.raises(RecursionError):
        asyncio.run(asyncio.wait_for(loop(1), timeout=5))


def test_async_failed_compute_releases_the_latch(cache):
    calls = {"n": 0}

    @cached(cache)
    async def flaky(x: int) -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return x

    async def run():
        with pytest.raises(RuntimeError, match="transient"):
            await flaky(3)
        return await flaky(3)

    assert asyncio.run(run()) == 3
    assert calls["n"] == 2


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork() is POSIX-only")
def test_forked_child_does_not_inherit_a_held_latch(cache):
    """A child forked while a parent thread holds a key's latch must compute that key, not wait
    forever on a lock whose owning thread does not exist in the child."""
    holding = threading.Event()
    release = threading.Event()

    @cached(cache)
    def slow(x: int) -> int:
        holding.set()
        release.wait(timeout=10)
        return x

    holder = threading.Thread(target=slow, args=(1,), daemon=True)
    holder.start()
    assert holding.wait(timeout=5), "the holder thread never entered the latched compute"

    def child() -> None:
        release.set()  # the child's own copy of the event; the parent's holder stays parked
        os._exit(0 if slow(1) == 1 else 1)

    proc = multiprocessing.get_context("fork").Process(target=child)
    with warnings.catch_warnings():
        # Forking a multi-threaded process is the hazard under test, not an accident here.
        warnings.simplefilter("ignore", DeprecationWarning)
        proc.start()
    proc.join(timeout=10)
    if proc.is_alive():
        proc.kill()
        release.set()
        pytest.fail("the forked child deadlocked on the latch its parent's thread held")
    release.set()
    holder.join(timeout=5)
    assert proc.exitcode == 0
    assert _LATCH.in_flight() == 0
