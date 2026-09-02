"""Tests for cache-only mode (`cache_only` / `EMBOSS_CACHE_ONLY` / `CacheMiss`)
and public key derivation (`cache_key` / `cache_keys`)."""

from __future__ import annotations

import asyncio
import copy
import pickle
from concurrent.futures import ThreadPoolExecutor

import diskcache
import pytest
from pydantic import BaseModel

from emboss import CacheMiss, cache_id, cache_key, cache_keys, cache_only, cached


@pytest.fixture
def cache(tmp_path):
    c = diskcache.Cache(str(tmp_path / "cache"))
    yield c
    c.close()


@pytest.fixture(autouse=True)
def _no_ambient_cache_only(monkeypatch):
    monkeypatch.delenv("EMBOSS_CACHE_ONLY", raising=False)


# ── cache-only mode: context manager ─────────────────────────────────────────


def test_genuine_miss_raises_and_does_not_execute(cache):
    calls = {"n": 0}

    @cached(cache)
    def f(x: int) -> int:
        calls["n"] += 1
        return x * 2

    with cache_only(), pytest.raises(CacheMiss):
        f(3)
    assert calls["n"] == 0  # the body never ran


def test_cached_value_returns_inside_cache_only(cache):
    calls = {"n": 0}

    @cached(cache)
    def f(x: int) -> int:
        calls["n"] += 1
        return x * 2

    assert f(3) == 6  # warm
    with cache_only():
        assert f(3) == 6  # served from cache — no raise
    assert calls["n"] == 1


class DriftModel(BaseModel):
    name: str


def test_schema_drift_raises_cache_miss_inside_cache_only(cache):
    """A stored shape that no longer rehydrates under the current class
    definition cannot be served, and a sealed sweep must not recompute — the
    drift is an honest CacheMiss, not a silent wrong-type return."""

    @cached(cache)
    def f() -> DriftModel:
        return DriftModel(name="x")

    f()
    key = next(iter(cache))
    cache.set(key, {"renamed_away": 1})  # predates DriftModel's required `name`

    with cache_only(), pytest.raises(CacheMiss):
        f()


def test_cache_miss_attributes_and_message(cache):
    @cached(cache)
    def f(token: str) -> str:
        return token

    secret = "sk-VERY-SECRET-TOKEN"
    with cache_only(), pytest.raises(CacheMiss) as exc_info:
        f(secret)
    err = exc_info.value
    assert isinstance(err, RuntimeError)
    assert err.func_name == "f"
    assert err.cache_id == cache_id(f)
    assert err.key == cache_key(f, secret)
    assert "cache-only mode" in str(err)
    assert err.key[:8] in str(err)
    assert "refusing to execute" in str(err)
    assert secret not in str(err)  # arguments never leak into the message


def test_context_manager_restores_state_on_exception(cache):
    calls = {"n": 0}

    @cached(cache)
    def f(x: int) -> int:
        calls["n"] += 1
        return x

    with pytest.raises(CacheMiss), cache_only():
        f(1)  # raises CacheMiss out of the block
    assert f(1) == 1  # mode was restored — this executes and caches
    assert calls["n"] == 1


def test_nested_blocks_restore_outer_state(cache):
    @cached(cache)
    def f(x: int) -> int:
        return x

    with cache_only():
        with cache_only(enabled=False):
            assert f(1) == 1  # inner block disables → executes
        with pytest.raises(CacheMiss):
            f(2)  # outer block active again


# ── cache-only mode: env var ─────────────────────────────────────────────────


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "Yes"])
def test_env_var_truthy_enables(cache, monkeypatch, value):
    @cached(cache)
    def f(x: int) -> int:
        return x

    # Set AFTER decoration — the flag must be read live per call, not at import
    # or decoration time.
    monkeypatch.setenv("EMBOSS_CACHE_ONLY", value)
    with pytest.raises(CacheMiss):
        f(1)


@pytest.mark.parametrize("value", ["", "0", "no", "false", "off"])
def test_env_var_falsy_disables(cache, monkeypatch, value):
    @cached(cache)
    def f(x: int) -> int:
        return x

    monkeypatch.setenv("EMBOSS_CACHE_ONLY", value)
    assert f(1) == 1


@pytest.mark.parametrize("value", ["on", "y", "enabled", "TRUE1"])
def test_env_var_unrecognized_value_raises(cache, monkeypatch, value):
    """A typo'd truthy-intent value must fail loudly, not silently unseal the run."""

    @cached(cache)
    def f(x: int) -> int:
        return x

    monkeypatch.setenv("EMBOSS_CACHE_ONLY", value)
    with pytest.raises(ValueError, match="EMBOSS_CACHE_ONLY"):
        f(1)


def test_context_manager_force_disables_env_var(cache, monkeypatch):
    @cached(cache)
    def f(x: int) -> int:
        return x

    monkeypatch.setenv("EMBOSS_CACHE_ONLY", "1")
    with cache_only(enabled=False):
        assert f(1) == 1  # env var overridden — executes
    with pytest.raises(CacheMiss):
        f(2)  # env var applies again outside the block


# ── cache-only mode: what still counts as a hit ──────────────────────────────


def test_also_accept_fallback_hit_does_not_raise_and_migrates(cache):
    calls = {"n": 0}

    @cached(cache)
    def fetch(x: int) -> int:
        calls["n"] += 1
        return x * 7

    assert fetch(3) == 21  # warm under the OLD identity
    old = cache_id(fetch)

    @cached(cache, also_accept=[old])
    def fetch_v2(x: int) -> int:
        calls["n"] += 1
        return x * 7

    with cache_only():
        assert fetch_v2(3) == 21  # reachable via also_accept → not a genuine miss
    assert calls["n"] == 1
    # The fallback hit still migrated forward (write-through): the value now
    # lives under fetch_v2's own key.
    assert cache.get(cache_key(fetch_v2, 3)) == 21

    with cache_only(), pytest.raises(CacheMiss):
        fetch_v2(4)  # different args → old identity has no entry either


def test_stored_none_is_a_hit_not_a_miss(cache):
    """A cached None (or any stored negative/known-miss result, e.g. what a
    KnownMiss-style backend writes) is a real value distinct from the _MISSING
    sentinel — cache-only mode returns it rather than raising."""
    calls = {"n": 0}

    @cached(cache)
    def lookup(q: str) -> str | None:
        calls["n"] += 1
        return None

    assert lookup("absent") is None  # warm: None is stored as a real entry
    with cache_only():
        assert lookup("absent") is None  # returned, not raised
    assert calls["n"] == 1


def test_backend_stored_negative_marker_is_a_hit(cache):
    """Pin the sentinel distinction at the Cache-protocol level: cache-only
    consults `cache.get(key, default=_MISSING)`, so ANY value a backend stored
    under the key — however miss-like — is a hit."""

    @cached(cache)
    def grade(x: int) -> dict:
        raise AssertionError("must not execute")

    cache.set(cache_key(grade, 5), {"known_miss": True})
    with cache_only():
        assert grade(5) == {"known_miss": True}


# ── cache-only mode: async parity ────────────────────────────────────────────


def test_async_miss_raises_on_await(cache):
    calls = {"n": 0}

    @cached(cache)
    async def f(x: int) -> int:
        calls["n"] += 1
        return x * 2

    async def run():
        with cache_only():
            await f(3)

    with pytest.raises(CacheMiss) as exc_info:
        asyncio.run(run())
    assert exc_info.value.func_name == "f"
    assert exc_info.value.key == cache_key(f, 3)
    assert calls["n"] == 0


def test_async_cached_value_returns_inside_cache_only(cache):
    calls = {"n": 0}

    @cached(cache)
    async def f(x: int) -> int:
        calls["n"] += 1
        return x * 2

    async def warm():
        return await f(3)

    async def sealed():
        with cache_only():
            return await f(3)

    assert asyncio.run(warm()) == 6
    assert asyncio.run(sealed()) == 6
    assert calls["n"] == 1


def test_cache_only_spans_asyncio_run(cache):
    @cached(cache)
    async def f(x: int) -> int:
        return x

    with cache_only(), pytest.raises(CacheMiss):
        asyncio.run(f(1))


# ── cache-only mode: scope boundaries (threads / coroutines / tasks) ─────────


def test_worker_threads_do_not_inherit_the_block(cache):
    """Documented escape: ThreadPoolExecutor workers start from a fresh context,
    so a `cache_only()` block does NOT seal them — `EMBOSS_CACHE_ONLY` does."""
    calls = {"n": 0}

    @cached(cache)
    def f(x: int) -> int:
        calls["n"] += 1
        return x

    with cache_only(), ThreadPoolExecutor(max_workers=1) as ex:
        assert ex.submit(f, 1).result() == 1  # executes — the seal does not reach it
    assert calls["n"] == 1


def test_asyncio_to_thread_inherits_the_block(cache):
    """`asyncio.to_thread` copies the caller's context, so the seal holds."""
    calls = {"n": 0}

    @cached(cache)
    def f(x: int) -> int:
        calls["n"] += 1
        return x

    async def run():
        with cache_only():
            await asyncio.to_thread(f, 1)

    with pytest.raises(CacheMiss):
        asyncio.run(run())
    assert calls["n"] == 0


def test_coroutine_created_inside_block_awaited_after_executes(cache):
    """Documented boundary: the mode is checked when the coroutine RUNS, so a
    bare coroutine created inside the block but awaited after it exits is no
    longer sealed."""
    calls = {"n": 0}

    @cached(cache)
    async def f(x: int) -> int:
        calls["n"] += 1
        return x * 2

    with cache_only():
        coro = f(3)  # created sealed…
    assert asyncio.run(coro) == 6  # …but runs unsealed → executes
    assert calls["n"] == 1


def test_concurrent_tasks_have_isolated_states(cache):
    """`create_task` copies the context at creation: a task created inside the
    block stays sealed even when awaited outside it, alongside an unsealed one
    on the same loop."""

    @cached(cache)
    async def f(x: int) -> int:
        return x

    async def main():
        with cache_only():
            sealed = asyncio.create_task(f(1))
        unsealed = asyncio.create_task(f(2))
        assert await unsealed == 2
        with pytest.raises(CacheMiss):
            await sealed

    asyncio.run(main())


# ── cache-only mode: CacheMiss ergonomics ────────────────────────────────────


def test_cache_miss_survives_pickle_and_copy(cache):
    """A CacheMiss crossing a multiprocessing boundary must round-trip —
    BaseException's default reduce would choke on the keyword-only __init__."""

    @cached(cache)
    def f(x: int) -> int:
        return x

    with cache_only(), pytest.raises(CacheMiss) as exc_info:
        f(1)
    err = exc_info.value
    for clone in (pickle.loads(pickle.dumps(err)), copy.copy(err)):
        assert isinstance(clone, CacheMiss)
        assert clone.func_name == err.func_name
        assert clone.cache_id == err.cache_id
        assert clone.key == err.key
        assert str(clone) == str(err)


def test_miss_leaves_cache_unwritten(cache):
    @cached(cache)
    def f(x: int) -> int:
        return x

    with cache_only(), pytest.raises(CacheMiss):
        f(3)
    assert cache.get(cache_key(f, 3), default="absent") == "absent"


# ── cache-only mode: pydantic model path ─────────────────────────────────────


class _User(BaseModel):
    name: str


def test_model_hit_returns_rehydrated_model_and_cold_call_raises(cache):
    calls = {"n": 0}

    @cached(cache)
    def get_user(uid: int) -> _User:
        calls["n"] += 1
        return _User(name=f"u{uid}")

    assert get_user(1) == _User(name="u1")  # warm: stored as a dict
    with cache_only():
        got = get_user(1)
        assert isinstance(got, _User)  # decode path runs on a sealed hit
        assert got.name == "u1"
        with pytest.raises(CacheMiss):
            get_user(2)  # cold model-annotated call raises before executing
    assert calls["n"] == 1


# ── cache-only mode: unsafe_manual_key ───────────────────────────────────────


def test_unsafe_manual_key_keys_and_miss_message(cache):
    @cached(cache, unsafe_manual_key="v1")
    def f(x: int) -> int:
        return x

    key = cache_key(f, 1)
    assert f(1) == 1
    assert cache.get(key) == 1  # cache_key matches the wrapper's storage

    with cache_only(), pytest.raises(CacheMiss) as exc_info:
        f(2)
    err = exc_info.value
    assert err.cache_id == "f:v1"
    assert "identity f:v1," in str(err)  # short manual key: no misleading ellipsis


# ── public key derivation ─────────────────────────────────────────────────────


def test_cache_key_matches_wrapper_storage(cache):
    @cached(cache)
    def f(x: int, scale: int = 1) -> int:
        return x * scale

    key = cache_key(f, 5, scale=3)
    assert cache.get(key) is None  # nothing written yet
    assert f(5, scale=3) == 15
    assert cache.get(key) == 15  # the wrapper wrote exactly this key


def test_cache_keys_includes_also_accept_fallbacks(cache):
    @cached(cache)
    def fetch(x: int) -> int:
        return x

    assert fetch(3) == 3
    old = cache_id(fetch)
    old_key = cache_key(fetch, 3)

    @cached(cache, also_accept=[old])
    def fetch_v2(x: int) -> int:
        return x

    key, accept_keys = cache_keys(fetch_v2, 3)
    assert key == cache_key(fetch_v2, 3)
    assert accept_keys == [old_key]
    assert cache.get(old_key) == 3  # the fallback key really holds the old entry


def test_cache_key_respects_default_encoder(cache):
    # Bare `object()` has no __dict__, so keying falls through to `default=`.
    # Under the package default (`str`) each instance would key on its memory
    # address; the pinned encoder collapses them — so cache_key on one instance
    # matching the wrapper's write for another proves both use the encoder.
    @cached(cache, default=lambda _obj: "pinned")
    def f(o: object) -> str:
        return "computed"

    key = cache_key(f, object())
    assert f(object()) == "computed"
    assert cache.get(key) == "computed"  # custom encoder used by both paths


def test_cache_key_rejects_unwrapped_function():
    def plain(x: int) -> int:
        return x

    with pytest.raises(TypeError, match="not an @cached-wrapped function"):
        cache_key(plain, 1)


def test_cache_key_rejects_non_callable_metadata():
    """A stray non-callable __emboss_keys__ (e.g. copied through an unrelated
    wrapper's __dict__ by functools.wraps) gets the curated TypeError, not a
    bare 'str is not callable'."""

    def impostor(x: int) -> int:
        return x

    impostor.__emboss_keys__ = "not-a-closure"
    with pytest.raises(TypeError, match="not an @cached-wrapped function"):
        cache_key(impostor, 1)
