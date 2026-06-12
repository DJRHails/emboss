"""Tests for cache identity (`cache_id`), `also_accept` migration, and `unsafe_manual_key`."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re

import diskcache
import pytest

from emboss import cache_id, cached


@pytest.fixture
def cache(tmp_path):
    c = diskcache.Cache(str(tmp_path / "cache"))
    yield c
    c.close()


def _key_for(identity: str, args: list) -> str:
    """Reconstruct the on-disk cache key for a `"name:body_hash"` identity + args."""
    name, _, body_hash = identity.partition(":")
    arg_hash = hashlib.md5((json.dumps(args) + json.dumps({})).encode()).hexdigest()
    return hashlib.md5(f"{name}{body_hash}{arg_hash}".encode()).hexdigest()


def test_cache_id_shape(cache):
    @cached(cache)
    def f(x: int) -> int:
        return x + 1

    cid = cache_id(f)
    assert re.fullmatch(r"f:[0-9a-f]{32}", cid)
    assert f.__emboss__.cache_id == cid
    assert f.__emboss__.name == "f"
    assert cid == f"f:{f.__emboss__.body_hash}"
    assert f.__emboss__.also_accept == ()


def test_cache_id_rejects_unwrapped_function():
    def plain(x: int) -> int:
        return x

    with pytest.raises(TypeError, match="not an @cached-wrapped function"):
        cache_id(plain)


def test_also_accept_migrates_renamed_function(cache):
    calls = {"n": 0}

    @cached(cache)
    def fetch(x: int) -> int:
        calls["n"] += 1
        return x * 7

    assert fetch(3) == 21
    assert calls["n"] == 1
    old = cache_id(fetch)

    @cached(cache, also_accept=[old])
    def fetch_v2(x: int) -> int:
        calls["n"] += 1
        return x * 7  # same behaviour — the old entry should be reused, not recomputed

    assert fetch_v2(3) == 21  # served from fetch's entry via the fallback key
    assert calls["n"] == 1

    # Write-through: drop the OLD entry — the value must now live under
    # fetch_v2's own key, so the next call still never runs the body.
    cache.delete(_key_for(old, [3]))
    assert fetch_v2(3) == 21
    assert calls["n"] == 1


def test_also_accept_different_args_still_miss(cache):
    calls = {"n": 0}

    @cached(cache)
    def f(x: int) -> int:
        calls["n"] += 1
        return x * 2

    assert f(1) == 2
    old = cache_id(f)

    @cached(cache, also_accept=[old])
    def g(x: int) -> int:
        calls["n"] += 1
        return x * 2

    assert (
        g(2) == 4
    )  # unseen args — neither the current nor the fallback key holds them
    assert calls["n"] == 2
    assert g(1) == 2  # the old args still migrate
    assert calls["n"] == 2


def test_also_accept_rejects_malformed_tokens(cache):
    for bad in ["no-colon-here", "name:", ":hash"]:
        with pytest.raises(ValueError, match=re.escape(repr(bad))):

            @cached(cache, also_accept=[bad])
            def f() -> int:
                return 1


def test_store_writes_exactly_one_key(cache):
    """No legacy twin entry: one call stores one cache entry."""

    @cached(cache)
    def f(x: int) -> int:
        return x + 1

    assert f(1) == 2
    assert len(list(cache)) == 1


def test_unsafe_manual_key_survives_body_edit(cache):
    calls = {"n": 0}

    @cached(cache, unsafe_manual_key="v1")
    def f(x: int) -> int:
        calls["n"] += 1
        return x + 100

    assert f(1) == 101
    assert calls["n"] == 1
    assert cache_id(f) == "f:v1"

    @cached(cache, unsafe_manual_key="v1")
    def f(x: int) -> int:  # noqa: F811 — intentional redef with an edited body
        calls["n"] += 1
        return x + 200  # changed constant — would re-key under source hashing

    assert f(1) == 101  # HIT: the manual key pins identity, stale-by-design
    assert calls["n"] == 1

    @cached(cache, unsafe_manual_key="v2")
    def f(x: int) -> int:  # noqa: F811 — intentional redef with a bumped key
        calls["n"] += 1
        return x + 200

    assert f(1) == 201  # bumping the manual key recomputes
    assert calls["n"] == 2


def test_unsafe_manual_key_rejects_empty_string(cache):
    with pytest.raises(ValueError, match="non-empty"):
        cached(cache, unsafe_manual_key="")


def test_also_accept_works_with_unsafe_manual_key(cache):
    calls = {"n": 0}

    @cached(cache)
    def f(x: int) -> int:
        calls["n"] += 1
        return x - 1

    assert f(9) == 8
    old = cache_id(f)

    @cached(cache, unsafe_manual_key="m1", also_accept=[old])
    def g(x: int) -> int:
        calls["n"] += 1
        return x - 1

    assert cache_id(g) == "g:m1"
    assert g(9) == 8  # migrated from f's source-keyed entry into the manual identity
    assert calls["n"] == 1


def test_async_also_accept_and_manual_key(cache):
    calls = {"n": 0}

    @cached(cache)
    async def f(x: int) -> int:
        calls["n"] += 1
        return x * 3

    assert asyncio.run(f(2)) == 6
    old = cache_id(f)

    @cached(cache, also_accept=[old])
    async def g(x: int) -> int:
        calls["n"] += 1
        return x * 3

    assert asyncio.run(g(2)) == 6  # migrated, not recomputed
    assert calls["n"] == 1

    @cached(cache, unsafe_manual_key="av1")
    async def h(x: int) -> int:
        calls["n"] += 1
        return x * 5

    assert asyncio.run(h(2)) == 10
    assert calls["n"] == 2

    @cached(cache, unsafe_manual_key="av1")
    async def h(x: int) -> int:  # noqa: F811 — intentional redef with an edited body
        calls["n"] += 1
        return x * 50

    assert asyncio.run(h(2)) == 10  # HIT under the pinned key despite the body edit
    assert calls["n"] == 2
