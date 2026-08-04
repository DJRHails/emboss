"""Tests for cache-only mode (`cache_only` / `EMBOSS_CACHE_ONLY` / `CacheMiss`)
and public key derivation (`cache_key` / `cache_keys`)."""

from __future__ import annotations

import asyncio

import diskcache
import pytest

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
