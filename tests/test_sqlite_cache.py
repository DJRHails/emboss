"""Tests for the `SqliteCache` backend — single-file, size-bounded, LRU, TTL."""

from __future__ import annotations

import time

import pytest

from emboss import SqliteCache, cached


@pytest.fixture
def cache(tmp_path):
    c = SqliteCache(tmp_path / "cache")
    yield c
    c.close()


def test_get_returns_default_on_miss(cache):
    assert cache.get("missing") is None
    assert cache.get("missing", default="fallback") == "fallback"


def test_set_then_get(cache):
    cache.set("k", {"a": 1})
    assert cache.get("k") == {"a": 1}


def test_set_overwrites_existing_key(cache):
    assert cache.set("k", "first") is True
    assert cache.set("k", "second") is True
    assert cache.get("k") == "second"


def test_delete(cache):
    cache.set("k", "v")
    assert cache.delete("k") is True
    assert cache.get("k") is None
    assert cache.delete("k") is False  # idempotent


def test_clear(cache):
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.clear() == 2
    assert cache.get("a") is None


def test_dunder_methods(cache):
    cache["k"] = "v"
    assert "k" in cache
    assert cache["k"] == "v"
    with pytest.raises(KeyError):
        _ = cache["missing"]
    del cache["k"]
    assert "k" not in cache
    with pytest.raises(KeyError):
        del cache["already_gone"]


def test_persists_across_reopen(tmp_path):
    with SqliteCache(tmp_path / "cache") as c:
        c.set("k", "v")
    c2 = SqliteCache(tmp_path / "cache")
    assert c2.get("k") == "v"
    c2.close()


def test_pickle_round_trip_for_complex_types(cache):
    payload = {"list": [1, 2, 3], "tup": (4, 5), "nested": {"k": [b"bytes"]}}
    out_in = cache.set("k", payload) and cache.get("k")
    assert out_in["tup"] == (4, 5)  # tuples survive pickle (unlike json)
    assert out_in["nested"]["k"][0] == b"bytes"


def test_non_string_keys(cache):
    cache.set((1, 2, 3), "tuple value")
    assert cache.get((1, 2, 3)) == "tuple value"


def test_expire_ttl(cache):
    cache.set("k", "v", expire=0.05)
    assert cache.get("k") == "v"
    time.sleep(0.07)
    assert cache.get("k") is None  # expired
    assert "k" not in cache


def test_size_limit_evicts_lru(tmp_path):
    val = "x" * 10_000  # ~10 KB pickled
    c = SqliteCache(tmp_path / "cache", size_limit=30_000)
    for i in range(20):
        c.set(f"key{i:02d}", val)
    total = c._conn.execute("SELECT COALESCE(SUM(size), 0) FROM cache").fetchone()[0]
    assert total <= 30_000  # bounded
    n = c._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    assert n < 20  # some evicted
    assert c.get("key19") == val  # the most-recent survives
    c.close()


def test_lru_keeps_recently_read(tmp_path):
    """A read refreshes recency, protecting an entry from eviction."""
    val = "x" * 10_000
    c = SqliteCache(tmp_path / "cache", size_limit=25_000)
    # age 'a' and 'b' into the past so 'a' is the oldest
    c.set("a", val)
    c._conn.execute("UPDATE cache SET atime=100 WHERE key='a'")
    c.set("b", val)
    c._conn.execute("UPDATE cache SET atime=200 WHERE key='b'")
    c.get("a")  # refresh 'a' -> now most recent (atime bump bypasses resolution: 0 -> now)
    c.set("c", val)  # over the cap -> evicts the LRU, which is now 'b'
    assert c.get("b") is None
    assert c.get("a") == val
    assert c.get("c") == val
    c.close()


def test_unbounded_when_size_limit_none(tmp_path):
    c = SqliteCache(tmp_path / "cache", size_limit=None)
    val = "x" * 10_000
    for i in range(20):
        c.set(f"key{i:02d}", val)
    assert all(c.get(f"key{i:02d}") == val for i in range(20))
    c.close()


def test_diskcache_kwargs_ignored(tmp_path):
    c = SqliteCache(tmp_path / "cache", timeout=30, eviction_policy="least-recently-used")
    c.set("k", "v")
    assert c.get("k") == "v"
    c.close()


def test_works_with_cached_decorator(tmp_path):
    cache = SqliteCache(tmp_path / "cache")
    calls = {"n": 0}

    @cached(cache)
    def f(x: int) -> int:
        calls["n"] += 1
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls["n"] == 1
    cache.close()
