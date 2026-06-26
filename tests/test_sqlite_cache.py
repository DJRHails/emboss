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


def test_size_limit_bounds_total(tmp_path):
    val = "x" * 10_000  # ~10 KB pickled
    c = SqliteCache(tmp_path / "cache", size_limit=30_000)
    for i in range(20):
        c.set(f"key{i:02d}", val)
    assert c.volume() <= 30_000  # bounded (trigger-maintained size)
    assert len(c) < 20  # some evicted
    assert c.get("key19") == val  # the most-recently-stored survives
    c.close()


def test_least_recently_stored_evicts_oldest(tmp_path):
    """Default policy evicts the oldest-stored entries first."""
    val = "x" * 10_000
    c = SqliteCache(tmp_path / "cache", size_limit=25_000)  # ~2 fit
    c.set("first", val)
    c.set("second", val)
    c.set("third", val)  # over cap -> evicts 'first' (oldest store_time)
    assert c.get("first") is None
    assert c.get("second") == val
    assert c.get("third") == val
    c.close()


def test_lru_keeps_recently_read(tmp_path):
    """least-recently-used: a read refreshes recency, protecting from eviction."""
    val = "x" * 10_000
    c = SqliteCache(tmp_path / "cache", size_limit=25_000, eviction_policy="least-recently-used")
    c.set("a", val)
    c._conn.execute("UPDATE Cache SET access_time = 100 WHERE key = 'a'")
    c.set("b", val)
    c._conn.execute("UPDATE Cache SET access_time = 200 WHERE key = 'b'")
    c.get("a")  # refresh 'a' (stale >60s -> access_time bumped to now)
    c.set("c", val)  # over cap -> evicts the LRU, now 'b'
    assert c.get("b") is None
    assert c.get("a") == val
    assert c.get("c") == val
    c.close()


def test_invalid_eviction_policy_raises(tmp_path):
    with pytest.raises(ValueError, match="eviction_policy"):
        SqliteCache(tmp_path / "cache", eviction_policy="nonsense")


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


def test_size_and_count_triggers_stay_accurate(tmp_path):
    c = SqliteCache(tmp_path / "cache", size_limit=None)
    c.set("a", b"x" * 100)
    c.set("b", b"y" * 200)
    assert len(c) == 2
    v = c.volume()
    assert v > 300  # pickled sizes, both counted
    c.set("a", b"x" * 1000)  # overwrite -> size adjusts, count unchanged
    assert len(c) == 2
    assert c.volume() > v
    c.delete("b")
    assert len(c) == 1
    c.close()


def test_large_values_spill_to_files(tmp_path):
    root = tmp_path / "cache"
    c = SqliteCache(root, min_file_size=100)
    c.set("small", "x" * 10)  # stays inline
    c.set("big", "y" * 5000)  # spills to a file
    spill_files = list((root / "store").glob("**/*")) if (root / "store").exists() else []
    spill_files = [p for p in spill_files if p.is_file()]
    assert len(spill_files) == 1
    assert c.get("big") == "y" * 5000  # read back from file
    assert c.get("small") == "x" * 10
    c.close()


def test_spill_file_removed_on_overwrite_and_delete(tmp_path):
    root = tmp_path / "cache"
    c = SqliteCache(root, min_file_size=100)
    c.set("k", "a" * 5000)

    def files():
        return [p for p in (root / "store").glob("**/*") if p.is_file()]

    assert len(files()) == 1
    c.set("k", "b" * 5000)  # overwrite -> old spill file removed, one remains
    assert len(files()) == 1
    assert c.get("k") == "b" * 5000
    c.delete("k")  # delete -> spill file removed
    assert files() == []
    c.close()


def test_expire_sweep(tmp_path):
    c = SqliteCache(tmp_path / "cache")
    c.set("keep", "v")
    c.set("gone", "v", expire=0.01)
    time.sleep(0.03)
    assert c.expire() == 1  # swept one expired entry
    assert len(c) == 1
    assert c.get("keep") == "v"
    c.close()


def test_iteration_and_len(tmp_path):
    c = SqliteCache(tmp_path / "cache")
    for k in ("a", "b", "c"):
        c.set(k, k)
    assert len(c) == 3
    assert sorted(c) == ["a", "b", "c"]  # __iter__ yields keys
    assert list(c.iterkeys()) == ["a", "b", "c"]  # store-time order
    c.close()


def test_concurrent_writers_share_db_safely(tmp_path):
    """Multiple connections (≈ multiple processes) writing the same DB must not
    raise 'database is locked' and must agree on the shared count/size."""
    import threading

    root = tmp_path / "cache"
    errors: list[Exception] = []

    def worker(wid: int) -> None:
        try:
            c = SqliteCache(root, size_limit=None)
            for i in range(50):
                c.set(f"w{wid}-k{i}", i)
            c.close()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    final = SqliteCache(root, size_limit=None)
    assert len(final) == 4 * 50  # trigger-maintained count is correct across writers
    final.close()


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
