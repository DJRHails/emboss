"""Tests for the `FileCache` backend — NFS-safe, file-per-key persistent cache."""

from __future__ import annotations

import os
import pickle

import pytest

from emboss import FileCache, cached
from emboss._file_cache import _key_to_path


@pytest.fixture
def cache(tmp_path):
    return FileCache(tmp_path / "cache")


def test_get_returns_default_on_miss(cache):
    assert cache.get("missing") is None
    assert cache.get("missing", default="fallback") == "fallback"


def test_set_then_get(cache):
    cache.set("k", {"a": 1})
    assert cache.get("k") == {"a": 1}


def test_set_overwrites_existing_key(cache):
    """set() overwrites via atomic rename (last writer wins)."""
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
    cache.set("c", 3)
    assert cache.clear() == 3
    assert cache.get("a") is None
    assert cache.get("b") is None


def test_corrupt_file_treated_as_miss(cache, tmp_path):
    """Partial writes shouldn't poison the cache — corrupted files behave as misses."""
    cache.set("k", "v")
    # Find the file and truncate it to corrupt the pickle.
    pkl_files = list((tmp_path / "cache").glob("**/*.pkl"))
    assert len(pkl_files) == 1
    pkl_files[0].write_bytes(b"not a pickle")
    assert cache.get("k") is None
    assert cache.get("k", default="fallback") == "fallback"


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


def test_context_manager_is_no_op(tmp_path):
    with FileCache(tmp_path / "cache") as c:
        c.set("k", "v")
    # Re-open: data persists, no close-required state.
    c2 = FileCache(tmp_path / "cache")
    assert c2.get("k") == "v"


def test_close_is_no_op(cache):
    cache.set("k", "v")
    cache.close()
    assert cache.get("k") == "v"


def test_nuke_drops_everything(cache):
    cache.set("k", "v")
    cache.nuke()
    assert cache.get("k") is None
    # Still usable after nuke
    cache.set("k", "fresh")
    assert cache.get("k") == "fresh"


def test_diskcache_kwargs_ignored(tmp_path):
    """FileCache must accept (and ignore) diskcache kwargs so existing call
    sites don't need updating."""
    c = FileCache(tmp_path / "cache", timeout=30, size_limit=2**30, eviction_policy="none")
    c.set("k", "v")
    assert c.get("k") == "v"


def test_works_with_cached_decorator(tmp_path):
    """End-to-end: @cached with a FileCache backend."""
    cache = FileCache(tmp_path / "cache")
    calls = {"n": 0}

    @cached(cache)
    def f(x: int) -> int:
        calls["n"] += 1
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls["n"] == 1


def test_sharding_caps_files_per_directory(cache, tmp_path):
    """Keys are sharded by first two chars to avoid the 'thousands of files in
    one NFS dir' performance cliff."""
    for i in range(20):
        cache.set(f"abc{i}", i)
    # All 'ab'-prefixed keys land in the same shard dir.
    shard = tmp_path / "cache" / "ab"
    assert shard.exists()
    assert len(list(shard.glob("*.pkl"))) == 20

    # A different prefix gets its own shard.
    cache.set("zzother", 99)
    assert (tmp_path / "cache" / "zz").exists()


def test_non_string_keys_hashed(cache):
    cache.set((1, 2, 3), "tuple value")
    assert cache.get((1, 2, 3)) == "tuple value"


def test_short_string_keys_do_not_collide(cache):
    """Regression: 1-char string keys all md5-mapped to `_/_.pkl` before fix."""
    cache.set("a", "alpha")
    cache.set("b", "bravo")
    cache.set("c", "charlie")
    assert cache.get("a") == "alpha"
    assert cache.get("b") == "bravo"
    assert cache.get("c") == "charlie"


def test_pickle_round_trip_for_complex_types(cache):
    payload = {"list": [1, 2, 3], "tup": (4, 5), "nested": {"k": [b"bytes"]}}
    cache.set("k", payload)
    out = cache.get("k")
    # tuples survive pickle (unlike json)
    assert out["tup"] == (4, 5)
    assert out["nested"]["k"][0] == b"bytes"


def test_no_residual_tempfiles_after_set(cache, tmp_path):
    cache.set("k", "v")
    # Atomic-rename leaves no .tmp files in the shard dir.
    tmp_leftovers = list((tmp_path / "cache").glob("**/.*.tmp"))
    assert tmp_leftovers == []


def test_unbounded_by_default_keeps_everything(tmp_path):
    c = FileCache(tmp_path / "cache")  # size_limit=None
    val = b"x" * 10_000
    for i in range(20):
        c.set(f"key{i:02d}", val)
    assert all(c.get(f"key{i:02d}") == val for i in range(20))


def test_size_limit_bounds_total(tmp_path):
    root = tmp_path / "cache"
    c = FileCache(root, size_limit=30_000)
    val = b"x" * 10_000  # ~10 KB pickled
    for i in range(20):
        c.set(f"key{i:02d}", val)
    total = sum(p.stat().st_size for p in root.glob("**/*.pkl"))
    assert total <= 30_000  # eviction kept it under the cap
    assert len(list(root.glob("**/*.pkl"))) < 20  # some were evicted


def test_lru_evicts_oldest_by_mtime(tmp_path):
    root = tmp_path / "cache"
    c = FileCache(root)  # unbounded during sets, so we control mtimes first
    val = b"x" * 10_000
    for name, mtime in (("old", 100.0), ("mid", 200.0), ("new", 300.0)):
        c.set(name, val)
        os.utime(_key_to_path(root, name), (mtime, mtime))
    c.size_limit = 25_000  # ~2 of the 3 entries fit (low-water 22.5 KB)
    c._evict()
    assert c.get("old") is None  # least-recently-used evicted first
    assert c.get("mid") == val
    assert c.get("new") == val


def test_get_bumps_mtime_to_protect_from_eviction(tmp_path):
    root = tmp_path / "cache"
    c = FileCache(root, size_limit=25_000)
    val = b"x" * 10_000
    for name, mtime in (("a", 100.0), ("b", 200.0)):
        c.set(name, val)
        os.utime(_key_to_path(root, name), (mtime, mtime))
    c.get("a")  # touch the older entry -> bumps its mtime to now (most recent)
    c.set("c", val)  # pushes over the cap, triggers eviction
    assert c.get("b") is None  # 'b' is now the least-recently-used -> evicted
    assert c.get("a") == val  # 'a' survived because it was read recently
    assert c.get("c") == val


def test_set_with_unpicklable_value_cleans_up_tempfile(cache, tmp_path):
    class Unpicklable:
        def __reduce__(self):
            raise pickle.PicklingError("nope")

    with pytest.raises(pickle.PicklingError):
        cache.set("k", Unpicklable())
    # Failed write must not leave a half-baked tempfile behind.
    leftovers = list((tmp_path / "cache").glob("**/.*.tmp"))
    assert leftovers == []
    # And the key must not appear to exist.
    assert cache.get("k") is None
