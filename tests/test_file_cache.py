"""Tests for the `FileCache` backend — NFS-safe, file-per-key persistent cache."""

from __future__ import annotations

import pickle

import pytest

from emboss import FileCache, cached


@pytest.fixture
def cache(tmp_path):
    return FileCache(tmp_path / "cache")


def test_get_returns_default_on_miss(cache):
    assert cache.get("missing") is None
    assert cache.get("missing", default="fallback") == "fallback"


def test_set_then_get(cache):
    cache.set("k", {"a": 1})
    assert cache.get("k") == {"a": 1}


def test_set_returns_false_on_existing_key(cache):
    """Concurrent-writer guard: cache values are pure functions of the key,
    so don't overwrite — the existing file is by construction equally correct."""
    assert cache.set("k", "first") is True
    assert cache.set("k", "second") is False
    assert cache.get("k") == "first"  # original retained


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
