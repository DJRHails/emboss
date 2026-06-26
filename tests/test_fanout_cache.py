"""Tests for the `FanoutCache` backend — a SqliteCache sharded across N DBs."""

from __future__ import annotations

import pytest

from emboss import Cache, FanoutCache, cached


@pytest.fixture
def cache(tmp_path):
    c = FanoutCache(tmp_path / "cache", shards=4)
    yield c
    c.close()


def test_satisfies_protocol(cache):
    assert isinstance(cache, Cache)


def test_set_get_delete(cache):
    cache.set("k", {"a": 1})
    assert cache.get("k") == {"a": 1}
    assert cache.delete("k") is True
    assert cache.get("k") is None


def test_creates_shard_dirs(tmp_path):
    c = FanoutCache(tmp_path / "cache", shards=4)
    shard_dirs = sorted(p.name for p in (tmp_path / "cache").iterdir() if p.is_dir())
    assert shard_dirs == ["0", "1", "2", "3"]
    c.close()


def test_routing_is_stable_across_instances(tmp_path):
    """A key written by one instance is found by another — routing must not depend
    on the salted built-in hash()."""
    c1 = FanoutCache(tmp_path / "cache", shards=8)
    c1.set("some-key", "value")
    c1.close()
    c2 = FanoutCache(tmp_path / "cache", shards=8)
    assert c2.get("some-key") == "value"
    c2.close()


def test_distributes_across_shards(cache, tmp_path):
    for i in range(200):
        cache.set(f"key-{i}", i)
    # every shard should hold some entries (200 keys over 4 shards)
    counts = [len(s) for s in cache._shards]
    assert all(n > 0 for n in counts)
    assert sum(counts) == 200


def test_len_volume_iter_aggregate(cache):
    for k in ("a", "b", "c"):
        cache.set(k, k * 100)
    assert len(cache) == 3
    assert cache.volume() > 0
    assert sorted(cache) == ["a", "b", "c"]


def test_clear_all_shards(cache):
    for i in range(50):
        cache.set(f"k{i}", i)
    assert cache.clear() == 50
    assert len(cache) == 0


def test_dunder_methods(cache):
    cache["k"] = "v"
    assert "k" in cache
    assert cache["k"] == "v"
    with pytest.raises(KeyError):
        _ = cache["missing"]
    del cache["k"]
    with pytest.raises(KeyError):
        del cache["already_gone"]


def test_size_limit_split_across_shards(tmp_path):
    c = FanoutCache(tmp_path / "cache", shards=4, size_limit=4 * 30_000)
    val = "x" * 10_000
    for i in range(200):
        c.set(f"key-{i}", val)
    # each shard bounded to ~30 KB -> total bounded to ~120 KB
    assert c.volume() <= 4 * 30_000 + 30_000  # within one shard's slack
    c.close()


def test_invalid_shards_raises(tmp_path):
    with pytest.raises(ValueError, match="shards"):
        FanoutCache(tmp_path / "cache", shards=0)


def test_works_with_cached_decorator(tmp_path):
    cache = FanoutCache(tmp_path / "cache", shards=4)
    calls = {"n": 0}

    @cached(cache)
    def f(x: int) -> int:
        calls["n"] += 1
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls["n"] == 1
    cache.close()
