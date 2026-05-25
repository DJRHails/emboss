"""Tests for the `Cache` protocol — runtime-checkable, structural typing."""

from __future__ import annotations

from typing import Any

import diskcache

from emboss import Cache, FileCache, cached


def test_filecache_satisfies_protocol(tmp_path):
    fc = FileCache(tmp_path / "fc")
    assert isinstance(fc, Cache)


def test_diskcache_satisfies_protocol(tmp_path):
    dc = diskcache.Cache(str(tmp_path / "dc"))
    try:
        assert isinstance(dc, Cache)
    finally:
        dc.close()


def test_arbitrary_get_set_object_satisfies_protocol():
    """Structural typing: no inheritance needed."""

    class DictCache:
        def __init__(self):
            self.store: dict[str, Any] = {}

        def get(self, key: str, default: Any = None) -> Any:
            return self.store.get(key, default)

        def set(self, key: str, value: Any) -> bool:
            self.store[key] = value
            return True

    dc = DictCache()
    assert isinstance(dc, Cache)


def test_object_missing_methods_fails_protocol():
    class JustGet:
        def get(self, key: str, default: Any = None) -> Any:
            return default

    assert not isinstance(JustGet(), Cache)


def test_cached_accepts_any_protocol_compliant_backend():
    """@cached(cache=...) should accept any Cache-protocol-compliant object."""

    class DictCache:
        def __init__(self):
            self.store: dict[str, Any] = {}

        def get(self, key: str, default: Any = None) -> Any:
            return self.store.get(key, default)

        def set(self, key: str, value: Any) -> bool:
            self.store[key] = value
            return True

    cache = DictCache()
    calls = {"n": 0}

    @cached(cache)
    def f(x: int) -> int:
        calls["n"] += 1
        return x * 3

    assert f(4) == 12
    assert f(4) == 12
    assert calls["n"] == 1
