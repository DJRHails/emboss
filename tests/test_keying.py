"""Tests for cache-key sensitivity: distinct args → distinct entries, identical args → hit."""

from __future__ import annotations

import diskcache
import pytest

from emboss import cached


@pytest.fixture
def cache(tmp_path):
    c = diskcache.Cache(str(tmp_path / "cache"))
    yield c
    c.close()


def test_distinct_args_distinct_entries(cache):
    calls = []

    @cached(cache)
    def f(x: int, y: int) -> int:
        calls.append((x, y))
        return x + y

    assert f(1, 2) == 3
    assert f(2, 1) == 3  # same sum, different args → recomputed
    assert f(1, 2) == 3  # cache hit
    assert calls == [(1, 2), (2, 1)]


def test_kwargs_participate_in_key(cache):
    calls = []

    @cached(cache)
    def f(x: int, scale: int = 1) -> int:
        calls.append((x, scale))
        return x * scale

    assert f(5) == 5
    assert f(5, scale=2) == 10  # different kwarg → recompute
    assert f(5) == 5  # hit
    assert calls == [(5, 1), (5, 2)]


def test_function_source_changes_invalidate(tmp_path):
    """If the function body changes, the cache key changes — automatic invalidation."""
    cache = diskcache.Cache(str(tmp_path / "cache"))
    try:
        @cached(cache)
        def f(x: int) -> int:
            return x * 2

        assert f(3) == 6

        # Redefine with a different body — new closure, new source hash.
        @cached(cache)
        def f(x: int) -> int:  # noqa: F811 — intentional redef
            return x * 3

        assert f(3) == 9  # not a stale 6 from the previous version
    finally:
        cache.close()
