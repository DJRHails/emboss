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


def test_whitespace_reformat_does_not_invalidate(cache):
    """Cosmetic reformatting (spacing, line breaks, comments) keeps the cache hit."""
    calls = []

    @cached(cache)
    def f(x: int) -> int:
        calls.append(x)
        return x + 1

    assert f(10) == 11
    assert calls == [10]

    # Same logic, reformatted: blank line, tightened operator spacing, a new comment.
    @cached(cache)
    def f(x: int) -> int:  # noqa: F811 — intentional reformatted redef
        calls.append(x)

        return x + 1  # an added comment that must not change the key

    assert f(10) == 11  # cache HIT despite the reformat
    assert calls == [10]  # body never re-ran


def test_string_literal_whitespace_invalidates(cache):
    """Whitespace *inside* a string literal is real content — it must change the key."""
    calls = []

    @cached(cache)
    def f() -> str:
        calls.append(1)
        return "hello world"

    assert f() == "hello world"

    @cached(cache)
    def f() -> str:  # noqa: F811 — intentional redef
        calls.append(2)
        return "hello  world"  # two spaces — a genuine change, not formatting

    assert f() == "hello  world"  # MISS — distinct string content
    assert calls == [1, 2]


def test_legacy_raw_source_key_no_longer_read(cache):
    """The implicit pre-0.3 raw-source fallback is gone — only `also_accept` migrates."""
    import hashlib
    import inspect
    import json

    calls = []

    @cached(cache)
    def f(x: int) -> int:
        calls.append(x)
        return x * 10

    # Plant a value under the old whitespace-sensitive key the 0.2 decorator used.
    raw_source = inspect.getsource(f.__wrapped__)
    raw_hash = hashlib.md5(raw_source.encode()).hexdigest()
    arg_hash = hashlib.md5((json.dumps([5]) + json.dumps({})).encode()).hexdigest()
    legacy_key = hashlib.md5(f"f{raw_hash}{arg_hash}".encode()).hexdigest()
    cache.set(legacy_key, 999)

    assert f(5) == 50  # MISS — the planted raw-source entry is ignored, body recomputes
    assert calls == [5]
