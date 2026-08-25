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


def test_docstring_edit_does_not_invalidate(cache):
    """Docstrings document behaviour, they don't implement it — editing one keeps the hit."""
    calls = []

    @cached(cache)
    def f(x: int) -> int:
        """Original docstring."""
        calls.append(x)
        return x + 1

    assert f(10) == 11
    assert calls == [10]

    @cached(cache)
    def f(x: int) -> int:  # noqa: F811 — intentional redef with a rewritten docstring
        """A completely rewritten docstring that must not change the key."""
        calls.append(x)
        return x + 1

    assert f(10) == 11  # cache HIT despite the docstring rewrite
    assert calls == [10]  # body never re-ran


def test_adding_a_docstring_does_not_invalidate(cache):
    """Going from no docstring to a docstring (or back) keeps the hit."""
    calls = []

    @cached(cache)
    def f(x: int) -> int:
        calls.append(x)
        return x * 2

    assert f(4) == 8

    @cached(cache)
    def f(x: int) -> int:  # noqa: F811 — intentional redef, docstring added
        """Newly added docstring."""
        calls.append(x)
        return x * 2

    assert f(4) == 8  # HIT — the docless canonical source is unchanged
    assert calls == [4]


def test_nested_docstrings_are_ignored(cache):
    """Docstrings of nested defs/classes are stripped too, at every level."""
    calls = []

    @cached(cache)
    def f(x: int) -> int:
        """Outer docstring."""

        def inner(y: int) -> int:
            """Inner docstring."""
            return y + 1

        calls.append(x)
        return inner(x)

    assert f(1) == 2

    @cached(cache)
    def f(x: int) -> int:  # noqa: F811 — intentional redef, nested docstring edited
        """Outer docstring, edited."""

        def inner(y: int) -> int:
            """Inner docstring, also edited."""
            return y + 1

        calls.append(x)
        return inner(x)

    assert f(1) == 2  # HIT — nested docstring edits don't change the key
    assert calls == [1]


def test_docstring_only_body_keys_like_pass(cache):
    """A body reduced to nothing by docstring-stripping keys identically to `pass`."""
    from emboss import cache_key

    @cached(cache)
    def f() -> None:
        """Only a docstring."""

    doc_only_key = cache_key(f)

    @cached(cache)
    def f() -> None:  # noqa: F811 — intentional redef
        pass

    assert cache_key(f) == doc_only_key


def test_doc_reading_function_keeps_docstring_sensitive_keying(cache):
    """A source reading `__doc__` implements behaviour with its docstring — edits invalidate."""
    from emboss import cache_keys

    calls = []

    @cached(cache)
    def prompt(x: int) -> str:
        """You are terse."""
        calls.append(x)
        return f"{prompt.__doc__}|{x}"

    assert prompt(1) == "You are terse.|1"
    _key, accept_keys = cache_keys(prompt, 1)
    assert accept_keys == []  # docstring stays in the hash → identities coincide, no fallback

    @cached(cache)
    def prompt(x: int) -> str:  # noqa: F811 — intentional redef with an edited docstring
        """You are verbose."""
        calls.append(x)
        return f"{prompt.__doc__}|{x}"

    assert prompt(1) == "You are verbose.|1"  # MISS — this docstring is load-bearing
    assert calls == [1, 1]
