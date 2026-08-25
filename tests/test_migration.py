"""Tests for cache identity (`cache_id`), `also_accept` migration, and `unsafe_manual_key`."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re

import diskcache
import pytest

from emboss import cache_id, cached


@pytest.fixture
def cache(tmp_path):
    c = diskcache.Cache(str(tmp_path / "cache"))
    yield c
    c.close()


def _key_for(identity: str, args: list) -> str:
    """Reconstruct the on-disk cache key for a `"name:body_hash"` identity + args."""
    name, _, body_hash = identity.partition(":")
    arg_hash = hashlib.md5((json.dumps(args) + json.dumps({})).encode()).hexdigest()
    return hashlib.md5(f"{name}{body_hash}{arg_hash}".encode()).hexdigest()


def test_cache_id_shape(cache):
    @cached(cache)
    def f(x: int) -> int:
        return x + 1

    cid = cache_id(f)
    assert re.fullmatch(r"f:[0-9a-f]{32}", cid)
    assert f.__emboss__.cache_id == cid
    assert f.__emboss__.name == "f"
    assert cid == f"f:{f.__emboss__.body_hash}"
    assert f.__emboss__.also_accept == ()


def test_cache_id_rejects_unwrapped_function():
    def plain(x: int) -> int:
        return x

    with pytest.raises(TypeError, match="not an @cached-wrapped function"):
        cache_id(plain)


def test_also_accept_migrates_renamed_function(cache):
    calls = {"n": 0}

    @cached(cache)
    def fetch(x: int) -> int:
        calls["n"] += 1
        return x * 7

    assert fetch(3) == 21
    assert calls["n"] == 1
    old = cache_id(fetch)

    @cached(cache, also_accept=[old])
    def fetch_v2(x: int) -> int:
        calls["n"] += 1
        return x * 7  # same behaviour — the old entry should be reused, not recomputed

    assert fetch_v2(3) == 21  # served from fetch's entry via the fallback key
    assert calls["n"] == 1

    # Write-through: drop the OLD entry — the value must now live under
    # fetch_v2's own key, so the next call still never runs the body.
    cache.delete(_key_for(old, [3]))
    assert fetch_v2(3) == 21
    assert calls["n"] == 1


def test_also_accept_different_args_still_miss(cache):
    calls = {"n": 0}

    @cached(cache)
    def f(x: int) -> int:
        calls["n"] += 1
        return x * 2

    assert f(1) == 2
    old = cache_id(f)

    @cached(cache, also_accept=[old])
    def g(x: int) -> int:
        calls["n"] += 1
        return x * 2

    assert (
        g(2) == 4
    )  # unseen args — neither the current nor the fallback key holds them
    assert calls["n"] == 2
    assert g(1) == 2  # the old args still migrate
    assert calls["n"] == 2


def test_also_accept_rejects_malformed_tokens(cache):
    for bad in ["no-colon-here", "name:", ":hash"]:
        with pytest.raises(ValueError, match=re.escape(repr(bad))):

            @cached(cache, also_accept=[bad])
            def f() -> int:
                return 1


def test_store_writes_exactly_one_key(cache):
    """No legacy twin entry: one call stores one cache entry."""

    @cached(cache)
    def f(x: int) -> int:
        return x + 1

    assert f(1) == 2
    assert len(list(cache)) == 1


def test_unsafe_manual_key_survives_body_edit(cache):
    calls = {"n": 0}

    @cached(cache, unsafe_manual_key="v1")
    def f(x: int) -> int:
        calls["n"] += 1
        return x + 100

    assert f(1) == 101
    assert calls["n"] == 1
    assert cache_id(f) == "f:v1"

    @cached(cache, unsafe_manual_key="v1")
    def f(x: int) -> int:  # noqa: F811 — intentional redef with an edited body
        calls["n"] += 1
        return x + 200  # changed constant — would re-key under source hashing

    assert f(1) == 101  # HIT: the manual key pins identity, stale-by-design
    assert calls["n"] == 1

    @cached(cache, unsafe_manual_key="v2")
    def f(x: int) -> int:  # noqa: F811 — intentional redef with a bumped key
        calls["n"] += 1
        return x + 200

    assert f(1) == 201  # bumping the manual key recomputes
    assert calls["n"] == 2


def test_unsafe_manual_key_rejects_empty_string(cache):
    with pytest.raises(ValueError, match="non-empty"):
        cached(cache, unsafe_manual_key="")


def test_also_accept_works_with_unsafe_manual_key(cache):
    calls = {"n": 0}

    @cached(cache)
    def f(x: int) -> int:
        calls["n"] += 1
        return x - 1

    assert f(9) == 8
    old = cache_id(f)

    @cached(cache, unsafe_manual_key="m1", also_accept=[old])
    def g(x: int) -> int:
        calls["n"] += 1
        return x - 1

    assert cache_id(g) == "g:m1"
    assert g(9) == 8  # migrated from f's source-keyed entry into the manual identity
    assert calls["n"] == 1


def test_async_also_accept_and_manual_key(cache):
    calls = {"n": 0}

    @cached(cache)
    async def f(x: int) -> int:
        calls["n"] += 1
        return x * 3

    assert asyncio.run(f(2)) == 6
    old = cache_id(f)

    @cached(cache, also_accept=[old])
    async def g(x: int) -> int:
        calls["n"] += 1
        return x * 3

    assert asyncio.run(g(2)) == 6  # migrated, not recomputed
    assert calls["n"] == 1

    @cached(cache, unsafe_manual_key="av1")
    async def h(x: int) -> int:
        calls["n"] += 1
        return x * 5

    assert asyncio.run(h(2)) == 10
    assert calls["n"] == 2

    @cached(cache, unsafe_manual_key="av1")
    async def h(x: int) -> int:  # noqa: F811 — intentional redef with an edited body
        calls["n"] += 1
        return x * 50

    assert asyncio.run(h(2)) == 10  # HIT under the pinned key despite the body edit
    assert calls["n"] == 2


def test_pre_docless_entries_migrate_forward(cache):
    """Entries keyed under the pre-0.12 docstring-sensitive identity still hit and migrate."""
    from emboss import cache_keys

    calls = []

    @cached(cache)
    def f(x: int) -> int:
        """A docstring, so the docful identity differs from the primary one."""
        calls.append(x)
        return x * 2

    key, accept_keys = cache_keys(f, 5)
    assert len(accept_keys) == 1  # exactly the implicit docstring-sensitive fallback
    cache.set(accept_keys[0], 10)  # simulate an entry written by emboss < 0.12

    assert f(5) == 10
    assert calls == []  # served from the fallback key, body never ran
    assert cache.get(key) == 10  # and migrated forward to the current key


def test_docful_fallback_key_matches_pre_0_12_formula(cache):
    """The implicit fallback key is byte-identical to what emboss < 0.12 wrote."""
    import ast
    import inspect
    import textwrap

    from emboss import cache_keys

    @cached(cache)
    def g(x: int) -> int:
        """Docstring that participates in the pre-0.12 hash."""
        return x + 3

    # Reproduce the pre-0.12 keying exactly: AST round-trip WITHOUT docstring stripping.
    raw_source = inspect.getsource(g.__wrapped__)
    docful_canon = ast.unparse(ast.parse(textwrap.dedent(raw_source)))
    docful_hash = hashlib.md5(docful_canon.encode()).hexdigest()
    arg_hash = hashlib.md5((json.dumps([4]) + json.dumps({})).encode()).hexdigest()
    pre_0_12_key = hashlib.md5(f"g{docful_hash}{arg_hash}".encode()).hexdigest()

    _key, accept_keys = cache_keys(g, 4)
    assert accept_keys == [pre_0_12_key]


def test_undocumented_function_has_no_implicit_fallback(cache):
    """No docstring → docful and docless identities coincide → no extra fallback key."""
    from emboss import cache_keys

    @cached(cache)
    def f(x: int) -> int:
        return x - 1

    _key, accept_keys = cache_keys(f, 1)
    assert accept_keys == []


def test_async_docstring_edit_keeps_hit_and_implicit_fallback(cache):
    """The `ast.AsyncFunctionDef` branch: async docstrings strip and get the fallback too."""
    from emboss import cache_keys

    calls = []

    @cached(cache)
    async def f(x: int) -> int:
        """Original async docstring."""
        calls.append(x)
        return x * 3

    assert asyncio.run(f(2)) == 6
    _key, accept_keys = cache_keys(f, 2)
    assert len(accept_keys) == 1  # exactly the implicit docstring-sensitive fallback

    @cached(cache)
    async def f(x: int) -> int:  # noqa: F811 — intentional redef with an edited docstring
        """Rewritten async docstring."""
        calls.append(x)
        return x * 3

    assert asyncio.run(f(2)) == 6  # HIT — the docstring edit doesn't re-key
    assert calls == [2]


def test_explicit_and_implicit_fallbacks_coexist(cache):
    """A documented function carries the explicit `also_accept` tokens AND the implicit fallback."""
    from emboss import cache_keys

    calls = []

    @cached(cache)
    def old_name(x: int) -> int:
        """Docstring shared by both definitions."""
        calls.append(x)
        return x + 7

    old_identity = cache_id(old_name)
    assert old_name(1) == 8

    @cached(cache, also_accept=[old_identity])
    def new_name(x: int) -> int:
        """Docstring shared by both definitions."""
        calls.append(x)
        return x + 7

    key, accept_keys = cache_keys(new_name, 1)
    assert len(accept_keys) == 2  # explicit token first, implicit pre-0.12 fallback after it
    assert accept_keys[0] == _key_for(old_identity, [1])
    assert new_name(1) == 8  # rename migration hits via the explicit token
    assert calls == [1]  # body never re-ran
    assert cache.get(key) == 8  # and the hit migrated forward to the current key


def test_unsafe_manual_key_with_docstring_has_no_implicit_fallback(cache):
    """A manual key pins the identity completely — no source-derived fallback sneaks in."""
    from emboss import cache_keys

    @cached(cache, unsafe_manual_key="pinned-v1")
    def f(x: int) -> int:
        """Docstring that must not create a fallback identity."""
        return x

    _key, accept_keys = cache_keys(f, 1)
    assert accept_keys == []


def test_fallback_hit_survives_migration_write_failure(cache, caplog):
    """The migration write-through is best-effort — a failing `set` must not eat the hit."""
    from emboss import cache_keys

    class ReadOnlyCache:
        def __init__(self, inner):
            self._inner = inner

        def get(self, key, default=None):
            return self._inner.get(key, default=default)

        def set(self, key, value):
            raise RuntimeError("attempt to write a readonly database")

    calls = []

    @cached(ReadOnlyCache(cache))
    def f(x: int) -> int:
        """Docstring so the implicit fallback exists."""
        calls.append(x)
        return x * 2

    _key, accept_keys = cache_keys(f, 5)
    cache.set(accept_keys[0], 10)  # plant a pre-0.12 entry, bypassing the read-only wrapper

    with caplog.at_level("WARNING", logger="emboss._cached"):
        assert f(5) == 10  # served from the fallback despite the failed migration write
    assert calls == []  # body never ran
    assert any("failed" in record.getMessage() for record in caplog.records)
