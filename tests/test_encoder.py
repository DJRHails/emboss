"""Tests for `safe_jsonable_encoder` — the function that turns arbitrary args into JSON-stable cache keys."""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path

import pytest
from pydantic import BaseModel

from emboss import cached, safe_jsonable_encoder


def test_primitives_pass_through():
    assert safe_jsonable_encoder(None) is None
    assert safe_jsonable_encoder(True) is True
    assert safe_jsonable_encoder(42) == 42
    assert safe_jsonable_encoder(3.14) == 3.14
    assert safe_jsonable_encoder("hi") == "hi"


def test_collections_recurse():
    assert safe_jsonable_encoder([1, 2, 3]) == [1, 2, 3]
    assert safe_jsonable_encoder((1, 2)) == [1, 2]  # tuple → list
    assert safe_jsonable_encoder({"a": 1}) == {"a": 1}
    assert safe_jsonable_encoder({3, 1, 2}) == [1, 2, 3]  # set → sorted list


def test_bytes_to_str():
    assert safe_jsonable_encoder(b"hello") == "hello"


def test_dates_to_isoformat():
    assert safe_jsonable_encoder(date(2026, 5, 22)) == "2026-05-22"
    assert safe_jsonable_encoder(datetime(2026, 5, 22, 14, 30)) == "2026-05-22T14:30:00"
    assert safe_jsonable_encoder(time(14, 30)) == "14:30:00"


def test_path_to_str():
    assert safe_jsonable_encoder(Path("/tmp/foo")) == "/tmp/foo"


def test_basemodel_dumps():
    class M(BaseModel):
        name: str
        n: int

    assert safe_jsonable_encoder(M(name="x", n=1)) == {"name": "x", "n": 1}


def test_nested_dict_with_non_str_keys():
    # Keys are stringified
    assert safe_jsonable_encoder({1: "a", 2: "b"}) == {"1": "a", "2": "b"}


def test_object_with_dunder_dict():
    class Bag:
        def __init__(self):
            self.x = 1
            self.y = [2, 3]

    encoded = safe_jsonable_encoder(Bag())
    assert encoded == {"x": 1, "y": [2, 3]}


# --- `default=` callable (json-style fallback for unhandled types) ---


class _NoDictNoRepr:
    """An object without `__dict__` (via __slots__) and without a stable __repr__.

    Stand-in for the real-world hazard the strict mode guards against: objects
    whose default `str(obj)` is `<X at 0x7f...>` would silently change every
    process invocation, busting the cache key.
    """

    __slots__ = ()


def test_default_str_is_package_default_and_preserves_loose_behaviour():
    """Calling without `default=` falls back to str — pre-0.2 behaviour."""
    obj = _NoDictNoRepr()
    encoded = safe_jsonable_encoder(obj)
    assert encoded == str(obj)
    assert encoded == safe_jsonable_encoder(obj, default=str)


def test_default_none_raises_on_unknown():
    """`default=None` mirrors `json.dumps(default=None)`: strict mode."""
    with pytest.raises(TypeError, match=r"cannot encode '?_NoDictNoRepr'?"):
        safe_jsonable_encoder(_NoDictNoRepr(), default=None)


def test_default_callable_invoked_on_unknown():
    sentinel_calls = []

    def my_default(obj):
        sentinel_calls.append(obj)
        return "custom!"

    assert safe_jsonable_encoder(_NoDictNoRepr(), default=my_default) == "custom!"
    assert len(sentinel_calls) == 1


def test_default_threads_through_nested_collections():
    """A list containing an unknown type should hit `default=` recursively."""
    obj = _NoDictNoRepr()
    encoded = safe_jsonable_encoder([1, obj, 2], default=lambda _: "<custom>")
    assert encoded == [1, "<custom>", 2]


def test_default_threads_through_dict_values():
    obj = _NoDictNoRepr()
    encoded = safe_jsonable_encoder({"a": obj}, default=lambda _: "<custom>")
    assert encoded == {"a": "<custom>"}


def test_default_does_not_override_builtin_handlers():
    """`default=` only fires for types no built-in handler matched."""
    called = []

    def my_default(obj):
        called.append(obj)
        return "would-replace"

    # str → handled, default never called
    assert safe_jsonable_encoder("hello", default=my_default) == "hello"
    # dict → handled, default never called
    assert safe_jsonable_encoder({"k": 1}, default=my_default) == {"k": 1}
    # BaseModel → handled, default never called
    class M(BaseModel):
        x: int

    assert safe_jsonable_encoder(M(x=1), default=my_default) == {"x": 1}
    assert called == []


def test_cached_threads_default_into_encoder(tmp_path):
    """`cached(..., default=...)` should propagate the encoder choice."""
    import diskcache

    cache = diskcache.Cache(str(tmp_path / "cache"))
    try:

        @cached(cache, default=None)  # strict
        def f(obj) -> str:
            return "ran"

        with pytest.raises(TypeError, match="cannot encode"):
            f(_NoDictNoRepr())  # raises during key construction
    finally:
        cache.close()
