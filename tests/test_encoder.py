"""Tests for `safe_jsonable_encoder` — the function that turns arbitrary args into JSON-stable cache keys."""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path

from pydantic import BaseModel

from emboss import safe_jsonable_encoder


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
