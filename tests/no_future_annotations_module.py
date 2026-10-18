"""Support module for annotation-resolution tests — deliberately NO
`from __future__ import annotations`, so a nested quoted name in a return
annotation arrives as `list['NestedRefModel']` (a builtin generic holding the
bare string, which `typing.get_type_hints` must turn into the class) and
exercises the non-PEP-563 resolution path. Not a test module."""

import typing

from pydantic import BaseModel

from emboss import cached


class NestedRefModel(BaseModel):
    name: str


def make_nested_ref_function(cache):
    @cached(cache)
    def g() -> list["NestedRefModel"]:
        return [NestedRefModel(name="nested")]

    return g


def make_optional_nested_ref_function(cache):
    @cached(cache)
    def h() -> list["NestedRefModel"] | None:
        return [NestedRefModel(name="optional")]

    return h


def make_typing_wrapped_nested_ref_function(cache):
    @cached(cache)
    def k() -> typing.Dict[str, list["NestedRefModel"]]:  # noqa: UP006 — typing alias IS the subject
        return {"a": [NestedRefModel(name="typing-wrapped")]}

    return k
