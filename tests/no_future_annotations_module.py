"""Support module for annotation-resolution tests — deliberately NO
`from __future__ import annotations`, so a nested quoted name in a return
annotation arrives as `list[ForwardRef('NestedRefModel')]` (an object, not a
string) and exercises the non-PEP-563 resolution path. Not a test module."""

from pydantic import BaseModel

from emboss import cached


class NestedRefModel(BaseModel):
    name: str


def make_nested_ref_function(cache):
    @cached(cache)
    def g() -> list["NestedRefModel"]:
        return [NestedRefModel(name="nested")]

    return g
