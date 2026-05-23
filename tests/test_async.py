"""Tests for async-function caching."""

from __future__ import annotations

import asyncio

import diskcache
import pytest

from emboss import cached


@pytest.fixture
def cache(tmp_path):
    c = diskcache.Cache(str(tmp_path / "cache"))
    yield c
    c.close()


def test_async_round_trip(cache):
    calls = {"n": 0}

    @cached(cache)
    async def f(x: int) -> int:
        calls["n"] += 1
        await asyncio.sleep(0)  # yield
        return x * 2

    async def run():
        a = await f(3)
        b = await f(3)
        return a, b

    a, b = asyncio.run(run())
    assert a == b == 6
    assert calls["n"] == 1


def test_async_none_caches(cache):
    calls = {"n": 0}

    @cached(cache)
    async def f(x: int) -> int | None:
        calls["n"] += 1
        return None if x < 0 else x

    async def run():
        return [await f(-1), await f(-1), await f(5), await f(5)]

    results = asyncio.run(run())
    assert results == [None, None, 5, 5]
    assert calls["n"] == 2  # one per distinct key, even for None
