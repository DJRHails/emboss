"""End-to-end round-trip tests for emboss.cached."""

from __future__ import annotations

import diskcache
import pytest
from pydantic import BaseModel

from emboss import cached


class M(BaseModel):
    name: str
    n: int = 0


@pytest.fixture
def cache(tmp_path):
    c = diskcache.Cache(str(tmp_path / "cache"))
    yield c
    c.close()


def test_plain_dict_round_trip(cache):
    calls = {"n": 0}

    @cached(cache)
    def f(x: int) -> dict:
        calls["n"] += 1
        return {"value": x * 2}

    assert f(3) == {"value": 6}
    assert f(3) == {"value": 6}
    assert calls["n"] == 1, "second call should be cached"


def test_basemodel_round_trip(cache):
    calls = {"n": 0}

    @cached(cache)
    def f() -> M:
        calls["n"] += 1
        return M(name="solo", n=1)

    r1 = f()
    r2 = f()
    assert isinstance(r1, M) and isinstance(r2, M)
    assert r1 == r2
    assert calls["n"] == 1


def test_list_of_basemodel_round_trip(cache):
    calls = {"n": 0}

    @cached(cache)
    def f() -> list[M]:
        calls["n"] += 1
        return [M(name="a"), M(name="b", n=2)]

    assert f() == f()
    assert calls["n"] == 1
    # All elements are still pydantic models, not dicts
    assert all(isinstance(m, M) for m in f())


def test_dict_of_basemodel_round_trip(cache):
    calls = {"n": 0}

    @cached(cache)
    def f() -> dict[str, M]:
        calls["n"] += 1
        return {"x": M(name="x", n=9)}

    assert f() == f()
    assert calls["n"] == 1
    assert isinstance(f()["x"], M)


def test_optional_basemodel_none_caches(cache):
    calls = {"n": 0}

    @cached(cache)
    def f(x: int) -> M | None:
        calls["n"] += 1
        return None if x < 0 else M(name="opt", n=x)

    assert f(-1) is None
    assert f(-1) is None
    assert f(5).n == 5
    assert f(5).n == 5
    # 2 distinct keys (one for -1, one for 5), each computed once
    assert calls["n"] == 2


def test_none_return_caches(cache):
    """Pre-emboss behaviour skipped caching None; we want None cached too."""
    calls = {"n": 0}

    @cached(cache)
    def f(x: str) -> str | None:
        calls["n"] += 1
        return None

    f("any")
    f("any")
    assert calls["n"] == 1
