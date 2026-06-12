"""End-to-end round-trip tests for emboss.cached."""

from __future__ import annotations

import logging

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


def test_env_var_sets_default_cache_dir(tmp_path, monkeypatch):
    """With no explicit cache, EMBOSS_CACHE_DIR controls where the cache lands."""
    cache_dir = tmp_path / "env-cache"
    monkeypatch.setenv("EMBOSS_CACHE_DIR", str(cache_dir))
    calls = {"n": 0}

    @cached()
    def f(x: int) -> dict:
        calls["n"] += 1
        return {"value": x + 1}

    assert f(1) == {"value": 2}
    assert f(1) == {"value": 2}
    assert calls["n"] == 1, "second call should hit the cache"
    assert (cache_dir / "cache.db").exists(), "cache should live at EMBOSS_CACHE_DIR"


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


def test_unresolvable_annotation_still_caches(cache):
    """`typing.get_type_hints` raises NameError on an unresolvable forward ref
    (e.g. a TYPE_CHECKING-only type); decoration must degrade to pass-through
    encoding, not crash."""
    calls = {"n": 0}

    @cached(cache)
    # The undefined name is the point of the test — see docstring.
    def f(x: int) -> "NoSuchType":  # ty: ignore[unresolved-reference]  # noqa: F821
        calls["n"] += 1
        return {"v": x}

    assert f(2) == {"v": 2}
    assert f(2) == {"v": 2}
    assert calls["n"] == 1


def test_pep563_model_encoding_stores_plain_dict(cache):
    """Under `from __future__ import annotations` (active in this module) the
    return annotation is a string; model detection must still engage and store
    a plain dict — a pickled model defined in `__main__` would fail to
    unpickle in a later process."""

    @cached(cache)
    def f(x: int) -> M:
        return M(name="p", n=x)

    assert f(1) == M(name="p", n=1)
    (key,) = list(cache)
    raw = cache.get(key)
    assert type(raw) is dict and raw == {"name": "p", "n": 1}


def test_unparseable_source_falls_back_with_warning(cache, caplog):
    """A lambda extracted mid-expression has no standalone-parseable source:
    keying falls back to the raw source (whitespace-sensitive) and says so at
    WARNING level."""
    with caplog.at_level(logging.WARNING, logger="emboss._cached"):
        # The dict literal is load-bearing: getsource() then yields a lone
        # `"double": ...` dict-item line, which cannot parse standalone.
        fns = {
            "double": cached(cache)(lambda x: x * 2),
        }

    assert "could not parse function source" in caplog.text
    assert fns["double"](3) == 6
    assert fns["double"](3) == 6  # still caches on the fallback key
