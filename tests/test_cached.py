"""End-to-end round-trip tests for emboss.cached."""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Mapping

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


def test_string_annotation_still_encodes_model(cache):
    """PEP 563 (`from __future__ import annotations`) turns the return annotation
    into a string. Detection must resolve it — otherwise the model pickles by
    class reference and the cached value dies with the defining module."""

    @cached(cache)
    def f() -> "M":  # noqa: UP037 — the string annotation IS the test subject
        return M(name="stringy", n=2)

    r1 = f()
    assert isinstance(r1, M) and r1.name == "stringy"
    # the STORED value is the model_dump dict, not a pickled M instance
    stored = cache[next(iter(cache))]
    assert isinstance(stored, dict) and stored["name"] == "stringy"
    assert isinstance(f(), M)  # decoded back to a model on a warm hit


def test_unresolvable_string_annotation_falls_back(cache, caplog):
    """A string annotation naming something not importable at decoration time
    (e.g. behind TYPE_CHECKING) must not raise — encoding stays off, with a
    warning naming the function (the values would die with their defining code)."""

    with caplog.at_level("WARNING"):

        @cached(cache)
        def f() -> "NotDefinedAnywhere":  # noqa: F821, UP037 — deliberately unresolvable
            return {"plain": True}

    assert any("did not resolve" in r.message for r in caplog.records)
    assert f() == {"plain": True}
    assert f() == {"plain": True}


def test_unresolvable_param_annotation_keeps_return_codec(cache):
    """One TYPE_CHECKING-gated *parameter* annotation must not disable the
    return codec — resolution is scoped to the return annotation alone."""

    @cached(cache)
    def f(client: "NotImportableAnywhere") -> M:  # noqa: F821, UP037 — the param IS the test
        return M(name="param-skew", n=1)

    assert isinstance(f(None), M)
    stored = _stored_values(cache)[0]
    assert isinstance(stored, dict) and stored["name"] == "param-skew"


def test_nested_forward_ref_without_future_import_encodes(cache):
    """In a module without PEP 563, `-> list["M"]` arrives holding a ForwardRef
    object (not a string); the codec must still resolve it rather than pickling
    the models by class reference."""
    from no_future_annotations_module import NestedRefModel, make_nested_ref_function

    g = make_nested_ref_function(cache)
    r = g()
    assert isinstance(r, list) and isinstance(r[0], NestedRefModel)
    stored = _stored_values(cache)[0]
    assert isinstance(stored[0], dict) and stored[0] == {"name": "nested"}
    assert isinstance(g()[0], NestedRefModel)  # warm hit rehydrates


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


@dataclasses.dataclass(frozen=True, kw_only=True)
class Score:
    value: float
    tags: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True, kw_only=True)
class Wrapped:
    inner: Score
    label: str


def _stored_values(cache):
    return [cache[k] for k in cache]


def test_dataclass_round_trip_stores_dict(cache):
    @cached(cache)
    def f() -> Score:
        return Score(value=0.5, tags=(1, 2))

    r = f()
    assert isinstance(r, Score) and r == Score(value=0.5, tags=(1, 2))
    stored = _stored_values(cache)[0]
    assert isinstance(stored, dict) and stored == {"value": 0.5, "tags": (1, 2)}
    assert isinstance(f(), Score)  # warm hit rehydrates


def test_nested_dataclass_fields_rehydrate(cache):
    @cached(cache)
    def f() -> Wrapped:
        return Wrapped(inner=Score(value=1.0), label="w")

    assert f() == f() == Wrapped(inner=Score(value=1.0), label="w")
    assert isinstance(f().inner, Score)
    stored = _stored_values(cache)[0]
    assert stored == {"inner": {"value": 1.0, "tags": ()}, "label": "w"}


def test_variadic_tuple_of_models_round_trip(cache):
    @cached(cache)
    def f() -> tuple[M, ...]:
        return (M(name="a"), M(name="b"))

    r = f()
    assert isinstance(r, tuple) and all(isinstance(m, M) for m in r)
    assert [m.name for m in f()] == ["a", "b"]
    stored = _stored_values(cache)[0]
    assert all(isinstance(v, dict) for v in stored)


def test_fixed_tuple_mixed_elements_round_trip(cache):
    @cached(cache)
    def f() -> tuple[M, int]:
        return (M(name="pair", n=1), 7)

    m, n = f()
    assert isinstance(m, M) and n == 7
    stored = _stored_values(cache)[0]
    assert isinstance(stored[0], dict) and stored[1] == 7


def test_list_of_dataclass_round_trip(cache):
    @cached(cache)
    def f() -> list[Score]:
        return [Score(value=2.0)]

    assert f() == [Score(value=2.0)]
    assert isinstance(f()[0], Score)


def test_optional_dataclass_round_trip(cache):
    @cached(cache)
    def f(x: int) -> Score | None:
        return Score(value=float(x)) if x else None

    assert f(0) is None
    assert f(3) == Score(value=3.0) and isinstance(f(3), Score)


@dataclasses.dataclass
class NotRebuildable:
    x: int
    derived: int = dataclasses.field(init=False, default=0)


def test_non_rebuildable_dataclass_keeps_raw_pickle(cache):
    """A dataclass whose field dict can't drive its constructor must NOT be
    dict-encoded — the warm hit would hand back a dict where the caller expects
    the dataclass. It stays on the raw-pickle passthrough."""

    @cached(cache)
    def f() -> NotRebuildable:
        return NotRebuildable(x=1)

    assert isinstance(f(), NotRebuildable)
    assert isinstance(f(), NotRebuildable)  # warm hit too
    assert isinstance(_stored_values(cache)[0], NotRebuildable)


def test_schema_drift_recomputes_instead_of_returning_dict(cache, caplog):
    """A stored field dict that no longer matches the dataclass constructor
    (fields renamed/removed since it was cached) is a warned miss that
    recomputes — never a raw dict handed to a caller expecting the dataclass."""

    @cached(cache)
    def f() -> Score:
        return Score(value=1.0)

    f()
    key = next(iter(cache))
    cache.set(key, {"value": 1.0, "removed_field": 3})  # simulate an old schema

    with caplog.at_level("WARNING"):
        r = f()
    assert isinstance(r, Score) and r == Score(value=1.0)
    assert any("treating as a miss" in rec.message for rec in caplog.records)
    assert _stored_values(cache)[0] == {"value": 1.0, "tags": ()}  # store healed
    assert isinstance(f(), Score)


def test_model_schema_drift_recomputes_instead_of_crashing(cache, caplog):
    """A stored field dict that no longer validates against the BaseModel (a
    required field added/renamed since it was cached) is a warned miss that
    recomputes — never a ValidationError leaking out of a warm hit. Same
    contract as the dataclass drift path."""

    @cached(cache)
    def f() -> M:
        return M(name="fresh", n=1)

    f()
    key = next(iter(cache))
    cache.set(key, {"renamed_away": True})  # predates M's required `name`

    with caplog.at_level("WARNING"):
        r = f()
    assert r == M(name="fresh", n=1)
    assert any("treating as a miss" in rec.message for rec in caplog.records)
    assert isinstance(f(), M)  # store healed


def test_schema_drift_warns_once_per_key(cache, caplog):
    """Drift warnings dedupe per key per decorated function — under cache_only
    or a read-only cache nothing heals the store, and one stale sweep must not
    warn on every call."""

    @cached(cache)
    def f() -> Score:
        return Score(value=1.0)

    f()
    key = next(iter(cache))
    with caplog.at_level("WARNING"):
        for _ in range(2):
            cache.set(key, {"value": 1.0, "removed_field": 3})
            assert isinstance(f(), Score)  # recomputes both times
    assert sum("treating as a miss" in r.message for r in caplog.records) == 1


@dataclasses.dataclass(frozen=True, kw_only=True)
class Refusal:
    error: str


class WrappedDict(dict):
    """Module-level so raw-pickled instances round-trip by class reference."""


def test_ambiguous_union_stays_raw_pickle(cache):
    """`A | B` with two codable members can't be told apart once dict-encoded —
    the decode of a stored dict couldn't pick the member. Such unions must stay
    on the raw-pickle passthrough (the pre-codec behaviour), instances intact."""

    @cached(cache)
    def f(ok: bool) -> Score | Refusal:
        return Score(value=1.0) if ok else Refusal(error="nope")

    assert isinstance(f(True), Score) and isinstance(f(False), Refusal)
    assert isinstance(f(True), Score) and isinstance(f(False), Refusal)  # warm hits
    assert all(isinstance(v, (Score, Refusal)) for v in _stored_values(cache))


def test_list_of_ambiguous_union_stays_raw_pickle(cache):
    @cached(cache)
    def f() -> list[Score | Refusal]:
        return [Score(value=2.0), Refusal(error="mid"), Score(value=3.0)]

    cold = f()
    warm = f()
    assert warm == cold
    assert isinstance(warm[1], Refusal) and isinstance(warm[2], Score)


def test_union_of_class_and_its_list_disambiguates_by_shape(cache):
    """`A | list[A]` IS unambiguous — a dict is the class, a list is the list —
    so both shapes encode and rehydrate."""

    @cached(cache)
    def f(many: bool) -> Score | list[Score]:
        return [Score(value=4.0)] if many else Score(value=5.0)

    assert f(False) == Score(value=5.0) and isinstance(f(False), Score)
    assert f(True) == [Score(value=4.0)] and isinstance(f(True)[0], Score)
    stored = _stored_values(cache)
    assert all(isinstance(v, (dict, list)) for v in stored)  # dict-encoded, no instances


def test_union_with_dict_member_stays_raw_pickle(cache):
    """`A | dict[...]` is dict-shape ambiguous: the class member itself stores as
    a dict, so a stored dict could be either member — decode would rebuild a
    plain-dict return as the class on the warm hit. Such unions must stay on the
    raw-pickle passthrough."""

    @cached(cache)
    def f(ok: bool) -> Refusal | dict[str, str]:
        return Refusal(error="bad") if ok else {"error": "just a dict"}

    assert isinstance(f(True), Refusal) and isinstance(f(True), Refusal)  # cold + warm
    assert f(False) == {"error": "just a dict"}
    assert isinstance(f(False), dict)  # warm hit: still the dict, not a Refusal


def test_union_with_typeddict_member_stays_raw_pickle(cache):
    """A TypedDict member is a plain dict at runtime, so it collides with a
    class member's dict encoding just like `dict[...]` does."""

    class ScoreTD(typing.TypedDict):
        value: float

    @cached(cache)
    def f(ok: bool) -> Score | ScoreTD:
        return Score(value=1.0) if ok else {"value": 2.0}

    assert isinstance(f(True), Score) and isinstance(f(True), Score)
    assert f(False) == {"value": 2.0}
    assert isinstance(f(False), dict) and not isinstance(f(False), Score)


def test_union_with_mapping_member_stays_raw_pickle(cache):
    """`Mapping[...]` members are dict-shaped at runtime even though their
    origin is `collections.abc.Mapping`, not `dict` — a stored dict could be
    either member, so the union must stay on the raw-pickle passthrough."""

    @cached(cache)
    def f(ok: bool) -> Refusal | Mapping[str, str]:
        return Refusal(error="bad") if ok else {"error": "just a dict"}

    assert isinstance(f(True), Refusal) and isinstance(f(True), Refusal)
    assert f(False) == {"error": "just a dict"}
    assert isinstance(f(False), dict) and not isinstance(f(False), Refusal)


def test_union_with_dict_subclass_member_stays_raw_pickle(cache):
    """A `dict` subclass member is claimed by `isinstance(value, dict)` on
    decode just like a plain dict, so it collides with a class member."""

    @cached(cache)
    def f(ok: bool) -> Refusal | WrappedDict:
        return Refusal(error="bad") if ok else WrappedDict({"error": "just a dict"})

    assert isinstance(f(True), Refusal) and isinstance(f(True), Refusal)
    assert f(False) == {"error": "just a dict"}
    assert not isinstance(f(False), Refusal)


def test_union_with_any_member_stays_raw_pickle(cache):
    """An `Any` member can hold a dict at runtime, so it collides with a class
    member's dict encoding — the union stays on the raw-pickle passthrough."""

    @cached(cache)
    def f(ok: bool) -> Refusal | typing.Any:
        return Refusal(error="bad") if ok else {"error": "just a dict"}

    assert isinstance(f(True), Refusal) and isinstance(f(True), Refusal)
    assert f(False) == {"error": "just a dict"}
    assert isinstance(f(False), dict) and not isinstance(f(False), Refusal)


def test_value_not_matching_annotation_passes_through(cache):
    @cached(cache)
    def f() -> Score:
        return {"already": "a dict"}  # type: ignore[return-value] — deliberate lie

    assert f() == {"already": "a dict"}
    assert f() == {"already": "a dict"}


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
