"""Internal: `@cached` decorator implementation. Public API lives in `emboss.__init__`."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import contextvars
import dataclasses
import functools
import hashlib
import inspect
import json
import logging
import os
import tempfile
import textwrap
import threading
import types
import typing
import weakref
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar, Union

from emboss._protocol import Cache
from emboss._sqlite_cache import SqliteCache

try:
    from pydantic import BaseModel
except ImportError:  # pragma: no cover — pydantic is optional for callers
    BaseModel = None  # type: ignore[assignment]

T = TypeVar("T")
logger = logging.getLogger(__name__)

# Sentinel for "key absent from cache" — lets None be a valid cached value.
_MISSING = object()

# Cache-only override for the current thread / async task. `None` means "no
# programmatic override — defer to the EMBOSS_CACHE_ONLY env var"; True/False
# is an explicit `cache_only(enabled=...)` block. A ContextVar gives the
# thread-local scoping `cache_only()` promises and additionally isolates
# concurrent asyncio tasks on one loop.
_CACHE_ONLY: contextvars.ContextVar[bool | None] = contextvars.ContextVar(
    "emboss_cache_only", default=None
)

_ENV_TRUTHY = frozenset({"1", "true", "yes"})
_ENV_FALSY = frozenset({"", "0", "false", "no", "off"})


def _cache_only_active() -> bool:
    """Whether cache-only mode is on right now.

    A `cache_only(enabled=...)` block wins; otherwise the `EMBOSS_CACHE_ONLY`
    environment variable decides ("1"/"true"/"yes" enable, unset/""/"0"/
    "false"/"no"/"off" disable, case-insensitively). Any other value raises
    `ValueError` rather than silently disabling the seal — a typo here would
    otherwise void the very guarantee the variable exists to enforce. The env
    var is read live on every call — not snapshotted at import — so it can be
    set or cleared mid-process (e.g. by a test harness).
    """
    override = _CACHE_ONLY.get()
    if override is not None:
        return override
    raw = os.environ.get("EMBOSS_CACHE_ONLY", "")
    value = raw.strip().lower()
    if value in _ENV_TRUTHY:
        return True
    if value in _ENV_FALSY:
        return False
    raise ValueError(
        f"EMBOSS_CACHE_ONLY={raw!r} is not a recognized value — use '1'/'true'/'yes' "
        "to enable cache-only mode, or unset it (or '0'/'false'/'no'/'off') to disable. "
        "Refusing to guess: a silently-ignored typo would disable the cache-only "
        "guarantee while you believe it is enforced."
    )


@contextlib.contextmanager
def cache_only(enabled: bool = True) -> Iterator[None]:
    """Scope cache-only mode to a block: genuine cache misses raise `CacheMiss`.

    Inside the block, a `@cached` call that cannot be served from its cache —
    neither the current key nor any `also_accept` fallback — raises `CacheMiss`
    instead of executing the wrapped function. Cached values (including a
    cached `None` or a stored negative/known-miss result) return normally, and
    `also_accept` hits still migrate forward to the current key.

    For async functions the mode is checked when the coroutine RUNS, not when
    it is created: `await`ing a missed call inside the block raises, and a task
    started with `asyncio.create_task` inside the block stays sealed (task
    creation copies the context) — but a bare coroutine created inside the
    block and only awaited after it exits executes normally.

    The scope is the calling thread's current context (isolated per async task
    on one loop). Worker threads do NOT inherit it: `threading.Thread` and
    `ThreadPoolExecutor` workers start from a fresh context and fall back to
    the env var, so seal a multi-threaded run with `EMBOSS_CACHE_ONLY=1`
    instead (`asyncio.to_thread` is safe — it copies the caller's context).

    `cache_only(enabled=False)` force-DISABLES the mode within the block, even
    when the `EMBOSS_CACHE_ONLY` env var turned it on process-wide — e.g. to
    carve out one deliberately-recomputed call from an otherwise sealed run.

    The prior state is restored on exit even when the block raises. Use it to
    prove a run performs zero uncached (external, paid) calls::

        with emboss.cache_only():
            run_all_monitors()  # any uncached call raises CacheMiss
    """
    token = _CACHE_ONLY.set(enabled)
    try:
        yield
    finally:
        _CACHE_ONLY.reset(token)


def _short_hash(value: str) -> str:
    """First 8 chars of a hash, with an ellipsis only when something was cut."""
    return value if len(value) <= 8 else f"{value[:8]}…"


class CacheMiss(RuntimeError):
    """A `@cached` call missed its cache while cache-only mode was active.

    `RuntimeError`, not `LookupError`: the caller performed a function call,
    not a lookup, and `except LookupError` fallback paths in callers would
    silently swallow exactly the "refused to execute" signal this exists to
    deliver.

    Attributes:
        func_name: `__name__` of the wrapped function.
        cache_id: the function's cache identity (`"name:body_hash"`).
        key: the storage key computed for this call's arguments.

    The message deliberately carries only the key and identity — never the
    call's arguments, which may be huge or contain secrets.
    """

    def __init__(self, *, func_name: str, cache_id: str, key: str) -> None:
        self.func_name = func_name
        self.cache_id = cache_id
        self.key = key
        name, _, body_hash = cache_id.partition(":")
        super().__init__(
            f"cache-only mode: no cached entry for {func_name!r} "
            f"(identity {name}:{_short_hash(body_hash)}, key {_short_hash(key)}); "
            "refusing to execute"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        # BaseException's default __reduce__ reconstructs via `cls(*self.args)`
        # — the formatted message, positionally — which this keyword-only
        # __init__ rejects. Without this override, pickling or copying a
        # CacheMiss (e.g. one crossing a multiprocessing boundary in a sealed
        # sweep) dies with an unrelated TypeError that masks the real miss.
        return (_rebuild_cache_miss, (self.func_name, self.cache_id, self.key))


def _rebuild_cache_miss(func_name: str, cache_id: str, key: str) -> CacheMiss:
    """Reconstruct a `CacheMiss` from pickle/copy (see `CacheMiss.__reduce__`)."""
    return CacheMiss(func_name=func_name, cache_id=cache_id, key=key)


class _InflightLatch:
    """Per-key compute-once latch for the threads of ONE process.

    A `@cached` miss holds its key's latch while it computes and stores, so a
    concurrent caller of the same key waits and then re-reads the cache instead
    of computing its own copy. Without it a fan-out that dispatches one key
    twice (a duplicated item, two arms sharing a cell) computes it twice and the
    later `set` supersedes the earlier — harmless for a deterministic function,
    but for a non-deterministic one (an LLM draw with no API seed) the value the
    first caller returned is silently replaced under every later reader (the
    touchstone keep-latest drift census measured 92% of its 43,852 conflicting
    cells as exactly this same-writer race, superseded within a minute).

    Locks are refcounted and dropped when the last holder leaves, so the table
    is bounded by the number of keys in flight, not the number ever seen. The
    per-key lock is re-entrant: a cached function that (mistakenly) recurses
    into its own key on the same thread hits the same `RecursionError` it did
    before, not a deadlock. Cross-process callers are NOT latched — that window
    is closed (best-effort) by the re-check before store in the wrapper.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, tuple[threading.RLock, int]] = {}

    @contextlib.contextmanager
    def hold(self, key: str) -> Iterator[None]:
        with self._guard:
            lock, refs = self._locks.get(key) or (threading.RLock(), 0)
            self._locks[key] = (lock, refs + 1)
        try:
            # The acquire sits inside the refcount's `finally`: an interrupted
            # wait (KeyboardInterrupt on the main thread) must drop its ref, or
            # the key stays in the table for the life of the process.
            lock.acquire()
            try:
                yield
            finally:
                lock.release()
        finally:
            with self._guard:
                lock, refs = self._locks[key]
                if refs == 1:
                    del self._locks[key]
                else:
                    self._locks[key] = (lock, refs - 1)

    def in_flight(self) -> int:
        """Number of keys currently latched (diagnostics and tests)."""
        with self._guard:
            return len(self._locks)


@dataclass
class _AsyncSlot:
    lock: asyncio.Lock
    refs: int = 0
    holder: asyncio.Task[Any] | None = None


class _AsyncInflightLatch:
    """The asyncio twin of `_InflightLatch`: one latch table per event loop.

    Coroutines on one loop interleave only at `await`s, so the table needs no
    lock of its own; `asyncio.Lock` does the waiting. The holder task is
    recorded so a same-task re-entry (a cached coroutine awaiting its own key)
    passes through instead of deadlocking on a non-reentrant `asyncio.Lock`.
    Tables are keyed weakly on the loop, so a loop that goes away takes its
    table with it.
    """

    def __init__(self) -> None:
        self._per_loop: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, dict[str, _AsyncSlot]
        ] = weakref.WeakKeyDictionary()

    @contextlib.asynccontextmanager
    async def hold(self, key: str) -> AsyncIterator[None]:
        slots = self._per_loop.setdefault(asyncio.get_running_loop(), {})
        slot = slots.get(key)
        if slot is None:
            slot = slots[key] = _AsyncSlot(lock=asyncio.Lock())
        me = asyncio.current_task()
        if slot.holder is not None and slot.holder is me:
            yield  # re-entrant: this task already holds the key
            return
        slot.refs += 1
        try:
            # The acquire sits inside the refcount's `finally`: a waiter
            # cancelled while queued (a `wait_for` timeout, a TaskGroup sibling
            # failing) must drop its ref, or the slot outlives every holder.
            await slot.lock.acquire()
            slot.holder = me
            try:
                yield
            finally:
                slot.holder = None
                slot.lock.release()
        finally:
            slot.refs -= 1
            if slot.refs == 0 and slots.get(key) is slot:
                del slots[key]

    def in_flight(self) -> int:
        """Number of keys latched on the running loop (diagnostics and tests)."""
        try:
            slots = self._per_loop.get(asyncio.get_running_loop())
        except RuntimeError:
            return 0
        return len(slots) if slots else 0


_LATCH = _InflightLatch()
_ASYNC_LATCH = _AsyncInflightLatch()


def _same_value(stored: Any, fresh: Any) -> bool:
    """Best-effort equality for the supersede notice — anything that cannot be compared
    (an array whose `==` is elementwise, a class raising in `__eq__`) counts as different, so
    the notice errs loud rather than quiet."""
    try:
        return bool(stored == fresh)
    except Exception:  # noqa: BLE001 — comparison failure is "not provably equal"
        return False


def safe_jsonable_encoder(
    obj: Any,
    *,
    default: Callable[[Any], Any] | None = str,
) -> Any:
    """Convert objects to JSON-serializable forms for cache keys.

    `default` mirrors `json.dumps(default=)`: called on values no built-in
    handler matches. `default=None` raises `TypeError` on unknown types
    (strict mode — useful when objects without `__dict__` might leak
    process-specific addresses into keys); `default=str` (the package
    default) preserves the pre-0.2 loose fallback.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [safe_jsonable_encoder(item, default=default) for item in obj]
    if isinstance(obj, dict):
        return {str(k): safe_jsonable_encoder(v, default=default) for k, v in obj.items()}
    if isinstance(obj, set):
        return sorted([safe_jsonable_encoder(item, default=default) for item in obj])
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="ignore")
    try:
        import arrow

        if isinstance(obj, arrow.Arrow):
            return obj.isoformat()
    except ImportError:
        pass
    try:
        from datetime import date, datetime, time

        if isinstance(obj, (datetime, date, time)):
            return obj.isoformat()
    except ImportError:
        pass
    try:
        from pathlib import Path

        if isinstance(obj, Path):
            return str(obj)
    except ImportError:
        pass
    if BaseModel is not None and isinstance(obj, BaseModel):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return safe_jsonable_encoder(obj.__dict__, default=default)
    if default is None:
        raise TypeError(
            f"safe_jsonable_encoder cannot encode {type(obj).__name__!r} for a cache key. "
            "Either convert to a primitive/dict in the caller, or pass `default=` "
            "(e.g. `default=str` for the loose fallback) when constructing the cache."
        )
    return default(obj)


def _is_basemodel_class(cls: Any) -> bool:
    return BaseModel is not None and isinstance(cls, type) and issubclass(cls, BaseModel)


@functools.cache
def _dataclass_rebuildable(cls: type) -> bool:
    """True when `cls(**field_dict)` reconstructs an instance from its stored
    fields: every stored field is settable through `__init__` and the constructor
    takes nothing else (no `init=False` fields, no `InitVar` init-only params).
    Anything else keeps the raw-pickle passthrough — storing a dict we could not
    rebuild would hand callers a silently wrong type on the warm hit."""
    stored = {f.name for f in dataclasses.fields(cls)}
    if any(not f.init for f in dataclasses.fields(cls)):
        return False
    try:
        params = inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return False
    return set(params) == stored


def _codable_dataclass(cls: Any) -> bool:
    return (
        isinstance(cls, type) and dataclasses.is_dataclass(cls) and _dataclass_rebuildable(cls)
    )


@functools.cache
def _dataclass_hints(cls: type) -> Mapping[str, Any] | None:
    """Resolved field annotations for recursive encode/decode of nested fields;
    `None` when a hint cannot be resolved (the field then passes through raw)."""
    try:
        return typing.get_type_hints(cls)
    except Exception:  # noqa: BLE001 — unresolvable hints → passthrough fields
        return None


def _member_shapes(a: Any) -> tuple[type, ...]:
    """The stored shapes (`list`/`tuple`/`dict`) a runtime value of member `a`
    can take — what `_decode`'s union branch would claim it by. Resolving to an
    origin type and testing `issubclass` covers TypedDicts and `dict`/`list`/
    `tuple` subclasses structurally; `Mapping[...]` origins map to the dict
    shape because their runtime values are (typically) dicts."""
    origin = typing.get_origin(a) or (a if isinstance(a, type) else None)
    if not isinstance(origin, type):
        return ()
    for shape in (list, tuple, dict):
        if issubclass(origin, shape):
            return (shape,)
    return (dict,) if issubclass(origin, Mapping) else ()


def _union_codec_members(args: tuple[Any, ...]) -> tuple[Any, ...] | None:
    """The union's codec-relevant members, or `None` when the union is ambiguous.

    A stored value must map back to exactly one member by *runtime* shape —
    `_decode` claims a stored dict for a model/dataclass member and a stored
    container by `isinstance` against the member's origin. Model and
    rebuildable-dataclass members store as dicts, so they collide with each
    other (`SmushRationale | RationaleRefusal`) and with any member whose
    runtime value can be a dict (`Model | dict[str, X]`, TypedDicts,
    `Mapping[...]`, `dict` subclasses); two members sharing a container origin
    (`list[A] | list[B]`) collide the same way, and an `Any`/`object` member
    can hold every shape at once. Such unions stay on the raw-pickle
    passthrough, exactly the pre-codec behaviour for them."""
    shapes: list[type] = []
    for a in args:
        if _is_basemodel_class(a) or _codable_dataclass(a):
            shapes.append(dict)
        elif a is Any or a is object:
            shapes.extend((list, tuple, dict))
        else:
            shapes.extend(_member_shapes(a))
    if len(shapes) != len(set(shapes)):
        return None
    return tuple(a for a in args if a is not type(None))


def _wants_codec(anno: Any, _depth: int = 0) -> bool:
    """Decoration-time gate: does the annotation mention a BaseModel or a
    rebuildable dataclass anywhere a value walk could reach one? Functions whose
    returns can't contain either stay on the zero-overhead passthrough path."""
    if _depth > 8 or anno is inspect.Parameter.empty or anno is None:
        return False
    if _is_basemodel_class(anno) or _codable_dataclass(anno):
        return True
    origin = typing.get_origin(anno)
    if origin in (Union, types.UnionType):
        members = _union_codec_members(typing.get_args(anno))
        return members is not None and any(_wants_codec(a, _depth + 1) for a in members)
    if origin in (list, tuple, dict):
        return any(
            _wants_codec(arg, _depth + 1)
            for arg in typing.get_args(anno)
            if arg is not Ellipsis
        )
    return False


def _resolve_return_annotation(func: Callable[..., Any], anno: Any) -> Any:
    """Resolve a string / forward-ref return annotation to real types.

    Under `from __future__ import annotations` every annotation arrives as a
    string, and without it a nested quoted name (`-> list["M"]`) arrives holding
    a `ForwardRef` — either way `_wants_codec` cannot see the model in it and
    the value would pickle by class reference, dying with the defining code.
    Resolution is scoped to the RETURN annotation via a stub function so an
    unresolvable *parameter* annotation (the common TYPE_CHECKING-gated client
    type) cannot disable the codec. A return annotation that itself fails to
    resolve (e.g. behind TYPE_CHECKING, or a class defined below the decorated
    function) keeps the raw-pickle passthrough — with a warning, because any
    model/dataclass it names will then be stored by class reference."""
    stub = types.FunctionType((lambda: None).__code__, getattr(func, "__globals__", {}))
    stub.__annotations__ = {"return": anno}
    # PEP 695 generics: `def f[T]() -> "list[T]"` resolves T off the function's
    # own type params, which the stub must inherit or the name never resolves.
    stub.__type_params__ = getattr(  # ty: ignore[unresolved-attribute] — writable since 3.12
        func, "__type_params__", ()
    )
    try:
        return typing.get_type_hints(stub).get("return", inspect.Parameter.empty)
    except Exception as exc:  # noqa: BLE001 — unresolvable annotation → passthrough
        logger.warning(
            "emboss.cached: return annotation %r of %s() did not resolve (%s); "
            "annotation-driven encoding stays off, so a model/dataclass return "
            "would pickle by class reference and die with its defining code.",
            anno,
            getattr(func, "__qualname__", func),
            exc,
        )
        return inspect.Parameter.empty


class _SchemaDrift(Exception):
    """A stored field dict no longer matches its dataclass constructor — the
    cached value predates a field rename/removal in the class definition."""


def _runtime_matches(value: Any, anno: Any) -> bool:
    """Does `value` plausibly inhabit union member `anno`? Used only to pick which
    member of a Union to encode/decode through."""
    if _is_basemodel_class(anno) or _codable_dataclass(anno):
        return isinstance(value, anno)
    origin = typing.get_origin(anno)
    return origin in (list, tuple, dict) and isinstance(value, origin)


def _encode(value: Any, anno: Any) -> Any:
    """Convert BaseModels and rebuildable dataclasses reachable through `anno`
    into plain dicts before pickling, recursing through `list`/`tuple`/`dict`
    containers and dataclass fields — so the stored bytes carry no class
    references, which would make the value unreadable once the defining code is
    gone. A value that doesn't match the annotation passes through untouched
    (the old raw-pickle behaviour)."""
    if value is None or anno is None:
        return value
    if _is_basemodel_class(anno):
        return value.model_dump() if isinstance(value, anno) else value
    if _codable_dataclass(anno):
        if not isinstance(value, anno):
            return value
        hints = _dataclass_hints(anno) or {}
        return {
            f.name: _encode(getattr(value, f.name), hints.get(f.name))
            for f in dataclasses.fields(value)
        }
    origin = typing.get_origin(anno)
    args = typing.get_args(anno)
    if origin in (Union, types.UnionType):
        members = _union_codec_members(args)
        for arg in members or ():
            if _wants_codec(arg) and _runtime_matches(value, arg):
                return _encode(value, arg)
        return value
    if origin is list and args and isinstance(value, list):
        return [_encode(v, args[0]) for v in value]
    if origin is tuple and args and isinstance(value, tuple):
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_encode(v, args[0]) for v in value)
        if len(args) == len(value):
            return tuple(_encode(v, a) for v, a in zip(value, args))
        return value
    if origin is dict and len(args) == 2 and isinstance(value, dict):
        return {k: _encode(v, args[1]) for k, v in value.items()}
    return value


def _decode(value: Any, anno: Any) -> Any:
    """Rehydrate dicts back into models/dataclasses on a cache hit, mirroring
    `_encode`'s walk. A value that doesn't match the expected *shape* passes
    through untouched; a field dict that matches the shape but no longer
    rehydrates under the current class definition raises `_SchemaDrift` (the
    wrapper treats it as a miss and recomputes)."""
    if value is None or anno is None:
        return value
    if _is_basemodel_class(anno):
        if not isinstance(value, dict):
            return value
        try:
            return anno.model_validate(value)
        except ValueError as exc:  # pydantic.ValidationError subclasses ValueError
            # Serving the raw field dict would hand the caller a wrong type on
            # a warm hit; the wrapper treats drift as a miss and recomputes.
            raise _SchemaDrift(f"{anno.__qualname__}: {exc}") from exc
    if _codable_dataclass(anno):
        if not isinstance(value, dict):
            return value
        hints = _dataclass_hints(anno) or {}
        kwargs = {k: _decode(v, hints.get(k)) for k, v in value.items()}
        try:
            return anno(**kwargs)
        except (TypeError, ValueError) as exc:
            # TypeError: fields renamed/removed since storing. ValueError: a
            # `__post_init__` that rejects the stored (drifted) values. Either
            # way the wrapper treats drift as a miss and recomputes.
            raise _SchemaDrift(f"{anno.__qualname__}: {exc}") from exc
    origin = typing.get_origin(anno)
    args = typing.get_args(anno)
    if origin in (Union, types.UnionType):
        members = _union_codec_members(args)
        drift: _SchemaDrift | None = None
        for arg in members or ():
            if (_is_basemodel_class(arg) or _codable_dataclass(arg)) and isinstance(value, dict):
                try:
                    return _decode(value, arg)
                except _SchemaDrift as exc:
                    drift = exc  # try the remaining members before giving up
                    continue
            if _wants_codec(arg) and _runtime_matches(value, arg):
                return _decode(value, arg)
        if drift is not None:
            raise drift
        return value
    if origin is list and args and isinstance(value, list):
        return [_decode(v, args[0]) for v in value]
    if origin is tuple and args and isinstance(value, tuple):
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(v, args[0]) for v in value)
        if len(args) == len(value):
            return tuple(_decode(v, a) for v, a in zip(value, args))
        return value
    if origin is dict and len(args) == 2 and isinstance(value, dict):
        return {k: _decode(v, args[1]) for k, v in value.items()}
    return value


def _references_doc(tree: ast.Module) -> bool:
    """True when the source reads `__doc__` anywhere — its docstrings are load-bearing.

    A function that consumes its own (or a nested def/class's) docstring at
    runtime — the docstring-as-prompt / docstring-as-data pattern — *implements*
    behaviour with it, so stripping would let a docstring edit change behaviour
    without changing the key (a silent stale hit). Detection is syntactic:
    an attribute or name read spelled `__doc__`. Indirect reads
    (`inspect.getdoc(f)`, `getattr(f, "__doc__")`) are not detected — pin such
    functions with `unsafe_manual_key` if their docstrings are load-bearing.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "__doc__":
            return True
        if isinstance(node, ast.Name) and node.id == "__doc__":
            return True
    return False


def _strip_docstrings(tree: ast.Module) -> ast.Module:
    """Remove the docstring statement from the module and every def/class in it.

    Only the docstring proper is dropped — the *first* statement of a body when
    it is a bare string constant. A body left empty by the removal gets `pass`
    so the tree still unparses. Mutates and returns `tree`.
    """
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return tree


def _canonical_source(raw_source: str, *, strip_docstrings: bool = True) -> str:
    """Normalize function source so cosmetic edits don't change the cache key.

    Round-trips through the AST (`ast.unparse(ast.parse(...))`), which discards
    formatting that never affects behaviour — indentation, line breaks, spacing,
    trailing commas, comments, quote style — and, by default, drops docstrings,
    which document behaviour but don't implement it. All other string-literal
    *contents* are preserved (so two functions whose only difference is the
    spaces inside a returned string still get distinct keys). Exception: when
    the source reads `__doc__` (see `_references_doc`), docstrings *are* the
    implementation, so they stay in the hash and edits invalidate as before.

    `strip_docstrings=False` reproduces the pre-0.12 docstring-sensitive
    canonicalization — the identity under which older caches were written. The
    decorator accepts that identity as an implicit read fallback so a docstring
    edit (or the 0.12 upgrade itself) doesn't orphan warm entries.

    Falls back to the raw source when it can't be parsed (e.g. the source is
    unavailable, or a decorator references names not importable at parse time),
    so keying degrades to the pre-0.3 whitespace-sensitive behaviour rather
    than crashing.
    """
    try:
        tree = ast.parse(textwrap.dedent(raw_source))
        if strip_docstrings and not _references_doc(tree):
            tree = _strip_docstrings(tree)
        return ast.unparse(tree)
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return raw_source


@dataclass(frozen=True, kw_only=True)
class EmbossInfo:
    """Cache-identity metadata attached to every `@cached` wrapper as `__emboss__`.

    `cache_id` is `f"{name}:{body_hash}"` — the function's identity in the
    keying scheme (`key = md5(name + body_hash + arg_hash)`), where `body_hash`
    is the md5 of the AST-canonical, docstring-stripped source, or the
    `unsafe_manual_key` when one was pinned. `also_accept` echoes the raw
    fallback identities passed at decoration time — the implicit pre-0.12
    docstring-sensitive fallback is *not* included; `cache_keys()` is the
    authority for the full fallback set the wrapper reads.
    """

    name: str
    body_hash: str
    cache_id: str
    also_accept: tuple[str, ...]


def cache_id(func: Callable[..., Any]) -> str:
    """Return the cache identity (`"name:body_hash"`) of an `@cached` function.

    The identity is what `also_accept` consumes: capture it *before* a rename
    or body edit, then pass it to the new definition so the warm cache entries
    keep matching (see `cached`).

    Raises:
        TypeError: if `func` is not an `@cached`-wrapped function (it lacks the
            `__emboss__` metadata attached at decoration time).
    """
    info = getattr(func, "__emboss__", None)
    if not isinstance(info, EmbossInfo):
        raise TypeError(
            f"{getattr(func, '__qualname__', func)!r} is not an @cached-wrapped function "
            "(missing __emboss__ metadata) — cache_id() only works on functions "
            "decorated with emboss.cached."
        )
    return info.cache_id


def cache_keys(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> tuple[str, list[str]]:
    """Return `(current key, fallback keys)` for a `@cached` call.

    The fallback keys are the explicit `also_accept` identities followed by the
    implicit pre-0.12 docstring-sensitive identity (when any docstring — the
    function's own or a nested def/class's — makes the two identities differ).
    The keys come from the wrapper's own keying closure (attached at decoration
    time), so they are byte-identical to what the wrapper reads and writes —
    including the decorator's `default=` encoder setting. `func` is
    positional-only so a wrapped function taking a `func=` kwarg still keys it.

    Raises:
        TypeError: if `func` is not an `@cached`-wrapped function.
    """
    keys_fn = getattr(func, "__emboss_keys__", None)
    if not callable(keys_fn):
        raise TypeError(
            f"{getattr(func, '__qualname__', func)!r} is not an @cached-wrapped function "
            "(missing __emboss_keys__ metadata) — cache_keys() only works on functions "
            "decorated with emboss.cached."
        )
    return keys_fn(args, kwargs)


def cache_key(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> str:
    """Return the storage key a `@cached` call reads and writes.

    `cache_key(fn, *args, **kwargs)` is the key `fn(*args, **kwargs)` uses —
    derive it to check "is this specific call cached?" (`key in cache` /
    `cache.get(key)`) or to migrate entries without recomputing them.

    Raises:
        TypeError: if `func` is not an `@cached`-wrapped function.
    """
    key, _accept_keys = cache_keys(func, *args, **kwargs)
    return key


def _parse_accept_token(token: str) -> tuple[str, str]:
    """Split an `also_accept` token into `(name, body_hash)`.

    Tokens are prior cache identities as returned by `cache_id` —
    `"name:body_hash"`. Malformed tokens fail at decoration time so a typo
    can't silently disable migration.
    """
    name, sep, body_hash = token.partition(":")
    if not sep or not name or not body_hash:
        raise ValueError(
            f"also_accept token {token!r} is not a cache identity of the form "
            "'name:body_hash' — pass the string returned by emboss.cache_id() for "
            "the old function (e.g. 'fetch_user:0123abcd...')."
        )
    return name, body_hash


def cached(
    cache: Cache | None = None,
    *,
    default: Callable[[Any], Any] | None = str,
    also_accept: list[str] | None = None,
    unsafe_manual_key: str | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Disk-backed memoization decorator.

    `cache` accepts any object satisfying the `Cache` protocol
    (`.get(key, default=...)` / `.set(key, value)`) — `emboss.SqliteCache`,
    `emboss.FileCache`, `diskcache.Cache` (optional: `pip install
    emboss[diskcache]`), or your own backend. Defaults to a fresh
    `emboss.SqliteCache` (see below).

    `default` is threaded into `safe_jsonable_encoder` for cache-key
    construction (see that function for semantics). The package default
    `str` preserves the loose 0.1 behaviour; pass `default=None` for strict
    mode that raises on unknown argument types.

    Keying ignores docstrings as well as comments and formatting — they
    document behaviour, they don't implement it — so editing a docstring never
    invalidates a warm cache. Exception: a source that reads `__doc__` (the
    docstring-as-prompt pattern) implements behaviour *with* its docstrings,
    so it keeps the docstring-sensitive keying and docstring edits invalidate
    as before; indirect reads (`inspect.getdoc`) aren't detected — pin those
    with `unsafe_manual_key`. Entries written by emboss < 0.12, keyed on the
    docstring-*sensitive* source, are still read via an implicit fallback
    identity and copied forward to the current key on first hit, so the
    upgrade itself keeps the cache warm.

    `also_accept` lists *old* cache identities — `cache_id` strings of the
    form `"name:body_hash"` — whose entries are still honoured. On a miss
    under the current key, each accepted identity is tried in order (then the
    implicit pre-0.12 fallback) and a hit is copied forward to the current key
    (write-through), so a behaviour-preserving rename or body refactor keeps
    its warm cache. Capture the identity with `emboss.cache_id(func)` before
    editing, or recover it afterwards with the `emboss id --rev <rev>` CLI.
    Malformed tokens raise `ValueError` at decoration time.

    `unsafe_manual_key` replaces the source-derived body hash with a fixed,
    caller-managed string. **WARNING — this opts out of emboss's
    invalidate-on-edit safety net**: editing the function body no longer
    invalidates its cache, so stale results are served until *you* bump the
    key string (e.g. `"v1"` → `"v2"`) — the caller is responsible for bumping
    it on every behaviour change. `also_accept` still works alongside it
    (e.g. to migrate source-keyed entries into a manual-key identity).

    Return annotations that mention a pydantic `BaseModel` or a rebuildable
    dataclass — directly, inside `list`/`tuple`/`dict` containers (nested to any
    depth), in a union, or in dataclass fields — have those values stored as
    plain dicts and rehydrated on read, so the cached bytes carry no class
    references: classes defined in `__main__` round-trip across script
    invocations, and a value outlives the module that defined its class. A
    dataclass with `init=False` fields or `InitVar` params can't be rebuilt from
    its field dict and keeps the raw-pickle passthrough. A stored field dict
    that no longer rehydrates under the current class definition (fields
    renamed/removed since it was cached) is treated as a warned miss: the
    function re-executes and the fresh store heals the stale shape (under
    cache-only mode it raises `CacheMiss` instead).

    When no `cache` is passed, a default `emboss.SqliteCache` is created at the
    directory named by the `EMBOSS_CACHE_DIR` environment variable; if unset, it
    falls back to a fresh temporary directory (ephemeral, like the previous
    `diskcache` default). The cache location never affects keying (keys are
    function identity + arguments).

    Under cache-only mode (the `EMBOSS_CACHE_ONLY` env var or a
    `emboss.cache_only()` block), a call whose key — and every `also_accept`
    fallback — misses raises `CacheMiss` instead of executing the function
    (see `cache_only`).

    Concurrent misses on one key compute once, and the first write wins. A
    miss holds a per-key in-flight latch (per process; threads and asyncio
    tasks alike) while it computes and stores, so a second caller of the same
    key waits and then reads the stored value instead of computing its own.
    Before storing, the miss re-reads the key: a value another writer landed
    while this call computed — a process or node the latch cannot see — is
    served instead and this call's result is discarded, with a warning when the
    two values differ (an info line when they are identical). For a
    deterministic function this only saves work; for a non-deterministic one
    (an LLM draw with no API seed) it is what keeps the value a caller returned
    from being silently replaced under every later reader. The cross-process
    half is best-effort: a peer's write inside the backend's index staleness,
    or one not yet synced in from another node, still supersedes on read.
    """
    if cache is None:
        cache_dir = os.environ.get("EMBOSS_CACHE_DIR") or tempfile.mkdtemp(prefix="emboss-")
        cache = SqliteCache(cache_dir)
    if unsafe_manual_key is not None and not unsafe_manual_key:
        raise ValueError(
            "unsafe_manual_key must be a non-empty string (e.g. 'v1') — "
            "omit it to key on the function source instead."
        )
    accepted_identities = tuple(_parse_accept_token(token) for token in (also_accept or []))

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # The function's identity is `name:body_hash`. The body hash is the
        # AST-canonical source (whitespace/comment/docstring-agnostic), unless
        # `unsafe_manual_key` pins it to a caller-managed string instead. The
        # pre-0.12 docstring-sensitive identity is accepted as an implicit read
        # fallback (after any explicit `also_accept` tokens) so entries written
        # under the old keying survive the change and migrate forward on hit.
        implicit_fallbacks: tuple[tuple[str, str], ...] = ()
        if unsafe_manual_key is not None:
            body_hash = unsafe_manual_key
        else:
            raw_source = inspect.getsource(func)
            body_hash = hashlib.md5(_canonical_source(raw_source).encode()).hexdigest()
            docful_hash = hashlib.md5(
                _canonical_source(raw_source, strip_docstrings=False).encode()
            ).hexdigest()
            if docful_hash != body_hash:
                implicit_fallbacks = ((func.__name__, docful_hash),)
        accepted = accepted_identities + implicit_fallbacks
        info = EmbossInfo(
            name=func.__name__,
            body_hash=body_hash,
            cache_id=f"{func.__name__}:{body_hash}",
            also_accept=tuple(also_accept or ()),
        )
        try:
            return_anno = inspect.signature(func).return_annotation
        except (TypeError, ValueError):
            return_anno = inspect.Parameter.empty
        if return_anno is not inspect.Parameter.empty and not _wants_codec(return_anno):
            # PEP 563 strings and nested forward refs hide models from the codec
            # gate; resolve the return annotation alone (never the parameters —
            # see `_resolve_return_annotation`) before giving up on encoding.
            return_anno = _resolve_return_annotation(func, return_anno)
        codec_anno = return_anno if _wants_codec(return_anno) else None
        is_async = asyncio.iscoroutinefunction(func)

        def _keys(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> tuple[str, list[str]]:
            """`(current key, fallback keys)` for one call's arguments."""
            json_args = [safe_jsonable_encoder(arg, default=default) for arg in args]
            json_kwargs = {
                k: safe_jsonable_encoder(v, default=default) for k, v in kwargs.items()
            }
            arg_hash = hashlib.md5(
                f"{json.dumps(json_args)}{json.dumps(json_kwargs)}".encode()
            ).hexdigest()
            key = hashlib.md5(f"{func.__name__}{body_hash}{arg_hash}".encode()).hexdigest()
            accept_keys = [
                hashlib.md5(f"{name}{accepted_hash}{arg_hash}".encode()).hexdigest()
                for name, accepted_hash in accepted
            ]
            return key, accept_keys

        def _lookup(key: str, accept_keys: list[str]) -> Any:
            """Read the current key; on miss, try each fallback key (explicit
            `also_accept`, then the implicit pre-0.12 identity) in order and
            migrate the first hit forward to the current key so later reads
            hit directly."""
            raw = cache.get(key, default=_MISSING)
            if raw is not _MISSING:
                return raw
            for accept_key in accept_keys:
                if accept_key == key:
                    continue
                raw = cache.get(accept_key, default=_MISSING)
                if raw is not _MISSING:
                    try:
                        cache.set(key, raw)
                    except Exception:  # noqa: BLE001 — migration is best-effort
                        # The hit is already in hand; failing to copy it forward
                        # (read-only backend, disk full) must not turn a cache
                        # hit into a crash. The next call retries the migration.
                        logger.warning(
                            "emboss: %s hit under fallback key %s but migrating it "
                            "to %s failed; serving the hit, the next call will retry",
                            info.cache_id,
                            accept_key,
                            key,
                            exc_info=True,
                        )
                    return raw
            return _MISSING

        # Drift keys already warned about: the recompute normally heals the
        # store, but under `cache_only` (or a read-only cache directory)
        # nothing supersedes the drifted entry — warn once per key, not per
        # call (same policy as LogCache's opaque-miss warnings).
        warned_drift: set[str] = set()

        def _serve(key: str, accept_keys: list[str]) -> Any:
            """The decoded cached value, or `_MISSING` — a hit that no longer
            rehydrates under the CURRENT class definition reads as a warned
            miss so the recompute heals the stale shape (under cache_only that
            raises CacheMiss downstream, naming the key honestly)."""
            raw = _lookup(key, accept_keys)
            if raw is _MISSING:
                return _MISSING
            try:
                return _decode(raw, codec_anno)
            except _SchemaDrift as drift:
                if key not in warned_drift:
                    warned_drift.add(key)
                    logger.warning(
                        "emboss: cached value for %s() no longer matches its "
                        "constructor (%s); treating as a miss and recomputing. "
                        "(Warned once per key per decorated function.)",
                        info.name,
                        drift,
                    )
                return _MISSING

        def _settle(key: str, accept_keys: list[str], result: T) -> T:
            """First write wins: re-read the key before storing, and if another
            writer landed a value while this call computed, serve THAT value and
            discard this call's result — `set` would otherwise supersede it under
            every later reader. The in-process latch makes this exact for the
            threads of one process; across processes and nodes it is best-effort
            (a peer's write within the backend's index staleness, or one not yet
            synced in, is still superseded). A stored value that no longer
            rehydrates is not a competing write — the fresh result heals it."""
            stored = _serve(key, accept_keys)
            if stored is not _MISSING:
                if _same_value(stored, result):
                    logger.info(
                        "emboss: %s() key %s was stored by another writer while this call "
                        "computed an identical value; serving the stored one.",
                        info.name,
                        _short_hash(key),
                    )
                else:
                    logger.warning(
                        "emboss: %s() key %s was stored by another writer while this call "
                        "computed, and the two values differ; keeping the first-written "
                        "value and discarding this call's result (a non-deterministic draw "
                        "would otherwise be replaced under every later reader).",
                        info.name,
                        _short_hash(key),
                    )
                return stored  # type: ignore[no-any-return]
            # Write back under the current key only — old identities are read
            # and migrated on hit, never written to.
            cache.set(key, _encode(result, codec_anno))
            return result

        def _compute(key: str, accept_keys: list[str], args: Any, kwargs: Any) -> T:
            """The latched miss path: re-check under the key's latch (a waiter
            finds the holder's write), then compute, then settle first-write-wins."""
            with _LATCH.hold(key):
                served = _serve(key, accept_keys)
                if served is not _MISSING:
                    return served  # type: ignore[no-any-return]
                if _cache_only_active():
                    raise CacheMiss(func_name=info.name, cache_id=info.cache_id, key=key)
                return _settle(key, accept_keys, func(*args, **kwargs))

        async def _compute_async(
            key: str, accept_keys: list[str], args: Any, kwargs: Any
        ) -> T:
            async with _ASYNC_LATCH.hold(key):
                served = _serve(key, accept_keys)
                if served is not _MISSING:
                    return served  # type: ignore[no-any-return]
                # Checked here — when the coroutine runs — not at call time, so
                # the raise pairs with the execution it prevents.
                if _cache_only_active():
                    raise CacheMiss(func_name=info.name, cache_id=info.cache_id, key=key)
                result = await func(*args, **kwargs)  # type: ignore[misc]
                return _settle(key, accept_keys, result)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            key, accept_keys = _keys(args, kwargs)
            served = _serve(key, accept_keys)
            if served is not _MISSING:
                if is_async:

                    async def return_cached():
                        return served

                    return return_cached()  # type: ignore[return-value]
                return served  # type: ignore[no-any-return]
            if is_async:
                return _compute_async(key, accept_keys, args, kwargs)  # type: ignore[return-value]
            return _compute(key, accept_keys, args, kwargs)

        wrapper.__emboss__ = info  # type: ignore[attr-defined]
        wrapper.__emboss_keys__ = _keys  # type: ignore[attr-defined]
        return wrapper

    return decorator
