"""Internal: `@cached` decorator implementation. Public API lives in `emboss.__init__`."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import contextvars
import functools
import hashlib
import inspect
import json
import logging
import os
import tempfile
import textwrap
import types
import typing
from collections.abc import Callable, Iterator, Mapping
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


def _model_info(annotation: Any) -> tuple[type | None, str]:
    """Return `(Model class, container)` extracted from a return annotation.

    `container` is one of `"none"` (single value), `"list"`, or `"dict"`.
    Returns `(None, "none")` when no BaseModel is in play (decorator falls back
    to pass-through encode/decode).
    """
    if BaseModel is None or annotation is inspect.Parameter.empty or annotation is None:
        return None, "none"
    if _is_basemodel_class(annotation):
        return annotation, "none"

    origin = typing.get_origin(annotation)
    if origin in (Union, types.UnionType):
        for arg in typing.get_args(annotation):
            if _is_basemodel_class(arg):
                return arg, "none"
        return None, "none"
    if origin is list:
        args = typing.get_args(annotation)
        if args and _is_basemodel_class(args[0]):
            return args[0], "list"
    if origin is dict:
        args = typing.get_args(annotation)
        if len(args) == 2 and _is_basemodel_class(args[1]):
            return args[1], "dict"
    return None, "none"


def _encode(value: Any, model_cls: type | None, container: str) -> Any:
    """Convert pydantic models to plain dicts before pickling."""
    if value is None or model_cls is None:
        return value
    if container == "list":
        return [v.model_dump() if isinstance(v, model_cls) else v for v in value]
    if container == "dict":
        return {k: (v.model_dump() if isinstance(v, model_cls) else v) for k, v in value.items()}
    if isinstance(value, model_cls):
        return value.model_dump()
    return value


def _decode(value: Any, model_cls: type | None, container: str) -> Any:
    """Rehydrate dicts into pydantic models on cache hit."""
    if value is None or model_cls is None:
        return value
    if container == "list":
        return [model_cls.model_validate(v) if isinstance(v, dict) else v for v in value]
    if container == "dict":
        return {k: (model_cls.model_validate(v) if isinstance(v, dict) else v) for k, v in value.items()}
    if isinstance(value, dict):
        return model_cls.model_validate(value)
    return value


def _canonical_source(raw_source: str) -> str:
    """Normalize function source so cosmetic edits don't change the cache key.

    Round-trips through the AST (`ast.unparse(ast.parse(...))`), which discards
    formatting that never affects behaviour — indentation, line breaks, spacing,
    trailing commas, comments, quote style — while preserving everything that
    does, including string-literal *contents* (so two functions whose only
    difference is the spaces inside a string still get distinct keys).

    Falls back to the raw source when it can't be parsed (e.g. the source is
    unavailable, or a decorator references names not importable at parse time),
    so keying degrades to the pre-0.3 whitespace-sensitive behaviour rather
    than crashing.
    """
    try:
        return ast.unparse(ast.parse(textwrap.dedent(raw_source)))
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return raw_source


@dataclass(frozen=True, kw_only=True)
class EmbossInfo:
    """Cache-identity metadata attached to every `@cached` wrapper as `__emboss__`.

    `cache_id` is `f"{name}:{body_hash}"` — the function's identity in the
    keying scheme (`key = md5(name + body_hash + arg_hash)`), where `body_hash`
    is the md5 of the AST-canonical source, or the `unsafe_manual_key` when one
    was pinned. `also_accept` echoes the raw fallback identities passed at
    decoration time.
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
    """Return `(current key, also_accept fallback keys)` for a `@cached` call.

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

    `also_accept` lists *old* cache identities — `cache_id` strings of the
    form `"name:body_hash"` — whose entries are still honoured. On a miss
    under the current key, each accepted identity is tried in order and a hit
    is copied forward to the current key (write-through), so a
    behaviour-preserving rename or body refactor keeps its warm cache.
    Capture the identity with `emboss.cache_id(func)` before editing, or
    recover it afterwards with the `emboss id --rev <rev>` CLI. Malformed
    tokens raise `ValueError` at decoration time.

    `unsafe_manual_key` replaces the source-derived body hash with a fixed,
    caller-managed string. **WARNING — this opts out of emboss's
    invalidate-on-edit safety net**: editing the function body no longer
    invalidates its cache, so stale results are served until *you* bump the
    key string (e.g. `"v1"` → `"v2"`) — the caller is responsible for bumping
    it on every behaviour change. `also_accept` still works alongside it
    (e.g. to migrate source-keyed entries into a manual-key identity).

    Detects `BaseModel` / `list[Model]` / `dict[str, Model]` return annotations
    and stores them as dicts (rehydrated on read) so model classes defined in
    `__main__` round-trip across script invocations.

    When no `cache` is passed, a default `emboss.SqliteCache` is created at the
    directory named by the `EMBOSS_CACHE_DIR` environment variable; if unset, it
    falls back to a fresh temporary directory (ephemeral, like the previous
    `diskcache` default). The cache location never affects keying (keys are
    function identity + arguments).

    Under cache-only mode (the `EMBOSS_CACHE_ONLY` env var or a
    `emboss.cache_only()` block), a call whose key — and every `also_accept`
    fallback — misses raises `CacheMiss` instead of executing the function
    (see `cache_only`).
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
        # AST-canonical source (whitespace/comment-agnostic), unless
        # `unsafe_manual_key` pins it to a caller-managed string instead.
        if unsafe_manual_key is not None:
            body_hash = unsafe_manual_key
        else:
            raw_source = inspect.getsource(func)
            body_hash = hashlib.md5(_canonical_source(raw_source).encode()).hexdigest()
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
        if isinstance(return_anno, str):
            # PEP 563 (`from __future__ import annotations`) delivers the annotation
            # as a string `_model_info` cannot see a BaseModel in — silently
            # disabling the model_dump encoding for every function in such a module
            # (its models then pickle by class reference and die with the defining
            # code). `get_type_hints` resolves it (including nested forward refs);
            # a name unresolvable at decoration time (e.g. behind TYPE_CHECKING, in
            # this or any parameter annotation) falls back to no model encoding, as
            # before.
            try:
                return_anno = typing.get_type_hints(func).get(
                    "return", inspect.Parameter.empty
                )
            except Exception:  # noqa: BLE001 — unresolvable annotation → passthrough
                return_anno = inspect.Parameter.empty
        model_cls, container = _model_info(return_anno)
        is_async = asyncio.iscoroutinefunction(func)

        def _keys(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> tuple[str, list[str]]:
            """`(current key, also_accept fallback keys)` for one call's arguments."""
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
                for name, accepted_hash in accepted_identities
            ]
            return key, accept_keys

        def _lookup(key: str, accept_keys: list[str]) -> Any:
            """Read the current key; on miss, try each `also_accept` fallback key
            in order and migrate the first hit forward to the current key so
            later reads hit directly."""
            raw = cache.get(key, default=_MISSING)
            if raw is not _MISSING:
                return raw
            for accept_key in accept_keys:
                if accept_key == key:
                    continue
                raw = cache.get(accept_key, default=_MISSING)
                if raw is not _MISSING:
                    cache.set(key, raw)
                    return raw
            return _MISSING

        def _store(key: str, encoded: Any) -> None:
            """Write back under the current key only — old identities are read
            (and migrated) via `also_accept`, never written to."""
            cache.set(key, encoded)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            key, accept_keys = _keys(args, kwargs)
            raw = _lookup(key, accept_keys)
            if raw is not _MISSING:
                decoded = _decode(raw, model_cls, container)
                if is_async:

                    async def return_cached():
                        return decoded

                    return return_cached()  # type: ignore[return-value]
                return decoded  # type: ignore[return-value]

            if is_async:

                async def execute():
                    # Checked here — when the coroutine runs — not at call
                    # time, so the raise pairs with the execution it prevents.
                    if _cache_only_active():
                        raise CacheMiss(func_name=info.name, cache_id=info.cache_id, key=key)
                    result = await func(*args, **kwargs)  # type: ignore[misc]
                    _store(key, _encode(result, model_cls, container))
                    return result

                return execute()  # type: ignore[return-value]

            if _cache_only_active():
                raise CacheMiss(func_name=info.name, cache_id=info.cache_id, key=key)
            result = func(*args, **kwargs)
            _store(key, _encode(result, model_cls, container))
            return result

        wrapper.__emboss__ = info  # type: ignore[attr-defined]
        wrapper.__emboss_keys__ = _keys  # type: ignore[attr-defined]
        return wrapper

    return decorator
