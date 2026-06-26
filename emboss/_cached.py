"""Internal: `@cached` decorator implementation. Public API lives in `emboss.__init__`."""

from __future__ import annotations

import ast
import asyncio
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
from collections.abc import Callable, Mapping
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
                    result = await func(*args, **kwargs)  # type: ignore[misc]
                    _store(key, _encode(result, model_cls, container))
                    return result

                return execute()  # type: ignore[return-value]

            result = func(*args, **kwargs)
            _store(key, _encode(result, model_cls, container))
            return result

        wrapper.__emboss__ = info  # type: ignore[attr-defined]
        return wrapper

    return decorator
