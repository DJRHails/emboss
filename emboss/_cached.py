"""Internal: `@cached` decorator implementation. Public API lives in `emboss.__init__`."""

from __future__ import annotations

import ast
import asyncio
import functools
import hashlib
import inspect
import json
import logging
import textwrap
import types
import typing
from collections.abc import Callable
from typing import Any, TypeVar, Union

import diskcache

from emboss._protocol import Cache

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


def cached(
    cache: Cache | None = None,
    *,
    default: Callable[[Any], Any] | None = str,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Disk-backed memoization decorator.

    `cache` accepts any object satisfying the `Cache` protocol
    (`.get(key, default=...)` / `.set(key, value)`) — `diskcache.Cache`,
    `emboss.FileCache`, or your own backend. Defaults to a fresh in-memory
    `diskcache.Cache()`.

    `default` is threaded into `safe_jsonable_encoder` for cache-key
    construction (see that function for semantics). The package default
    `str` preserves the loose 0.1 behaviour; pass `default=None` for strict
    mode that raises on unknown argument types.

    Detects `BaseModel` / `list[Model]` / `dict[str, Model]` return annotations
    and stores them as dicts (rehydrated on read) so model classes defined in
    `__main__` round-trip across script invocations.
    """
    if cache is None:
        cache = diskcache.Cache()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        raw_source = inspect.getsource(func)
        # Primary key uses the AST-canonical source (whitespace/comment-agnostic);
        # the legacy key uses the raw source — the pre-0.3 scheme — so existing
        # entries keep matching and get migrated forward on first read.
        raw_hash = hashlib.md5(raw_source.encode()).hexdigest()
        canon_hash = hashlib.md5(_canonical_source(raw_source).encode()).hexdigest()
        try:
            return_anno = inspect.signature(func).return_annotation
        except (TypeError, ValueError):
            return_anno = inspect.Parameter.empty
        model_cls, container = _model_info(return_anno)
        is_async = asyncio.iscoroutinefunction(func)

        def _keys(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[str, str]:
            """`(canonical key, legacy raw-source key)` for one call's arguments."""
            json_args = [safe_jsonable_encoder(arg, default=default) for arg in args]
            json_kwargs = {
                k: safe_jsonable_encoder(v, default=default) for k, v in kwargs.items()
            }
            arg_hash = hashlib.md5(
                f"{json.dumps(json_args)}{json.dumps(json_kwargs)}".encode()
            ).hexdigest()
            key = hashlib.md5(f"{func.__name__}{canon_hash}{arg_hash}".encode()).hexdigest()
            legacy = hashlib.md5(f"{func.__name__}{raw_hash}{arg_hash}".encode()).hexdigest()
            return key, legacy

        def _lookup(key: str, legacy_key: str) -> Any:
            """Read the canonical key; on miss, fall back to the legacy raw-source
            key and migrate the value forward so later reads hit directly."""
            raw = cache.get(key, default=_MISSING)
            if raw is _MISSING and legacy_key != key:
                raw = cache.get(legacy_key, default=_MISSING)
                if raw is not _MISSING:
                    cache.set(key, raw)
            return raw

        def _store(key: str, legacy_key: str, encoded: Any) -> None:
            """Write back under both the canonical and raw-source keys, so callers
            on either keying scheme find the entry."""
            cache.set(key, encoded)
            if legacy_key != key:
                cache.set(legacy_key, encoded)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            key, legacy_key = _keys(args, kwargs)
            raw = _lookup(key, legacy_key)
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
                    _store(key, legacy_key, _encode(result, model_cls, container))
                    return result

                return execute()  # type: ignore[return-value]

            result = func(*args, **kwargs)
            _store(key, legacy_key, _encode(result, model_cls, container))
            return result

        return wrapper

    return decorator
