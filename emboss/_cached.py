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
import textwrap
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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

    Falls back to the raw source when it can't be parsed as a standalone
    block even after `textwrap.dedent` (e.g. a lambda extracted from the
    middle of an expression), so keying degrades to the pre-0.3
    whitespace-sensitive behaviour rather than crashing. The fallback is
    logged at WARNING level because cosmetic edits then re-key the function.
    """
    try:
        return ast.unparse(ast.parse(textwrap.dedent(raw_source)))
    except (SyntaxError, ValueError, TypeError, RecursionError):
        logger.warning(
            "emboss: could not parse function source for canonical keying; "
            "falling back to raw source (cosmetic edits will re-key). Source begins: %r",
            raw_source[:80],
        )
        return raw_source


@dataclass(frozen=True, kw_only=True)
class EmbossInfo:
    """Cache-identity metadata attached to every `@cached` wrapper as `__emboss__`.

    `cache_id` is `f"{name}:{body_hash}"` — the function's identity in the
    keying scheme (see `_derive_key`), where `body_hash` is the md5 of the
    AST-canonical source, or the `unsafe_manual_key` when one was pinned.
    `also_accept` echoes the raw fallback identities passed at decoration time.
    """

    name: str
    body_hash: str
    also_accept: tuple[str, ...]

    @property
    def cache_id(self) -> str:
        """`f"{name}:{body_hash}"` — the token `also_accept` consumes."""
        return f"{self.name}:{self.body_hash}"


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
    `"name:body_hash"`, exactly one colon (manual keys cannot contain `":"`,
    so no valid identity has two). Malformed tokens — wrong shape, extra
    colons, or any whitespace (the classic copy-paste artifact, which would
    otherwise silently never match a key) — fail at decoration time instead
    of silently disabling migration. Rejecting extra colons also keeps a
    token's `manual=False` derivation colon-free, which `_derive_key` relies
    on for collision-freedom.
    """
    name, sep, body_hash = token.partition(":")
    if (
        not sep
        or not name
        or not body_hash
        or ":" in body_hash
        or any(ch.isspace() for ch in token)
    ):
        raise ValueError(
            f"also_accept token {token!r} is not a cache identity of the form "
            "'name:body_hash' (exactly one ':', no whitespace) — pass the string "
            "returned by emboss.cache_id() for the old function "
            "(e.g. 'fetch_user:0123abcd...')."
        )
    return name, body_hash


def _derive_key(name: str, body_hash: str, arg_hash: str, *, manual: bool) -> str:
    """Derive the on-disk cache key for one identity + argument hash.

    Source-hashed identities keep the historical undelimited preimage
    (`name + body_hash + arg_hash`) so on-disk caches written by earlier
    releases stay valid; `body_hash` is then a fixed-width 32-hex md5, which
    makes the name/body boundary unambiguous. Manual identities
    (`unsafe_manual_key`) carry arbitrary-length hashes, so they get an
    explicit `":"` delimiter after the name. Because `":"` is rejected in
    manual keys, accept-token halves, and Python identifiers, every manual
    preimage contains exactly one colon (which pins `name` and, with the
    fixed-width `arg_hash` suffix, `body_hash`) and no undelimited preimage
    contains any — so distinct identities cannot share a preimage in either
    form (without the delimiter, `ab` + key `"bX"` and `abb` + key `"X"`
    share a key).
    """
    sep = ":" if manual else ""
    return hashlib.md5(f"{name}{sep}{body_hash}{arg_hash}".encode()).hexdigest()


def cached(
    cache: Cache | None = None,
    *,
    default: Callable[[Any], Any] | None = str,
    also_accept: Sequence[str] | None = None,
    unsafe_manual_key: str | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Disk-backed memoization decorator.

    `cache` accepts any object satisfying the `Cache` protocol
    (`.get(key, default=...)` / `.set(key, value)`) — `diskcache.Cache`,
    `emboss.FileCache`, or your own backend. Defaults to a `diskcache.Cache`
    rooted at `$EMBOSS_CACHE_DIR`, or a temporary directory when unset
    (see below).

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
    caller-managed string (non-empty, no whitespace or `":"` — it becomes the
    `body_hash` half of the cache identity, which `also_accept` tokens must
    round-trip). **WARNING — this opts out of emboss's
    invalidate-on-edit safety net**: editing the function body no longer
    invalidates its cache, so stale results are served until *you* bump the
    key string (e.g. `"v1"` → `"v2"`) — the caller is responsible for bumping
    it on every behaviour change. `also_accept` still works alongside it
    (e.g. to migrate source-keyed entries into a manual-key identity).

    Detects `BaseModel` / `list[Model]` / `dict[str, Model]` return annotations
    and stores them as dicts (rehydrated on read) so model classes defined in
    `__main__` round-trip across script invocations.

    When no `cache` is passed, the default cache directory is read from the
    `EMBOSS_CACHE_DIR` environment variable at cache-creation time; if unset,
    `diskcache` falls back to a temporary directory as before. The cache
    location never affects keying (keys are function identity + arguments).
    """
    if cache is None:
        cache = diskcache.Cache(os.environ.get("EMBOSS_CACHE_DIR"))
    if unsafe_manual_key is not None and (
        not unsafe_manual_key
        or ":" in unsafe_manual_key
        or any(ch.isspace() for ch in unsafe_manual_key)
    ):
        # A ":" would let the identity parse ambiguously — "f" + "v:1" reads back
        # as name "f", hash "v:1", whose undelimited derivation collides with the
        # written key of name "fv", hash "1" (see _derive_key).
        raise ValueError(
            "unsafe_manual_key must be a non-empty string without whitespace or ':' "
            "(e.g. 'v1') — omit it to key on the function source instead."
        )
    if isinstance(also_accept, str):
        raise TypeError(
            "also_accept must be a sequence of cache identities, not a bare string — "
            f"wrap it in a list: also_accept=[{also_accept!r}]"
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
            also_accept=tuple(also_accept or ()),
        )
        is_manual = unsafe_manual_key is not None
        # `typing.get_type_hints` (not `inspect.signature`) so PEP 563 modules —
        # `from __future__ import annotations` stringifies every annotation — still
        # get model detection. Unresolvable annotations (e.g. forward refs to
        # TYPE_CHECKING-only imports) degrade to pass-through encoding, as before.
        try:
            return_anno = typing.get_type_hints(func).get("return", inspect.Parameter.empty)
        except Exception as exc:  # annotation expressions can raise anything
            logger.debug(
                "emboss: could not resolve type hints for %s (%s: %s); "
                "pydantic model encoding disabled for it.",
                func.__name__,
                type(exc).__name__,
                exc,
            )
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
            key = _derive_key(info.name, info.body_hash, arg_hash, manual=is_manual)
            # A token's origin — source hash or manual key — is indistinguishable
            # from its text, so try both derivations. The spurious form can only
            # match a key written for the same identity string (colon-freedom of
            # manual keys and token halves keeps the preimage spaces disjoint —
            # see _derive_key), so it's one harmless extra `get` on the miss path.
            accept_keys = [
                _derive_key(name, accepted_hash, arg_hash, manual=manual)
                for name, accepted_hash in accepted_identities
                for manual in (False, True)
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
                    logger.debug(
                        "emboss: migrated cache entry for %s from fallback key %s to %s",
                        info.name,
                        accept_key,
                        key,
                    )
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
