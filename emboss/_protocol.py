"""Public `Cache` protocol — the minimal interface `@cached` needs.

Any object with `.get(key, default=...)` and `.set(key, value)` can be
passed as `cached(cache=...)` — `diskcache.Cache`, `emboss.FileCache`,
a Redis adapter, an in-memory dict-wrapper for tests, etc. Structural
typing means no inheritance is required.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Cache(Protocol):
    """Minimal cache backend interface."""

    def get(self, key: str, default: Any = None) -> Any: ...

    def set(self, key: str, value: Any) -> Any: ...
