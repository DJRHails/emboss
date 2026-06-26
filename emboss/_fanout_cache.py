"""FanoutCache — shard a `SqliteCache` across N databases to cut write-lock
contention on one host.

A single SQLite DB serialises writers on one write lock. Under heavy concurrent
writes (a process pool, an async fleet) that lock becomes the bottleneck.
`FanoutCache` (diskcache's analogue) spreads entries over `shards` independent
`SqliteCache` databases, each with its own lock, so writes to different keys
usually proceed in parallel.

Keys are routed by a **stable, process-independent** hash (md5 of `str(key)`),
NOT Python's salted `hash()`, so every process and node agrees on which shard
owns a key — essential for a shared on-disk cache. Reads/writes delegate to the
owning shard; `len()` / `volume()` / `clear()` / `expire()` / iteration aggregate
across shards. `size_limit` is split evenly across shards.

Each shard is a `SqliteCache`, so all of its properties carry over (multi-process
safety, trigger-maintained size, file spillover, `auto_vacuum`, eviction policy).
This is a local backend — for a cache shared across hosts use `FileCache`.
"""

from __future__ import annotations

import hashlib
import itertools
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from emboss._sqlite_cache import _MISSING, SqliteCache


class FanoutCache:
    """A `SqliteCache` sharded across `shards` databases by a stable key hash."""

    def __init__(
        self,
        directory: str | os.PathLike[str] = ".cache",
        shards: int = 8,
        size_limit: int | None = 2**30,
        **kwargs: Any,
    ) -> None:
        if shards < 1:
            raise ValueError(f"shards must be >= 1, got {shards}")
        self.directory = Path(directory)
        self.shards = shards
        per_shard = size_limit // shards if size_limit is not None else None
        width = len(str(shards - 1))
        self._shards: list[SqliteCache] = [
            SqliteCache(self.directory / f"{i:0{width}d}", size_limit=per_shard, **kwargs)
            for i in range(shards)
        ]

    def _shard(self, key: Any) -> SqliteCache:
        digest = hashlib.md5(str(key).encode()).digest()
        return self._shards[int.from_bytes(digest[:8], "big") % self.shards]

    def get(self, key: Any, default: Any = None) -> Any:
        return self._shard(key).get(key, default)

    def set(self, key: Any, value: Any, expire: float | None = None, **kwargs: Any) -> bool:
        return self._shard(key).set(key, value, expire=expire, **kwargs)

    def delete(self, key: Any) -> bool:
        return self._shard(key).delete(key)

    def clear(self) -> int:
        return sum(shard.clear() for shard in self._shards)

    def expire(self, now: float | None = None) -> int:
        return sum(shard.expire(now) for shard in self._shards)

    def volume(self) -> int:
        return sum(shard.volume() for shard in self._shards)

    def __len__(self) -> int:
        return sum(len(shard) for shard in self._shards)

    def __iter__(self) -> Iterator[str]:
        return itertools.chain.from_iterable(self._shards)

    iterkeys = __iter__

    def __contains__(self, key: Any) -> bool:
        return key in self._shard(key)

    def __getitem__(self, key: Any) -> Any:
        value = self._shard(key).get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        self._shard(key).set(key, value)

    def __delitem__(self, key: Any) -> None:
        if not self._shard(key).delete(key):
            raise KeyError(key)

    def __enter__(self) -> FanoutCache:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def close(self) -> None:
        for shard in self._shards:
            shard.close()
