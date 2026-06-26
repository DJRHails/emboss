"""SQLite-backed cache — a first-party, dependency-free `diskcache.Cache` work-alike.

Why this exists: emboss used to depend on `diskcache`. `SqliteCache` provides the
same single-file, size-bounded, LRU-evicting cache using only the stdlib
(`sqlite3`), so emboss core has zero runtime dependencies. `diskcache.Cache`
stays a drop-in compatible backend (`pip install emboss[diskcache]`) — both
satisfy the `Cache` protocol; pick diskcache when you need its richer feature set
(tags, transactions, stats).

Versus `FileCache`: one SQLite file instead of one file per key — far fewer inodes
and faster bulk operations, but it is NOT safe to share across hosts. SQLite
file-locking breaks over NFS, and a single growing DB can't be replicated by a
file syncer without torn-read corruption. Use `SqliteCache` for a bounded *local*
cache; use `FileCache` when the cache must live on a network mount or be
replicated across machines.

Bounded by `size_limit` bytes (default 1 GiB, like diskcache): least-recently-used
entries are evicted once the total exceeds it, down to ~90%. Optional per-entry
`expire` (TTL in seconds) is honoured on read.

Benchmark (local SSD, 512-byte values), order of magnitude:
    set ~10^4 ops/s     get ~10^5 ops/s
Reads are fast — SQLite serves them from its in-memory page cache. To keep reads
cheap the LRU access time is only rewritten when it is stale by more than
``_ATIME_RESOLUTION_S`` (coarse LRU, minimal write amplification).

Implements the subset of `diskcache.Cache` that `@cached` needs, plus ergonomic
dunders: ``get``/``set``/``delete``/``clear``/``close``, ``in``/``[]``/``[]=``/
``del``, and the context-manager protocol. Extra diskcache kwargs
(``eviction_policy``, ``timeout``, ...) are accepted and ignored so existing call
sites port unchanged.
"""

from __future__ import annotations

import os
import pickle
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_MISSING = object()
_DEFAULT_SIZE_LIMIT = 2**30  # 1 GiB, matching diskcache's default
_ATIME_RESOLUTION_S = 60.0  # don't rewrite the LRU timestamp more often than this


class SqliteCache:
    """Single-file SQLite cache: size-bounded, LRU-evicting, TTL-aware."""

    def __init__(
        self,
        directory: str | os.PathLike[str] = ".cache",
        size_limit: int | None = _DEFAULT_SIZE_LIMIT,
        **_kwargs: Any,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "cache.db"
        self.size_limit = size_limit
        self._lock = threading.Lock()
        # isolation_level=None → autocommit; check_same_thread=False + our own lock
        # so a cache shared across threads (e.g. an async worker pool) is safe.
        self._conn = sqlite3.connect(
            str(self.path), isolation_level=None, check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "  key TEXT PRIMARY KEY,"
            "  value BLOB NOT NULL,"
            "  size INTEGER NOT NULL,"
            "  atime REAL NOT NULL,"
            "  expire REAL"
            ")"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS cache_atime ON cache (atime)")
        # Running byte total so `set` doesn't aggregate the whole table each call.
        # Approximate (over-counts overwrites); `_evict_locked` re-sums authoritatively.
        self._size: int = self._conn.execute(
            "SELECT COALESCE(SUM(size), 0) FROM cache"
        ).fetchone()[0]

    def get(self, key: Any, default: Any = None) -> Any:
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT value, expire, atime FROM cache WHERE key=?", (str(key),)
            ).fetchone()
            if row is None:
                return default
            value, expire, atime = row
            if expire is not None and expire < now:
                self._conn.execute("DELETE FROM cache WHERE key=?", (str(key),))
                return default
            if now - atime > _ATIME_RESOLUTION_S:  # coarse LRU, low write amplification
                self._conn.execute("UPDATE cache SET atime=? WHERE key=?", (now, str(key)))
        return pickle.loads(value)

    def set(self, key: Any, value: Any, expire: float | None = None, **_kwargs: Any) -> bool:
        blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        now = time.time()
        exp = now + expire if expire else None
        with self._lock:
            self._conn.execute(
                "INSERT INTO cache (key, value, size, atime, expire) VALUES (?,?,?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET "
                "value=excluded.value, size=excluded.size, atime=excluded.atime, "
                "expire=excluded.expire",
                (str(key), blob, len(blob), now, exp),
            )
            if self.size_limit is not None:
                self._size += len(blob)  # approximate; corrected on eviction
                if self._size > self.size_limit:
                    self._evict_locked()
        return True

    def _evict_locked(self) -> None:
        """Drop least-recently-used rows until the total is back under ~90% of the
        limit. Re-sums authoritatively (correcting the running estimate); the
        ``cache_atime`` index makes the ordered scan cheap."""
        assert self.size_limit is not None  # only called when eviction is enabled
        total = self._conn.execute("SELECT COALESCE(SUM(size), 0) FROM cache").fetchone()[0]
        self._size = total
        if total <= self.size_limit:
            return
        low_water = self.size_limit * 0.9
        freed = 0
        victims: list[tuple[str]] = []
        for k, s in self._conn.execute("SELECT key, size FROM cache ORDER BY atime ASC"):
            if total - freed <= low_water:
                break
            victims.append((k,))
            freed += s
        if victims:
            self._conn.executemany("DELETE FROM cache WHERE key=?", victims)
            self._size = total - freed

    def delete(self, key: Any) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT size FROM cache WHERE key=?", (str(key),)
            ).fetchone()
            cur = self._conn.execute("DELETE FROM cache WHERE key=?", (str(key),))
            if row is not None:
                self._size = max(0, self._size - row[0])
        return cur.rowcount > 0

    def clear(self) -> int:
        with self._lock:
            n = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            self._conn.execute("DELETE FROM cache")
            self._size = 0
        return int(n)

    def __contains__(self, key: Any) -> bool:
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT expire FROM cache WHERE key=?", (str(key),)
            ).fetchone()
        return row is not None and (row[0] is None or row[0] >= now)

    def __getitem__(self, key: Any) -> Any:
        value = self.get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        self.set(key, value)

    def __delitem__(self, key: Any) -> None:
        if not self.delete(key):
            raise KeyError(key)

    def __enter__(self) -> SqliteCache:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
