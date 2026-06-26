"""SQLite-backed cache — a first-party, dependency-free `diskcache.Cache` work-alike.

Why this exists: emboss used to depend on `diskcache`. `SqliteCache` provides the
same single-file, size-bounded, evicting cache using only the stdlib (`sqlite3`),
so emboss core has zero runtime dependencies. `diskcache.Cache` stays a drop-in
compatible backend (`pip install emboss[diskcache]`); pick it for its richer
feature set (tags, atomic counters, queues, stats).

Versus `FileCache`: one SQLite DB instead of one file per key — far fewer inodes
and faster bulk operations, but NOT safe to share across hosts (SQLite locking
breaks over NFS; a single DB can't be replicated by a file syncer). Use
`SqliteCache` for a bounded *local* cache; use `FileCache` for a cache on a
network mount or replicated across machines. For heavy concurrent write load on
one host, `FanoutCache` shards across several `SqliteCache` databases.

Modelled on diskcache, so it shares its robustness properties:

- **Multi-process safe.** `busy_timeout` + `BEGIN IMMEDIATE` writes + a retry on
  "database is locked", so several processes on one host can share the DB.
- **Accurate, shared size/count.** Maintained by SQLite triggers in a `Settings`
  table (not an in-process estimate), so `size_limit` is enforced correctly even
  across processes; `len()` / `volume()` are O(1).
- **Large values spill to files.** Values >= `min_file_size` (32 KB) are written
  beside the DB and referenced by filename, keeping the DB small and fast.
- **`auto_vacuum=FULL`**, so the DB shrinks on disk as entries are evicted.
- Bounded by `size_limit` bytes (default 1 GiB); evicts per `eviction_policy`
  (`least-recently-stored` default — cheap reads; or `least-recently-used`).
  Optional per-entry `expire` (TTL); `expire()` sweeps stale rows.

Benchmark (local SSD, 512-byte values), order of magnitude:
    set ~10^4 ops/s     get ~10^5 ops/s

Implements the subset of `diskcache.Cache` that `@cached` needs plus ergonomic
dunders and iteration: `get`/`set`/`delete`/`clear`/`close`, `in`/`[]`/`[]=`/
`del`, `len()`, iteration over keys, `volume()`, and the context-manager protocol.
Extra diskcache kwargs (`timeout`, `tag`, ...) are accepted and ignored so call
sites port unchanged.
"""

from __future__ import annotations

import contextlib
import os
import pickle
import shutil
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_MISSING = object()
_DEFAULT_SIZE_LIMIT = 2**30  # 1 GiB, matching diskcache's default
_DEFAULT_MIN_FILE_SIZE = 2**15  # 32 KB — values this big or larger spill to files
_ATIME_RESOLUTION_S = 60.0  # LRU: don't rewrite access_time more often than this
_BUSY_TIMEOUT_MS = 30_000  # SQLite waits this long on a locked DB before erroring
_LOCK_RETRIES = 50  # extra retries on residual "database is locked"
_LOCK_RETRY_SLEEP_S = 0.005
_CULL_BATCH = 64  # rows evicted per cull iteration

_MODE_BLOB = 0  # value stored inline in the DB
_MODE_FILE = 1  # value spilled to a file (filename column)

# Each policy: the index to create, and the ORDER BY used to pick eviction
# victims. `least-recently-used` also rewrites access_time on read.
_EVICTION_POLICIES = {
    "least-recently-stored": "store_time",
    "least-recently-used": "access_time",
}


class SqliteCache:
    """Single-DB cache: multi-process-safe, size-bounded, file-spilling, evicting."""

    def __init__(
        self,
        directory: str | os.PathLike[str] = ".cache",
        size_limit: int | None = _DEFAULT_SIZE_LIMIT,
        eviction_policy: str = "least-recently-stored",
        min_file_size: int = _DEFAULT_MIN_FILE_SIZE,
        **_kwargs: Any,
    ) -> None:
        if eviction_policy not in _EVICTION_POLICIES:
            raise ValueError(
                f"eviction_policy must be one of {sorted(_EVICTION_POLICIES)}, "
                f"got {eviction_policy!r}"
            )
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._store = self.directory / "store"  # spilled value files live here
        self.path = self.directory / "cache.db"
        self.size_limit = size_limit
        self.eviction_policy = eviction_policy
        self.min_file_size = min_file_size
        self._lock = threading.Lock()  # serialises this process's threads
        self._conn = sqlite3.connect(
            str(self.path), isolation_level=None, check_same_thread=False
        )
        self._init_db()

    # ── schema / pragmas ──────────────────────────────────────────────────────

    def _init_db(self) -> None:
        c = self._conn
        # auto_vacuum must be set before any table exists (else it needs a full
        # VACUUM to take effect), so apply it first on a fresh DB.
        c.execute("PRAGMA auto_vacuum = FULL")
        c.execute("PRAGMA journal_mode = WAL")
        c.execute("PRAGMA synchronous = NORMAL")
        c.execute("PRAGMA mmap_size = 67108864")  # 64 MB
        c.execute("PRAGMA cache_size = -8000")  # ~8 MB page cache
        c.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        c.execute(
            "CREATE TABLE IF NOT EXISTS Settings (key TEXT PRIMARY KEY, value INTEGER NOT NULL)"
        )
        c.execute("INSERT OR IGNORE INTO Settings (key, value) VALUES ('size', 0), ('count', 0)")
        c.execute(
            "CREATE TABLE IF NOT EXISTS Cache ("
            "  rowid INTEGER PRIMARY KEY,"
            "  key TEXT NOT NULL,"
            "  store_time REAL NOT NULL,"
            "  access_time REAL NOT NULL,"
            "  expire_time REAL,"
            "  size INTEGER NOT NULL,"
            "  mode INTEGER NOT NULL,"
            "  filename TEXT,"
            "  value BLOB"
            ")"
        )
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS Cache_key ON Cache (key)")
        c.execute("CREATE INDEX IF NOT EXISTS Cache_store_time ON Cache (store_time)")
        c.execute("CREATE INDEX IF NOT EXISTS Cache_access_time ON Cache (access_time)")
        c.execute("CREATE INDEX IF NOT EXISTS Cache_expire_time ON Cache (expire_time)")
        # Triggers keep count + size accurate in the DB (shared across processes),
        # so eviction / len() / volume() never scan the table.
        for name, op, expr in (
            ("count_insert", "INSERT", "value = value + 1"),
            ("count_delete", "DELETE", "value = value - 1"),
        ):
            c.execute(
                f"CREATE TRIGGER IF NOT EXISTS Settings_{name} AFTER {op} ON Cache BEGIN "
                f"UPDATE Settings SET {expr} WHERE key = 'count'; END"
            )
        c.execute(
            "CREATE TRIGGER IF NOT EXISTS Settings_size_insert AFTER INSERT ON Cache BEGIN "
            "UPDATE Settings SET value = value + NEW.size WHERE key = 'size'; END"
        )
        c.execute(
            "CREATE TRIGGER IF NOT EXISTS Settings_size_update AFTER UPDATE ON Cache BEGIN "
            "UPDATE Settings SET value = value + NEW.size - OLD.size WHERE key = 'size'; END"
        )
        c.execute(
            "CREATE TRIGGER IF NOT EXISTS Settings_size_delete AFTER DELETE ON Cache BEGIN "
            "UPDATE Settings SET value = value - OLD.size WHERE key = 'size'; END"
        )

    # ── transactions / retry (multi-process safety) ──────────────────────────

    def _write(self, body) -> Any:
        """Run `body(conn)` inside a BEGIN IMMEDIATE transaction, retrying on a
        residual 'database is locked' (busy_timeout handles most waiting)."""
        for attempt in range(_LOCK_RETRIES + 1):
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    result = body(self._conn)
                    self._conn.execute("COMMIT")
                    return result
                except BaseException:
                    self._conn.execute("ROLLBACK")
                    raise
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc) or attempt == _LOCK_RETRIES:
                    raise
                time.sleep(_LOCK_RETRY_SLEEP_S)

    # ── file spillover ───────────────────────────────────────────────────────

    def _spill_path(self) -> tuple[str, Path]:
        name = uuid.uuid4().hex
        rel = os.path.join(name[:2], name[2:])
        return rel, self._store / rel

    def _write_spill(self, blob: bytes) -> str:
        rel, full = self._spill_path()
        full.parent.mkdir(parents=True, exist_ok=True)
        tmp = full.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            f.write(blob)
        os.replace(tmp, full)
        return rel

    def _read_spill(self, rel: str) -> Any:
        with open(self._store / rel, "rb") as f:
            return pickle.load(f)

    def _remove_spill(self, rel: str | None) -> None:
        if rel:
            with contextlib.suppress(OSError):
                (self._store / rel).unlink()

    # ── core API ─────────────────────────────────────────────────────────────

    def get(self, key: Any, default: Any = None) -> Any:
        now = time.time()
        db_key = str(key)
        with self._lock:
            row = self._conn.execute(
                "SELECT rowid, expire_time, access_time, mode, filename, value "
                "FROM Cache WHERE key = ?",
                (db_key,),
            ).fetchone()
            if row is None:
                return default
            rowid, expire_time, access_time, mode, filename, value = row
            if expire_time is not None and expire_time < now:
                self._write(lambda c: self._delete_row(c, rowid, filename))
                return default
            if (
                self.eviction_policy == "least-recently-used"
                and now - access_time > _ATIME_RESOLUTION_S
            ):
                self._write(
                    lambda c: c.execute(
                        "UPDATE Cache SET access_time = ? WHERE rowid = ?", (now, rowid)
                    )
                )
        return self._read_spill(filename) if mode == _MODE_FILE else pickle.loads(value)

    def set(self, key: Any, value: Any, expire: float | None = None, **_kwargs: Any) -> bool:
        now = time.time()
        db_key = str(key)
        blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        size = len(blob)
        expire_time = now + expire if expire else None
        if size >= self.min_file_size:
            mode, filename, stored = _MODE_FILE, self._write_spill(blob), None
        else:
            mode, filename, stored = _MODE_BLOB, None, sqlite3.Binary(blob)

        def body(c: sqlite3.Connection) -> None:
            old = c.execute(
                "SELECT filename FROM Cache WHERE key = ?", (db_key,)
            ).fetchone()
            c.execute(
                "INSERT INTO Cache (key, store_time, access_time, expire_time, size, mode, "
                "filename, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET store_time=excluded.store_time, "
                "access_time=excluded.access_time, expire_time=excluded.expire_time, "
                "size=excluded.size, mode=excluded.mode, filename=excluded.filename, "
                "value=excluded.value",
                (db_key, now, now, expire_time, size, mode, filename, stored),
            )
            if old is not None and old[0] and old[0] != filename:
                self._remove_spill(old[0])  # an overwritten spilled value
            if self.size_limit is not None:
                self._cull(c)

        with self._lock:
            try:
                self._write(body)
            except BaseException:
                self._remove_spill(filename)  # don't orphan the spill file on failure
                raise
        return True

    def _delete_row(self, c: sqlite3.Connection, rowid: int, filename: str | None) -> None:
        c.execute("DELETE FROM Cache WHERE rowid = ?", (rowid,))
        self._remove_spill(filename)

    def _cull(self, c: sqlite3.Connection) -> None:
        """Evict by `eviction_policy` until size is back under ~90% of the limit.
        Reads the trigger-maintained size (O(1), shared across processes)."""
        assert self.size_limit is not None
        size = c.execute("SELECT value FROM Settings WHERE key = 'size'").fetchone()[0]
        if size <= self.size_limit:
            return
        low_water = self.size_limit * 0.9
        order = _EVICTION_POLICIES[self.eviction_policy]
        while size > low_water:
            rows = c.execute(
                f"SELECT rowid, filename, size FROM Cache ORDER BY {order} LIMIT ?",
                (_CULL_BATCH,),
            ).fetchall()
            if not rows:
                break
            for rowid, filename, sz in rows:
                c.execute("DELETE FROM Cache WHERE rowid = ?", (rowid,))
                self._remove_spill(filename)
                size -= sz
                if size <= low_water:
                    break

    def delete(self, key: Any) -> bool:
        db_key = str(key)

        def body(c: sqlite3.Connection) -> bool:
            row = c.execute(
                "SELECT rowid, filename FROM Cache WHERE key = ?", (db_key,)
            ).fetchone()
            if row is None:
                return False
            self._delete_row(c, row[0], row[1])
            return True

        with self._lock:
            return self._write(body)

    def clear(self) -> int:
        with self._lock:
            n = self._conn.execute("SELECT value FROM Settings WHERE key = 'count'").fetchone()[0]
            self._write(lambda c: c.execute("DELETE FROM Cache"))
            shutil.rmtree(self._store, ignore_errors=True)
        return int(n)

    def expire(self, now: float | None = None) -> int:
        """Remove all expired entries; return the count removed."""
        cutoff = time.time() if now is None else now

        def body(c: sqlite3.Connection) -> int:
            rows = c.execute(
                "SELECT rowid, filename FROM Cache WHERE expire_time IS NOT NULL "
                "AND expire_time < ?",
                (cutoff,),
            ).fetchall()
            for rowid, filename in rows:
                self._delete_row(c, rowid, filename)
            return len(rows)

        with self._lock:
            return self._write(body)

    def volume(self) -> int:
        """Total bytes of cached values (trigger-maintained)."""
        with self._lock:
            return int(
                self._conn.execute("SELECT value FROM Settings WHERE key = 'size'").fetchone()[0]
            )

    def __len__(self) -> int:
        with self._lock:
            return int(
                self._conn.execute("SELECT value FROM Settings WHERE key = 'count'").fetchone()[0]
            )

    def __iter__(self) -> Iterator[str]:
        """Iterate keys, oldest-stored first. Snapshots up front so the cache can
        be mutated during iteration."""
        with self._lock:
            keys = [
                r[0] for r in self._conn.execute("SELECT key FROM Cache ORDER BY store_time")
            ]
        return iter(keys)

    iterkeys = __iter__

    def __contains__(self, key: Any) -> bool:
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT expire_time FROM Cache WHERE key = ?", (str(key),)
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
