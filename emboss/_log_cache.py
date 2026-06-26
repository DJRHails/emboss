"""LogCache — a log-structured, replication-safe cache with few inodes.

`FileCache` is replication-safe but writes one file per key — millions of inodes
for a big cache, and slow `du`/`rsync`/Syncthing scans. The obvious fix
(bundling many keys into one file per prefix) makes sync *worse*: two nodes both
rewriting the same `<prefix>` file means a syncer's last-write-wins discards a
whole bundle of one node's entries.

`LogCache` gets few inodes AND conflict-free sync by giving every writer its own
files. Entries are sharded by a stable hash of the key into 256 prefixes; within
a prefix each node appends to **its own** log:

    directory/<prefix>/<writer_id>.log     # written by exactly ONE node
    directory/<prefix>/<writer_id>.lock    # flock; mutually excludes append/compact

Because no file is ever written by two nodes, a file syncer (Syncthing, rsync)
just ships each node's logs around — last-write-wins never fires, so a sync
conflict can't lose data. Same-node processes sharing a node-log are serialised
by the per-writer lock file (a stable inode, never replaced, so it still excludes
an append racing a compaction's atomic rename). Inodes are bounded by
(#prefixes x #writers), not entry count.

- **Reads** consult a per-prefix in-memory index (built by scanning that
  prefix's logs once; rebuilt when a log's size changes, e.g. a peer's appends
  arrived via sync). A miss just recomputes — safe under eventual consistency.
- **Writes** append a length-framed record; a torn tail from a crash is ignored.
- **Deletes** append a tombstone.
- **Compaction** rewrites *this node's own* log (under its lock, atomic rename),
  dropping superseded / tombstoned / expired records; it never touches a peer's
  log, so it stays conflict-free. Auto-runs when a log passes `max_log_bytes`;
  also exposed as `compact()`. `size_limit` is a best-effort bound on each
  (prefix, writer) log's live bytes, applied at compaction (per-log, like
  `FanoutCache`'s per-shard split — not a single global cap).

Use `LogCache` for **many small entries on a shared/synced mount**, where
`FileCache`'s inode count hurts. For a bounded local cache use `SqliteCache`.

Benchmark (local SSD, `python scripts/bench.py`), order of magnitude:
    set ~10^3-10^4 ops/s     get ~10^4 ops/s
(get re-stats the prefix's logs to notice a peer's appends, so it trails
`SqliteCache`'s indexed read.) The win is inodes: ~256 files for *any* number of
entries — 20k entries used 512 files here, vs `FileCache`'s 20,000.

`writer_id` defaults to the hostname and **must be unique per node** (override it
if two nodes could share a hostname). Implements the `Cache` protocol subset
`@cached` needs plus dunders, `len()`, iteration, and `volume()`.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import pickle
import shutil
import socket
import struct
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

_MISSING = object()
_HEADER = struct.Struct(">I")  # 4-byte big-endian record length
_DEFAULT_MAX_LOG_BYTES = 4 * 2**20  # compact a node's per-prefix log past 4 MB
_PREFIX_WIDTH = 2  # 2 hex chars -> 256 shard dirs


class _Record(NamedTuple):
    key: str
    value: Any
    expire_time: float | None
    store_time: float
    deleted: bool


def _iter_records(path: Path) -> Iterator[_Record]:
    """Yield records from a log, stopping at the first torn/corrupt tail."""
    with open(path, "rb") as f:
        while True:
            header = f.read(_HEADER.size)
            if len(header) < _HEADER.size:
                return
            (length,) = _HEADER.unpack(header)
            blob = f.read(length)
            if len(blob) < length:
                return  # truncated final write (crash mid-append)
            try:
                yield _Record(*pickle.loads(blob))
            except Exception:  # noqa: BLE001 — a corrupt tail ends the log
                return


def _frame(rec: _Record) -> bytes:
    blob = pickle.dumps(tuple(rec), protocol=pickle.HIGHEST_PROTOCOL)
    return _HEADER.pack(len(blob)) + blob


def _sanitize(writer_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in writer_id) or "node"


def _flock(fileobj: Any, op: int) -> None:
    """Best-effort advisory lock (POSIX); no-op where fcntl is unavailable."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover — non-POSIX
        return
    with contextlib.suppress(OSError):
        fcntl.flock(fileobj.fileno(), op)


class LogCache:
    """Log-structured cache: per-writer, prefix-sharded append logs."""

    def __init__(
        self,
        directory: str | os.PathLike[str] = ".cache",
        size_limit: int | None = None,
        writer_id: str | None = None,
        max_log_bytes: int = _DEFAULT_MAX_LOG_BYTES,
        **_kwargs: Any,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.size_limit = size_limit
        self.max_log_bytes = max_log_bytes
        self.writer_id = _sanitize(writer_id or socket.gethostname())
        self._lock = threading.Lock()  # serialise this process's threads
        # Per-prefix: {prefix: {key: _Record}} + the per-log sizes the index was
        # built from, so we rebuild only when a log changed (incl. a peer's).
        self._index: dict[str, dict[str, _Record]] = {}
        self._sig: dict[str, dict[str, int]] = {}

    # ── layout ────────────────────────────────────────────────────────────────

    def _prefix(self, key: Any) -> str:
        return hashlib.md5(str(key).encode()).hexdigest()[:_PREFIX_WIDTH]

    def _log_path(self, prefix: str) -> Path:
        return self.directory / prefix / f"{self.writer_id}.log"

    @contextlib.contextmanager
    def _writer_lock(self, prefix: str) -> Iterator[None]:
        """Exclude same-node appends/compactions on this writer's log. The lock
        file is never replaced, so the flock stays valid across compaction's
        rename (unlike locking the log file itself)."""
        lock_path = self.directory / prefix / f"{self.writer_id}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(lock_path, "a+")  # noqa: SIM115
        try:
            import fcntl

            _flock(f, fcntl.LOCK_EX)
            yield
        except ImportError:  # pragma: no cover — non-POSIX: no cross-process lock
            yield
        finally:
            with contextlib.suppress(ImportError):
                import fcntl

                _flock(f, fcntl.LOCK_UN)
            f.close()

    # ── index ─────────────────────────────────────────────────────────────────

    def _current_sig(self, prefix: str) -> dict[str, int]:
        pdir = self.directory / prefix
        if not pdir.is_dir():
            return {}
        sig: dict[str, int] = {}
        for log in pdir.glob("*.log"):
            with contextlib.suppress(OSError):
                sig[log.name] = log.stat().st_size
        return sig

    def _ensure_index(self, prefix: str) -> dict[str, _Record]:
        sig = self._current_sig(prefix)
        if prefix in self._index and self._sig.get(prefix) == sig:
            return self._index[prefix]
        index: dict[str, _Record] = {}
        pdir = self.directory / prefix
        if pdir.is_dir():
            for log in sorted(pdir.glob("*.log")):
                with contextlib.suppress(OSError):
                    for rec in _iter_records(log):
                        cur = index.get(rec.key)
                        if cur is None or rec.store_time >= cur.store_time:
                            index[rec.key] = rec
        self._index[prefix] = index
        self._sig[prefix] = sig
        return index

    @staticmethod
    def _live(rec: _Record | None, now: float) -> bool:
        return (
            rec is not None
            and not rec.deleted
            and (rec.expire_time is None or rec.expire_time >= now)
        )

    # ── append / compaction ────────────────────────────────────────────────────

    def _append(self, prefix: str, rec: _Record) -> int:
        path = self._log_path(prefix)
        with self._writer_lock(prefix):
            with open(path, "ab") as f:
                f.write(_frame(rec))
            return path.stat().st_size

    def compact(self, prefix: str | None = None) -> None:
        """Rewrite this node's own log(s), dropping dead/superseded/expired records
        (and, if `size_limit` is set, oldest-stored live records). Never touches a
        peer's log, so it stays conflict-free under sync."""
        with self._lock:
            if prefix is not None:
                self._compact_prefix(prefix)
            else:
                for pdir in list(self.directory.iterdir()):
                    if pdir.is_dir():
                        self._compact_prefix(pdir.name)

    def _compact_prefix(self, prefix: str) -> None:
        path = self._log_path(prefix)
        if not path.exists():
            return
        now = time.time()
        with self._writer_lock(prefix):
            latest: dict[str, _Record] = {}
            for rec in _iter_records(path):
                cur = latest.get(rec.key)
                if cur is None or rec.store_time >= cur.store_time:
                    latest[rec.key] = rec
            keep = [r for r in latest.values() if self._live(r, now)]
            if self.size_limit is not None:
                keep = self._trim_to_limit(keep)
            keep.sort(key=lambda r: r.store_time)
            tmp = path.with_name(path.name + ".compact.tmp")
            with open(tmp, "wb") as out:
                for rec in keep:
                    out.write(_frame(rec))
            os.replace(tmp, path)
        self._index.pop(prefix, None)  # force rebuild
        self._sig.pop(prefix, None)

    def _trim_to_limit(self, records: list[_Record]) -> list[_Record]:
        """Best-effort size bound over THIS node's live records for one prefix."""
        assert self.size_limit is not None

        def vsize(rec: _Record) -> int:
            return len(pickle.dumps(rec.value, protocol=pickle.HIGHEST_PROTOCOL))

        if sum(vsize(r) for r in records) <= self.size_limit:
            return records
        kept: list[_Record] = []
        running = 0
        for rec in sorted(records, key=lambda r: r.store_time, reverse=True):  # newest first
            running += vsize(rec)
            if running > self.size_limit:
                break
            kept.append(rec)
        return kept

    # ── core API ────────────────────────────────────────────────────────────

    def get(self, key: Any, default: Any = None) -> Any:
        now = time.time()
        prefix = self._prefix(key)
        with self._lock:
            rec = self._ensure_index(prefix).get(str(key))
        return rec.value if rec is not None and self._live(rec, now) else default

    def set(self, key: Any, value: Any, expire: float | None = None, **_kwargs: Any) -> bool:
        now = time.time()
        prefix = self._prefix(key)
        rec = _Record(str(key), value, now + expire if expire else None, now, False)
        with self._lock:
            log_size = self._append(prefix, rec)
            index = self._ensure_index(prefix)
            index[rec.key] = rec
            self._sig.setdefault(prefix, {})[self._log_path(prefix).name] = log_size
            if log_size > self.max_log_bytes:
                self._compact_prefix(prefix)
        return True

    def delete(self, key: Any) -> bool:
        now = time.time()
        prefix = self._prefix(key)
        with self._lock:
            index = self._ensure_index(prefix)
            if not self._live(index.get(str(key)), now):
                return False
            tombstone = _Record(str(key), None, None, now, True)
            log_size = self._append(prefix, tombstone)
            index[tombstone.key] = tombstone
            self._sig.setdefault(prefix, {})[self._log_path(prefix).name] = log_size
        return True

    def clear(self) -> int:
        """Drop this node's view of the cache (removes local logs). Peers' logs
        re-appear via sync — clear is node-local for a replicated cache."""
        with self._lock:
            n = sum(1 for _ in self._iter_live())
            for pdir in list(self.directory.iterdir()):
                if pdir.is_dir():
                    shutil.rmtree(pdir, ignore_errors=True)
            self._index.clear()
            self._sig.clear()
        return n

    def volume(self) -> int:
        """Total bytes of the live value payloads (across all writers)."""
        with self._lock:
            return sum(
                len(pickle.dumps(rec.value, protocol=pickle.HIGHEST_PROTOCOL))
                for rec in self._iter_live()
            )

    def _iter_live(self) -> Iterator[_Record]:
        now = time.time()
        for pdir in self.directory.iterdir():
            if not pdir.is_dir():
                continue
            for rec in self._ensure_index(pdir.name).values():
                if self._live(rec, now):
                    yield rec

    def __len__(self) -> int:
        with self._lock:
            return sum(1 for _ in self._iter_live())

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            keys = [rec.key for rec in self._iter_live()]
        return iter(keys)

    iterkeys = __iter__

    def __contains__(self, key: Any) -> bool:
        return self.get(key, _MISSING) is not _MISSING

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

    def __enter__(self) -> LogCache:
        return self

    def __exit__(self, *_args: Any) -> None:
        pass

    def close(self) -> None:
        pass
