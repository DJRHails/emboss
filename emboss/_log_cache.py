"""LogCache — a log-structured, replication-safe cache with few inodes.

`FileCache` is replication-safe but writes one file per key — millions of inodes
for a big cache, and slow `du`/`rsync`/Syncthing scans. The obvious fix
(bundling many keys into one file per prefix) makes sync *worse*: two nodes both
rewriting the same `<prefix>` file means a syncer's last-write-wins discards a
whole bundle of one node's entries.

`LogCache` gets few inodes AND conflict-free sync by giving every writer its own
files. Entries are sharded by a stable hash of the key into prefixes; within a
prefix each node appends to **its own** log:

    directory/<prefix>/<writer_id>.log       # written by exactly ONE node
    directory/<prefix>/<writer_id>.lock      # flock; excludes append/compact races
    directory/<prefix>/spill/<sha256>.val    # large values — shared, content-addressed pool
    directory/<prefix>/<writer_id>.spill/*   # legacy layout — migrated into the pool on sight

Because no LOG is ever written by two nodes — and any two nodes writing the same
content-addressed pool file write identical bytes — a file syncer just ships each
node's files around: last-write-wins never loses data. Same-node processes
sharing a node-log are serialised by the per-writer lock file. Inodes are bounded by ~(#prefixes x #writers), not entry count.

- **Reads** consult a per-prefix in-memory index (built by scanning that
  prefix's logs once; rebuilt when a log's size changes — a peer's appends). The
  freshness check is throttled by `index_ttl`, so warm reads are an in-memory
  lookup. A miss just recomputes — safe under eventual consistency.
- **Writes** append a length-framed record; a torn tail from a crash is ignored.
- **Large values spill to a shared, content-addressed pool** (`min_file_size`,
  default 32 KB): the record holds a filename reference instead of the value,
  keeping the log small (so a 100 MB value doesn't balloon the append log). The
  name is the blob's sha256, so identical values collapse to ONE file per prefix
  across every writer, and re-spilling existing bytes is a no-op. Two nodes
  "conflicting" on a pool file write identical bytes, so a syncer's
  last-write-wins is harmless there — the no-two-writers rule only needs to hold
  for the logs. Pool files are deleted ONLY by the consolidation mark-and-sweep:
  the reference count that justifies a deletion is DERIVED from every log in the
  prefix at that moment (never stored or adjusted incrementally — a maintained
  on-disk counter would be shared mutable state a syncer could corrupt), and a
  zero-reference file must also outlive a grace window, because an in-flight
  `set()` spills before its record lands and a not-yet-synced peer log may still
  reference the content (a swept file a lagging peer references degrades to a
  miss-and-recompute — the module's standing eventual-consistency contract).
- **Deletes** append a tombstone.
- **Compaction** rewrites *this node's own* log (under its lock, atomic rename),
  dropping superseded / tombstoned / expired records. It never touches spill
  files (they are shared) or a peer's files. Auto-runs past `max_log_bytes`;
  also `compact()`.
- **Consolidation / GC** merges *all* writers' logs in a prefix into THIS node's
  single log, dropping dead records and pruning the now-redundant peer logs (and
  their spill files) — the missing cross-writer GC. Without it, file count is
  ~(#prefixes x #writers) and grows forever as nodes come and go (decommissioned
  hosts, one-shot bulk-import writers, containers that fell back to a random
  hostname). Sync-safe: it snapshots each peer log's `(size, mtime)` first and
  re-stats before delete, so a peer/local append that lands DURING consolidation
  is never deleted (its newer records win on read; the next pass collects it).
  Consolidation is also the migration point for the legacy per-writer spill
  layout (files adopted into the pool via hardlink, records repointed, the
  namespaces pruned with their logs — detected at index build, triggered on the
  next write) and the pool's mark-and-sweep GC. With the pre-pool uuid names
  every pass re-copied every foreign value, which (amplified by a syncer
  resurrecting pruned peer logs) once grew a 45 GB cache to 160 GB in a day;
  content addressing makes passes idempotent. Auto-runs past
  `max_writers_per_prefix` (with a cooldown); also `consolidate()`.

Tunables (defaults chosen via `python scripts/bench.py`):
- `index_ttl` (1.0 s) — index reuse before re-stat'ing for peers' appends; the
  dominant read lever (~10^4/s always-fresh -> ~10^5/s throttled), at the cost of
  up to `index_ttl` of staleness on cross-node writes (own writes immediate).
- `prefix_width` (2 -> 256 shards) — inodes vs per-log size. Aim ~1k entries per
  prefix: width 1 (<~10k entries), 2 (~10k-2M), 3 (>~2M); avoid >=4 (the parent
  dir then holds 65k+ subdirs — the cliff). Must match across a directory.
- `min_file_size` (32 KB) — values this big or larger spill to side files.
- `max_log_bytes` (4 MB) — per-log size that triggers compaction.
- `max_writers_per_prefix` (8) — distinct logs in a prefix before a write
  auto-consolidates them into one (bounds inode growth as writers accumulate).
  `0` disables auto-consolidation (still callable explicitly via `consolidate()`).

Benchmark (local SSD, 512-byte values), order of magnitude:
    set ~10^4 ops/s     get ~10^5 ops/s (index_ttl=1.0; ~10^4/s if index_ttl=0)
The headline win is inodes: ~256 files for *any* number of small entries.

`writer_id` defaults to the hostname and **must be unique per node**. Implements
the `Cache` protocol subset `@cached` needs plus dunders, `len()`, iteration,
`volume()`.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import pickle
import re
import shutil
import socket
import struct
import threading
import time
import uuid
from collections.abc import Iterator, Set as AbstractSet
from pathlib import Path
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

_MISSING = object()
_HEADER = struct.Struct(">I")  # 4-byte big-endian record length
_PROTO = b"\x80"  # pickle PROTO opcode: every blob (protocol >= 2) starts with it
_DEFAULT_MAX_LOG_BYTES = 4 * 2**20  # compact a node's per-prefix log past 4 MB
_DEFAULT_PREFIX_WIDTH = 2  # 2 hex chars -> 256 shard dirs
_DEFAULT_INDEX_TTL = 1.0  # seconds a per-prefix index is reused before re-stat'ing
_DEFAULT_MIN_FILE_SIZE = 2**15  # 32 KB — values this big or larger spill to files
_DEFAULT_MAX_WRITERS_PER_PREFIX = 8  # logs in a prefix before auto-consolidation
_SHARED_SPILL_DIR = "spill"  # the per-prefix shared, content-addressed spill pool
# Auto-consolidation cooldown per prefix: when pruning is blocked (busy peers, a syncer
# resurrecting logs) the writer count can sit above the bound indefinitely, and without a
# cooldown EVERY write re-triggers consolidation — a per-write O(prefix) storm. One minute
# keeps the fold responsive while bounding the worst case to one pass per prefix per minute.
_AUTO_CONSOLIDATE_COOLDOWN_S = 60.0


class _Record(NamedTuple):
    key: str
    value: Any  # the value (inline) or None when spilled
    expire_time: float | None
    store_time: float
    deleted: bool
    spill: str | None  # relative path to a side file holding the value, or None


def _parse_frame(data: bytes, pos: int) -> tuple[_Record, int] | None:
    """Try to parse one `[length][blob]` frame at `pos`; return `(record, end)` or
    `None` if the header is short, the blob runs past EOF, or the blob does not
    unpickle into a 6-field record."""
    if pos + _HEADER.size > len(data):
        return None
    (length,) = _HEADER.unpack(data[pos : pos + _HEADER.size])
    blob_start = pos + _HEADER.size
    blob_end = blob_start + length
    if blob_end > len(data):
        return None  # truncated / corrupt length field
    try:
        return _Record(*pickle.loads(data[blob_start:blob_end])), blob_end
    except Exception:  # noqa: BLE001 — torn/corrupt frame; caller resyncs
        return None


def _resync(data: bytes, after: int) -> int | None:
    """Find the start of the next valid frame strictly after byte `after`.

    Every blob begins with the pickle PROTO opcode (`0x80`), so each `0x80` in the
    stream is a candidate blob start whose frame begins `_HEADER.size` bytes
    earlier. Return the first such frame-start that re-parses; `None` at EOF. The
    search begins past `after`'s header so the returned frame is strictly after the
    torn one (guaranteeing forward progress)."""
    search = after + _HEADER.size + 1
    while True:
        idx = data.find(_PROTO, search)
        if idx < 0:
            return None
        frame_start = idx - _HEADER.size
        if frame_start > after and _parse_frame(data, frame_start) is not None:
            return frame_start
        search = idx + 1


class _ScanResult(NamedTuple):
    records: list[_Record]  # valid records in on-disk order, resynced past any tears
    tear_at: int | None  # byte offset of the first torn/corrupt frame, or None
    recovered: int  # valid records found *after* a tear (would be stranded by a stop-reader)


def _read_records(path: Path) -> _ScanResult:
    """Read every valid record from a log, **skipping torn/corrupt frames and resyncing** to the
    next valid frame rather than stopping at the first bad one.

    A torn frame (a crash or a concurrent container teardown mid-append) can land *mid-log* with
    valid frames after it. Stopping at the first bad frame — the old behaviour — stranded every
    record past the tear: invisible to reads, so each silently re-executes and re-bills forever.
    Here a bad frame is skipped (resynced via the PROTO-opcode anchor). A benign truncated *final*
    write (the documented crash-mid-append case) recovers nothing past it (`recovered == 0`).

    `OSError` from the read propagates — a caller that must not treat an unreadable log as empty
    (consolidation, which would then prune it) relies on that."""
    data = path.read_bytes()  # OSError propagates by design (see docstring)
    n = len(data)
    pos = 0
    records: list[_Record] = []
    tear_at: int | None = None
    recovered = 0
    while pos + _HEADER.size <= n:
        parsed = _parse_frame(data, pos)
        if parsed is not None:
            rec, end = parsed
            if tear_at is not None:
                recovered += 1
            records.append(rec)
            pos = end
            continue
        if tear_at is None:
            tear_at = pos
        nxt = _resync(data, pos)
        if nxt is None:
            break
        pos = nxt
    return _ScanResult(records, tear_at, recovered)


def _iter_records(path: Path) -> Iterator[_Record]:
    """Yield a log's records (resyncing past torn frames — see `_read_records`). Warns when a
    mid-log tear stranded later records, so the silent re-bill becomes visible; a benign
    truncated final write stays quiet. `OSError` propagates on first iteration."""
    scan = _read_records(path)
    if scan.recovered:
        logger.warning(
            "emboss.LogCache: torn/corrupt frame(s) in %s starting at byte %d — resynced past "
            "them and recovered %d later record(s) that a stop-at-first-tear reader would hide "
            "(they were re-executing on every read). compact()/consolidate() rewrites the log to "
            "drop the malformed frame(s) permanently.",
            path,
            scan.tear_at,
            scan.recovered,
        )
    yield from scan.records


def _frame(rec: _Record) -> bytes:
    blob = pickle.dumps(tuple(rec), protocol=pickle.HIGHEST_PROTOCOL)
    return _HEADER.pack(len(blob)) + blob


def _sanitize(writer_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in writer_id) or "node"


# Docker's default hostname is the 12-hex short container id — an EPHEMERAL identity.
_CONTAINER_HOSTNAME = re.compile(r"^[0-9a-f]{12}$")


def _default_writer_id() -> str:
    """The hostname — unless it is a bare container id, which collapses to a shared id.

    A container that inherits no explicit `writer_id` and no stable hostname would mint a
    brand-new writer namespace per container, growing the writer-log count forever — and each
    one crossing `max_writers_per_prefix` auto-consolidates, adopting every other writer's
    spills before the container dies (the incident that grew a 45 GB cache to 160 GB in a day,
    27,880 writer logs). Ephemeral containers share one 'container-orphan' writer instead:
    same-id writers are flock-serialised by design, so this is safe, merely slower under
    contention — and the warning makes the missing explicit `writer_id` visible."""
    hostname = socket.gethostname()
    if not _CONTAINER_HOSTNAME.match(hostname):
        return hostname
    logger.warning(
        "emboss.LogCache: hostname %r looks like a bare container id and no writer_id was "
        "given — using the shared 'container-orphan' writer id (pass an explicit stable "
        "writer_id per node).",
        hostname,
    )
    return "container-orphan"


def _flock(fileobj: Any, op: int) -> None:
    """Best-effort advisory lock (POSIX); no-op where fcntl is unavailable."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover — non-POSIX
        return
    with contextlib.suppress(OSError):
        fcntl.flock(fileobj.fileno(), op)


class LogCache:
    """Log-structured cache: per-writer, prefix-sharded append logs + spillover."""

    def __init__(
        self,
        directory: str | os.PathLike[str] = ".cache",
        size_limit: int | None = None,
        writer_id: str | None = None,
        max_log_bytes: int = _DEFAULT_MAX_LOG_BYTES,
        prefix_width: int = _DEFAULT_PREFIX_WIDTH,
        index_ttl: float = _DEFAULT_INDEX_TTL,
        min_file_size: int = _DEFAULT_MIN_FILE_SIZE,
        max_writers_per_prefix: int = _DEFAULT_MAX_WRITERS_PER_PREFIX,
        **_kwargs: Any,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.size_limit = size_limit
        self.max_log_bytes = max_log_bytes
        self.prefix_width = prefix_width
        self.index_ttl = index_ttl
        self.min_file_size = min_file_size
        self.max_writers_per_prefix = max_writers_per_prefix
        self.writer_id = _sanitize(writer_id or _default_writer_id())
        self._lock = threading.Lock()  # serialise this process's threads
        self._index: dict[str, dict[str, _Record]] = {}
        self._sig: dict[str, dict[str, int]] = {}
        self._checked: dict[str, float] = {}  # monotonic time of last freshness check
        self._consolidated_at: dict[str, float] = {}  # per-prefix auto-consolidation cooldown
        # Prefixes seen holding legacy per-writer `<writer>.spill/` dirs (the pre-pool
        # layout): detection happens at index build, and the next write kicks the
        # consolidation that migrates them into the shared pool — standard operation
        # itself never handles the legacy layout.
        self._needs_migration: set[str] = set()

    # ── layout ────────────────────────────────────────────────────────────────

    def _prefix(self, key: Any) -> str:
        return hashlib.md5(str(key).encode()).hexdigest()[: self.prefix_width]

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

    # ── spillover (large values -> side files) ─────────────────────────────────

    def _spill_write(self, prefix: str, blob: bytes) -> str:
        """Write a value blob to the prefix's **shared, content-addressed** spill
        pool; return its path relative to `directory` (so any reader can resolve
        it).

        The pool is `<prefix>/spill/<sha256>.val`, shared by ALL writers — the
        per-writer namespace existed for uuid-named files, where only the owning
        writer knew a file's lifecycle and the no-two-writers-one-file rule kept
        sync conflict-free. Content addressing dissolves both needs: any two
        writers producing the same name produce identical bytes (a syncer's
        last-write-wins between identical contents is a no-op), identical values
        collapse to ONE file per prefix across every writer, and re-spilling
        bytes already on disk costs nothing — which is what stops consolidation
        from duplicating the corpus (uuid names re-copied every foreign value on
        every pass; that grew a 45 GB cache to 160 GB in a day). An existing
        file is trusted as complete: it only appears via the atomic rename
        below, and same hash = same bytes. Shared files are GC'd only by the
        consolidation mark-and-sweep, never eagerly — several records (and
        several nodes) may reference one file."""
        rel_dir = f"{prefix}/{_SHARED_SPILL_DIR}"
        full_dir = self.directory / rel_dir
        full_dir.mkdir(parents=True, exist_ok=True)
        name = f"{hashlib.sha256(blob).hexdigest()}.val"
        full = full_dir / name
        rel = f"{rel_dir}/{name}"
        if full.exists():
            return rel
        # uuid tmp suffix: two writers spilling the same content must not collide
        # mid-write; every rename lands on the same (identical) final file.
        tmp = full.with_name(f"{name}.{uuid.uuid4().hex}.tmp")
        with open(tmp, "wb") as f:
            f.write(blob)
        os.replace(tmp, full)
        return rel

    def _pool_adopt(self, prefix: str, legacy_rel: str) -> str | None:
        """Bring a legacy per-writer-namespace spill into the shared pool without
        copying bytes when possible: hardlink it to its content-addressed name
        (same filesystem — free, and the inode survives the source namespace's
        later prune), with a byte-copy fallback across filesystems. Returns the
        pool rel path, or None when the source vanished (sync lag) — the caller
        drops the record rather than dangle a reference."""
        src = self.directory / legacy_rel
        h = hashlib.sha256()
        try:
            with open(src, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
        except OSError:
            return None
        rel_dir = f"{prefix}/{_SHARED_SPILL_DIR}"
        full_dir = self.directory / rel_dir
        full_dir.mkdir(parents=True, exist_ok=True)
        name = f"{h.hexdigest()}.val"
        full = full_dir / name
        rel = f"{rel_dir}/{name}"
        if full.exists():
            return rel
        tmp = full.with_name(f"{name}.{uuid.uuid4().hex}.tmp")
        try:
            os.link(src, tmp)
        except OSError:
            try:
                shutil.copyfile(src, tmp)
            except OSError:
                Path(tmp).unlink(missing_ok=True)
                return None
        os.replace(tmp, full)
        return rel

    def _spill_read(self, rel: str) -> Any:
        with open(self.directory / rel, "rb") as f:
            return pickle.load(f)

    def _spill_delete(self, rel: str) -> None:
        with contextlib.suppress(OSError):
            (self.directory / rel).unlink()

    def _is_shared_spill(self, prefix: str, rel: str) -> bool:
        return rel.startswith(f"{prefix}/{_SHARED_SPILL_DIR}/")



    def _rec_size(self, rec: _Record) -> int:
        if rec.spill:
            try:
                return (self.directory / rec.spill).stat().st_size
            except OSError:
                return 0
        return len(pickle.dumps(rec.value, protocol=pickle.HIGHEST_PROTOCOL))

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
        # Throttle the on-disk freshness check: within index_ttl of the last
        # check, reuse the in-memory index (own writes are applied to it directly,
        # so only a peer's writes can be missed — for up to index_ttl).
        now = time.monotonic()
        if (
            prefix in self._index
            and self.index_ttl > 0
            and now - self._checked.get(prefix, 0.0) < self.index_ttl
        ):
            return self._index[prefix]
        self._checked[prefix] = (
            now  # record the freshness check (incl. the first build)
        )
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
        # Legacy layout detection: a per-writer `<writer>.spill/` dir means this prefix
        # predates the shared pool — flag it so the next write kicks the consolidation
        # that migrates it. Detection only; no legacy handling on the read/write paths.
        if pdir.is_dir() and any(True for _ in pdir.glob("*.spill")):
            self._needs_migration.add(prefix)
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
        and their spill files (and, with `size_limit`, oldest-stored live records).
        Never touches a peer's files, so it stays conflict-free under sync."""
        with self._lock:
            if prefix is not None:
                self._compact_prefix(prefix)
            else:
                for pdir in list(self.directory.iterdir()):
                    if pdir.is_dir():
                        self._compact_prefix(pdir.name)

    def _compact_prefix(self, prefix: str) -> None:
        # Compaction never touches spill files: pool files are shared across writers
        # (and nodes), so only the consolidation mark-and-sweep — which reads EVERY
        # log — can know a file is unreferenced.
        path = self._log_path(prefix)
        if not path.exists():
            return
        now = time.time()
        with self._writer_lock(prefix):
            scan = _read_records(path)
            latest: dict[str, _Record] = {}
            for rec in scan.records:
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
        if scan.recovered:  # a real mid-log tear was healed; a benign final tear stays quiet
            logger.warning(
                "emboss.LogCache: compacted %s past a torn frame at byte %d — dropped the "
                "malformed frame(s) permanently and resynced %d later record(s).",
                path,
                scan.tear_at,
                scan.recovered,
            )
        self._index.pop(prefix, None)  # force rebuild
        self._sig.pop(prefix, None)

    # ── consolidation (cross-writer GC) ────────────────────────────────────────

    def consolidate(self, prefix: str | None = None) -> None:
        """Merge EVERY writer's log in a prefix into this node's single log,
        dropping dead/superseded/expired records, then prune the now-redundant
        peer logs (and their spills) — the cross-writer GC `compact()` lacks.

        Sync-safe: a peer/local append that lands during the merge is detected by
        a `(size, mtime)` re-stat and left intact (its newer records win on read;
        the next pass collects it), so a concurrent write is never lost. This is
        also the **legacy-spill migration point**: a record still referencing a
        per-writer `<writer>.spill/` file (the pre-pool layout) has its value
        adopted into the shared pool (hardlinked when the filesystem allows) and
        is repointed, and the legacy namespaces die with their pruned logs — so
        standard operation never handles legacy layout. A spill not on disk
        (sync lag) drops its record (a cache miss that recomputes) rather than
        dangle. Finally the pool is mark-and-swept: every log was just read, so
        the surviving reference set is complete, and unreferenced files past the
        grace window are deleted."""
        with self._lock:
            if prefix is not None:
                self._consolidate_prefix(prefix)
            else:
                for pdir in list(self.directory.iterdir()):
                    if pdir.is_dir():
                        self._consolidate_prefix(pdir.name)

    def _consolidate_prefix(self, prefix: str) -> None:
        pdir = self.directory / prefix
        if not pdir.is_dir():
            return
        now = time.time()
        target = self._log_path(prefix)  # our log — the consolidation destination
        with self._writer_lock(prefix):  # serialise same-node writes to our log
            # Snapshot every source log's identity BEFORE the merge. A peer (or a
            # same-node process holding a different writer_id) may append to its
            # own log concurrently; we compare this snapshot to a re-stat before
            # deleting, so a write that lands mid-consolidation is never lost.
            snapshot: dict[str, tuple[int, int]] = {}
            for log in pdir.glob("*.log"):
                with contextlib.suppress(OSError):
                    st = log.stat()
                    snapshot[log.name] = (st.st_size, st.st_mtime_ns)
            keep, unread = self._collect_live_across_logs(pdir, snapshot, now)
            consolidated = self._resolve_spills(prefix, keep)
            self._write_consolidated(target, consolidated)
            self._prune_consolidated_sources(pdir, snapshot, target.name, unread)
            self._sweep_shared_spills(pdir, consolidated, unread, now)
            self._sweep_orphaned_namespaces(pdir, consolidated, unread, now)
        self._index.pop(prefix, None)  # force rebuild
        self._sig.pop(prefix, None)
        self._checked.pop(prefix, None)
        self._needs_migration.discard(prefix)
        self._consolidated_at[prefix] = time.time()

    # A shared-pool file must outlive any in-flight `set()` that wrote it before its
    # record landed, any not-yet-synced peer log that references it, and any read in
    # progress. One hour of grace covers all three with a wide margin; an unreferenced
    # file merely waits one more consolidation pass.
    _SHARED_SPILL_GRACE_S = 3600.0

    def _sweep_shared_spills(
        self,
        pdir: Path,
        consolidated: list[_Record],
        unread: AbstractSet[str],
        now: float,
    ) -> None:
        """Mark-and-sweep GC of the prefix's shared spill pool.

        Shared-pool files are never deleted eagerly (several records — and several
        nodes — may reference one content hash), so consolidation is where they are
        collected: it has just merged EVERY readable log, so `consolidated` is the
        complete local reference set. Skipped entirely when any source log could not
        be read — its references are unknown, and deleting a file it points at would
        turn recoverable state into misses. A file younger than the grace window is
        kept even when unreferenced: an in-flight `set()` spills BEFORE appending its
        record, and a syncer may deliver a peer's spill before its log. A swept file
        that a lagging peer still references degrades to a miss-and-recompute, the
        module's standing eventual-consistency contract."""
        if unread:
            return
        pool = pdir / _SHARED_SPILL_DIR
        if not pool.is_dir():
            return
        prefix = pdir.name
        kept = {r.spill for r in consolidated if r.spill}
        for val in pool.glob("*.val"):
            rel = f"{prefix}/{_SHARED_SPILL_DIR}/{val.name}"
            if rel in kept:
                continue
            try:
                if now - val.stat().st_mtime < self._SHARED_SPILL_GRACE_S:
                    continue
            except OSError:
                continue
            self._spill_delete(rel)

    def _sweep_orphaned_namespaces(
        self,
        pdir: Path,
        consolidated: list[_Record],
        unread: AbstractSet[str],
        now: float,
    ) -> None:
        """Collect legacy `<writer>.spill/` namespaces whose LOG is already gone.

        Prune-with-log covers a namespace whose log this pass consolidated; a dir
        that outlived its log (pre-pool passes, a crash between the log unlink and
        the rmtree, a syncer copying a dir without its log) is otherwise immortal —
        nothing references its files, yet its existence re-arms the migration flag
        on every index build, re-triggering a pointless consolidation per cooldown
        forever. Same guards as the pool sweep: skipped when any source log was
        unreadable (unknown references), skipped while any consolidated record
        still points into the dir (`_resolve_spills` repoints them, so this only
        bites mid-migration edge cases), and postponed while the dir or any file
        in it is younger than the grace window. The dir's own mtime matters: at
        the empty moment — a syncer that created the dir before copying its
        files, a pre-pool writer between mkdir and its first spill write (no log
        exists yet on a first large `set()`) — there are no files to be young,
        yet sweeping would yank the dir out from under the writer mid-flight. A
        dir's mtime updates on entry creation/removal, so aged orphans (empty or
        not) still sweep on the first pass."""
        if unread:
            return
        prefix = pdir.name
        kept = {r.spill for r in consolidated if r.spill}
        for sdir in pdir.glob("*.spill"):
            writer = sdir.name.removesuffix(".spill")
            if (pdir / f"{writer}.log").exists():
                continue  # owned: the log-prune path removes them together
            try:
                entries = list(sdir.iterdir())
            except OSError:
                continue
            if any(f"{prefix}/{sdir.name}/{entry.name}" in kept for entry in entries):
                continue
            try:
                if any(
                    now - p.stat().st_mtime < self._SHARED_SPILL_GRACE_S
                    for p in (sdir, *entries)
                ):
                    continue
            except OSError:
                continue
            shutil.rmtree(sdir, ignore_errors=True)

    def _collect_live_across_logs(
        self, pdir: Path, sources: dict[str, tuple[int, int]], now: float
    ) -> tuple[list[_Record], AbstractSet[str]]:
        """Merge all source logs into the live set: latest `store_time` wins per
        key (a newer overwrite/tombstone under ANY writer beats an older one).
        `sorted(sources)` makes the merge order deterministic for tie handling.

        Returns `(keep, unread)`: the records to keep, and the names of source
        logs that could NOT be fully read — a transient read error must not make
        a log prunable, or its unmerged records would be lost."""
        latest: dict[str, _Record] = {}
        unread: set[str] = set()
        for name in sorted(sources):
            try:
                scan = _read_records(pdir / name)  # OSError → unread (below), never prune
            except OSError:
                unread.add(name)  # transient read error → never prune this source
                continue
            if scan.recovered:  # real mid-log tear; a benign final tear is dropped quietly
                logger.warning(
                    "emboss.LogCache: consolidating %s past a torn frame at byte %d — merging "
                    "%d recovered record(s) into %s and dropping the malformed frame(s).",
                    pdir / name,
                    scan.tear_at,
                    scan.recovered,
                    self.writer_id,
                )
            for rec in scan.records:
                cur = latest.get(rec.key)
                if cur is None or rec.store_time >= cur.store_time:
                    latest[rec.key] = rec
        keep = [r for r in latest.values() if self._live(r, now)]
        if self.size_limit is not None:
            keep = self._trim_to_limit(keep)
        keep.sort(key=lambda r: r.store_time)
        return keep, unread

    def _resolve_spills(self, prefix: str, keep: list[_Record]) -> list[_Record]:
        """Make every kept spill reference point at an existing pool file.

        A pool reference is kept when its file exists and dropped when it does
        not (sync lag: the log arrived before the value; the record recomputes
        on next read rather than dangle). A **legacy** per-writer reference (the
        pre-pool layout) is migrated: its bytes are adopted into the pool under
        their content hash — hardlinked when the filesystem allows, so migration
        moves no bytes and the inode survives the legacy namespace's prune — and
        the record repointed. Content addressing makes all of this idempotent:
        re-consolidating (e.g. against a syncer that resurrects pruned peer
        logs) can never duplicate the corpus. The raw bytes are already pickled
        and are never unpickled/repickled."""
        consolidated: list[_Record] = []
        migrated = 0
        for rec in keep:
            if not rec.spill:
                consolidated.append(rec)
                continue
            if self._is_shared_spill(prefix, rec.spill):
                if (self.directory / rec.spill).exists():
                    consolidated.append(rec)
                continue  # pool file absent (sync lag) → drop, never dangle
            new_rel = self._pool_adopt(prefix, rec.spill)
            if new_rel is None:
                continue  # legacy spill absent → drop, never dangle
            migrated += 1
            consolidated.append(rec._replace(spill=new_rel, value=None))
        if migrated:
            logger.info(
                "emboss.LogCache: migrated %d legacy per-writer spill(s) in %s/ into the "
                "shared pool.",
                migrated,
                prefix,
            )
        return consolidated

    @staticmethod
    def _write_consolidated(target: Path, consolidated: list[_Record]) -> None:
        """Write the merged records to our log atomically (fsync + rename). An
        empty result means nothing live remains → drop our log entirely rather
        than leave a zero-record file lying around."""
        if not consolidated:
            target.unlink(missing_ok=True)
            return
        tmp = target.with_name(target.name + ".consolidate.tmp")
        with open(tmp, "wb") as out:
            for rec in consolidated:
                out.write(_frame(rec))
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, target)

    def _prune_consolidated_sources(
        self,
        pdir: Path,
        snapshot: dict[str, tuple[int, int]],
        target_name: str,
        unread: AbstractSet[str],
    ) -> None:
        """Delete each source log (and its spill dir + lock) now fully represented
        in our consolidated log — but ONLY if it is byte-identical to the snapshot.
        A changed `(size, mtime)` means it was appended to during the merge: those
        newer records aren't in our log, so we leave it (they win on read; the
        next pass folds them in). This is the sync-safety guarantee. A source we
        could not fully read (`unread`) is likewise kept — its unmerged records
        aren't in our log, so deleting it would lose them."""
        for name, (size, mtime) in snapshot.items():
            if name == target_name or name in unread:
                continue  # our own log is the destination / a partial read → keep
            src = pdir / name
            try:
                st = src.stat()
            except OSError:
                continue  # already gone (e.g. a peer compacted/cleared it)
            if (st.st_size, st.st_mtime_ns) != (size, mtime):
                continue  # appended-to mid-consolidation → keep; never lose a write
            writer = name[:-4]  # strip ".log"
            src.unlink(missing_ok=True)
            shutil.rmtree(pdir / f"{writer}.spill", ignore_errors=True)
            (pdir / f"{writer}.lock").unlink(missing_ok=True)

    def _trim_to_limit(self, records: list[_Record]) -> list[_Record]:
        """Best-effort size bound over THIS node's live records for one prefix."""
        assert self.size_limit is not None
        if sum(self._rec_size(r) for r in records) <= self.size_limit:
            return records
        kept: list[_Record] = []
        running = 0
        for rec in sorted(
            records, key=lambda r: r.store_time, reverse=True
        ):  # newest first
            running += self._rec_size(rec)
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
        if rec is None or not self._live(rec, now):
            return default
        if rec.spill:
            try:
                return self._spill_read(rec.spill)
            except OSError:
                return default  # spill not present yet (e.g. log synced before it) -> miss
            except Exception:  # noqa: BLE001 — corrupt/unpicklable spill: warn, miss (don't crash)
                logger.warning(
                    "emboss.LogCache: spill file %s for key %r is present but "
                    "unreadable (corrupt/partial write); treating as a miss.",
                    rec.spill,
                    key,
                )
                return default
        return rec.value

    def set(
        self, key: Any, value: Any, expire: float | None = None, **_kwargs: Any
    ) -> bool:
        now = time.time()
        prefix = self._prefix(key)
        expire_time = now + expire if expire else None
        blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        if len(blob) >= self.min_file_size:
            rec = _Record(
                str(key), None, expire_time, now, False, self._spill_write(prefix, blob)
            )
        else:
            rec = _Record(str(key), value, expire_time, now, False, None)
        with self._lock:
            index = self._ensure_index(prefix)
            # Spill lifecycle is entirely the consolidation sweep's job: pool files
            # are shared across writers and nodes, so no write path may delete one
            # (a superseded value's file lingers at most until the next sweep; a
            # spill orphaned by an append failure below likewise).
            log_size = self._append(prefix, rec)
            index[rec.key] = rec
            self._sig.setdefault(prefix, {})[self._log_path(prefix).name] = log_size
            if log_size > self.max_log_bytes:
                self._compact_prefix(prefix)
            elif (
                self.max_writers_per_prefix
                and (
                    len(self._sig.get(prefix, {})) > self.max_writers_per_prefix
                    or prefix in self._needs_migration
                )
                and now - self._consolidated_at.get(prefix, 0.0)
                > _AUTO_CONSOLIDATE_COOLDOWN_S
            ):
                # Cheap GC trigger: `self._sig[prefix]` is the per-prefix
                # `{logname: size}` already maintained for the index, so its
                # length counts distinct writer logs without a fresh glob on the
                # hot write path. Past the bound — or when the prefix still holds
                # the legacy per-writer spill layout — fold every writer's log
                # into ours, migrate legacy spills into the pool, and prune the
                # redundant peers (sync-safe). The cooldown stops the trigger
                # from storming when pruning is blocked (busy peers / a syncer
                # resurrecting logs keep the count above the bound), which used
                # to cost a full consolidation on EVERY write.
                self._consolidate_prefix(prefix)
        return True

    def delete(self, key: Any) -> bool:
        now = time.time()
        prefix = self._prefix(key)
        with self._lock:
            index = self._ensure_index(prefix)
            old = index.get(str(key))
            if not self._live(old, now):
                return False
            tombstone = _Record(str(key), None, None, now, True, None)
            log_size = self._append(prefix, tombstone)
            index[tombstone.key] = tombstone
            self._sig.setdefault(prefix, {})[self._log_path(prefix).name] = log_size
            # The tombstoned value's pool file is collected by the consolidation
            # sweep once nothing references it — never deleted on the write path.
        return True

    def clear(self) -> int:
        """Drop this node's view of the cache (removes local files). Peers' files
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
            return sum(self._rec_size(rec) for rec in self._iter_live())

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
