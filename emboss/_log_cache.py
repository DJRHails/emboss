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
  `set()` spills before its record lands (a dedup hit refreshes the file's age)
  and a not-yet-synced peer log may still reference the content (age counts from
  `max(mtime, ctime)`, so a synced-in file gets its grace from local arrival; a
  swept file a lagging peer references degrades to a miss-and-recompute — the
  module's standing eventual-consistency contract).
- **Deletes** append a tombstone.
- **Compaction** rewrites *this node's own* log (under its lock, atomic rename),
  dropping superseded / tombstoned / expired records. It never touches spill
  files (they are shared) or a peer's files. Auto-runs past `max_log_bytes` — a
  prefix holding ONLY this writer's log consolidates instead (cooldown-bounded,
  compacting in between), so a lone writer's pool still gets swept; also
  `compact()`.
- **Consolidation / GC** compacts THIS node's log against every writer's records
  (a record superseded under ANY writer is dropped), migrates our own legacy
  per-writer spill layout into the pool, and mark-and-sweeps the shared pool.
  **It never touches a peer's log by default.** Under a file syncer a peer's log
  here is a REPLICA whose owner may hold an un-synced tail: byte-stable locally
  does not mean complete, no local check can tell the difference, and a deletion
  here propagates back and destroys the owner's original (the 2026-07-10
  fleet-cache clobber — a janitor consolidating torn mid-sync replicas deleted
  the owners' logs, losing every record the replicas had not yet received).
  Cross-writer GC — folding a writer's log into ours and removing it — runs only
  for writers the caller explicitly names in `consolidate(prune_writer_ids=...)`,
  asserting they will never write again (a decommissioned host, a one-shot
  bulk importer, a retired container id). Even then the pruned log is
  snapshotted `(size, mtime)` first and re-stat'd before delete, so an append
  landing DURING the pass is never lost. With the pre-pool uuid names every
  pass re-copied every foreign value, which once grew a 45 GB cache to 160 GB
  in a day; content addressing makes passes idempotent. The size trigger
  auto-runs a (prune-free) pass on a lone-writer prefix; also `consolidate()`.

Tunables (defaults chosen via `python scripts/bench.py`):
- `index_ttl` (1.0 s) — index reuse before re-stat'ing for peers' appends; the
  dominant read lever (~10^4/s always-fresh -> ~10^5/s throttled), at the cost of
  up to `index_ttl` of staleness on cross-node writes (own writes immediate).
- `prefix_width` (2 -> 256 shards) — inodes vs per-log size. Aim ~1k entries per
  prefix: width 1 (<~10k entries), 2 (~10k-2M), 3 (>~2M); avoid >=4 (the parent
  dir then holds 65k+ subdirs — the cliff). Must match across a directory.
- `min_file_size` (32 KB) — values this big or larger spill to side files.
- `max_log_bytes` (4 MB) — per-log size that triggers compaction.

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
import io
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
from collections.abc import Callable, Iterator, MutableSet, Set as AbstractSet
from concurrent.futures import ThreadPoolExecutor
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
_SHARED_SPILL_DIR = "spill"  # the per-prefix shared, content-addressed spill pool
# Auto-consolidation cooldown per prefix: when the trigger condition cannot clear (a live
# set that cannot shrink below max_log_bytes, a legacy namespace that resists removal),
# without a cooldown EVERY write re-triggers consolidation — a per-write O(prefix) storm.
# One minute keeps the pass responsive while bounding the worst case to one per minute.
_AUTO_CONSOLIDATE_COOLDOWN_S = 60.0
# Batch compact/consolidate fan the prefixes across a thread pool: the work is file-I/O
# bound (log reads, hashing and renames release the GIL), per-prefix state is independent,
# and the per-(prefix, writer) flock — which excludes same-process threads too, each task
# opening its own fd — already serialises same-prefix races. Not a knob: an internal
# ceiling balancing disk queue depth against fd/flock pressure (a 4,096-prefix production
# pass dropped from ~6 minutes serial to well under one).
_BATCH_POOL_WIDTH = 16


class _Record(NamedTuple):
    # Positional on-disk format: `_frame` pickles a plain 6-tuple and `_parse_frame`
    # rebuilds by position — never reorder or grow the fields without a migration.
    key: str
    value: Any  # the value (inline) or None when spilled
    expire_time: float | None
    store_time: float
    deleted: bool
    spill: str | None  # relative path to a side file holding the value, or None


def _record_from(obj: Any) -> _Record | None:
    """Build a `_Record` from an unpickled value, or `None` if it is not a genuine
    6-field record with the right field types. Arity alone (what `_Record(*obj)`
    checks) is not enough: `_resync` probes arbitrary byte offsets, so a false
    anchor can unpickle into any 6-element iterable (e.g. the string `"abcdef"`).
    Admitting one would seat a poison record in the index whose mistyped
    `store_time`/`expire_time` raises `TypeError` on the read path — the exact
    'a corrupt log breaks reads' outcome resync exists to prevent."""
    if not (isinstance(obj, tuple) and len(obj) == 6):
        return None
    key, _value, expire_time, store_time, deleted, spill = obj
    if not isinstance(key, str):
        return None
    if expire_time is not None and not isinstance(expire_time, (int, float)):
        return None
    if not isinstance(store_time, (int, float)):
        return None
    if not isinstance(deleted, bool):
        return None
    if spill is not None and not isinstance(spill, str):
        return None
    return _Record(*obj)


def _parse_frame(data: bytes, pos: int) -> tuple[_Record, int] | None:
    """Try to parse one `[length][blob]` frame at `pos`; return `(record, end)` or
    `None` if the header is short, the blob runs past EOF, the pickle does not
    consume exactly `length` bytes, or it does not decode into a 6-field record.

    The exact-length check is what makes `_resync` trustworthy: a genuine frame's
    blob is a single pickle with no trailing bytes, so any false anchor whose
    `length` header overshoots the real pickle is rejected here rather than seated
    as a spurious record."""
    if pos + _HEADER.size > len(data):
        return None
    (length,) = _HEADER.unpack(data[pos : pos + _HEADER.size])
    blob_start = pos + _HEADER.size
    blob_end = blob_start + length
    if blob_end > len(data):
        return None  # truncated / corrupt length field
    reader = io.BytesIO(data[blob_start:blob_end])
    try:
        obj = pickle.Unpickler(reader).load()
    except Exception:  # noqa: BLE001 — torn/corrupt frame; caller resyncs
        return None
    if reader.tell() != length:
        return None  # trailing bytes → not a genuine frame boundary
    rec = _record_from(obj)
    return (rec, blob_end) if rec is not None else None


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
    # valid records found *after* a tear (would be stranded by a stop-reader)
    recovered: int


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
    brand-new writer namespace per container, growing the writer-log count forever (the
    incident that reached 27,880 writer logs — and, under the since-removed writer-count
    auto-consolidation, grew a 45 GB cache to 160 GB in a day). Ephemeral containers share
    one 'container-orphan' writer instead.
    On a single host that is safe — same-id writers sharing a filesystem are flock-serialised,
    merely slower under contention. Ephemeral containers on DIFFERENT hosts whose directories
    are joined by a file syncer would share one log across nodes, the one thing the design
    cannot survive (the syncer's last-write-wins may discard one side's appends), so multi-host
    container fleets MUST pass an explicit stable `writer_id` per node — the warning makes the
    missing one visible."""
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


try:
    import fcntl as _fcntl_probe  # noqa: F401 — availability probe; call sites import lazily

    _HAS_FCNTL = True
except ImportError:  # pragma: no cover — non-POSIX
    _HAS_FCNTL = False


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
        **_kwargs: Any,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.size_limit = size_limit
        self.max_log_bytes = max_log_bytes
        self.prefix_width = prefix_width
        self.index_ttl = index_ttl
        self.min_file_size = min_file_size
        self.writer_id = _sanitize(writer_id or _default_writer_id())
        self._lock = threading.Lock()  # serialise this process's threads
        self._index: dict[str, dict[str, _Record]] = {}
        self._sig: dict[str, dict[str, int]] = {}
        self._checked: dict[str, float] = {}  # monotonic time of last freshness check
        # per-prefix auto-consolidation cooldown (monotonic timestamps)
        self._consolidated_at: dict[str, float] = {}
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
        bytes already on disk writes no new file — which is what stops
        consolidation from duplicating the corpus (uuid names re-copied every
        foreign value on every pass; that grew a 45 GB cache to 160 GB in a
        day). An existing file is trusted only when its SIZE matches the blob:
        rename is atomic for the name, not the data, so a crash (or a
        non-atomic syncer) can leave a truncated file under the right name — a
        size mismatch is warned about and re-spilled (a fresh tmp, atomically
        renamed over the poisoned name), so re-storing the value heals
        truncation rather than trusting it forever. (Same-SIZE corruption is
        not detected here; a corrupt spill surfaces as a warned read-time
        miss.) A dedup hit also refreshes the
        file's age (`os.utime`), so the sweep's grace window protects this
        in-flight `set()` exactly as it would a fresh spill. Shared files are
        GC'd only by the consolidation mark-and-sweep, never eagerly — several
        records (and several nodes) may reference one file."""
        rel_dir = f"{prefix}/{_SHARED_SPILL_DIR}"
        full_dir = self.directory / rel_dir
        full_dir.mkdir(parents=True, exist_ok=True)
        name = f"{hashlib.sha256(blob).hexdigest()}.val"
        full = full_dir / name
        rel = f"{rel_dir}/{name}"
        try:
            found = full.stat().st_size
            if found == len(blob):
                os.utime(full)  # grace window covers this not-yet-appended record
                return rel
            logger.warning(
                "emboss.LogCache: pool file %s exists with size %d, expected %d — "
                "rewriting it (crash artifact or non-atomic sync delivery).",
                full,
                found,
                len(blob),
            )
        except OSError:
            pass  # absent (or unstat-able): write it below
        # uuid tmp suffix: two writers spilling the same content must not collide
        # mid-write; every rename lands on the same (identical) final file.
        tmp = full.with_name(f"{name}.{uuid.uuid4().hex}.tmp")
        with open(tmp, "wb") as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())  # rename is atomic for the name, not the data
        os.replace(tmp, full)
        return rel

    def _pool_adopt(self, prefix: str, legacy_rel: str) -> str | None:
        """Bring a legacy per-writer-namespace spill into the shared pool without
        copying bytes when possible: hardlink it to its content-addressed name
        (same filesystem — free, and the inode survives the source namespace's
        later prune), with a byte-copy fallback across filesystems. Returns the
        pool rel path, or None when the source VANISHED (sync lag) — the caller
        drops the record rather than dangle a reference. Any other `OSError`
        (permissions, I/O, disk full) propagates: the source bytes are still on
        disk, so the caller must keep the namespace for a retry instead of
        treating a transient local fault as sync lag and pruning the only copy."""
        src = self.directory / legacy_rel
        h = hashlib.sha256()
        size = 0
        try:
            with open(src, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
                    size += len(chunk)
        except FileNotFoundError:
            return None
        rel_dir = f"{prefix}/{_SHARED_SPILL_DIR}"
        full_dir = self.directory / rel_dir
        full_dir.mkdir(parents=True, exist_ok=True)
        name = f"{h.hexdigest()}.val"
        full = full_dir / name
        rel = f"{rel_dir}/{name}"
        try:
            found = full.stat().st_size
            if found == size:
                return rel
            logger.warning(
                "emboss.LogCache: pool file %s exists with size %d, expected %d — "
                "re-adopting over it (crash artifact or non-atomic sync delivery).",
                full,
                found,
                size,
            )
        except OSError:
            pass  # absent (or unstat-able): adopt it below
        tmp = full.with_name(f"{name}.{uuid.uuid4().hex}.tmp")
        try:
            os.link(src, tmp)
        except FileNotFoundError:
            return None  # source vanished between the hash and the link
        except OSError:  # cross-filesystem / links unsupported → byte-copy
            try:
                shutil.copyfile(src, tmp)
                with open(tmp, "rb+") as f:
                    os.fsync(f.fileno())  # see _spill_write: durable before named
            except FileNotFoundError:
                Path(tmp).unlink(missing_ok=True)
                return None
            except OSError:
                Path(tmp).unlink(missing_ok=True)
                raise
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
        # Legacy layout detection: OUR OWN `<writer_id>.spill/` dir means this prefix
        # predates the shared pool — flag it so the next write kicks the consolidation
        # that migrates it. Only our own namespace arms the flag: a peer's is the
        # peer's to migrate (its log references it, and consolidation no longer
        # touches peer files), so arming on it would re-trigger a pass forever.
        if pdir.is_dir() and (pdir / f"{self.writer_id}.spill").is_dir():
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
        (and, with `size_limit`, oldest-stored live records). Spill files are shared
        and left to the consolidation mark-and-sweep. Never touches a peer's files,
        so it stays conflict-free under sync."""
        if prefix is not None:
            with self._lock:
                self._compact_prefix(prefix)
            return
        self._run_batch(self._compact_prefix)

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
        # a real mid-log tear was healed; a benign final tear stays quiet
        if scan.recovered:
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

    def consolidate(
        self,
        prefix: str | None = None,
        prune_writer_ids: AbstractSet[str] | None = None,
    ) -> None:
        """Compact this node's log against EVERY writer's records (a record
        superseded under any writer is dropped from ours), migrate our own
        legacy spill layout into the pool, and mark-and-sweep the pool.

        **Peer logs are never touched by default.** Under a file syncer a peer's
        log here is a replica whose owner may hold an un-synced tail — locally
        byte-stable is NOT complete, and deleting (or rewriting) the replica
        propagates back and destroys the owner's original along with every
        record the replica never had. A torn tail in a peer log is likewise
        expected mid-sync state, not damage to repair.

        `prune_writer_ids` is the explicit cross-writer GC: for exactly those
        writers — which the caller asserts will NEVER write again (a
        decommissioned host, a one-shot bulk importer, a retired container id)
        — the winning records are folded into our log and the writer's log,
        lock, and legacy namespace are removed. A pruned log is still guarded
        by a `(size, mtime)` snapshot re-stat, so an append landing during the
        pass (the assertion being wrong) is never lost — the log is kept and
        the next pass retries.

        Legacy-spill migration: a kept record referencing a per-writer
        `<writer>.spill/` file (the pre-pool layout) has its value adopted into
        the shared pool (hardlinked when the filesystem allows) and is
        repointed; our own namespace is removed once its records are repointed,
        a pruned writer's dies with its log, and an unpruned peer's is left
        alone (its log still references it). A spill not on disk (sync lag)
        drops its record (a cache miss that recomputes) rather than dangle.
        Finally the pool is mark-and-swept against every live record across ALL
        logs (skipped when any log was unreadable — the reference set would be
        incomplete): unreferenced files past the grace window are deleted."""
        prune = frozenset(_sanitize(w) for w in prune_writer_ids or ())
        if prefix is not None:
            with self._lock:
                self._consolidate_prefix(prefix, prune)
            return
        self._run_batch(lambda p: self._consolidate_prefix(p, prune))

    def _run_batch(self, per_prefix: Callable[[str], None]) -> None:
        """Run a per-prefix pass over every prefix dir, fanned across a thread pool.

        Deliberately NOT under `self._lock`: holding it for a whole multi-minute batch
        blocks every same-process `get()`/`set()`. On-disk safety never needed it — every
        file mutation sits under the per-(prefix, writer) flock, which also excludes this
        process's own threads (each task opens its own lock fd) — and the in-memory maps
        tolerate racing single-op mutations as bounded `index_ttl`-style staleness (a
        just-invalidated index rebuilds on the next read). Where `fcntl` is unavailable
        the flock is a no-op and nothing would exclude a racing same-process `set()`, so
        the batch keeps the old serial pass under `self._lock` (which, as before #16,
        stops at the first raising prefix). On the fanned path one prefix's failure
        doesn't abort the others; every suppressed failure is logged with its prefix and
        the first error re-raises once the batch completes."""
        prefixes = [pdir.name for pdir in self.directory.iterdir() if pdir.is_dir()]
        if not prefixes:
            return
        if (
            not _HAS_FCNTL
        ):  # non-POSIX: no flock, so the batch must not race same-process ops
            with self._lock:
                for prefix in prefixes:
                    per_prefix(prefix)
            return
        errors: list[tuple[str, BaseException]] = []
        with ThreadPoolExecutor(
            max_workers=min(_BATCH_POOL_WIDTH, len(prefixes))
        ) as pool:
            futures = [(prefix, pool.submit(per_prefix, prefix)) for prefix in prefixes]
            for prefix, future in futures:
                try:
                    future.result()
                except BaseException as err:  # noqa: BLE001 — finish the batch, raise after
                    errors.append((prefix, err))
        if not errors:
            return
        for prefix, err in errors[1:]:
            logger.warning(
                "emboss.LogCache: batch pass failed for prefix %s/ — suppressed in favour "
                "of the first error, which re-raises: %r",
                prefix,
                err,
            )
        raise errors[0][1]

    def _consolidate_prefix(
        self, prefix: str, prune: AbstractSet[str] = frozenset()
    ) -> None:
        pdir = self.directory / prefix
        if not pdir.is_dir():
            return
        now = time.time()
        target = self._log_path(prefix)  # our log — the consolidation destination
        prune_logs = {f"{w}.log" for w in prune} - {target.name}
        with self._writer_lock(prefix):  # serialise same-node writes to our log
            # Snapshot every source log's identity BEFORE the merge. A pruned
            # writer (or a same-node process holding that writer_id) may append
            # concurrently; we compare this snapshot to a re-stat before
            # deleting, so a write that lands mid-consolidation is never lost.
            snapshot: dict[str, tuple[int, int]] = {}
            for log in pdir.glob("*.log"):
                with contextlib.suppress(OSError):
                    st = log.stat()
                    snapshot[log.name] = (st.st_size, st.st_mtime_ns)
            keep, unread = self._collect_live_across_logs(pdir, snapshot, now)
            if target.name in unread:
                # `unread` protects sources from the prune, but the DESTINATION is
                # rewritten unconditionally — with our own records missing from the
                # merge, that rewrite would clobber them. Nothing has changed yet:
                # abort the whole pass and let a later one retry.
                self._consolidated_at[prefix] = time.monotonic()  # cooldown the retry
                logger.warning(
                    "emboss.LogCache: could not read our own log %s — aborting the "
                    "consolidation pass (nothing was changed; it retries later).",
                    target,
                )
                return
            # Our rewritten log carries only records WE are responsible for: the
            # per-key winners that live in our own log or in a log being pruned.
            # A winner in an unpruned peer log stays exactly where it is.
            ours = [rec for rec, src in keep if src == target.name or src in prune_logs]
            if self.size_limit is not None:
                ours = self._trim_to_limit(ours)
            ours.sort(key=lambda r: r.store_time)
            consolidated, failed_adoptions = self._resolve_spills(prefix, ours)
            # A namespace whose adoption hit a transient fault keeps its log too:
            # its records were dropped from the merge, so pruning the log would
            # lose them along with the still-on-disk legacy bytes.
            keep_sources = set(unread) | {f"{w}.log" for w in failed_adoptions}
            self._write_consolidated(target, consolidated)
            self._prune_consolidated_sources(
                pdir, snapshot, target.name, keep_sources, prune_logs
            )
            # The pool's reference set spans every log — records staying put in
            # unpruned peer logs keep their spills exactly as our own do. Failed
            # adoptions do NOT blind the sweep (unlike unreadable logs): legacy
            # references never point into the pool, and every pool-referencing
            # record from those same logs was merged into `consolidated` — so
            # the pool reference set is still complete.
            kept_spills = {r.spill for r in consolidated if r.spill} | {
                rec.spill
                for rec, src in keep
                if rec.spill and src != target.name and src not in prune_logs
            }
            self._sweep_shared_spills(pdir, kept_spills, unread, now)
            self._sweep_legacy_namespaces(
                pdir, kept_spills, unread, failed_adoptions, prune, now
            )
            if self.writer_id not in failed_adoptions:
                # Our own legacy namespace is never pruned with a log (our log is
                # the destination), and only our own records ever referenced it —
                # all just repointed into the pool — so it is removed here or never.
                own_legacy = pdir / f"{self.writer_id}.spill"
                shutil.rmtree(own_legacy, ignore_errors=True)
                if own_legacy.exists():
                    logger.warning(
                        "emboss.LogCache: could not remove migrated legacy "
                        "namespace %s — the prefix re-triggers migration until "
                        "it is removable.",
                        own_legacy,
                    )
        self._index.pop(prefix, None)  # force rebuild
        self._sig.pop(prefix, None)
        self._checked.pop(prefix, None)
        self._needs_migration.discard(prefix)
        self._consolidated_at[prefix] = time.monotonic()

    # A shared-pool file must outlive any in-flight `set()` that wrote it before its
    # record landed, any not-yet-synced peer log that references it, and any read in
    # progress. One hour of grace covers all three with a wide margin; an unreferenced
    # file merely waits one more consolidation pass.
    _SHARED_SPILL_GRACE_S = 3600.0

    def _sweep_shared_spills(
        self,
        pdir: Path,
        kept_spills: AbstractSet[str | None],
        unread: AbstractSet[str],
        now: float,
    ) -> None:
        """Mark-and-sweep GC of the prefix's shared spill pool.

        Shared-pool files are never deleted eagerly (several records — and several
        nodes — may reference one content hash), so consolidation is where they are
        collected: it has just read EVERY readable log, so `kept_spills` — the
        spill references of every live record across all logs, wherever the
        record lives — is the complete local reference set. The `.val` sweep is
        skipped when any source
        log could not be read — its references are unknown, and deleting a file it
        points at would turn recoverable state into misses. A file younger than the grace window is
        kept even when unreferenced: an in-flight `set()` spills BEFORE appending its
        record (a dedup hit refreshes the file's age for the same reason), and a
        syncer may deliver a peer's spill before its log — age is measured from
        `max(mtime, ctime)` because a syncer preserves the peer's (old) mtime but
        ctime is stamped locally on arrival and cannot be backdated. A swept file
        that a lagging peer still references degrades to a miss-and-recompute, the
        module's standing eventual-consistency contract. Crash-orphaned `*.tmp`
        files (a crash between write and rename) are referenced by nothing, so they
        are reaped past the grace window even when a log was unreadable."""
        pool = pdir / _SHARED_SPILL_DIR
        if not pool.is_dir():
            return
        for leftover in pool.glob("*.tmp"):
            with contextlib.suppress(OSError):
                st = leftover.stat()
                if now - max(st.st_mtime, st.st_ctime) >= self._SHARED_SPILL_GRACE_S:
                    leftover.unlink()
        if unread:
            return
        prefix = pdir.name
        for val in pool.glob("*.val"):
            rel = f"{prefix}/{_SHARED_SPILL_DIR}/{val.name}"
            if rel in kept_spills:
                continue
            try:
                st = val.stat()
                if now - max(st.st_mtime, st.st_ctime) < self._SHARED_SPILL_GRACE_S:
                    continue
            except OSError:
                continue
            self._spill_delete(rel)

    def _sweep_legacy_namespaces(
        self,
        pdir: Path,
        kept_spills: AbstractSet[str | None],
        unread: AbstractSet[str],
        failed_adoptions: AbstractSet[str],
        prune: AbstractSet[str],
        now: float,
    ) -> None:
        """Collect legacy `<writer>.spill/` namespaces — file by file, then the dir.

        Scope: **our own namespace and the writers being pruned, nothing else.**
        An unpruned peer's namespace is still referenced by that peer's log
        (which stays), and under a syncer it is a replica of files the peer
        owns — deleting it here deletes it on the peer. Our own pre-pool
        leftovers were otherwise immortal (measured ~35 GB across 4,096
        `bonbon.spill` dirs after full migration), and a dir that outlived its
        records re-armed the migration flag on every index build.
        `_resolve_spills` has just repointed every kept record into the shared
        pool, so an in-scope legacy file is garbage unless a kept record still
        names it (a mid-migration edge). Same guards as the pool sweep: skipped
        when any source log was unreadable (unknown references), per-file kept-set
        check, and a grace window per file by `max(mtime, ctime)` — a still-running
        pre-pool writer, or a syncer that delivered the namespace before its log
        (old mtimes are preserved; ctime is stamped on local arrival). The dir goes
        once it is empty — so the migration flag stops re-arming — but only when
        the dir itself is past the grace window, giving an empty namespace (a
        syncer creates the dir before copying files in) the same grace as its
        files. A namespace whose adoption failed this pass is skipped outright:
        its records were dropped from the merge (so the kept set cannot
        vouch for its files, however old), and the retry that its kept log
        guarantees needs the legacy bytes still on disk."""
        if unread:
            return
        prefix = pdir.name
        in_scope = {self.writer_id} | set(prune)
        for sdir in pdir.glob("*.spill"):
            owner = sdir.name.removesuffix(".spill")
            if owner not in in_scope:
                continue  # an unpruned peer's namespace — never ours to touch
            if owner in failed_adoptions:
                continue  # adoption retries next pass — its bytes must survive
            try:
                # Stat the dir BEFORE unlinking below refreshes its timestamps.
                dstat = sdir.stat()
                entries = list(sdir.iterdir())
            except OSError:
                continue
            dir_young = (
                now - max(dstat.st_mtime, dstat.st_ctime) < self._SHARED_SPILL_GRACE_S
            )
            remaining = len(entries)
            for entry in entries:
                if f"{prefix}/{sdir.name}/{entry.name}" in kept_spills:
                    continue
                try:
                    st = entry.stat()
                    if now - max(st.st_mtime, st.st_ctime) < self._SHARED_SPILL_GRACE_S:
                        continue
                except OSError:
                    continue
                with contextlib.suppress(OSError):
                    entry.unlink()
                    remaining -= 1
            if remaining == 0 and not dir_young:
                with contextlib.suppress(OSError):
                    sdir.rmdir()

    def _collect_live_across_logs(
        self, pdir: Path, sources: dict[str, tuple[int, int]], now: float
    ) -> tuple[list[tuple[_Record, str]], AbstractSet[str]]:
        """Merge all source logs into the live set: latest `store_time` wins per
        key (a newer overwrite/tombstone under ANY writer beats an older one).
        `sorted(sources)` makes the merge order deterministic for tie handling.

        Returns `(keep, unread)`: the live `(record, source-log name)` winners —
        provenance decides which records our rewrite may carry and which belong
        to a peer log we must not touch — and the names of source logs that
        could NOT be fully read: a transient read error must not make a log
        prunable, or its unmerged records would be lost."""
        latest: dict[str, tuple[_Record, str]] = {}
        unread: set[str] = set()
        for name in sorted(sources):
            try:
                # OSError → unread (below), never prune
                scan = _read_records(pdir / name)
            except OSError as exc:
                unread.add(name)  # transient read error → never prune this source
                logger.warning(
                    "emboss.LogCache: could not read source log %s during "
                    "consolidation (%s) — keeping it, and skipping the pool's "
                    ".val sweep, until it reads cleanly.",
                    pdir / name,
                    exc,
                )
                continue
            # A mid-log tear in OUR OWN log is healed by the rewrite; in a peer
            # log it is expected mid-sync replica state and the file is left
            # exactly as it is (the owner holds the complete original).
            if scan.recovered and name == self._log_path(pdir.name).name:
                logger.warning(
                    "emboss.LogCache: consolidating %s past a torn frame at byte %d — "
                    "keeping %d recovered record(s) and dropping the malformed frame(s).",
                    pdir / name,
                    scan.tear_at,
                    scan.recovered,
                )
            for rec in scan.records:
                cur = latest.get(rec.key)
                if cur is None or rec.store_time >= cur[0].store_time:
                    latest[rec.key] = (rec, name)
        keep = [(r, src) for r, src in latest.values() if self._live(r, now)]
        keep.sort(key=lambda pair: pair[0].store_time)
        return keep, unread

    def _resolve_spills(
        self, prefix: str, keep: list[_Record]
    ) -> tuple[list[_Record], AbstractSet[str]]:
        """Make every kept spill reference point at an existing pool file.

        A pool reference is kept when its file exists and dropped when it does
        not (sync lag: the log arrived before the value; the record recomputes
        on next read rather than dangle) — but a transient stat FAULT is not
        absence: the record is kept, so a local blip can never cascade into
        pruned logs and swept files. A **legacy** per-writer reference (the
        pre-pool layout) is migrated: its bytes are adopted into the pool under
        their content hash — hardlinked when the filesystem allows, so migration
        moves no bytes and the inode survives the legacy namespace's prune — and
        the record repointed. Content addressing makes all of this idempotent:
        re-consolidating (e.g. against a syncer that resurrects pruned peer
        logs) can never duplicate the corpus. The raw bytes are already pickled
        and are never unpickled/repickled.

        Returns `(consolidated, failed_adoptions)`: the resolved records, and
        the writer namespaces whose adoption hit a TRANSIENT fault (permissions,
        I/O — not sync lag): their legacy bytes are still on disk, so the caller
        must keep those namespaces (and their logs) for a retry — see
        `_defer_failed_adoption` for how the log is recovered from the spill
        path and how own/out-of-contract references are kept instead."""
        consolidated: list[_Record] = []
        failed: set[str] = set()
        migrated = 0
        dropped = 0
        for rec in keep:
            if not rec.spill:
                consolidated.append(rec)
                continue
            if self._is_shared_spill(prefix, rec.spill):
                try:
                    (self.directory / rec.spill).stat()
                except FileNotFoundError:
                    dropped += 1  # pool file absent (sync lag) → drop, never dangle
                except OSError as exc:
                    # A transient stat fault is NOT absence: keep the record
                    # (its path also keeps the file out of the sweep); reads
                    # degrade to a miss until the fault clears.
                    logger.warning(
                        "emboss.LogCache: could not stat pool spill %s (%s) — "
                        "keeping its record; reads miss until it is reachable.",
                        rec.spill,
                        exc,
                    )
                    consolidated.append(rec)
                else:
                    consolidated.append(rec)
                continue
            try:
                new_rel = self._pool_adopt(prefix, rec.spill)
            except OSError as exc:
                self._defer_failed_adoption(rec, exc, consolidated, failed)
                continue
            if new_rel is None:
                dropped += 1  # legacy spill absent (sync lag) → drop, never dangle
                continue
            migrated += 1
            consolidated.append(rec._replace(spill=new_rel, value=None))
        if migrated:
            logger.info(
                "emboss.LogCache: migrated %d legacy per-writer spill(s) in %s/ into the "
                "shared pool.",
                migrated,
                prefix,
            )
        if dropped:
            logger.info(
                "emboss.LogCache: dropped %d record(s) in %s/ whose spill files were "
                "absent (sync lag or swept); they recompute on next read.",
                dropped,
                prefix,
            )
        return consolidated, failed

    def _defer_failed_adoption(
        self,
        rec: _Record,
        exc: OSError,
        consolidated: list[_Record],
        failed: MutableSet[str],
    ) -> None:
        """Bookkeeping for a legacy adoption that hit a transient fault.

        The log to protect is recovered from the spill path via the layout
        invariant that a record referencing `<w>.spill/…` is stored only in
        `<w>.log` (writers spill exclusively into their own namespace): a peer
        namespace is deferred through `failed`, which keeps it and its log from
        the prune. Our OWN record — or one whose reference does not parse as
        `<prefix>/<w>.spill/…` at all, where the holding log is unknown — is
        instead carried into our rewritten log un-repointed: the rewrite must
        never shed a record whose bytes may still be on disk, and an
        unreachable spill already degrades to a read-time miss."""
        assert rec.spill is not None
        namespace = rec.spill.split("/")[1] if "/" in rec.spill else ""
        if namespace.endswith(".spill"):
            failed.add(namespace.removesuffix(".spill"))
        if namespace.endswith(".spill") and namespace != f"{self.writer_id}.spill":
            action = "keeping its namespace and log for a retry on a later pass"
        else:
            consolidated.append(rec)  # our rewrite must not shed the record
            action = "keeping its record un-repointed for a retry on a later pass"
        logger.warning(
            "emboss.LogCache: could not adopt legacy spill %s into the pool (%s) — %s.",
            rec.spill,
            exc,
            action,
        )

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
        prune_logs: AbstractSet[str],
    ) -> None:
        """Delete each ALLOWLISTED source log (and its spill dir + lock) now fully
        represented in our consolidated log — but ONLY if it is byte-identical to
        the snapshot. A log outside `prune_logs` is never touched: the caller has
        not asserted its writer is dead, and under a syncer it is a replica whose
        owner may hold an un-synced tail. A changed `(size, mtime)` means it was
        appended to during the merge: those newer records aren't in our log, so
        we leave it (they win on read; the next pass folds them in). A source we
        could not fully read (`unread`) is likewise kept — its unmerged records
        aren't in our log, so deleting it would lose them."""
        for name, (size, mtime) in snapshot.items():
            if name == target_name or name in unread or name not in prune_logs:
                continue  # destination / partial read / not asserted dead → keep
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
            except FileNotFoundError:
                return default  # spill not present yet (log synced before it) -> miss
            except (
                OSError
            ) as exc:  # persistent I/O error (EACCES/EIO): warn, don't hide
                logger.warning(
                    "emboss.LogCache: spill file %s for key %r could not be read "
                    "(%s); treating as a miss.",
                    rec.spill,
                    key,
                    exc,
                )
                return default
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
            # (a superseded value's file lingers until the next consolidation —
            # guaranteed for a lone-writer prefix by the size trigger below and
            # for a crowded one by the writer-count trigger; a stable in-between
            # prefix sweeps only via an explicit consolidate(). A spill orphaned
            # by an append failure below likewise waits for a sweep).
            log_size = self._append(prefix, rec)
            index[rec.key] = rec
            self._sig.setdefault(prefix, {})[self._log_path(prefix).name] = log_size
            last_pass = self._consolidated_at.get(prefix)
            cooled_down = (
                last_pass is None
                or time.monotonic() - last_pass > _AUTO_CONSOLIDATE_COOLDOWN_S
            )
            if log_size > self.max_log_bytes:
                if len(self._sig.get(prefix, {})) <= 1 and cooled_down:
                    # A lone-writer prefix's only sweep: with a single log,
                    # consolidation degenerates to compaction plus the pool GC.
                    # (`_sig` may be stale; a hidden peer is simply left alone —
                    # a prune-free pass touches nothing of theirs.) The cooldown
                    # bounds the pathological case — a live set that cannot
                    # shrink below `max_log_bytes` — to one full pass per
                    # window, compacting in between.
                    self._consolidate_prefix(prefix)
                else:
                    self._compact_prefix(prefix)
            elif prefix in self._needs_migration and cooled_down:
                # Migrate-on-sight: the prefix still holds OUR OWN legacy
                # per-writer spill layout — repoint our records into the shared
                # pool and remove our namespace. (There is deliberately no
                # writer-count trigger any more: an implicit cross-writer fold
                # pruned peer logs that, under a syncer, were mid-sync replicas
                # — the 2026-07-10 clobber. Cross-writer GC is explicit-only via
                # `consolidate(prune_writer_ids=...)`.) The cooldown stops the
                # trigger from storming when the namespace resists removal.
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
        """Drop THIS writer's contribution to the cache: our log, lock, and
        legacy namespace in every prefix. Returns the number of live records
        dropped (per-key winners within our own logs).

        Peer files are never touched. They are not ours to delete — and under a
        file syncer they are replicas of files the peers own, so a local rmtree
        would propagate and destroy every node's originals (the old
        whole-directory clear assumed peers' files "re-appear via sync"; a
        syncer treats the deletion as the newest change and applies it
        everywhere instead). Peers' records therefore remain readable after
        clear; on a replicated tree our own deletion propagates, dropping this
        writer's records fleet-wide. Shared pool files also stay — a peer may
        reference them; orphans are collected by the consolidation sweep."""
        now = time.time()
        dropped = 0
        with self._lock:
            for pdir in list(self.directory.iterdir()):
                if not pdir.is_dir():
                    continue
                own_log = pdir / f"{self.writer_id}.log"
                with contextlib.suppress(OSError):
                    latest: dict[str, _Record] = {}
                    for rec in _read_records(own_log).records:
                        cur = latest.get(rec.key)
                        if cur is None or rec.store_time >= cur.store_time:
                            latest[rec.key] = rec
                    dropped += sum(1 for r in latest.values() if self._live(r, now))
                own_log.unlink(missing_ok=True)
                (pdir / f"{self.writer_id}.lock").unlink(missing_ok=True)
                shutil.rmtree(pdir / f"{self.writer_id}.spill", ignore_errors=True)
            self._index.clear()
            self._sig.clear()
            self._checked.clear()
        return dropped

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
