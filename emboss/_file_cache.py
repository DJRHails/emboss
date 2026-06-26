"""File-per-key persistent cache — NFS-safe, replication-safe.

Why this exists: a single-DB cache (`SqliteCache` / `diskcache.Cache`) can't be
shared across hosts. SQLite file-locking breaks over NFS — two nodes both opening
`SqliteCache('.data/cache')` on a VAST/NFS mount get
`sqlite3.OperationalError: locking protocol` — and one growing DB file can't be
replicated by a file syncer (Syncthing, rsync) without torn-read corruption.
`FileCache` stores each `(key -> value)` as its own file written via
`tempfile + os.replace`; there's no shared lock to contend on and each entry
syncs independently. That makes it the backend for a cache that lives on a network
mount or is replicated across machines.

The cost is one file per key — many inodes, and slower bulk/`du`/scan operations
at very large entry counts. For a bounded *local* cache prefer `SqliteCache`
(one file, index-driven eviction). The file-per-key vs single-file tradeoff is
fundamental: bundling entries back into one file would reintroduce the very lock
contention `FileCache` exists to avoid, so `SqliteCache` *is* the bundled variant.

Eviction: unbounded by default (`size_limit=None`). When `size_limit` (bytes) is
set, least-recently-used entries — ordered by file mtime, bumped on read — are
evicted once the total exceeds it, down to ~90%. Eviction is a full directory
scan, so it is best-effort and amortized (it runs only when over the limit, then
clears headroom); `SqliteCache` evicts far more cheaply via its index. Across a
syncing fleet, eviction is per-node and deletions propagate, so the bound is
enforced approximately, by convergence.

Concurrent writers race on the same path, but POSIX rename is atomic so the last
writer wins; cache values are pure functions of the key, so any winning version is
equally correct.

Benchmark (local SSD, 512-byte values), order of magnitude:
    set ~10^4 ops/s     get ~10^4 ops/s

Implements the subset of `diskcache.Cache` that `@cached` needs, plus ergonomic
dunders: `get`/`set`/`delete`/`clear`/`close`, `in`/`[]`/`[]=`/`del`, context
manager. Extra diskcache kwargs (`timeout`, `eviction_policy`, ...) are accepted
and ignored so existing call sites port unchanged.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import pickle
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

_MISSING = object()


class _Entry(NamedTuple):
    """On-disk record: the original key paired with its value, so keys are
    recoverable for iteration / `transfer`. Files written before this format
    held the bare value; `get` reads both (an `_Entry` -> its value; anything
    else -> the legacy raw value)."""

    key: Any
    value: Any


def _key_to_path(root: Path, key: Any) -> Path:
    """Map a cache key to a file path.

    String keys are sharded by their first two characters to cap any one
    directory at a few hundred entries (avoids the "thousands of files in one
    NFS dir" performance cliff). Non-string keys *and* string keys shorter than
    two characters are md5-hashed first — otherwise every one-character key would
    collide on the same `_/_.pkl` path.
    """
    if not isinstance(key, str) or len(key) < 2:
        key = hashlib.md5(repr(key).encode()).hexdigest()
    shard = key[:2]
    rest = key[2:] or "_"
    safe_rest = rest.replace("/", "_").replace("\x00", "_")
    return root / shard / f"{safe_rest}.pkl"


class FileCache:
    """File-per-key persistent cache — NFS-safe via atomic rename, optionally
    bounded by an LRU `size_limit`."""

    def __init__(
        self,
        directory: str | os.PathLike[str] = ".cache",
        size_limit: int | None = None,
        **_kwargs: Any,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.size_limit = size_limit
        # Running total of cache bytes, tracked only when eviction is on. Lazily
        # initialised by a one-time scan on first write so a pre-existing cache is
        # accounted for; `_evict` recomputes it authoritatively each pass.
        self._size: int | None = None

    def get(self, key: Any, default: Any = None) -> Any:
        path = _key_to_path(self.directory, key)
        if not path.exists():
            return default
        if self.size_limit is not None:
            with contextlib.suppress(OSError):
                os.utime(path)  # LRU: mark recently used (bump mtime)
        try:
            with path.open("rb") as f:
                obj = pickle.load(f)
        except (EOFError, pickle.UnpicklingError, OSError):
            # Partial write or corrupted file; treat as a miss. Caller recomputes
            # and atomic-renames a fresh copy over it.
            return default
        return obj.value if isinstance(obj, _Entry) else obj  # else: legacy raw value

    def set(self, key: Any, value: Any, **_kwargs: Any) -> bool:
        path = _key_to_path(self.directory, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        old_size = path.stat().st_size if path.exists() else 0
        # Atomic overwrite: tempfile in the SAME directory as the target so the
        # rename is atomic on POSIX and NFS-safe. Cross-directory renames on NFS
        # can fall back to copy+delete and are not atomic.
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(_Entry(key, value), f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        if self.size_limit is not None:
            self._account_and_evict(path.stat().st_size - old_size)
        return True

    def _account_and_evict(self, delta: int) -> None:
        assert self.size_limit is not None  # only called when eviction is enabled
        if self._size is None:
            self._size = self._scan_size()
        self._size += delta
        if self._size > self.size_limit:
            self._evict()

    def _scan_size(self) -> int:
        total = 0
        for p in self.directory.glob("**/*.pkl"):
            with contextlib.suppress(OSError):
                total += p.stat().st_size
        return total

    def _evict(self) -> None:
        """Evict least-recently-used entries (oldest mtime first) until the total
        is back under ~90% of `size_limit`. Recomputes the authoritative total in
        the same scan, so the in-memory estimate self-corrects."""
        entries: list[tuple[float, int, Path]] = []
        total = 0
        for p in self.directory.glob("**/*.pkl"):
            try:
                st = p.stat()
            except OSError:
                continue
            entries.append((st.st_mtime, st.st_size, p))
            total += st.st_size
        self._size = total
        if self.size_limit is None or total <= self.size_limit:
            return
        low_water = self.size_limit * 0.9
        for _mtime, size, p in sorted(entries):  # oldest first
            if self._size <= low_water:
                break
            try:
                p.unlink()
                self._size -= size
            except OSError:
                pass

    def delete(self, key: Any) -> bool:
        path = _key_to_path(self.directory, key)
        try:
            path.unlink()
            self._size = None  # invalidate the running total; re-scanned on next evict
            return True
        except FileNotFoundError:
            return False

    def clear(self) -> int:
        """Remove all `.pkl` entries; return the count removed.

        Walks the entire tree (not just one shard level) so a future deeper layout
        won't silently leak entries.
        """
        count = 0
        for pkl in self.directory.glob("**/*.pkl"):
            with contextlib.suppress(OSError):
                pkl.unlink()
                count += 1
        for sub in sorted(
            (p for p in self.directory.glob("**/*") if p.is_dir()),
            key=lambda p: len(p.parts),
            reverse=True,
        ):
            with contextlib.suppress(OSError):
                sub.rmdir()
        self._size = 0
        return count

    def nuke(self) -> None:
        """Drop the entire cache directory and recreate it. Tests-only escape hatch."""
        shutil.rmtree(self.directory, ignore_errors=True)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._size = 0

    def __len__(self) -> int:
        return sum(1 for _ in self.directory.glob("**/*.pkl"))

    def __iter__(self) -> Iterator[Any]:
        """Yield stored keys. Entries written before key-recovery (bare values)
        are skipped — only `_Entry`-format files carry a recoverable key."""
        keys = []
        for pkl in self.directory.glob("**/*.pkl"):
            try:
                with pkl.open("rb") as f:
                    obj = pickle.load(f)
            except (EOFError, pickle.UnpicklingError, OSError):
                continue
            if isinstance(obj, _Entry):
                keys.append(obj.key)
        return iter(keys)

    iterkeys = __iter__

    def volume(self) -> int:
        """Total bytes on disk across all entry files."""
        total = 0
        for pkl in self.directory.glob("**/*.pkl"):
            with contextlib.suppress(OSError):
                total += pkl.stat().st_size
        return total

    def __contains__(self, key: Any) -> bool:
        return _key_to_path(self.directory, key).exists()

    def __getitem__(self, key: Any) -> Any:
        value = self.get(key, default=_MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        self.set(key, value)

    def __delitem__(self, key: Any) -> None:
        if not self.delete(key):
            raise KeyError(key)

    def __enter__(self) -> FileCache:
        return self

    def __exit__(self, *_args: Any) -> None:
        pass

    def close(self) -> None:
        pass
