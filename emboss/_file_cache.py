"""File-per-key persistent cache. NFS-safe alternative to `diskcache.Cache`.

Why this exists: `diskcache` stores entries in a SQLite database, and SQLite
over NFS has broken file-locking semantics. Two cluster nodes both calling
`diskcache.Cache('.data/cache')` on a VAST mount get
`sqlite3.OperationalError: locking protocol` and the second one fails. The
file-per-key + atomic-rename design used here sidesteps the problem: each
`(key → value)` is its own file, written via `tempfile + os.replace`;
there's no shared lock to contend on. Concurrent writers race on the same
file path, but POSIX rename is atomic and the loser is overwritten with
identical-or-newer content (cache values are by construction pure functions
of the input key, so any winning version is equally correct).

Implemented API (the subset of `diskcache.Cache` `@cached` uses, plus a few
ergonomic dunder methods):

- `get(key, default=None)`
- `set(key, value)`
- `__contains__` / `__getitem__` / `__setitem__` / `__delitem__`
- `delete(key)`, `clear()`, `close()`
- Context manager (no-op)

The constructor accepts and ignores diskcache's other kwargs (`timeout`,
`size_limit`, `eviction_policy`, ...) so call sites that pass them
unchanged still work.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any

_MISSING = object()


def _key_to_path(root: Path, key: Any) -> Path:
    """Map a cache key to a file path.

    String keys are sharded by their first two characters to cap any one
    directory at a few hundred entries (avoids the "thousands of files in
    one NFS dir" performance cliff). Non-string keys *and* string keys
    shorter than two characters are md5-hashed first — otherwise every
    one-character key would collide on the same `_/_.pkl` path.
    """
    if not isinstance(key, str) or len(key) < 2:
        key = hashlib.md5(repr(key).encode()).hexdigest()
    shard = key[:2]
    rest = key[2:] or "_"
    safe_rest = rest.replace("/", "_").replace("\x00", "_")
    return root / shard / f"{safe_rest}.pkl"


class FileCache:
    """File-per-key persistent cache. NFS-safe via atomic rename."""

    def __init__(self, directory: str | os.PathLike[str] = ".cache", **_kwargs: Any) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def get(self, key: Any, default: Any = None) -> Any:
        path = _key_to_path(self.directory, key)
        if not path.exists():
            return default
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError, OSError):
            # Partial write or corrupted file; treat as a miss. Caller will
            # recompute and atomic-rename a fresh copy over it.
            return default

    def set(self, key: Any, value: Any, **_kwargs: Any) -> bool:
        path = _key_to_path(self.directory, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Concurrent-writer guard: cache values are pure functions of key, so
        # an existing file is by construction equally correct. Skip the
        # write entirely rather than risk overwriting a peer's just-written
        # bytes with our own. Matches the docstring invariant.
        if path.exists():
            return False
        # Atomic write: tempfile in the SAME directory as the target so the
        # rename is atomic on POSIX and NFS-safe. Cross-directory renames on
        # NFS can fall back to copy+delete and are not atomic.
        fd, tmp = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        return True

    def delete(self, key: Any) -> bool:
        path = _key_to_path(self.directory, key)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False

    def clear(self) -> int:
        """Remove all `.pkl` entries; return the count removed.

        Walks the entire tree (not just a single shard level) so a future
        layout change with deeper nesting won't silently leak entries.
        """
        count = 0
        for pkl in self.directory.glob("**/*.pkl"):
            with contextlib.suppress(OSError):
                pkl.unlink()
                count += 1
        # Best-effort cleanup of empty shard dirs left behind.
        for sub in sorted(
            (p for p in self.directory.glob("**/*") if p.is_dir()),
            key=lambda p: len(p.parts),
            reverse=True,
        ):
            with contextlib.suppress(OSError):
                sub.rmdir()
        return count

    def nuke(self) -> None:
        """Drop the entire cache directory and recreate it. Tests-only escape hatch."""
        shutil.rmtree(self.directory, ignore_errors=True)
        self.directory.mkdir(parents=True, exist_ok=True)

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
