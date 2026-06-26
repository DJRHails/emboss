"""emboss — On-Disk Input-keyed Cache.

Dependency-free disk-backed memoization, with auto-detection of pydantic v2
`BaseModel` return types (encoded via `model_dump`, decoded via `model_validate`)
so models defined in `__main__` round-trip across script invocations.

Three backends, all satisfying the `Cache` protocol:

- `SqliteCache` — single-file, size-bounded, LRU (the default; stdlib only).
- `FileCache` — file-per-key, NFS-safe and replication-safe (shared mounts /
  Syncthing across machines).
- `diskcache.Cache` — still supported as a drop-in (`pip install emboss[diskcache]`).

Usage::

    from emboss import cached, SqliteCache

    cache = SqliteCache("/tmp/my-cache")

    @cached(cache)
    def expensive(url: str) -> dict:
        return requests.get(url).json()

    # pydantic BaseModel returns are auto-encoded / decoded
    from pydantic import BaseModel

    class User(BaseModel):
        name: str

    @cached(cache)
    def get_user(uid: int) -> User | None:
        return User.model_validate(requests.get(f"/users/{uid}").json())

See README.md for the full feature list.
"""

from importlib.metadata import PackageNotFoundError, version

from emboss._cached import cache_id, cached, safe_jsonable_encoder
from emboss._file_cache import FileCache
from emboss._protocol import Cache
from emboss._sqlite_cache import SqliteCache

# Single source of truth: the version declared in pyproject.toml, read back from
# installed package metadata — so `__version__` can never drift from the release
# (a hardcoded string here once shipped as 0.3.0 inside the 0.4.0 wheel).
try:
    __version__ = version("emboss")
except PackageNotFoundError:  # running from a source tree that isn't installed
    __version__ = "0+unknown"
__all__ = [
    "Cache",
    "FileCache",
    "SqliteCache",
    "cache_id",
    "cached",
    "safe_jsonable_encoder",
]
