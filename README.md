# emboss

**O**n-**D**isk **I**nput-keyed **C**ache — disk-backed memoization with pydantic-aware encoding.

Version: 0.5.0

```bash
pip install emboss              # core — zero runtime dependencies (stdlib only)
pip install emboss[pydantic]    # + pydantic v2 BaseModel support
pip install emboss[diskcache]   # + use diskcache.Cache as a backend (now optional)
```

## Why

`functools.lru_cache` is per-process. A disk cache survives invocations but pickling values as-is breaks the moment your cached return type is a pydantic `BaseModel` defined in `__main__` (the new process can't unpickle `__main__.MyModel`). `emboss` fixes that by detecting BaseModel return annotations and converting to/from plain dicts at the cache boundary.

It ships its own dependency-free backends (`SqliteCache`, `FileCache`) — `diskcache` is no longer required, but stays a drop-in option.

Plus: a `None`-aware sentinel so functions returning `None` actually cache instead of re-running every call.

## Quick start

```python
from emboss import cached, SqliteCache

cache = SqliteCache("/tmp/my-cache")

@cached(cache)
def fetch(url: str) -> dict:
    import requests
    return requests.get(url).json()

fetch("https://api.example.com/users/1")  # network
fetch("https://api.example.com/users/1")  # cached, no network
```

## Default cache directory

Passing a cache is optional. With `@cached()` (no cache argument), emboss creates a default `SqliteCache` at the directory named by the `EMBOSS_CACHE_DIR` environment variable; if unset, it falls back to a fresh temporary directory.

```bash
EMBOSS_CACHE_DIR=.data/cache python my_script.py
```

The cache location never affects keying — keys are function source + arguments either way.

## Pydantic BaseModel returns

`emboss` reads the function's return type annotation. If it sees a `BaseModel`, `list[BaseModel]`, `dict[str, BaseModel]`, or `BaseModel | None`, it serialises via `model.model_dump()` before pickling and rehydrates via `Model.model_validate(...)` on read. The cached value on disk is a plain dict — round-trips cleanly across process boundaries, even for models defined in `__main__`.

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str

@cached(cache)
def get_user(uid: int) -> User | None:
    ...

@cached(cache)
def list_users() -> list[User]:
    ...

@cached(cache)
def users_by_id() -> dict[str, User]:
    ...
```

Functions returning non-BaseModel types continue to pickle as-is — fully backward-compatible.

## None caching

```python
@cached(cache)
def lookup(query: str) -> str | None:
    return external_api(query)

lookup("missing")  # returns None, cached
lookup("missing")  # returns cached None, no re-run
```

The previous behaviour (skip-cache-on-None) is replaced by a `_MISSING` sentinel internally so `None` is a valid cached value.

## Cache key

Arguments are converted via `safe_jsonable_encoder` (recursive JSON-friendly conversion handling sets, bytes, dates, `Path`, BaseModel, and objects with `__dict__`), then hashed with the function source + name. Re-decorating the same function body → same key; changing the function body → new key (transparent cache invalidation on code change).

### Custom or strict encoder (`default=`)

`safe_jsonable_encoder` mirrors `json.dumps(default=)`: pass a callable that handles types no built-in handler matched, or `None` for strict mode that raises on unknown types.

```python
# strict mode — raise on anything we can't serialise
@cached(cache, default=None)
def f(x: dict) -> str:
    ...

# custom fallback — e.g. include a deterministic hash for opaque objects
def my_default(obj):
    return obj.cache_key() if hasattr(obj, "cache_key") else hashlib.md5(repr(obj).encode()).hexdigest()

@cached(cache, default=my_default)
def g(complicated_input) -> dict:
    ...
```

The package default is `default=str`, which preserves the loose 0.1 behaviour of falling back to `str(obj)`. Use strict mode when your inputs include objects without `__dict__` whose `str(obj)` includes a memory address — those addresses change every process invocation and would silently bust the cache key.

## Cache identity & migration

Every `@cached` function has a stable identity — `"name:body_hash"` — which combines with the per-call argument hash to form the cache key. `cache_id()` returns it, and `func.__emboss__` carries the full metadata:

```python
from emboss import cache_id

@cached(cache)
def fetch_user(uid: int) -> dict:
    ...

cache_id(fetch_user)   # "fetch_user:3f2a9c..." (32-hex hash of the AST-canonical source)
```

The `emboss id` CLI prints the same token without writing a script — handy for capturing an identity before (or after) an edit:

```bash
emboss id mypkg.users:fetch_user            # dotted module path
emboss id src/mypkg/users.py:fetch_user     # file path
emboss id --rev HEAD~1 src/mypkg/users.py:fetch_user   # identity as of a git revision
```

`--rev` reads the file out of git (`git show`), so you can recover the pre-edit identity even if you forgot to capture it first. It imports the module to do this, so it's best-effort: if the module's imports fail, it reports the error and exits 1.

### `also_accept` — keep warm entries through a rename or refactor

Renaming a function or editing its body changes its identity, so existing entries stop matching. When the *behaviour* is unchanged, pass the old identity and emboss falls back to the old keys on a miss — copying each hit forward to the new key (write-through), so the fallback can be dropped once the cache has migrated:

```python
old_id = cache_id(fetch_user)   # capture before the rename, e.g. "fetch_user:3f2a9c..."

@cached(cache, also_accept=["fetch_user:3f2a9c..."])
def get_user(uid: int) -> dict:
    ...  # same behaviour, new name — old entries are reused, not recomputed
```

Different arguments still miss as usual — migration only redirects keys, never serves a value computed for other inputs. Malformed tokens (anything not `"name:body_hash"`) raise `ValueError` at decoration time.

### `unsafe_manual_key` — opt out of source-based invalidation

`unsafe_manual_key` pins the identity to a fixed string instead of the source hash:

```python
@cached(cache, unsafe_manual_key="v1")
def summarise(text: str) -> str:
    ...  # edit freely — entries keyed on "summarise:v1" keep matching
```

**Warning: this disables emboss's invalidate-on-edit safety net.** Editing the body no longer invalidates the cache, so stale results are served until *you* bump the key (`"v1"` → `"v2"`). Use it only when you accept that responsibility — e.g. a hot cache you must not re-bill for cosmetic-but-not-quite-canonical churn. `also_accept` works alongside it, e.g. to migrate source-keyed entries into a manual-key identity.

## Backends (`Cache` protocol)

`cached` accepts any object satisfying the runtime-checkable `Cache` protocol — `.get(key, default=...)` and `.set(key, value)`. Structural typing, no inheritance:

```python
@runtime_checkable
class Cache(Protocol):
    def get(self, key: str, default: Any = None) -> Any: ...
    def set(self, key: str, value: Any) -> Any: ...
```

emboss ships two dependency-free backends; `diskcache.Cache` still works if you `pip install emboss[diskcache]`.

| backend | storage | bounded? | shared across hosts? | use when |
|---|---|---|---|---|
| `SqliteCache` *(default)* | one SQLite file | ✅ `size_limit` + LRU + TTL | ❌ SQLite locking breaks on NFS | a bounded **local** cache |
| `FileCache` | one file per key | ✅ optional `size_limit` + LRU | ✅ atomic rename, syncable | a cache on a **network mount / replicated** across machines |
| `diskcache.Cache` | SQLite (+ tags, stats) | ✅ | ❌ | you want diskcache's richer feature set |

Throughput (local SSD, 512-byte values, order of magnitude — `python scripts/bench.py`):

| backend | set | get |
|---|---|---|
| `SqliteCache` | ~10⁴/s | ~10⁵/s |
| `FileCache` | ~10⁴/s | ~10⁴/s |

### `SqliteCache` — bounded, dependency-free (the default)

```python
from emboss import SqliteCache, cached

cache = SqliteCache(".data/cache", size_limit=2**30)  # 1 GiB, LRU-evicted

@cached(cache)
def expensive(x: int) -> dict:
    ...
```

One SQLite file; `size_limit` bytes (default 1 GiB) with least-recently-used eviction, plus optional per-entry TTL (`cache.set(key, value, expire=3600)`). Stdlib only — this is what replaced the `diskcache` dependency.

### `FileCache` — NFS-safe / replication-safe

```python
from emboss import FileCache, cached

cache = FileCache(".data/cache")                     # unbounded
cache = FileCache(".data/cache", size_limit=2**30)   # bounded + LRU

@cached(cache)
def expensive(x: int) -> dict:
    ...
```

A single-file cache can't be shared across hosts: SQLite file-locking breaks over NFS — two nodes on the same VAST mount get `sqlite3.OperationalError: locking protocol` — and one growing DB can't be replicated by a file syncer (Syncthing, rsync) without torn-read corruption. `FileCache` writes one file per key via `tempfile + os.replace` (atomic, NFS-safe), so each entry is independent and syncs cleanly. The cost is one inode per key; for a bounded *local* cache prefer `SqliteCache`.

`size_limit` (bytes) is optional — `None` (default) is unbounded; when set, least-recently-used entries (by file mtime, bumped on read) are evicted past the limit. Eviction is a full-directory scan (best-effort, amortized), and across a syncing fleet it's per-node with deletions propagating.

Both backends implement the `diskcache.Cache` subset `@cached` uses (`get`, `set`, `__contains__`, `__getitem__`, `__setitem__`, `__delitem__`, `delete`, `clear`, `close`, context-manager) and accept-and-ignore extra diskcache kwargs (`timeout`, `eviction_policy`, ...) so call sites switch with no code changes.

## Async support

```python
@cached(cache)
async def fetch_async(url: str) -> dict:
    async with httpx.AsyncClient() as c:
        return (await c.get(url)).json()
```

Cache hits return a fresh awaitable wrapping the cached value, so the call site keeps `await`-ing as normal.

## Daily-rolling caches

The cache instance you pass is yours to manage. For per-entry expiry use `SqliteCache`'s TTL (`cache.set(key, value, expire=86400)`). For a coarser "fresh every day", point the directory at today's date:

```python
from datetime import date
from emboss import SqliteCache
cache = SqliteCache(f"/tmp/my-cache-{date.today()}")
```

Each new day → new dir → effectively fresh cache. Old dirs land in `/tmp` and get reaped by the OS.

## License

MIT.
