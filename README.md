# emboss

**O**n-**D**isk **I**nput-keyed **C**ache — disk-backed memoization with pydantic-aware encoding.

```bash
pip install emboss              # core (just diskcache)
pip install emboss[pydantic]    # + pydantic v2 BaseModel support
```

## Why

`functools.lru_cache` is per-process. `diskcache` survives invocations but pickles values as-is — which breaks the moment your cached return type is a pydantic `BaseModel` defined in `__main__` (the new process can't unpickle `__main__.MyModel`). `emboss` fixes that by detecting BaseModel return annotations and converting to/from plain dicts at the cache boundary.

Plus: a `None`-aware sentinel so functions returning `None` actually cache instead of re-running every call.

## Quick start

```python
import diskcache
from emboss import cached

cache = diskcache.Cache("/tmp/my-cache")

@cached(cache)
def fetch(url: str) -> dict:
    import requests
    return requests.get(url).json()

fetch("https://api.example.com/users/1")  # network
fetch("https://api.example.com/users/1")  # cached, no network
```

## Default cache directory

Passing a `diskcache.Cache` is optional. With `@cached()` (no cache argument), emboss creates one at the directory named by the `EMBOSS_CACHE_DIR` environment variable; if unset, `diskcache` falls back to a temporary directory.

```bash
EMBOSS_CACHE_DIR=.data/cache python my_script.py
```

The cache location never affects keying — keys are function identity + arguments either way (see [Cache identity & migration](#cache-identity--migration)).

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

Arguments are converted via `safe_jsonable_encoder` (recursive JSON-friendly conversion handling sets, bytes, dates, `Path`, BaseModel, and objects with `__dict__`), then hashed with the function's cache identity — its name plus the hash of its AST-canonical source (see [Cache identity & migration](#cache-identity--migration)). Re-decorating the same function body → same key; changing the function body → new key (transparent cache invalidation on code change, unless you opt out with `unsafe_manual_key`).

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

@cached(cache, also_accept=[old_id])   # a literal "fetch_user:3f2a9c..." works too
def get_user(uid: int) -> dict:
    ...  # same behaviour, new name — old entries are reused, not recomputed
```

Different arguments still miss as usual — migration only redirects keys, never serves a value computed for other inputs. Malformed tokens (anything not `"name:body_hash"`, or containing whitespace) raise `ValueError` at decoration time.

Migration via `also_accept` is the *only* fallback: the implicit raw-source fallback that pre-0.3 entries were read through has been removed, so entries that only exist under a pre-0.3 key are reachable solely by declaring their old identity in `also_accept`.

### `unsafe_manual_key` — opt out of source-based invalidation

`unsafe_manual_key` pins the identity to a fixed string (non-empty, no whitespace) instead of the source hash:

```python
@cached(cache, unsafe_manual_key="v1")
def summarise(text: str) -> str:
    ...  # edit freely — entries keyed on "summarise:v1" keep matching
```

**Warning: this disables emboss's invalidate-on-edit safety net.** Editing the body no longer invalidates the cache, so stale results are served until *you* bump the key (`"v1"` → `"v2"`). Use it only when you accept that responsibility — e.g. a hot cache you must not re-bill for cosmetic-but-not-quite-canonical churn. `also_accept` works alongside it, e.g. to migrate source-keyed entries into a manual-key identity.

## Pluggable backends (`Cache` protocol)

`cached` accepts any object satisfying the runtime-checkable `Cache` protocol:

```python
from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class Cache(Protocol):
    def get(self, key: str, default: Any = None) -> Any: ...
    def set(self, key: str, value: Any) -> Any: ...
```

Structural typing — no inheritance required. `diskcache.Cache`, `emboss.FileCache`, and any custom Redis / in-memory adapter you write all work out of the box.

## `FileCache` backend — NFS-safe alternative to diskcache

```python
from emboss import FileCache, cached

cache = FileCache(".data/cache")

@cached(cache)
def expensive(x: int) -> dict:
    ...
```

`diskcache` stores entries in SQLite, and SQLite over NFS has broken file-locking — two cluster nodes hitting the same `.data/cache` mount on VAST get `sqlite3.OperationalError: locking protocol`. `FileCache` writes one file per key via `tempfile + os.replace` (atomic rename, NFS-safe), with `(key, value)` pickled. Concurrent writers race on the same file path but POSIX rename is atomic and the winning version is by construction equally correct (cache values are pure functions of the key).

Drop-in for the subset of `diskcache.Cache` API `@cached` uses (`get`, `set`, `__contains__`, `__getitem__`, `__setitem__`, `__delitem__`, `delete`, `clear`, `close`, context-manager). Extra diskcache kwargs (`timeout`, `size_limit`, `eviction_policy`) are accepted and ignored so call sites switch with no code changes.

## Async support

```python
@cached(cache)
async def fetch_async(url: str) -> dict:
    async with httpx.AsyncClient() as c:
        return (await c.get(url)).json()
```

Cache hits return a fresh awaitable wrapping the cached value, so the call site keeps `await`-ing as normal.

## Daily-rolling caches

The `diskcache.Cache` instance you pass is yours to manage. A common pattern for "expire daily" without thinking about it:

```python
from datetime import date
import diskcache
cache = diskcache.Cache(f"/tmp/my-cache-{date.today()}")
```

Each new day → new dir → effectively fresh cache. Old dirs land in `/tmp` and get reaped by the OS.

## License

MIT.
