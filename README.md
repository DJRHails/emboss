# emboss

**O**n-**D**isk **I**nput-keyed **C**ache — disk-backed memoization with pydantic-aware encoding.

Version: 0.2.0

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
