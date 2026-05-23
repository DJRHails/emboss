# odic

**O**n-**D**isk **I**nput-keyed **C**ache — disk-backed memoization with pydantic-aware encoding.

Version: 0.1.0

```bash
pip install odic              # core (just diskcache)
pip install odic[pydantic]    # + pydantic v2 BaseModel support
```

## Why

`functools.lru_cache` is per-process. `diskcache` survives invocations but pickles values as-is — which breaks the moment your cached return type is a pydantic `BaseModel` defined in `__main__` (the new process can't unpickle `__main__.MyModel`). `odic` fixes that by detecting BaseModel return annotations and converting to/from plain dicts at the cache boundary.

Plus: a `None`-aware sentinel so functions returning `None` actually cache instead of re-running every call.

## Quick start

```python
import diskcache
from odic import cached

cache = diskcache.Cache("/tmp/my-cache")

@cached(cache)
def fetch(url: str) -> dict:
    import requests
    return requests.get(url).json()

fetch("https://api.example.com/users/1")  # network
fetch("https://api.example.com/users/1")  # cached, no network
```

## Pydantic BaseModel returns

`odic` reads the function's return type annotation. If it sees a `BaseModel`, `list[BaseModel]`, `dict[str, BaseModel]`, or `BaseModel | None`, it serialises via `model.model_dump()` before pickling and rehydrates via `Model.model_validate(...)` on read. The cached value on disk is a plain dict — round-trips cleanly across process boundaries, even for models defined in `__main__`.

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
