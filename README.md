# emboss

**O**n-**D**isk **I**nput-keyed **C**ache — disk-backed memoization with pydantic-aware encoding.

Version: 0.8.0

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

emboss ships three dependency-free backends; `diskcache.Cache` still works if you `pip install emboss[diskcache]`.

| backend | storage | bounded? | shared across hosts? | use when |
|---|---|---|---|---|
| `SqliteCache` *(default)* | one SQLite DB | ✅ `size_limit` + eviction + TTL | ❌ SQLite locking breaks on NFS | a bounded **local** cache (multi-process-safe) |
| `FanoutCache` | N sharded SQLite DBs | ✅ | ❌ | a local cache under **heavy concurrent writes** |
| `FileCache` | one file per key | ✅ optional `size_limit` + LRU | ✅ atomic rename, syncable | a cache on a **network mount / replicated** across machines |
| `LogCache` | per-writer append logs | best-effort | ✅ conflict-free, **few inodes** | **many small entries** on a shared/synced mount |
| `diskcache.Cache` | SQLite (+ tags, stats) | ✅ | ❌ | you want diskcache's richer feature set |

Throughput (local SSD, 512-byte values, order of magnitude — `python scripts/bench.py`):

| backend | set | get |
|---|---|---|
| `SqliteCache` | ~10⁴/s | ~10⁵/s |
| `FileCache` | ~10⁴/s | ~10⁴/s |
| `LogCache` | ~10⁴/s | ~10⁵/s (warm index) |

`SqliteCache` is multi-process-safe (SQLite `busy_timeout` + `BEGIN IMMEDIATE` + a "database is locked" retry), keeps `size`/`count` accurate via DB triggers (so `size_limit` holds across processes and `len()`/`volume()` are O(1)), spills values ≥ 32 KB to side files to keep the DB small, and runs `auto_vacuum=FULL` so the DB shrinks as entries are evicted.

### `SqliteCache` — bounded, dependency-free (the default)

```python
from emboss import SqliteCache, cached

cache = SqliteCache(".data/cache", size_limit=2**30)  # 1 GiB, LRU-evicted

@cached(cache)
def expensive(x: int) -> dict:
    ...
```

One SQLite DB; `size_limit` bytes (default 1 GiB) with eviction, plus optional per-entry TTL (`cache.set(key, value, expire=3600)`) and an `expire()` sweep. Stdlib only — this is what replaced the `diskcache` dependency.

**Eviction policy** (`eviction_policy=`): `least-recently-stored` (default) orders victims by store time and needs **no write on read**; `least-recently-used` refreshes recency on read but, when reads land > 60 s apart, each read becomes a write transaction — benchmarks at **~8–9× slower reads** in that worst case (≈ equal to LRS in steady state, where a 60 s throttle suppresses the rewrites). Default to LRS unless you specifically need recency-aware eviction.

### `FanoutCache` — sharded for write concurrency

```python
from emboss import FanoutCache, cached

cache = FanoutCache(".data/cache", shards=8)  # 8 independent SQLite DBs

@cached(cache)
def expensive(x: int) -> dict:
    ...
```

A single SQLite DB serialises writers on one lock. `FanoutCache` spreads entries across `shards` independent `SqliteCache` databases (each with its own lock), so writes to different keys mostly proceed in parallel — useful for a process pool or async fleet hammering one cache. Routing uses a **stable** md5 hash of the key (not Python's salted `hash()`), so every process/node agrees on the shard. `size_limit` is split evenly across shards; `len()`/`volume()`/`clear()`/iteration aggregate.

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

### `LogCache` — replication-safe with few inodes

```python
from emboss import LogCache, cached

cache = LogCache(".data/cache")  # writer_id defaults to the hostname

@cached(cache)
def expensive(x: int) -> dict:
    ...
```

`FileCache` is sync-safe but writes **one file per key** — millions of inodes for a big cache, and slow `du`/`rsync`/Syncthing scans. The naive fix (bundle keys into one file per prefix) makes sync *worse*: two nodes rewriting the same bundle means a syncer's last-write-wins drops a whole node's chunk of entries.

`LogCache` gets few inodes **and** conflict-free sync by giving every writer its own files. Entries are sharded into 256 prefixes; within each, a node appends to `directory/<prefix>/<writer_id>.log`. Because **no file is ever written by two nodes**, a syncer just ships each node's logs around — last-write-wins never fires, so a conflict can't lose data. Reads merge a prefix's logs (cached in memory; rebuilt when a peer's log grows); deletes append tombstones; compaction (auto past `max_log_bytes`, or `compact()`) rewrites *this node's own* log dropping dead records. Same-node processes are serialised by a per-writer lock file.

**Consolidation / GC** is the missing cross-writer collector: `consolidate()` (auto past `max_writers_per_prefix`, default 8) merges writers' logs in a prefix into this node's single log, drops dead records, and prunes the now-redundant peer logs — bounding the file count (≈ #prefixes × #writers) that otherwise grows forever as writers come and go (decommissioned hosts, one-shot bulk-import jobs, containers that fell back to a random hostname). It is safe against concurrent *appends*: it snapshots each peer log's `(size, mtime)` and re-`stat`s before deleting, so a peer/local append that lands *during* consolidation is never deleted (its newer records win on read; the next pass folds them in). Foreign spilled values are adopted into the shared content-addressed pool (hardlinked when possible), so the consolidated log stays self-contained after the peers are gone.

In the benchmark, **20,000 entries used 512 files (vs `FileCache`'s 20,000)** — the inode count is capped at #prefixes × #writers, independent of entry count. Reads are an in-memory index lookup: with the default `index_ttl=1.0s` (which throttles the freshness re-`stat` for peers' appends) LogCache benches as the **fastest backend (~380k get/s)**; the trade is up to `index_ttl` of staleness on cross-process/node writes (own writes are immediate, and 1 s is well under Syncthing's latency).

**Large values spill to side files** (`min_file_size`, default 32 KB), keeping the append log small — verified on touchstone's real cache, where a **185 MB** value spilled to a side file and the log stayed at **120 bytes**. Spills live in a per-prefix **shared, content-addressed pool** (`<prefix>/spill/<sha256>.val`): identical values collapse to one file across writers, and two nodes "conflicting" on a pool file write identical bytes, so a syncer's last-write-wins is harmless there. Pool files are deleted only by the consolidation mark-and-sweep (references derived from every log, plus a grace window) — never on the write path.

Tunables (`python scripts/bench.py` picked the defaults): `index_ttl` (1.0 s), `prefix_width` (2 → 256 shards; use 3 above ~2M entries), `max_log_bytes` (4 MB), `min_file_size` (32 KB), `max_writers_per_prefix` (8; `0` disables auto-consolidation), and — replicated mode only — `replicated_stale_ttl` (12 h) and `replicated_spill_grace` (24 h). `writer_id` **must be unique per node** (defaults to the hostname); `prefix_width` must match across writers of a directory.

#### Replicated trees — the `.replicated` marker

> **WARNING — pre-0.8 consolidation must never run on a replicated tree.** It rewrites and deletes *every* writer's logs; run it on more than one node of a file-replicated fabric (Syncthing, etc.) and the same-named rewritten files diverge, the syncer resolves last-writer-wins, and records are silently lost. This caused real fleet data loss on 2026-07-10 ([#22](https://github.com/DJRHails/emboss/issues/22)): per-shard writer logs deleted or stubbed, ≥4,015 LLM calls re-billed. On a replicated tree, either upgrade every node to ≥0.8 **and** drop the marker, or don't consolidate at all.

Touch a `.replicated` file at the cache root (next to the prefix dirs) and `consolidate()` — including the auto-trigger in `set()` — switches to replication-safe semantics. **Whoever operates the sync fabric owns the marker**: the orchestrator that configures the folder for replication drops it in the same breath (it replicates to every node like any other file); remove it only after the folder is permanently un-shared. The marker is checked live on every pass, no restarts needed.

Replication-safe semantics — the invariants:

- **Never rewrite or truncate a foreign writer's file.** The only mutations are rewrites of the node's *own* log, creation of new files, and whole-file deletion of inputs that were folded.
- **Fold-eligible inputs**: the node's own log (sole writer — always safe); every `*.sync-conflict-*` log **regardless of age** (the syncer minted its unique name, nobody appends to it); foreign logs idle past `replicated_stale_ttl` (default 12 h; env `EMBOSS_REPLICATED_STALE_TTL`, seconds — mtime reflects the origin node's last append, which syncers preserve).
- **Outputs before inputs**: merged records land in the node's own log exactly as in exclusive mode, fsync'd (file + directory) *before* any input file is unlinked.
- **A fresh foreign log is invisible** — read normally on the read path, never mutated or deleted by consolidation. A *torn* fresh foreign log has its readable records recovered into the local log for durability; the file itself is never repaired in place.
- **Pool GC** still derives references from *all* logs present, with the grace window raised to `replicated_spill_grace` (default 24 h; env `EMBOSS_REPLICATED_SPILL_GRACE`) — a spill can replicate ahead of the log that references it.

These invariants make concurrent consolidation on every node **idempotent and coordination-free**: each node rewrites only its own uniquely-named file, and the deletions (stale/conflict inputs, unreferenced pool files) are whole-file unlinks that converge under sync whichever node performs them. A staleness misfire — folding a log whose owner was still appending, hidden by sync lag or clock skew — is self-healing: the delete-vs-modify race surfaces as a sync-conflict copy, which the next pass on any node folds by name. Note that `clear()` remains node-local in intent but its deletions *do* propagate through a syncer — don't run it on a replicated tree you don't mean to empty everywhere.

### Migrating between backends — `transfer`

```python
from emboss import transfer, SqliteCache, LogCache

transfer(SqliteCache(".old"), LogCache(".new"))   # returns the count copied
```

`transfer(source, destination, *, clear_source=False)` copies every entry **verbatim** (the stored encoding), so a cache written by `@cached` stays readable by `@cached` on the destination. Use it to switch backends (`diskcache` → `SqliteCache`), re-shard a `FanoutCache`, or consolidate logs. The source must be iterable — every emboss backend (`SqliteCache`, `FanoutCache`, `LogCache`, `FileCache`) and `diskcache.Cache` is. (`FileCache` stores each entry's key alongside its value so keys are recoverable; entries written by older versions held only the value and are skipped.)

All backends implement the `diskcache.Cache` subset `@cached` uses (`get`, `set`, `__contains__`, `__getitem__`, `__setitem__`, `__delitem__`, `delete`, `clear`, `close`, context-manager) and accept-and-ignore extra diskcache kwargs (`timeout`, ...) so call sites switch with no code changes.

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
