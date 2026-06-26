#!/usr/bin/env python3
"""Reusable cache-backend benchmark across a few workload domains.

Run: `python scripts/bench.py`. Reports set/get throughput per (domain, backend),
the inode footprint per backend (LogCache vs FileCache), and the SqliteCache
eviction-policy read cost. These numbers back the order-of-magnitude figures
quoted in each backend's module docstring. diskcache is included only if
installed (`pip install emboss[diskcache]`).
"""

from __future__ import annotations

import random
import tempfile
import time
from pathlib import Path

from emboss import FanoutCache, FileCache, LogCache, SqliteCache

# (label, value_size_bytes, n_entries)
DOMAINS = [
    ("small values  (512 B x 5000)", 512, 5000),
    ("large values  (64 KB x 1000)", 64 * 1024, 1000),
    ("many tiny     (64 B x 20000)", 64, 20000),
]


def _backends():
    yield "SqliteCache", lambda d: SqliteCache(d)
    yield "FanoutCache", lambda d: FanoutCache(d)
    yield "FileCache", lambda d: FileCache(d)
    yield "LogCache", lambda d: LogCache(d, writer_id="bench")
    try:
        import diskcache

        yield "diskcache", lambda d: diskcache.Cache(d)
    except ImportError:
        pass


def _bench(make, value_size: int, n: int) -> tuple[float, float]:
    value = b"x" * value_size
    random.seed(1)
    get_keys = [f"key-{random.randrange(n):06d}" for _ in range(min(n, 5000))]
    cache = make(tempfile.mkdtemp(prefix="emboss-bench-"))
    t0 = time.perf_counter()
    for i in range(n):
        cache.set(f"key-{i:06d}", value)
    set_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for k in get_keys:
        cache.get(k)
    get_s = time.perf_counter() - t0
    if hasattr(cache, "close"):
        cache.close()
    return n / set_s, len(get_keys) / get_s


def _inode_footprint(n: int = 20000) -> None:
    value = b"x" * 64
    print(f"\n[inode footprint — {n} tiny entries]")
    for name, make in (("FileCache", lambda d: FileCache(d)),
                       ("LogCache", lambda d: LogCache(d, writer_id="bench"))):
        d = Path(tempfile.mkdtemp(prefix="emboss-bench-"))
        cache = make(str(d))
        for i in range(n):
            cache.set(f"key-{i:06d}", value)
        files = sum(1 for _ in d.rglob("*") if _.is_file())
        print(f"  {name:12s} {files:>7,} files for {n:,} entries")


def _bench_policy(policy: str, stale: bool) -> float:
    value = b"x" * 512
    cache = SqliteCache(tempfile.mkdtemp(prefix="emboss-bench-"), size_limit=None,
                        eviction_policy=policy)
    for i in range(5000):
        cache.set(f"key-{i:05d}", value)
    if stale:
        cache._conn.execute("UPDATE Cache SET access_time = 0")
    keys = [f"key-{random.randrange(5000):05d}" for _ in range(5000)]
    t0 = time.perf_counter()
    for k in keys:
        cache.get(k)
    dt = time.perf_counter() - t0
    cache.close()
    return 5000 / dt


def _tune_logcache() -> None:
    """Simple sweep over LogCache's tunables to pick sane defaults."""
    value = b"x" * 512
    n = 5000
    keys = [f"key-{random.randrange(n):05d}" for _ in range(5000)]

    print("\n[LogCache index_ttl -> get throughput (read freshness throttle)]")
    for ttl in (0.0, 0.5, 1.0):
        c = LogCache(tempfile.mkdtemp(prefix="emboss-bench-"), writer_id="b", index_ttl=ttl)
        for i in range(n):
            c.set(f"key-{i:05d}", value)
        t0 = time.perf_counter()
        for k in keys:
            c.get(k)
        print(f"  index_ttl={ttl:<4} get {len(keys) / (time.perf_counter() - t0):>9,.0f}/s")

    print("\n[LogCache prefix_width -> set/get/inodes (20000 tiny entries)]")
    for width in (1, 2, 3):
        d = Path(tempfile.mkdtemp(prefix="emboss-bench-"))
        c = LogCache(str(d), writer_id="b", prefix_width=width)
        t0 = time.perf_counter()
        for i in range(20000):
            c.set(f"key-{i:06d}", b"x" * 64)
        sets = 20000 / (time.perf_counter() - t0)
        files = sum(1 for _ in d.rglob("*") if _.is_file())
        print(f"  prefix_width={width} ({16**width:>4} shards)  set {sets:>8,.0f}/s   files {files:>5}")

    print("\n[LogCache max_log_bytes -> set throughput (overwrite-heavy: 5000x one key)]")
    for mlb in (2**20, 4 * 2**20, 16 * 2**20):
        c = LogCache(tempfile.mkdtemp(prefix="emboss-bench-"), writer_id="b", max_log_bytes=mlb)
        t0 = time.perf_counter()
        for _ in range(5000):
            c.set("hot", b"x" * 512)
        print(f"  max_log_bytes={mlb // 2**20:>2}MB  set {5000 / (time.perf_counter() - t0):>9,.0f}/s")


def main() -> None:
    for label, size, n in DOMAINS:
        print(f"\n[{label}]")
        for name, make in _backends():
            sets, gets = _bench(make, size, n)
            print(f"  {name:12s} set {sets:>9,.0f}/s   get {gets:>9,.0f}/s")
    _inode_footprint()
    print("\n[SqliteCache eviction-policy read throughput]")
    print(f"  least-recently-stored          get {_bench_policy('least-recently-stored', False):>9,.0f}/s")
    print(f"  least-recently-used (warm)     get {_bench_policy('least-recently-used', False):>9,.0f}/s")
    print(f"  least-recently-used (rewrites) get {_bench_policy('least-recently-used', True):>9,.0f}/s")
    _tune_logcache()


if __name__ == "__main__":
    main()
