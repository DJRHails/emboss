#!/usr/bin/env python3
"""Micro-benchmark the cache backends (set/get throughput, 512-byte values).

Run: `python scripts/bench.py`. diskcache is benchmarked only if installed.
The order-of-magnitude results here back the numbers quoted in each backend's
module docstring.
"""

from __future__ import annotations

import random
import tempfile
import time
from pathlib import Path

from emboss import FileCache, SqliteCache

VALUE = b"x" * 512
N_SET = 5000
N_GET = 5000
random.seed(1)
GET_KEYS = [f"key-{random.randrange(N_SET):05d}" for _ in range(N_GET)]


def _bench(make_cache, name: str) -> None:
    d = tempfile.mkdtemp(prefix="emboss-bench-")
    cache = make_cache(d)
    t0 = time.perf_counter()
    for i in range(N_SET):
        cache.set(f"key-{i:05d}", VALUE)
    set_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for k in GET_KEYS:
        cache.get(k)
    get_s = time.perf_counter() - t0
    print(f"  {name:22s} set {N_SET / set_s:>9,.0f}/s   get {N_GET / get_s:>9,.0f}/s")
    close = getattr(cache, "close", None)
    if close:
        close()


def _bench_policy_reads(policy: str, stale: bool) -> float:
    """get throughput for a SqliteCache eviction policy. `stale=True` ages every
    entry past the LRU access-time resolution so each get triggers a rewrite —
    the worst case for least-recently-used reads."""
    d = tempfile.mkdtemp(prefix="emboss-bench-")
    cache = SqliteCache(d, size_limit=None, eviction_policy=policy)
    for i in range(N_SET):
        cache.set(f"key-{i:05d}", VALUE)
    if stale:
        cache._conn.execute("UPDATE Cache SET access_time = 0")
    t0 = time.perf_counter()
    for k in GET_KEYS:
        cache.get(k)
    get_s = time.perf_counter() - t0
    cache.close()
    return N_GET / get_s


def main() -> None:
    print(f"[backend throughput, {len(VALUE)}-byte values, local disk]")
    _bench(lambda d: SqliteCache(d), "SqliteCache")
    _bench(lambda d: FileCache(Path(d)), "FileCache")
    try:
        import diskcache

        _bench(lambda d: diskcache.Cache(d), "diskcache (optional)")
    except ImportError:
        print("  diskcache             not installed (pip install emboss[diskcache])")

    print("\n[SqliteCache eviction-policy read throughput]")
    print(f"  least-recently-stored          get {_bench_policy_reads('least-recently-stored', False):>9,.0f}/s")
    print(f"  least-recently-used (warm)     get {_bench_policy_reads('least-recently-used', False):>9,.0f}/s")
    print(f"  least-recently-used (rewrites) get {_bench_policy_reads('least-recently-used', True):>9,.0f}/s")


if __name__ == "__main__":
    main()
