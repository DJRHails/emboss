"""transfer() — copy entries from one cache into another.

Migrate between backends (e.g. `diskcache.Cache` -> `SqliteCache`, or `FileCache`
-> `LogCache`), re-shard a `FanoutCache`, or consolidate one `LogCache` into
another. Values are copied **verbatim** (the stored encoding), so a cache written
by `@cached` stays readable by `@cached` on the destination — no re-encoding, and
the function-identity keys are preserved.

The source must be **iterable over keys** (`SqliteCache`, `FanoutCache`,
`LogCache`, and `diskcache.Cache` are; `FileCache` is not — it can't recover
original keys from its hashed paths, so it works as a `transfer` *destination*
but not a *source*).
"""

from __future__ import annotations

from typing import Any

_MISSING = object()


def transfer(source: Any, destination: Any, *, clear_source: bool = False) -> int:
    """Copy every live entry from `source` into `destination`; return the count.

    :param source: an iterable cache (yields keys) with a ``.get(key, default)``.
    :param destination: a cache with a ``.set(key, value)``.
    :param clear_source: if true, ``source.clear()`` after a successful copy.
    :return: number of entries copied.

    Entries that expire or are evicted between iteration and read are skipped.
    """
    count = 0
    for key in source:
        value = source.get(key, _MISSING)
        if value is _MISSING:  # expired/evicted between iter and get
            continue
        destination.set(key, value)
        count += 1
    if clear_source:
        source.clear()
    return count
