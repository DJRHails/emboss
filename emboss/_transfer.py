"""transfer() — copy entries from one cache into another.

Migrate between backends (e.g. `diskcache.Cache` -> `SqliteCache`, or `FileCache`
-> `LogCache`), re-shard a `FanoutCache`, or consolidate one `LogCache` into
another. Values are copied **verbatim** (the stored encoding), so a cache written
by `@cached` stays readable by `@cached` on the destination — no re-encoding, and
the function-identity keys are preserved.

The source must be **iterable over keys** — every emboss backend (`SqliteCache`,
`FanoutCache`, `LogCache`, `FileCache`) and `diskcache.Cache` is. (`FileCache`
entries written before key-recovery hold only the value and are skipped.)
"""

from __future__ import annotations

import logging
from typing import Any

_MISSING = object()

logger = logging.getLogger(__name__)


def transfer(source: Any, destination: Any, *, clear_source: bool = False) -> int:
    """Copy every live entry from `source` into `destination`; return the count.

    :param source: an iterable cache (yields keys) with a ``.get(key, default)``.
    :param destination: a cache with a ``.set(key, value)``.
    :param clear_source: if true, ``source.clear()`` after a successful copy —
        **skipped, with a warning, when any key was left behind**: a key that
        iterates as live but reads as a miss may be an entry that expired
        mid-transfer, or a `LogCache` record whose value is unreadable in this
        environment (import skew) and perfectly readable elsewhere. Deleting the
        only copy of what wasn't transferred is never safe, so the source is
        kept for the caller to clear once the skipped keys are accounted for.
    :return: number of entries copied.
    """
    count = 0
    skipped = 0
    for key in source:
        value = source.get(key, _MISSING)
        if value is _MISSING:  # expired mid-transfer, or unreadable here (import skew)
            skipped += 1
            continue
        destination.set(key, value)
        count += 1
    if skipped:
        logger.warning(
            "emboss.transfer: %d key(s) iterated as live but read as a miss "
            "(expired mid-transfer, or values unreadable in this environment — "
            "import skew)%s.",
            skipped,
            "; clear_source skipped to preserve them" if clear_source else "",
        )
    if clear_source and not skipped:
        source.clear()
    return count
