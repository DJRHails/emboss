"""Tests for the `LogCache` backend — per-writer, prefix-sharded append logs."""

from __future__ import annotations

import logging
import shutil
import os
import threading
import time

import pytest

from emboss import Cache, LogCache, cached
from emboss._log_cache import _HEADER, _frame, _iter_records, _read_records, _Record


@pytest.fixture
def cache(tmp_path):
    return LogCache(tmp_path / "cache", writer_id="test")


def test_satisfies_protocol(cache):
    assert isinstance(cache, Cache)


def test_get_default_on_miss(cache):
    assert cache.get("missing") is None
    assert cache.get("missing", default="fb") == "fb"


def test_set_get_overwrite(cache):
    cache.set("k", {"a": 1})
    assert cache.get("k") == {"a": 1}
    cache.set("k", {"a": 2})  # latest wins
    assert cache.get("k") == {"a": 2}


def test_delete(cache):
    cache.set("k", "v")
    assert cache.delete("k") is True
    assert cache.get("k") is None
    assert cache.delete("k") is False  # tombstoned / absent


def test_dunders(cache):
    cache["k"] = "v"
    assert "k" in cache
    assert cache["k"] == "v"
    with pytest.raises(KeyError):
        _ = cache["missing"]
    del cache["k"]
    assert "k" not in cache
    with pytest.raises(KeyError):
        del cache["already_gone"]


def test_persists_across_reopen(tmp_path):
    LogCache(tmp_path / "c", writer_id="a").set("k", "v")
    assert LogCache(tmp_path / "c", writer_id="a").get("k") == "v"


def test_expire_ttl(cache):
    cache.set("k", "v", expire=0.03)
    assert cache.get("k") == "v"
    time.sleep(0.05)
    assert cache.get("k") is None


def test_iteration_len_volume(cache):
    for k in ("a", "b", "c"):
        cache.set(k, k * 100)
    assert len(cache) == 3
    assert sorted(cache) == ["a", "b", "c"]
    assert cache.volume() > 0


def test_few_inodes_for_many_entries(tmp_path):
    c = LogCache(tmp_path / "c", writer_id="a")
    for i in range(1000):
        c.set(f"k{i}", i)
    logs = list((tmp_path / "c").glob("**/*.log"))
    assert len(logs) <= 256  # bounded by #prefixes, not #entries (1000)
    assert len(c) == 1000


# ── the replication-safety properties ─────────────────────────────────────────


def test_two_writers_use_separate_files(tmp_path):
    a = LogCache(tmp_path / "c", writer_id="nodeA")
    b = LogCache(tmp_path / "c", writer_id="nodeB")
    a.set("x", 1)
    b.set("y", 2)
    names = {p.name for p in (tmp_path / "c").glob("**/*.log")}
    assert names <= {"nodeA.log", "nodeB.log"}  # one file per writer; never shared


def test_reader_picks_up_peer_writes(tmp_path):
    reader = LogCache(tmp_path / "c", writer_id="R", index_ttl=0)  # always re-check
    assert reader.get("k") is None  # builds an (empty) index for the prefix
    LogCache(tmp_path / "c", writer_id="W").set("k", "v")  # a peer appends its log
    assert reader.get("k") == "v"  # reader notices the log grew, rebuilds, finds it


def test_index_ttl_defers_peer_visibility(tmp_path):
    """With index_ttl > 0, a peer's write is visible only after the TTL — the
    read-speed/staleness trade. Own writes are always immediate."""
    reader = LogCache(tmp_path / "c", writer_id="R", index_ttl=10.0)
    assert reader.get("k") is None  # caches the (empty) index for ~10s
    LogCache(tmp_path / "c", writer_id="W").set("k", "v")
    assert reader.get("k") is None  # still stale within the TTL
    reader.index_ttl = 0  # force a re-check
    assert reader.get("k") == "v"


def test_same_key_two_writers_latest_wins(tmp_path):
    a = LogCache(tmp_path / "c", writer_id="A")
    b = LogCache(tmp_path / "c", writer_id="B")
    a.set("k", "from-A")
    time.sleep(0.01)
    b.set("k", "from-B")  # later store_time
    assert LogCache(tmp_path / "c", writer_id="R").get("k") == "from-B"
    # both writers' files exist — no overwrite, so a syncer can't lose either
    assert {p.name for p in (tmp_path / "c").glob("**/*.log")} == {"A.log", "B.log"}


# ── durability / compaction ───────────────────────────────────────────────────


def test_torn_tail_ignored(tmp_path):
    c = LogCache(tmp_path / "c", writer_id="A")
    c.set("k", "good")
    log = c._log_path(c._prefix("k"))
    with open(log, "ab") as f:
        f.write(b"\x00\x00\x10\x00partial-record")  # claims 4 KB, supplies a few bytes
    assert LogCache(tmp_path / "c", writer_id="B").get("k") == "good"


def _write_log(path, *records):
    """Write raw framed records to a fresh log; return the byte offset of each."""
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets, buf = [], bytearray()
    for rec in records:
        offsets.append(len(buf))
        buf += _frame(rec)
    path.write_bytes(bytes(buf))
    return offsets


def _rec(key, value):
    return _Record(key, value, None, time.time(), False, None)


def test_mid_log_tear_recovers_later_records(tmp_path):
    """The bug: a torn frame mid-log must NOT strand every record after it."""
    log = tmp_path / "00" / "test.log"
    before, torn, after = _rec("a", "A"), _rec("b", "B"), _rec("c", "C")
    offs = _write_log(log, before, torn, after)
    # Corrupt the middle frame's blob in place (keep its length header intact) so it
    # fails to unpickle but the file stays the same size — a genuine mid-log tear.
    raw = bytearray(log.read_bytes())
    blob_start = offs[1] + _HEADER.size
    raw[blob_start : blob_start + 3] = b"\xff\xff\xff"
    log.write_bytes(bytes(raw))

    keys = [r.key for r in _iter_records(log)]
    assert keys == ["a", "c"]  # the torn "b" is skipped, "c" is recovered (not stranded)


def test_mid_log_tear_via_cache_get(tmp_path):
    """End-to-end: a value written after a mid-log tear stays readable."""
    reader = LogCache(tmp_path / "c", writer_id="R", index_ttl=0)
    # Two keys in the same prefix shard so they land in one log (the tear between).
    prefix, first, second = None, None, None
    seen: dict[str, str] = {}
    for i in range(10000):
        k = f"key{i}"
        p = reader._prefix(k)
        if p in seen:
            prefix, first, second = p, seen[p], k
            break
        seen[p] = k
    assert prefix is not None, "no colliding prefix found"
    log = tmp_path / "c" / prefix / "W.log"
    offs = _write_log(log, _rec(first, "v1"), _rec("torn", "x"), _rec(second, "v3"))
    raw = bytearray(log.read_bytes())
    bs = offs[1] + _HEADER.size
    raw[bs : bs + 3] = b"\xff\xff\xff"  # corrupt the middle frame's blob in place
    log.write_bytes(bytes(raw))
    assert reader.get(first) == "v1"
    assert reader.get(second) == "v3"  # None under the old stop-at-first-tear reader


def test_mid_log_tear_warns(tmp_path, caplog):
    log = tmp_path / "00" / "test.log"
    offs = _write_log(log, _rec("a", "A"), _rec("b", "B"), _rec("c", "C"))
    raw = bytearray(log.read_bytes())
    bs = offs[1] + _HEADER.size
    raw[bs : bs + 3] = b"\xff\xff\xff"
    log.write_bytes(bytes(raw))
    with caplog.at_level("WARNING"):
        list(_iter_records(log))
    assert any("torn/corrupt frame" in r.message for r in caplog.records)


def test_truncated_final_write_stays_quiet(tmp_path, caplog):
    """A truncated *final* frame recovers nothing past it — the documented benign
    crash-mid-append case — so it must not warn."""
    log = tmp_path / "00" / "test.log"
    _write_log(log, _rec("a", "A"))
    with open(log, "ab") as f:
        f.write(_HEADER.pack(4096) + b"partial")  # claims 4 KB, supplies 7 bytes
    with caplog.at_level("WARNING"):
        keys = [r.key for r in _iter_records(log)]
    assert keys == ["a"]
    assert not any("torn/corrupt frame" in r.message for r in caplog.records)


def test_compaction_drops_torn_frame_permanently_and_warns(tmp_path, caplog):
    """A rewrite (compaction) removes the malformed frame from disk — the healed log has no tear
    and reads clean without resync — and logs that it did so."""
    log = tmp_path / "00" / "A.log"
    offs = _write_log(log, _rec("a", "A"), _rec("b", "B"), _rec("c", "C"))
    raw = bytearray(log.read_bytes())
    bs = offs[1] + _HEADER.size
    raw[bs : bs + 3] = b"\xff\xff\xff"  # corrupt the middle frame
    log.write_bytes(bytes(raw))

    c = LogCache(tmp_path, writer_id="A")
    with caplog.at_level("WARNING"):
        c.compact("00")
    assert any("dropped the malformed frame" in r.message for r in caplog.records)
    # After the rewrite the log is clean: no tear, only the two good records, and reads without
    # having to recover anything past a tear.
    scan = _read_records(log)
    assert scan.tear_at is None and scan.recovered == 0
    assert {r.key for r in scan.records} == {"a", "c"}


def test_compaction_of_benign_final_tear_stays_quiet(tmp_path, caplog):
    """A truncated *final* frame recovers nothing past it — the documented benign crash-mid-append
    case. Compaction drops it while rewriting, but (like the read path) must NOT warn: nothing was
    stranded, so a WARNING on every routine teardown tail would be noise."""
    log = tmp_path / "00" / "A.log"
    _write_log(log, _rec("a", "A"), _rec("b", "B"))
    with open(log, "ab") as f:
        f.write(_HEADER.pack(4096) + b"partial")  # torn final frame: claims 4 KB, supplies 7 bytes

    c = LogCache(tmp_path, writer_id="A")
    with caplog.at_level("WARNING"):
        c.compact("00")
    assert not any(
        "torn frame" in r.message or "malformed frame" in r.message for r in caplog.records
    )
    scan = _read_records(log)
    assert scan.tear_at is None and scan.recovered == 0
    assert {r.key for r in scan.records} == {"a", "b"}


def test_read_records_propagates_oserror(tmp_path):
    """`_read_records` must NOT swallow a read error into an empty result — consolidation relies
    on the OSError to mark a source unread and never prune it."""
    with pytest.raises(OSError):
        _read_records(tmp_path / "does-not-exist" / "x.log")


def test_corrupt_spill_is_a_warned_miss(tmp_path, caplog):
    """A record whose spill file is present but unreadable must miss (recompute),
    not crash the get()."""
    c = LogCache(tmp_path / "c", writer_id="A", min_file_size=1)  # force every value to spill
    c.set("k", "value-that-spills")
    rec = c._ensure_index(c._prefix("k"))["k"]
    (c.directory / rec.spill).write_bytes(b"\x00not-a-pickle")  # corrupt the spill
    fresh = LogCache(tmp_path / "c", writer_id="A", min_file_size=1, index_ttl=0)
    with caplog.at_level("WARNING"):
        assert fresh.get("k", "MISS") == "MISS"
    assert any("unreadable" in r.message for r in caplog.records)


def test_compaction_shrinks_and_keeps_latest(tmp_path):
    c = LogCache(tmp_path / "c", writer_id="A")
    for i in range(200):
        c.set("k", i)  # 199 superseded records pile up
    log = c._log_path(c._prefix("k"))
    before = log.stat().st_size
    c.compact()
    assert log.stat().st_size < before  # dead records dropped
    assert c.get("k") == 199  # latest survives
    assert len(c) == 1


def test_auto_compaction_on_large_log(tmp_path):
    c = LogCache(tmp_path / "c", writer_id="A", max_log_bytes=2000)
    for i in range(500):
        c.set(
            "k", "x" * 50
        )  # repeatedly overwrite -> log would grow without compaction
    log = c._log_path(c._prefix("k"))
    assert log.stat().st_size < 10_000  # auto-compaction kept it small
    assert c.get("k") == "x" * 50


def test_size_limit_best_effort_per_log(tmp_path):
    """size_limit bounds each (prefix, writer) log's live bytes at compaction —
    best-effort and per-log (like FanoutCache's per-shard split), not global."""
    c = LogCache(tmp_path / "c", writer_id="A", size_limit=5000)
    for i in range(50):
        c.set(f"k{i}", "x" * 1000)  # spread across prefixes
    c.compact()
    for log in (tmp_path / "c").glob("**/*.log"):
        assert log.stat().st_size <= 5000 + 1500  # per-log bound + framing slack


def _spills(root):
    return list(root.glob("**/spill/*.val"))


def _sweep_now(cache, prefix):
    """Consolidate with the sweep grace collapsed to zero (test-only shortcut)."""
    cache._SHARED_SPILL_GRACE_S = 0.0
    cache.consolidate(prefix)


def test_large_value_spills_to_file(tmp_path):
    root = tmp_path / "c"
    c = LogCache(root, writer_id="A", min_file_size=100)
    c.set("big", "x" * 5000)
    assert len(_spills(root)) == 1
    log = c._log_path(c._prefix("big"))
    assert log.stat().st_size < 500  # the log holds a reference, not the 5 KB value
    assert c.get("big") == "x" * 5000  # read back from the spill file
    c.set("small", "y")  # stays inline
    assert len(_spills(root)) == 1


def test_superseded_spill_collected_by_sweep(tmp_path):
    """An overwrite never deletes on the write path (the pool is shared across
    writers and nodes); the superseded value's file waits for the consolidation
    mark-and-sweep, which derives references from every log."""
    root = tmp_path / "c"
    c = LogCache(root, writer_id="A", min_file_size=100)
    c.set("k", "a" * 5000)
    assert len(_spills(root)) == 1
    c.set("k", "b" * 5000)  # supersedes; the old pool file lingers until the sweep
    assert len(_spills(root)) == 2
    assert c.get("k") == "b" * 5000
    _sweep_now(c, c._prefix("k"))
    assert len(_spills(root)) == 1  # sweep collected the unreferenced value
    assert c.get("k") == "b" * 5000


def test_spill_removed_on_delete(tmp_path):
    root = tmp_path / "c"
    c = LogCache(root, writer_id="A", min_file_size=100)
    c.set("k", "a" * 5000)
    c.delete("k")
    _sweep_now(c, c._prefix("k"))
    assert _spills(root) == []


def test_expired_spill_collected_by_sweep_not_compaction(tmp_path):
    """Compaction rewrites only OUR log and cannot know a pool file is globally
    unreferenced; the consolidation sweep — which reads every log — collects it."""
    root = tmp_path / "c"
    keys = _SAME_PREFIX_KEYS[:2]  # one prefix, so one sweep covers both
    c = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    c.set(keys[0], "x" * 5000, expire=0.01)
    c.set(keys[1], "y" * 5000)
    time.sleep(0.03)
    assert len(_spills(root)) == 2  # expiry is lazy — both spill files still present
    c.compact()
    assert len(_spills(root)) == 2  # compaction leaves the shared pool alone
    _sweep_now(c, c._prefix(keys[0]))
    assert len(_spills(root)) == 1  # the sweep dropped the expired one
    assert c.get(keys[1]) == "y" * 5000


def test_spilled_value_transfers(tmp_path):
    from emboss import SqliteCache, transfer

    src = LogCache(tmp_path / "src", writer_id="A", min_file_size=100)
    src.set("big", "z" * 5000)
    dst = SqliteCache(tmp_path / "dst")
    assert transfer(src, dst) == 1
    assert dst.get("big") == "z" * 5000  # large value survives a cross-backend transfer
    dst.close()


def test_works_with_cached_decorator(tmp_path):
    cache = LogCache(tmp_path / "c", writer_id="A")
    calls = {"n": 0}

    @cached(cache)
    def f(x: int) -> int:
        calls["n"] += 1
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls["n"] == 1


# ── consolidation (cross-writer GC) ───────────────────────────────────────────


# With prefix_width=1 these single-char keys all hash to the same shard ("8"), so
# each writer below leaves a log in ONE prefix dir — exactly the accumulation a
# consolidate must collapse. (Verified: md5(k)[:1] == "8" for each.)
_SAME_PREFIX_KEYS = ["d", "f", "i", "k", "p"]


def test_consolidate_merges_many_writers_into_one(tmp_path):
    """N writers in a prefix → one log after consolidate, every live entry kept."""
    root = tmp_path / "c"
    prefix = LogCache(root, prefix_width=1)._prefix(_SAME_PREFIX_KEYS[0])
    for w, k in zip("ABCDE", _SAME_PREFIX_KEYS):  # each writer sets a colliding key
        LogCache(root, writer_id=w, prefix_width=1).set(k, k * 3)
    before = list((root / prefix).glob("*.log"))
    assert len(before) == 5  # five writer logs piled up in the one prefix

    gc = LogCache(root, writer_id="GC", prefix_width=1)
    gc.consolidate()

    after = list((root / prefix).glob("*.log"))
    assert len(after) < len(before)  # file count dropped
    assert {p.name for p in after} == {"GC.log"}  # all folded into our single log
    reader = LogCache(root, writer_id="R", prefix_width=1, index_ttl=0)
    for k in _SAME_PREFIX_KEYS:
        assert reader.get(k) == k * 3  # every writer's live entry still reads back


def test_consolidate_cross_writer_supersede(tmp_path):
    root = tmp_path / "c"
    LogCache(root, writer_id="A").set("k", 1)  # older
    time.sleep(0.01)
    LogCache(root, writer_id="B").set("k", 2)  # newer store_time wins
    gc = LogCache(root, writer_id="A")  # consolidate into A's existing log
    gc.consolidate()
    assert LogCache(root, writer_id="R", index_ttl=0).get("k") == 2
    # exactly one surviving record for the key (the latest) in our single log
    prefix = gc._prefix("k")
    recs = list(_iter_records(gc._log_path(prefix)))
    assert len(recs) == 1
    assert recs[0].value == 2


def test_consolidate_tombstone_wins(tmp_path):
    root = tmp_path / "c"
    LogCache(root, writer_id="A").set("k", "v")  # older set
    time.sleep(0.01)
    LogCache(root, writer_id="B").delete("k")  # newest = tombstone for A's key
    gc = LogCache(root, writer_id="A")
    prefix = gc._prefix("k")
    before = sum(p.stat().st_size for p in (root / prefix).glob("*.log"))
    gc.consolidate()
    assert LogCache(root, writer_id="R", index_ttl=0).get("k") is None  # key absent
    target = gc._log_path(prefix)
    after = target.stat().st_size if target.exists() else 0
    assert after < before  # dead key dropped → log shrank (or removed entirely)
    # nothing live remains in our log: either it's gone or holds no records
    assert not target.exists() or list(_iter_records(target)) == []


def test_consolidate_keeps_changed_peer_log(tmp_path):
    """Sync-safety: a peer log appended-to AFTER the snapshot but BEFORE the prune
    must NOT be deleted, and its new entry must survive (its newer record wins)."""
    root = tmp_path / "c"
    k_a, k_p, k_p2 = _SAME_PREFIX_KEYS[:3]  # all share one prefix
    a = LogCache(root, writer_id="A", prefix_width=1)
    a.set(k_a, 1)
    peer = LogCache(root, writer_id="PEER", prefix_width=1)
    peer.set(k_p, 1)
    prefix = a._prefix(k_a)
    peer_log = root / prefix / "PEER.log"

    # Make the prune step see PEER.log as changed since the snapshot: a concurrent
    # peer append lands AFTER the snapshot was taken but BEFORE the prune runs.
    real_prune = LogCache._prune_consolidated_sources

    def append_then_prune(self, pdir, snapshot, target_name, unread):
        peer.set(k_p2, 2)  # a concurrent peer append lands mid-consolidation
        return real_prune(self, pdir, snapshot, target_name, unread)

    monkey = LogCache(root, writer_id="A", prefix_width=1)
    object.__setattr__(
        monkey, "_prune_consolidated_sources", append_then_prune.__get__(monkey)
    )
    monkey.consolidate(prefix)

    assert peer_log.exists()  # changed peer log was NOT deleted (no lost write)
    reader = LogCache(root, writer_id="R", prefix_width=1, index_ttl=0)
    assert reader.get(k_p2) == 2  # the concurrent append survives
    assert reader.get(k_p) == 1
    assert reader.get(k_a) == 1


def test_prune_stat_decision_unchanged_vs_changed(tmp_path):
    """Directly pin the prune decision: an unchanged source is deleted, a source
    whose (size, mtime) differs from the snapshot is left intact."""
    root = tmp_path / "c"
    gc = LogCache(root, writer_id="GC", prefix_width=1)
    pdir = root / "00"
    pdir.mkdir(parents=True)
    stable = pdir / "STABLE.log"
    changed = pdir / "CHANGED.log"
    stable.write_bytes(b"x")
    changed.write_bytes(b"x")
    snapshot = {}
    for log in (stable, changed):
        st = log.stat()
        snapshot[log.name] = (st.st_size, st.st_mtime_ns)
    # mutate CHANGED after snapshotting it
    time.sleep(0.01)
    changed.write_bytes(b"xy")

    gc._prune_consolidated_sources(pdir, snapshot, "GC.log", set())
    assert not stable.exists()  # unchanged → fully represented → deleted
    assert changed.exists()  # changed since snapshot → kept (sync-safe)


def test_consolidate_foreign_spill_reread_byte_correct(tmp_path):
    root = tmp_path / "c"
    big = "z" * 5000
    b = LogCache(root, writer_id="B", prefix_width=1, min_file_size=100)
    b.set("big", big)
    prefix = b._prefix("big")
    assert len(_spills(root)) == 1  # B spilled the large value into the shared pool

    a = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    a.consolidate(prefix)

    assert LogCache(root, writer_id="R", prefix_width=1, index_ttl=0).get("big") == big
    assert len(_spills(root)) == 1  # the pool file was never copied or moved
    assert {p.name for p in (root / prefix).glob("*.log")} == {"A.log"}


def test_consolidate_twice_never_grows_the_pool(tmp_path):
    """The incident regression guard: with uuid spill names every consolidation
    pass re-copied every foreign value (45 GB -> 160 GB in a day once a syncer
    kept resurrecting pruned peer logs). Content addressing makes passes
    idempotent — the file count must not grow, whatever the pass count."""
    root = tmp_path / "c"
    for w in ("B", "C", "D"):
        LogCache(root, writer_id=w, prefix_width=1, min_file_size=100).set(
            f"k-{w}", w * 5000
        )
    a = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    prefixes = {a._prefix(f"k-{w}") for w in ("B", "C", "D")}
    for _ in range(3):
        for prefix in prefixes:
            a.consolidate(prefix)
    assert len(_spills(root)) == 3  # one content-addressed file per distinct value
    reader = LogCache(root, writer_id="R", prefix_width=1, index_ttl=0)
    assert all(reader.get(f"k-{w}") == w * 5000 for w in ("B", "C", "D"))


def test_identical_values_share_one_pool_file(tmp_path):
    """Cross-writer dedup at write time: the same bytes under any writer land on
    the same content-addressed name — unified, never copied per writer."""
    root = tmp_path / "c"
    big = "same" * 2500
    keys = _SAME_PREFIX_KEYS[:2]
    LogCache(root, writer_id="A", prefix_width=1, min_file_size=100).set(keys[0], big)
    LogCache(root, writer_id="B", prefix_width=1, min_file_size=100).set(keys[1], big)
    assert len(_spills(root)) == 1  # two writers, two keys, ONE file
    reader = LogCache(root, writer_id="R", prefix_width=1, index_ttl=0)
    assert reader.get(keys[0]) == big
    assert reader.get(keys[1]) == big


def test_legacy_namespace_spills_migrate_on_consolidation(tmp_path):
    """A prefix holding the pre-pool per-writer layout is unified on sight: the
    legacy file's bytes are adopted into the pool (hardlink — same inode), the
    record is repointed, and the legacy namespace dies with its pruned log."""
    root = tmp_path / "c"
    b = LogCache(root, writer_id="B", prefix_width=1, min_file_size=100)
    b.set("big", "z" * 5000)
    prefix = b._prefix("big")
    # Rebuild the legacy layout by hand: move the pool file into B's namespace
    # and repoint B's record at it (as a pre-pool writer would have written).
    pool_file = next((root / prefix / "spill").glob("*.val"))
    legacy_dir = root / prefix / "B.spill"
    legacy_dir.mkdir()
    legacy_rel = f"{prefix}/B.spill/{pool_file.name}"
    pool_file.rename(root / legacy_rel)
    (root / prefix / "spill").rmdir()
    log_path = root / prefix / "B.log"
    recs = [r._replace(spill=legacy_rel) for r in _iter_records(log_path)]
    log_path.write_bytes(b"".join(_frame(r) for r in recs))

    a = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    assert a.get("big") == "z" * 5000  # legacy refs still resolve pre-migration
    a.consolidate(prefix)

    assert not legacy_dir.exists()  # legacy namespace pruned with its log
    assert len(list((root / prefix / "spill").glob("*.val"))) == 1  # adopted
    assert LogCache(root, writer_id="R", prefix_width=1, index_ttl=0).get(
        "big"
    ) == "z" * 5000


def test_consolidate_missing_foreign_spill_drops_record(tmp_path):
    """A foreign spill absent on disk (sync lag) → drop that record rather than
    write a dangling spill reference."""
    root = tmp_path / "c"
    big_k, keep_k = [k for k in _SAME_PREFIX_KEYS if k != "k"][:2]  # share a prefix
    b = LogCache(root, writer_id="B", prefix_width=1, min_file_size=100)
    b.set(big_k, "z" * 5000)  # B spills the large value
    prefix = b._prefix(big_k)
    # a second live key keeps the consolidated log non-empty, so a (buggy) dangling
    # ref would actually be written — making this assertion able to catch it.
    LogCache(root, writer_id="B", prefix_width=1).set(keep_k, "ok")
    # simulate the spill file lagging behind the log (log synced, spill not yet)
    shutil.rmtree(root / prefix / "spill")

    a = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    a.consolidate(prefix)

    target = a._log_path(prefix)
    keys = [r.key for r in _iter_records(target)]
    assert big_k not in keys  # record genuinely DROPPED, not kept with a dead ref
    assert all(r.spill is None for r in _iter_records(target))  # no dangling spill ref
    reader = LogCache(root, writer_id="R", prefix_width=1, index_ttl=0)
    assert reader.get(big_k) is None  # spilled value gone (recomputes on next read)
    assert reader.get(keep_k) == "ok"  # the other live entry survived
    assert not (root / prefix / "spill").exists()  # we wrote no spill at all


def test_consolidate_empty_removes_log(tmp_path):
    root = tmp_path / "c"
    a = LogCache(root, writer_id="A", prefix_width=1)
    a.set("k", "v")
    b = LogCache(root, writer_id="B", prefix_width=1)
    time.sleep(0.01)
    b.delete("k")  # newest is a tombstone → nothing live
    gc = LogCache(root, writer_id="GC", prefix_width=1)
    gc.consolidate()
    prefix = gc._prefix("k")
    assert not gc._log_path(prefix).exists()  # empty result → our log dropped
    assert len(LogCache(root, writer_id="R", prefix_width=1, index_ttl=0)) == 0


def test_auto_consolidate_on_writer_count(tmp_path):
    """Past max_writers_per_prefix, the next set folds the prefix into one log."""
    root = tmp_path / "c"
    # 3 peer writers each leave a log in the shared prefix (same key → same prefix)
    for w in ("w0", "w1", "w2"):
        LogCache(root, writer_id=w, prefix_width=1, max_writers_per_prefix=0).set(
            "k", w
        )
    prefix = LogCache(root, prefix_width=1)._prefix("k")
    assert len({p.name for p in (root / prefix).glob("*.log")}) == 3

    # our writer with a small bound: its set sees 3 peers + itself > 2 → consolidate
    writer = LogCache(
        root, writer_id="ME", prefix_width=1, max_writers_per_prefix=2, index_ttl=0
    )
    writer.get("k")  # build the index so _sig[prefix] is populated with peer logs
    writer.set("k", "ME")  # this write trips the auto-consolidation
    assert {p.name for p in (root / prefix).glob("*.log")} == {"ME.log"}  # back to one
    assert LogCache(root, writer_id="R", prefix_width=1, index_ttl=0).get("k") == "ME"


def test_consolidate_sweeps_superseded_value(tmp_path):
    """A peer overwrites our spilled key with a newer value. The consolidation
    sweep — the only pool deleter — must collect our now-unreferenced file."""
    root = tmp_path / "c"
    a = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    a.set("d", "x" * 5000)  # A's value for "d"
    prefix = a._prefix("d")
    time.sleep(0.01)
    LogCache(root, writer_id="B", prefix_width=1, min_file_size=100).set(
        "d", "y" * 5000
    )  # newer
    assert len(_spills(root)) == 2

    gc = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    _sweep_now(gc, prefix)

    assert (
        LogCache(root, writer_id="R", prefix_width=1, index_ttl=0).get("d")
        == "y" * 5000
    )
    assert len(_spills(root)) == 1  # superseded value swept, winner kept


def test_sweep_grace_protects_inflight_spill(tmp_path):
    """A pool file not yet referenced by any log (a concurrent set() wrote it
    before appending its record) must survive the sweep — the grace window is
    what protects the imminent record from a glob-and-delete."""
    root = tmp_path / "c"
    a = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    a.set("d", "v" * 5000)  # a real spilled record so the log is non-empty
    prefix = a._prefix("d")
    inflight = root / prefix / "spill" / "inflight.val"
    inflight.write_bytes(b"not-yet-logged")  # simulate an in-flight concurrent spill

    a.consolidate(prefix)  # default grace: the young unreferenced file survives

    assert inflight.exists()
    assert a.get("d") == "v" * 5000
    _sweep_now(a, prefix)  # grace collapsed: now it is genuinely orphaned -> swept
    assert not inflight.exists()


def test_consolidate_keeps_unreadable_source(tmp_path, monkeypatch):
    """A source log that errors mid-read is never pruned — its unmerged records
    would otherwise be lost when its (size, mtime) still matches the snapshot."""
    import emboss._log_cache as m

    root = tmp_path / "c"
    LogCache(root, writer_id="A", prefix_width=1).set("d", 1)
    LogCache(root, writer_id="PEER", prefix_width=1).set("f", 2)
    prefix = LogCache(root, prefix_width=1)._prefix("d")
    peer_log = root / prefix / "PEER.log"
    real_read = m._read_records

    def flaky(path):
        if path.name == "PEER.log":
            raise OSError("transient read failure")
        return real_read(path)

    monkeypatch.setattr(m, "_read_records", flaky)  # the parse entrypoint consolidation uses
    LogCache(root, writer_id="A", prefix_width=1).consolidate(prefix)
    monkeypatch.undo()

    assert peer_log.exists()  # unreadable source kept, not pruned
    reader = LogCache(root, writer_id="R", prefix_width=1, index_ttl=0)
    assert reader.get("f") == 2  # its records survive (read from the kept log)
    assert reader.get("d") == 1


def test_consolidate_drops_expired(tmp_path):
    """consolidate honours TTL: an expired record is dropped from the merge."""
    root = tmp_path / "c"
    gone, keep = _SAME_PREFIX_KEYS[:2]
    LogCache(root, writer_id="A", prefix_width=1).set(gone, 1, expire=0.01)
    LogCache(root, writer_id="B", prefix_width=1).set(keep, 2)
    prefix = LogCache(root, prefix_width=1)._prefix(gone)
    time.sleep(0.03)

    LogCache(root, writer_id="A", prefix_width=1).consolidate(prefix)

    keys = [r.key for r in _iter_records(root / prefix / "A.log")]
    assert gone not in keys  # expired entry dropped
    assert keep in keys


def test_auto_consolidate_disabled_with_zero_bound(tmp_path):
    """max_writers_per_prefix=0 disables the auto-GC trigger: peer logs accumulate
    untouched no matter how many appear."""
    root = tmp_path / "c"
    for w in ("w0", "w1", "w2", "w3"):
        LogCache(root, writer_id=w, prefix_width=1, max_writers_per_prefix=0).set(
            "k", w
        )
    prefix = LogCache(root, prefix_width=1)._prefix("k")
    me = LogCache(
        root, writer_id="ME", prefix_width=1, max_writers_per_prefix=0, index_ttl=0
    )
    me.get("k")  # populate _sig with every peer log
    me.set("k", "ME")  # would trip auto-consolidation if it were enabled

    logs = {p.name for p in (root / prefix).glob("*.log")}
    assert logs == {"w0.log", "w1.log", "w2.log", "w3.log", "ME.log"}  # nothing folded


def test_consolidate_then_cached_and_transfer(tmp_path):
    from emboss import SqliteCache, transfer

    root = tmp_path / "c"
    cache = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    calls = {"n": 0}

    @cached(cache)
    def f(x: int) -> int:
        calls["n"] += 1
        return x * 2

    assert f(3) == 6
    cache.set("blob", "q" * 5000)  # a spilled value to exercise re-spill on transfer
    cache.consolidate()
    assert f(3) == 6  # still a hit after consolidate (no recompute)
    assert calls["n"] == 1
    assert cache.get("blob") == "q" * 5000

    dst = SqliteCache(tmp_path / "dst")
    assert transfer(cache, dst) >= 1
    assert (
        dst.get("blob") == "q" * 5000
    )  # spilled value survives transfer post-consolidate
    dst.close()


def test_default_writer_id_collapses_bare_container_hostname(monkeypatch):
    """A 12-hex docker hostname must not mint an ephemeral writer namespace —
    orphaned containers share one id (flock-serialised, safe) with a warning."""
    import emboss._log_cache as m

    monkeypatch.setattr(m.socket, "gethostname", lambda: "2863d6c454a5")
    assert m._default_writer_id() == "container-orphan"
    monkeypatch.setattr(m.socket, "gethostname", lambda: "bonbon")
    assert m._default_writer_id() == "bonbon"


def test_orphaned_legacy_namespace_is_swept(tmp_path):
    """A legacy `<writer>.spill/` dir whose log is GONE must be collected: nothing
    references its files, and its existence re-arms the migration flag forever."""
    root = tmp_path / "c"
    a = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    a.set("d", "v" * 5000)  # a live record so the consolidated log is non-empty
    prefix = a._prefix("d")
    orphan = root / prefix / "ghost.spill"
    orphan.mkdir()
    stale = orphan / "leftover.val"
    stale.write_bytes(b"pre-pool leftovers")
    old = time.time() - 7200
    os.utime(stale, (old, old))  # past the grace window

    a.consolidate(prefix)
    assert not orphan.exists()  # orphaned namespace collected
    assert a.get("d") == "v" * 5000

    young_orphan = root / prefix / "ghost2.spill"
    young_orphan.mkdir()
    (young_orphan / "fresh.val").write_bytes(b"maybe a pre-pool writer still runs")
    a.consolidate(prefix)
    assert young_orphan.exists()  # grace window postpones the sweep


def test_batch_consolidate_parallel_equivalence(tmp_path):
    """consolidate() with no prefix fans across a pool — the outcome must match the
    serial pass: every pre-batch value stays readable, and a concurrent same-process
    writer's appends are neither corrupted nor lost (flock-serialised; a log appended
    to mid-pass survives pruning via the re-stat guard)."""
    root = tmp_path / "c"
    for w in ("A", "B", "C"):
        writer = LogCache(root, writer_id=w, prefix_width=1, min_file_size=100)
        for i in range(12):
            writer.set(f"{w}-{i}", f"{w}-{i}-" + "v" * 3000)
    gc = LogCache(root, writer_id="GC", prefix_width=1, min_file_size=100)

    stop = threading.Event()
    wrote = 0

    def churn() -> None:  # a live writer racing the batch
        nonlocal wrote
        live = LogCache(root, writer_id="LIVE", prefix_width=1, min_file_size=100)
        while not stop.is_set():
            live.set(f"live-{wrote % 4}", f"live-{wrote}-" + "x" * 3000)
            wrote += 1

    t = threading.Thread(target=churn, daemon=True)
    t.start()
    try:
        gc.consolidate()
    finally:
        stop.set()
        t.join(timeout=5)

    reader = LogCache(root, writer_id="R", prefix_width=1, index_ttl=0)
    for w in ("A", "B", "C"):
        for i in range(12):
            assert reader.get(f"{w}-{i}") == f"{w}-{i}-" + "v" * 3000
    for k in range(min(wrote, 4)):  # no live append was lost or corrupted by the batch
        got = reader.get(f"live-{k}")
        assert got is not None
        n = int(got.split("-")[1])
        assert n % 4 == k and got == f"live-{n}-" + "x" * 3000


def test_batch_pass_survives_prefix_failures(tmp_path, caplog):
    """Failing prefixes must not abort the batch: every prefix is still attempted,
    the first error re-raises once the batch completes, and the other failures are
    logged rather than silently dropped."""
    cache = LogCache(tmp_path / "c", writer_id="W", prefix_width=1)
    for i in range(64):
        cache.set(f"k{i}", i)
    prefixes = sorted(p.name for p in (tmp_path / "c").iterdir() if p.is_dir())
    assert len(prefixes) > 2
    bad = set(prefixes[:2])
    attempted: list[str] = []
    real = cache._consolidate_prefix

    def flaky(prefix: str) -> None:
        attempted.append(prefix)  # list.append is atomic under the GIL
        if prefix in bad:
            raise RuntimeError(f"boom-{prefix}")
        real(prefix)

    cache._consolidate_prefix = flaky
    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match=r"boom-"):
        cache.consolidate()
    assert sorted(attempted) == prefixes
    assert sum("suppressed" in r.getMessage() for r in caplog.records) == 1


def test_batch_falls_back_to_serial_without_fcntl(tmp_path, monkeypatch):
    """Where fcntl is unavailable the flock is a no-op and nothing would exclude a
    racing same-process set(), so the batch must keep the serial pass under the
    process lock instead of fanning out."""
    import emboss._log_cache as log_cache_module

    monkeypatch.setattr(log_cache_module, "_HAS_FCNTL", False)
    monkeypatch.setattr(log_cache_module, "ThreadPoolExecutor", None)  # fan-out would TypeError
    cache = LogCache(tmp_path / "c", writer_id="W", prefix_width=1, min_file_size=100)
    for i in range(24):
        cache.set(f"k{i}", f"k{i}-" + "v" * 3000)
    cache.consolidate()
    reader = LogCache(tmp_path / "c", writer_id="R", prefix_width=1, index_ttl=0)
    for i in range(24):
        assert reader.get(f"k{i}") == f"k{i}-" + "v" * 3000
