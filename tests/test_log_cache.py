"""Tests for the `LogCache` backend — per-writer, prefix-sharded append logs."""

from __future__ import annotations

import logging
import os
import pickle
import shutil
import threading
import time
from pathlib import Path

import pytest

from emboss import Cache, LogCache, cached
from emboss._log_cache import (
    _HEADER,
    _frame,
    _iter_records,
    _parse_frame,
    _read_records,
    _Record,
)


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
    assert keys == [
        "a",
        "c",
    ]  # the torn "b" is skipped, "c" is recovered (not stranded)


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
        f.write(
            _HEADER.pack(4096) + b"partial"
        )  # torn final frame: claims 4 KB, supplies 7 bytes

    c = LogCache(tmp_path, writer_id="A")
    with caplog.at_level("WARNING"):
        c.compact("00")
    assert not any(
        "torn frame" in r.message or "malformed frame" in r.message
        for r in caplog.records
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
    c = LogCache(
        tmp_path / "c", writer_id="A", min_file_size=1
    )  # force every value to spill
    c.set("k", "value-that-spills")
    rec = c._ensure_index(c._prefix("k"))["k"]
    (c.directory / rec.spill).write_bytes(b"\x00not-a-pickle")  # corrupt the spill
    fresh = LogCache(tmp_path / "c", writer_id="A", min_file_size=1, index_ttl=0)
    with caplog.at_level("WARNING"):
        assert fresh.get("k", "MISS") == "MISS"
    assert any("unreadable" in r.message for r in caplog.records)


def test_missing_spill_is_a_quiet_miss(tmp_path, caplog):
    """A record whose spill file isn't present yet (sync lag) misses quietly — only
    a present-but-unreadable spill or a persistent I/O error warns."""
    c = LogCache(tmp_path / "c", writer_id="A", min_file_size=1)
    c.set("k", "value-that-spills")
    rec = c._ensure_index(c._prefix("k"))["k"]
    (c.directory / rec.spill).unlink()  # spill not synced yet
    fresh = LogCache(tmp_path / "c", writer_id="A", min_file_size=1, index_ttl=0)
    with caplog.at_level("WARNING"):
        assert fresh.get("k", "MISS") == "MISS"
    assert not caplog.records  # genuine sync-lag miss stays quiet


def test_iter_records_propagates_read_error(tmp_path, monkeypatch):
    """A read error must NOT be swallowed into an empty iteration: callers rely on
    it propagating (`_collect_live_across_logs` marks the source `unread` so prune
    keeps it; `_compact_prefix` aborts instead of rewriting an empty log)."""
    log = tmp_path / "00" / "test.log"
    _write_log(log, _rec("a", "A"))
    orig_read = Path.read_bytes

    def boom(self):
        if self == log:
            raise OSError("simulated transient I/O error")
        return orig_read(self)

    monkeypatch.setattr(Path, "read_bytes", boom)
    with pytest.raises(OSError, match="simulated transient"):
        list(_iter_records(log))


def test_unreadable_source_is_not_pruned_on_consolidate(tmp_path, monkeypatch):
    """End-to-end guard: an allowlisted log that errors on read during
    consolidate() must survive — its unmerged records aren't in our log, so
    pruning it would lose them. This is the safety the `unread` set provides."""
    reader = LogCache(tmp_path / "c", writer_id="R")
    peer = LogCache(tmp_path / "c", writer_id="PEER")
    key = "k"
    peer.set(key, "peer-value")
    prefix = reader._prefix(key)
    peer_log = tmp_path / "c" / prefix / "PEER.log"
    assert peer_log.exists()

    orig_read = Path.read_bytes

    def boom(self):
        if self == peer_log:
            raise OSError("simulated transient I/O error")
        return orig_read(self)

    monkeypatch.setattr(Path, "read_bytes", boom)
    # the read error is caught into `unread`, not raised — even when the caller
    # asserted the writer dead, an unreadable log is never pruned
    reader.consolidate(prefix, prune_writer_ids={"PEER"})
    assert peer_log.exists()  # not pruned — its record was never merged


def test_parse_frame_rejects_non_record_pickle():
    """A blob that unpickles cleanly but is not a 6-field record of the right types
    must be rejected, so a false resync anchor can't seat a poison record."""
    six_char_string = pickle.dumps("abcdef")  # a 6-element iterable, wrong shape
    assert _parse_frame(_HEADER.pack(len(six_char_string)) + six_char_string, 0) is None

    mistyped = pickle.dumps((123, None, None, "not-a-float", False, None))
    assert _parse_frame(_HEADER.pack(len(mistyped)) + mistyped, 0) is None


def test_parse_frame_rejects_trailing_bytes():
    """An over-long `length` header (pickle plus trailing garbage) is not a genuine
    frame boundary and must be rejected — the exact-length check."""
    blob = _frame(_rec("k", "v"))[_HEADER.size :]  # just the pickle bytes
    padded = blob + b"\x00\x00\x00\x00"  # trailing bytes the pickle won't consume
    assert _parse_frame(_HEADER.pack(len(padded)) + padded, 0) is None


def test_resync_skips_0x80_inside_value_blob(tmp_path):
    """A `0x80` byte living inside a value payload is not a frame start; resync must
    step over it and land on the real next frame."""
    log = tmp_path / "00" / "test.log"
    payload = b"\x80\x80 not a frame \x80\x95\x80"
    offs = _write_log(log, _rec("torn", "t"), _rec("real", payload))
    raw = bytearray(log.read_bytes())
    raw[offs[0] + _HEADER.size] = 0xFF  # corrupt the first frame's blob
    log.write_bytes(bytes(raw))

    records = list(_iter_records(log))
    assert [r.key for r in records] == ["real"]
    assert records[0].value == payload  # recovered intact, no false anchor mid-blob


def test_iter_records_terminates_on_all_0x80(tmp_path):
    """A buffer that is nothing but `0x80` candidates must terminate (forward
    progress) and yield no phantom records."""
    log = tmp_path / "00" / "test.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_bytes(b"\x80" * 500)
    assert list(_iter_records(log)) == []


def test_compact_aborts_on_read_error_without_truncating(tmp_path, monkeypatch):
    """A read error during compaction must abort — never `os.replace` an empty log
    over the real one (the pre-fix bug: swallowed `OSError` -> `keep == []` ->
    truncation). The record must survive intact."""
    c = LogCache(tmp_path / "c", writer_id="A")
    c.set("k", "v")
    log = c._log_path(c._prefix("k"))
    before = log.read_bytes()
    orig_read = Path.read_bytes

    def boom(self):
        if self == log:
            raise OSError("simulated transient I/O error")
        return orig_read(self)

    monkeypatch.setattr(Path, "read_bytes", boom)
    with pytest.raises(OSError, match="simulated transient"):
        c.compact(c._prefix("k"))
    monkeypatch.undo()
    assert log.read_bytes() == before  # real log intact, not truncated to empty
    assert c.get("k") == "v"


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


def _sweep_now(cache, prefix, prune_writer_ids=None):
    """Consolidate with the sweep grace collapsed to zero (test-only shortcut)."""
    cache._SHARED_SPILL_GRACE_S = 0.0
    cache.consolidate(prefix, prune_writer_ids=prune_writer_ids)


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
    """N retired writers in a prefix → one log after an allowlisted consolidate,
    every live entry kept."""
    root = tmp_path / "c"
    prefix = LogCache(root, prefix_width=1)._prefix(_SAME_PREFIX_KEYS[0])
    for w, k in zip("ABCDE", _SAME_PREFIX_KEYS):  # each writer sets a colliding key
        LogCache(root, writer_id=w, prefix_width=1).set(k, k * 3)
    before = list((root / prefix).glob("*.log"))
    assert len(before) == 5  # five writer logs piled up in the one prefix

    gc = LogCache(root, writer_id="GC", prefix_width=1)
    gc.consolidate(prune_writer_ids=set("ABCDE"))  # caller asserts them retired

    after = list((root / prefix).glob("*.log"))
    assert len(after) < len(before)  # file count dropped
    assert {p.name for p in after} == {"GC.log"}  # all folded into our single log
    reader = LogCache(root, writer_id="R", prefix_width=1, index_ttl=0)
    for k in _SAME_PREFIX_KEYS:
        assert reader.get(k) == k * 3  # every writer's live entry still reads back


def test_consolidate_cross_writer_supersede(tmp_path):
    """Our record superseded under ANY writer is dropped from our rewrite; the
    winning record stays in the peer log it lives in (untouched)."""
    root = tmp_path / "c"
    LogCache(root, writer_id="A").set("k", 1)  # older
    time.sleep(0.01)
    LogCache(root, writer_id="B").set("k", 2)  # newer store_time wins
    gc = LogCache(root, writer_id="A")  # consolidate A's own log
    prefix = gc._prefix("k")
    b_bytes = (root / prefix / "B.log").read_bytes()
    gc.consolidate()
    assert LogCache(root, writer_id="R", index_ttl=0).get("k") == 2
    # A's only record was superseded → its rewritten log is empty and dropped
    assert not gc._log_path(prefix).exists()
    assert (root / prefix / "B.log").read_bytes() == b_bytes  # winner untouched


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
    """Sync-safety even under the allowlist: a pruned-writer log appended-to AFTER
    the snapshot but BEFORE the prune must NOT be deleted (the dead-writer
    assertion was wrong), and its new entry must survive."""
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

    def append_then_prune(self, pdir, snapshot, target_name, unread, prune_logs):
        peer.set(k_p2, 2)  # a concurrent peer append lands mid-consolidation
        return real_prune(self, pdir, snapshot, target_name, unread, prune_logs)

    monkey = LogCache(root, writer_id="A", prefix_width=1)
    object.__setattr__(
        monkey, "_prune_consolidated_sources", append_then_prune.__get__(monkey)
    )
    monkey.consolidate(prefix, prune_writer_ids={"PEER"})

    assert peer_log.exists()  # changed peer log was NOT deleted (no lost write)
    reader = LogCache(root, writer_id="R", prefix_width=1, index_ttl=0)
    assert reader.get(k_p2) == 2  # the concurrent append survives
    assert reader.get(k_p) == 1
    assert reader.get(k_a) == 1


def test_prune_stat_decision_unchanged_vs_changed(tmp_path):
    """Directly pin the prune decision: an unchanged ALLOWLISTED source is
    deleted, a changed one is left intact, and a source outside the allowlist
    is never touched however stable it is."""
    root = tmp_path / "c"
    gc = LogCache(root, writer_id="GC", prefix_width=1)
    pdir = root / "00"
    pdir.mkdir(parents=True)
    stable = pdir / "STABLE.log"
    changed = pdir / "CHANGED.log"
    bystander = pdir / "BYSTANDER.log"
    for log in (stable, changed, bystander):
        log.write_bytes(b"x")
    snapshot = {}
    for log in (stable, changed, bystander):
        st = log.stat()
        snapshot[log.name] = (st.st_size, st.st_mtime_ns)
    # mutate CHANGED after snapshotting it
    time.sleep(0.01)
    changed.write_bytes(b"xy")

    gc._prune_consolidated_sources(
        pdir, snapshot, "GC.log", set(), {"STABLE.log", "CHANGED.log"}
    )
    assert not stable.exists()  # allowlisted + unchanged → deleted
    assert changed.exists()  # changed since snapshot → kept (sync-safe)
    assert bystander.exists()  # not allowlisted → never touched


def test_consolidate_foreign_spill_reread_byte_correct(tmp_path):
    root = tmp_path / "c"
    big = "z" * 5000
    b = LogCache(root, writer_id="B", prefix_width=1, min_file_size=100)
    b.set("big", big)
    prefix = b._prefix("big")
    assert len(_spills(root)) == 1  # B spilled the large value into the shared pool

    a = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    a.consolidate(prefix, prune_writer_ids={"B"})

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
            a.consolidate(prefix, prune_writer_ids={"B", "C", "D"})
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
    a.consolidate(prefix, prune_writer_ids={"B"})

    assert not legacy_dir.exists()  # legacy namespace pruned with its log
    assert len(list((root / prefix / "spill").glob("*.val"))) == 1  # adopted
    assert (
        LogCache(root, writer_id="R", prefix_width=1, index_ttl=0).get("big")
        == "z" * 5000
    )


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
    a.consolidate(prefix, prune_writer_ids={"B"})

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
    gc.consolidate(prune_writer_ids={"A", "B"})
    prefix = gc._prefix("k")
    assert not gc._log_path(prefix).exists()  # empty result → our log dropped
    assert len(LogCache(root, writer_id="R", prefix_width=1, index_ttl=0)) == 0


def test_explicit_prune_folds_many_writers(tmp_path):
    """The explicit allowlist is the only cross-writer fold: naming every retired
    writer collapses the prefix to one log."""
    root = tmp_path / "c"
    for w in ("w0", "w1", "w2"):
        LogCache(root, writer_id=w, prefix_width=1).set("k", w)
    prefix = LogCache(root, prefix_width=1)._prefix("k")
    assert len({p.name for p in (root / prefix).glob("*.log")}) == 3

    writer = LogCache(root, writer_id="ME", prefix_width=1, index_ttl=0)
    writer.set("k", "ME")
    writer.consolidate(prefix, prune_writer_ids={"w0", "w1", "w2"})
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

    monkeypatch.setattr(
        m, "_read_records", flaky
    )  # the parse entrypoint consolidation uses
    LogCache(root, writer_id="A", prefix_width=1).consolidate(prefix)
    monkeypatch.undo()

    assert peer_log.exists()  # unreadable source kept, not pruned
    reader = LogCache(root, writer_id="R", prefix_width=1, index_ttl=0)
    assert reader.get("f") == 2  # its records survive (read from the kept log)
    assert reader.get("d") == 1


def test_consolidate_drops_expired(tmp_path):
    """consolidate honours TTL: an expired record is dropped from the rewrite."""
    root = tmp_path / "c"
    gone, keep = _SAME_PREFIX_KEYS[:2]
    LogCache(root, writer_id="A", prefix_width=1).set(gone, 1, expire=0.01)
    LogCache(root, writer_id="B", prefix_width=1).set(keep, 2)
    prefix = LogCache(root, prefix_width=1)._prefix(gone)
    time.sleep(0.03)

    LogCache(root, writer_id="A", prefix_width=1).consolidate(prefix)

    # A's only record expired → its rewritten log is empty and dropped;
    # B's live record stays in B's own (untouched) log.
    assert not (root / prefix / "A.log").exists()
    assert LogCache(root, writer_id="R", prefix_width=1, index_ttl=0).get(keep) == 2


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
    """A legacy `<writer>.spill/` dir whose log is GONE is collected once the
    caller names its writer in the prune allowlist (nothing references its
    files, but only the caller can assert the writer is dead). Age counts from
    LOCAL arrival (`max(mtime, ctime)`): a syncer-delivered dir keeps its old
    mtimes, and reaping it before its log can arrive would leave that log's
    records dangling."""
    root = tmp_path / "c"
    a = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    a.set("d", "v" * 5000)  # a live record so the consolidated log is non-empty
    prefix = a._prefix("d")
    orphan = root / prefix / "ghost.spill"
    orphan.mkdir()
    stale = orphan / "leftover.val"
    stale.write_bytes(b"pre-pool leftovers")
    old = time.time() - 7200
    os.utime(stale, (old, old))  # syncer-preserved mtimes (file AND dir);
    os.utime(orphan, (old, old))  # only ctime betrays the local arrival
    young_orphan = root / prefix / "ghost2.spill"
    young_orphan.mkdir()
    (young_orphan / "fresh.val").write_bytes(b"maybe a pre-pool writer still runs")
    empty_orphan = root / prefix / "ghost3.spill"
    empty_orphan.mkdir()  # a syncer creates the dir before copying files in

    ghosts = {"ghost", "ghost2", "ghost3"}
    a.consolidate(prefix, prune_writer_ids=ghosts)  # default grace
    assert stale.exists()  # survives: age counts from local arrival (ctime)
    assert young_orphan.exists()  # grace window postpones the sweep
    assert empty_orphan.exists()  # the dir's own stat grants an empty one grace

    _sweep_now(a, prefix)  # grace collapsed but NOT allowlisted → still kept
    assert orphan.exists()
    _sweep_now(a, prefix, prune_writer_ids=ghosts)  # allowlisted → all collected
    assert not orphan.exists()
    assert not young_orphan.exists()
    assert not empty_orphan.exists()
    assert a.get("d") == "v" * 5000


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
    first_write = threading.Event()  # batch must not win the race outright
    wrote = 0

    def churn() -> None:  # a live writer racing the batch
        nonlocal wrote
        live = LogCache(root, writer_id="LIVE", prefix_width=1, min_file_size=100)
        while not stop.is_set():
            live.set(f"live-{wrote % 4}", f"live-{wrote}-" + "x" * 3000)
            wrote += 1
            first_write.set()

    t = threading.Thread(target=churn, daemon=True)
    t.start()
    try:
        assert first_write.wait(
            timeout=10
        )  # ensure the read-back below is never vacuous
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

    def flaky(prefix: str, prune=frozenset()) -> None:
        attempted.append(prefix)  # list.append is atomic under the GIL
        if prefix in bad:
            raise RuntimeError(f"boom-{prefix}")
        real(prefix, prune)

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
    monkeypatch.setattr(
        log_cache_module, "ThreadPoolExecutor", None
    )  # fan-out would TypeError
    cache = LogCache(tmp_path / "c", writer_id="W", prefix_width=1, min_file_size=100)
    for i in range(24):
        cache.set(f"k{i}", f"k{i}-" + "v" * 3000)
    cache.consolidate()
    reader = LogCache(tmp_path / "c", writer_id="R", prefix_width=1, index_ttl=0)
    for i in range(24):
        assert reader.get(f"k{i}") == f"k{i}-" + "v" * 3000


def test_live_writers_own_legacy_leftovers_are_swept(tmp_path):
    """Pre-pool leftovers under the LIVE writer's own `<writer>.spill/` dir must be
    collected too — prune-with-log never fires for a live log, and these were the
    single biggest residue after full migration (~35 GB in production)."""
    root = tmp_path / "c"
    a = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    a.set("d", "v" * 5000)  # live record; A.log stays alive
    prefix = a._prefix("d")
    own_legacy = root / prefix / "A.spill"
    own_legacy.mkdir()
    stale = own_legacy / "prepool.val"
    stale.write_bytes(b"unreferenced pre-pool leftover")
    old = time.time() - 7200
    os.utime(stale, (old, old))

    a.consolidate(prefix)

    assert not own_legacy.exists()  # file swept, emptied dir removed
    assert a.get("d") == "v" * 5000
    assert (root / prefix / "A.log").exists()  # the live log is untouched


# ── spill-pool hardening (post-0.7.0 review fixes) ────────────────────────────


def _to_legacy(root, prefix, writer):
    """Rebuild the pre-pool layout for `writer`: move its pool file into
    `<writer>.spill/` and repoint its log's records (as a pre-pool writer
    would have written them). Returns (legacy_dir, legacy_file)."""
    pool_file = next((root / prefix / "spill").glob("*.val"))
    legacy_dir = root / prefix / f"{writer}.spill"
    legacy_dir.mkdir()
    legacy_rel = f"{prefix}/{writer}.spill/{pool_file.name}"
    pool_file.rename(root / legacy_rel)
    (root / prefix / "spill").rmdir()
    log_path = root / prefix / f"{writer}.log"
    recs = [r._replace(spill=legacy_rel) for r in _iter_records(log_path)]
    log_path.write_bytes(b"".join(_frame(r) for r in recs))
    return legacy_dir, root / legacy_rel


def test_own_legacy_namespace_removed_after_self_migration(tmp_path):
    """The consolidating writer's OWN legacy dir dies once its records are
    repointed — otherwise every index rebuild re-flags the prefix for migration
    and every write past the cooldown re-runs a full consolidation, forever."""
    root = tmp_path / "c"
    a = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    a.set("d", "z" * 5000)
    prefix = a._prefix("d")
    legacy_dir, _ = _to_legacy(root, prefix, "A")

    fresh = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    fresh.consolidate(prefix)

    assert not legacy_dir.exists()  # own namespace removed with the migration
    assert len(list((root / prefix / "spill").glob("*.val"))) == 1  # adopted
    assert fresh.get("d") == "z" * 5000
    fresh._index.clear()
    fresh._checked.clear()
    fresh._ensure_index(prefix)
    assert prefix not in fresh._needs_migration  # no perpetual re-migration flag


def test_reset_heals_corrupt_pool_file(tmp_path, caplog):
    """A crash can leave a truncated file under a content-addressed name.
    Re-storing the value must rewrite it (size mismatch), not trust the name
    forever — and warn: the mismatch is evidence of a crash artifact or a
    non-atomic syncer, which an operator needs surfaced."""
    root = tmp_path / "c"
    c = LogCache(root, writer_id="A", min_file_size=100, index_ttl=0)
    big = "z" * 5000
    c.set("k", big)
    pool_file = next(root.glob("**/spill/*.val"))
    pool_file.write_bytes(b"")  # crash artifact: the right name, no data
    assert c.get("k", "MISS") == "MISS"
    with caplog.at_level("WARNING"):
        c.set("k", big)  # the recompute path re-stores the same value
    assert any("rewriting it" in r.message for r in caplog.records)
    assert pool_file.stat().st_size > 0
    assert c.get("k") == big  # healed, not poisoned


def test_dedup_hit_refreshes_pool_file_age(tmp_path):
    """Re-spilling existing content must refresh the file's mtime, so the
    sweep's grace window protects the in-flight record like a fresh spill."""
    root = tmp_path / "c"
    c = LogCache(root, writer_id="A", min_file_size=100)
    big = "z" * 5000
    c.set("k", big)
    pool_file = next(root.glob("**/spill/*.val"))
    old = time.time() - 7200
    os.utime(pool_file, (old, old))
    c.set("k", big)  # dedup hit: no new file, but the age must refresh
    assert pool_file.stat().st_mtime > old + 1


def test_transient_adopt_error_keeps_legacy_source(tmp_path, caplog):
    """A legacy spill that errors on adoption (EACCES — not sync-lag-absent)
    must keep its namespace and log: the bytes are intact on disk, and a later
    pass retries instead of pruning the only local copy."""
    root = tmp_path / "c"
    b = LogCache(root, writer_id="B", prefix_width=1, min_file_size=100)
    b.set("d", "z" * 5000)
    prefix = b._prefix("d")
    legacy_dir, legacy_file = _to_legacy(root, prefix, "B")
    legacy_file.chmod(0)  # transient local fault, NOT a vanished file

    a = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    with caplog.at_level("WARNING"):
        a.consolidate(prefix, prune_writer_ids={"B"})
    legacy_file.chmod(0o644)

    assert legacy_dir.exists()  # namespace kept for a retry, not destroyed
    assert (root / prefix / "B.log").exists()  # its log kept too
    assert any("could not adopt" in r.message for r in caplog.records)
    reader = LogCache(root, writer_id="R", prefix_width=1, index_ttl=0)
    assert reader.get("d") == "z" * 5000  # value survives in the kept log
    a.consolidate(prefix, prune_writer_ids={"B"})  # fault cleared → retry prunes
    assert not legacy_dir.exists()
    assert (
        LogCache(root, writer_id="R2", prefix_width=1, index_ttl=0).get("d")
        == "z" * 5000
    )


def test_consolidation_aborts_when_own_log_unreadable(tmp_path, monkeypatch, caplog):
    """`unread` guards sources; the DESTINATION rewrite must abort too — a
    transient read error on our own log must not clobber our records."""
    import emboss._log_cache as m

    root = tmp_path / "c"
    a = LogCache(root, writer_id="A", prefix_width=1)
    a.set("d", "mine")
    prefix = a._prefix("d")
    real_read = m._read_records

    def flaky(path):
        if path.name == "A.log":
            raise OSError("transient read failure")
        return real_read(path)

    monkeypatch.setattr(m, "_read_records", flaky)
    with caplog.at_level("WARNING"):
        LogCache(root, writer_id="A", prefix_width=1).consolidate(prefix)
    monkeypatch.undo()

    assert (root / prefix / "A.log").exists()  # not rewritten or unlinked
    assert any("aborting the consolidation pass" in r.message for r in caplog.records)
    assert LogCache(root, writer_id="R", prefix_width=1, index_ttl=0).get("d") == "mine"


def test_unreadable_source_log_protects_pool_from_sweep(tmp_path, monkeypatch):
    """A pool file referenced only by an unreadable log must survive the sweep —
    its references are unknown, so deleting it would turn recoverable state
    into misses. (The sweep must be skipped entirely on any unread log.)"""
    import emboss._log_cache as m

    root = tmp_path / "c"
    peer = LogCache(root, writer_id="PEER", prefix_width=1, min_file_size=100)
    peer.set("d", "z" * 5000)  # PEER's log holds the only reference
    prefix = peer._prefix("d")
    LogCache(root, writer_id="A", prefix_width=1).set("f", "inline")  # same prefix
    pool_file = next((root / prefix / "spill").glob("*.val"))
    real_read = m._read_records

    def flaky(path):
        if path.name == "PEER.log":
            raise OSError("transient read failure")
        return real_read(path)

    monkeypatch.setattr(m, "_read_records", flaky)
    gc = LogCache(root, writer_id="A", prefix_width=1)
    _sweep_now(gc, prefix)  # grace collapsed: only the unread guard protects it
    monkeypatch.undo()

    assert pool_file.exists()  # sweep skipped: reference set incomplete
    assert (
        LogCache(root, writer_id="R", prefix_width=1, index_ttl=0).get("d")
        == "z" * 5000
    )


def test_lone_writer_size_trigger_sweeps_pool(tmp_path, monkeypatch):
    """A single-writer prefix never crosses the writer bound, so the
    max_log_bytes trigger must consolidate (compact + sweep) — superseded pool
    files cannot accumulate forever without a manual consolidate()."""
    import emboss._log_cache as m

    monkeypatch.setattr(m, "_AUTO_CONSOLIDATE_COOLDOWN_S", 0.0)  # sweep every trigger
    root = tmp_path / "c"
    c = LogCache(root, writer_id="A", min_file_size=100, max_log_bytes=500)
    c._SHARED_SPILL_GRACE_S = 0.0  # let the sweep act immediately
    for i in range(20):
        c.set("k", f"{i}-" * 3000)  # each overwrite spills a distinct value
    assert c.get("k") == "19-" * 3000
    assert len(_spills(root)) <= 6  # 20 without the sweep; only the tail lingers


def test_lone_writer_size_trigger_honors_cooldown(tmp_path, monkeypatch):
    """Within the cooldown window the size trigger compacts instead of
    consolidating — the full pass (pool glob + fsync'd rewrite) is bounded to
    one per window even when the log cannot shrink below max_log_bytes."""
    root = tmp_path / "c"
    c = LogCache(root, writer_id="A", min_file_size=100, max_log_bytes=500)
    c._SHARED_SPILL_GRACE_S = 0.0
    passes = {"n": 0}
    real = LogCache._consolidate_prefix

    def counting(self, p):
        passes["n"] += 1
        real(self, p)

    monkeypatch.setattr(LogCache, "_consolidate_prefix", counting)
    for i in range(20):
        c.set("k", f"{i}-" * 3000)
    assert passes["n"] == 1  # one full pass; compaction covers the window
    assert c.get("k") == "19-" * 3000


def test_multiwriter_size_trigger_compacts_not_consolidates(tmp_path):
    """With peers in the prefix the size trigger must compact — a regression
    to always-consolidate would run peer-pruning passes on every large write
    (the per-write storm, via a side door)."""
    root = tmp_path / "c"
    peer = LogCache(root, writer_id="PEER", prefix_width=1, min_file_size=100)
    peer.set("d", "z" * 5000)
    prefix = peer._prefix("d")
    c = LogCache(
        root, writer_id="A", prefix_width=1, min_file_size=100, max_log_bytes=500
    )
    for i in range(10):
        c.set("d", f"{i}-" * 3000)  # same prefix; each write crosses the bound
    assert (root / prefix / "PEER.log").exists()  # never pruned on the write path


def test_orphaned_pool_tmp_reaped_by_sweep(tmp_path):
    """A crash between the tmp write and the rename must not leak the tmp
    forever: the sweep reaps pool `*.tmp` files past the grace window."""
    root = tmp_path / "c"
    c = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    c.set("d", "z" * 5000)
    prefix = c._prefix("d")
    orphan = root / prefix / "spill" / ("f" * 64 + ".val.deadbeef.tmp")
    orphan.write_bytes(b"crash leftover")

    c.consolidate(prefix)  # default grace: a young tmp may still be mid-spill
    assert orphan.exists()
    _sweep_now(c, prefix)  # grace collapsed: the crash leftover is reaped
    assert not orphan.exists()
    assert c.get("d") == "z" * 5000  # the real pool file untouched


def test_synced_in_spill_with_old_mtime_gets_grace(tmp_path):
    """A syncer preserves the peer's (old) mtime; the grace window must count
    from LOCAL arrival (ctime), or a spill delivered before its log is swept
    immediately and the log arrives dangling."""
    root = tmp_path / "c"
    c = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    c.set("d", "z" * 5000)  # keeps the prefix non-empty
    prefix = c._prefix("d")
    synced = root / prefix / "spill" / ("c" * 64 + ".val")
    synced.write_bytes(b"peer content, arrived before its log")
    old = time.time() - 7200
    os.utime(synced, (old, old))  # syncer-preserved mtime; ctime = local arrival

    c.consolidate(prefix)  # default grace

    assert synced.exists()  # survives: age counted from local arrival


def test_absent_spill_drop_is_logged(tmp_path, caplog):
    """Consolidation shedding records (absent pool files) must say so — a mass
    drop otherwise reads as a mysterious recompute storm later."""
    root = tmp_path / "c"
    b = LogCache(root, writer_id="B", prefix_width=1, min_file_size=100)
    b.set("d", "z" * 5000)
    prefix = b._prefix("d")
    LogCache(root, writer_id="B", prefix_width=1).set("f", "ok")
    shutil.rmtree(root / prefix / "spill")  # the value lags behind the log

    a = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    with caplog.at_level("INFO"):
        a.consolidate(prefix, prune_writer_ids={"B"})
    assert any("dropped 1 record" in r.message for r in caplog.records)


def test_migration_trigger_cooldown_suppresses_storm(tmp_path, monkeypatch):
    """When our legacy namespace resists removal and the migration flag stays
    armed, the cooldown must keep consolidation to one pass per window, not one
    per write."""
    root = tmp_path / "c"
    me = LogCache(root, writer_id="ME", prefix_width=1, min_file_size=100, index_ttl=0)
    me.set("d", "z" * 5000)
    prefix = me._prefix("d")
    _to_legacy(root, prefix, "ME")  # our own pre-pool layout arms the flag
    passes = {"n": 0}

    def counting(self, p, prune=frozenset()):
        passes["n"] += 1  # never calls the real pass, so the namespace persists
        self._consolidated_at[p] = time.monotonic()  # the pass stamps the cooldown

    monkeypatch.setattr(LogCache, "_consolidate_prefix", counting)
    me = LogCache(root, writer_id="ME", prefix_width=1, min_file_size=100, index_ttl=0)
    k1, k2, k3 = _SAME_PREFIX_KEYS[1:4]  # same prefix as "d"
    me.get("d")  # index build detects our legacy namespace
    me.set(k1, "v")  # trips the migration trigger → one pass
    me.get("d")  # namespace still there → flag re-arms
    me.set(k2, "v")  # within the cooldown → suppressed
    assert passes["n"] == 1  # no per-write storm
    me._consolidated_at[prefix] = time.monotonic() - 61.0  # cooldown elapsed
    me.set(k3, "v")
    assert passes["n"] == 2


def test_write_path_triggers_legacy_migration(tmp_path):
    """'Migrated on sight' for OUR OWN pre-pool layout: index build detects our
    legacy namespace and the NEXT WRITE kicks the migration — no explicit
    consolidate() call. A PEER's legacy namespace is the peer's to migrate and
    must neither arm the flag nor be touched."""
    root = tmp_path / "c"
    me = LogCache(root, writer_id="ME", prefix_width=1, min_file_size=100)
    me.set("d", "z" * 5000)
    prefix = me._prefix("d")
    own_legacy, _ = _to_legacy(root, prefix, "ME")

    fresh = LogCache(
        root, writer_id="ME", prefix_width=1, min_file_size=100, index_ttl=0
    )
    assert fresh.get("d") == "z" * 5000  # index build flags our legacy layout
    fresh.set("f", "small")  # the next write in the prefix kicks the migration

    assert not own_legacy.exists()
    assert len(list((root / prefix / "spill").glob("*.val"))) == 1  # adopted
    reader = LogCache(root, writer_id="R", prefix_width=1, index_ttl=0)
    assert reader.get("d") == "z" * 5000
    assert reader.get("f") == "small"


def test_peer_legacy_namespace_never_arms_migration(tmp_path):
    """A PEER's legacy namespace must not arm the migration flag: consolidation
    no longer touches peer files, so arming on it would re-run a full pass every
    cooldown window forever without ever clearing the condition."""
    root = tmp_path / "c"
    b = LogCache(root, writer_id="B", prefix_width=1, min_file_size=100)
    b.set("d", "z" * 5000)
    prefix = b._prefix("d")
    peer_legacy, peer_file = _to_legacy(root, prefix, "B")

    me = LogCache(root, writer_id="ME", prefix_width=1, min_file_size=100, index_ttl=0)
    assert me.get("d") == "z" * 5000  # index build sees the peer namespace
    assert prefix not in me._needs_migration  # …and does not arm on it
    me.set("f", "small")  # a write runs no migration pass

    assert peer_legacy.exists()  # untouched
    assert peer_file.exists()
    assert (
        LogCache(root, writer_id="R", prefix_width=1, index_ttl=0).get("d")
        == "z" * 5000
    )


# ── spill-pool hardening, second review round (0.7.1 fixes) ───────────────────


def test_failed_own_adoption_keeps_record_and_namespace(tmp_path, caplog):
    """A failed adoption of OUR OWN legacy spill must carry the un-repointed
    record into the rewritten log (the rewrite is unconditional — shedding it
    loses the key), keep the namespace (rmtree would destroy the only copy),
    and complete the migration on a later pass."""
    root = tmp_path / "c"
    a = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    a.set("d", "z" * 5000)
    prefix = a._prefix("d")
    legacy_dir, legacy_file = _to_legacy(root, prefix, "A")
    old = time.time() - 7200  # past the grace window: only the failed-adoption
    os.utime(legacy_file, (old, old))  # skip protects the bytes from the sweep
    legacy_file.chmod(0)  # transient local fault, NOT a vanished file

    fresh = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    with caplog.at_level("WARNING"):
        fresh.consolidate(prefix)
    legacy_file.chmod(0o644)

    assert legacy_file.exists()  # own namespace kept — the only copy survives
    recs = list(_iter_records(root / prefix / "A.log"))
    assert any(r.key == "d" for r in recs)  # the rewrite did not shed the record
    assert any("could not adopt" in r.message for r in caplog.records)
    fresh.consolidate(prefix)  # fault cleared → the retry migrates and removes
    assert not legacy_dir.exists()
    assert len(list((root / prefix / "spill").glob("*.val"))) == 1  # adopted
    assert (
        LogCache(root, writer_id="R", prefix_width=1, index_ttl=0).get("d")
        == "z" * 5000
    )


def test_failed_peer_adoption_keeps_legacy_bytes(tmp_path, caplog):
    """A PEER namespace whose adoption failed is skipped by the legacy sweep even
    past the grace window: its records were dropped from the merge (so the kept
    set cannot vouch for its files), and the retry its kept log guarantees needs
    the bytes still on disk — sweeping them would turn a transient fault into a
    permanent loss."""
    root = tmp_path / "c"
    b = LogCache(root, writer_id="B", prefix_width=1, min_file_size=100)
    b.set("d", "z" * 5000)
    prefix = b._prefix("d")
    legacy_dir, legacy_file = _to_legacy(root, prefix, "B")
    legacy_file.chmod(0)  # adoption (a content hash) fails; stat still works

    a = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    with caplog.at_level("WARNING"):
        # grace collapsed: only the failed-adoption skip protects the bytes
        _sweep_now(a, prefix, prune_writer_ids={"B"})
    legacy_file.chmod(0o644)

    assert legacy_file.exists()  # bytes survive for the retry
    assert (root / prefix / "B.log").exists()  # with the log that references them
    assert any("could not adopt" in r.message for r in caplog.records)
    a.consolidate(prefix, prune_writer_ids={"B"})  # fault cleared → retry prunes
    assert not legacy_dir.exists()
    assert (
        LogCache(root, writer_id="R", prefix_width=1, index_ttl=0).get("d")
        == "z" * 5000
    )


def test_transient_pool_stat_fault_keeps_record(tmp_path, caplog):
    """A transient stat fault on a pool file is NOT sync-lag absence: the
    record must be kept (protecting the file from the sweep too) — dropping it
    would rewrite our log without it, prune the peer log that held the only
    other reference, and let a later sweep destroy the healthy file."""
    root = tmp_path / "c"
    b = LogCache(root, writer_id="B", prefix_width=1, min_file_size=100)
    b.set("d", "z" * 5000)
    prefix = b._prefix("d")
    pool_dir = root / prefix / "spill"
    pool_dir.chmod(0)  # stat inside now fails EACCES — absence would be ENOENT

    a = LogCache(root, writer_id="A", prefix_width=1)
    with caplog.at_level("WARNING"):
        a.consolidate(prefix, prune_writer_ids={"B"})
    pool_dir.chmod(0o755)

    assert any("could not stat pool spill" in r.message for r in caplog.records)
    assert len(list(pool_dir.glob("*.val"))) == 1  # never swept
    assert (
        LogCache(root, writer_id="R", prefix_width=1, index_ttl=0).get("d")
        == "z" * 5000
    )


def test_orphan_peer_legacy_dir_reaped(tmp_path):
    """A peer `.spill/` dir with no matching log (a crash between a prune's
    unlink and rmtree, or a syncer resurrecting it) is reaped once its writer
    is named in the prune allowlist and the grace window has passed. Without
    the allowlist it is left alone — and it never arms the migration flag."""
    root = tmp_path / "c"
    c = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    c.set("d", "z" * 5000)
    prefix = c._prefix("d")
    orphan = root / prefix / "GONE.spill"
    orphan.mkdir()
    (orphan / "stranded.val").write_bytes(b"referenced by no log")

    # default grace: a fresh dir may be spill-before-log
    c.consolidate(prefix, prune_writer_ids={"GONE"})
    assert orphan.exists()
    _sweep_now(c, prefix)  # grace collapsed but not allowlisted → kept
    assert orphan.exists()
    _sweep_now(c, prefix, prune_writer_ids={"GONE"})  # allowlisted → reaped
    assert not orphan.exists()
    fresh = LogCache(root, writer_id="A", prefix_width=1, index_ttl=0)
    assert fresh.get("d") == "z" * 5000
    assert prefix not in fresh._needs_migration  # peer dirs never arm the flag


def test_consolidation_abort_sets_cooldown(tmp_path, monkeypatch, caplog):
    """A persistently unreadable own log must abort once per cooldown window,
    not once per write — the abort path stamps the cooldown. (Reached via the
    migration trigger, whose within-cooldown writes skip the pass entirely.)"""
    import emboss._log_cache as m

    root = tmp_path / "c"
    me = LogCache(root, writer_id="ME", prefix_width=1, min_file_size=100)
    me.set("d", "z" * 5000)
    prefix = me._prefix("d")
    _to_legacy(root, prefix, "ME")  # our legacy namespace arms the trigger
    real_read = m._read_records

    def flaky(path):
        if path.name == "ME.log":
            raise OSError("persistent read failure")
        return real_read(path)

    monkeypatch.setattr(m, "_read_records", flaky)
    me = LogCache(root, writer_id="ME", prefix_width=1, min_file_size=100, index_ttl=0)
    k1, k2 = _SAME_PREFIX_KEYS[1:3]  # same prefix as "d"
    with caplog.at_level("WARNING"):
        me.set(k1, "v")  # migration trigger → pass → abort + cooldown stamp
        me.set(k2, "v")  # within the cooldown: no second pass, no second abort
    aborts = [r for r in caplog.records if "aborting the consolidation" in r.message]
    assert len(aborts) == 1
    assert (root / prefix / "ME.log").exists()  # never rewritten blind


def test_cross_filesystem_adopt_falls_back_to_copy(tmp_path, monkeypatch):
    """When hardlinks are unavailable (EXDEV) adoption byte-copies: the value
    lands in the pool, the record repoints, and the namespace still prunes."""
    import errno

    root = tmp_path / "c"
    b = LogCache(root, writer_id="B", prefix_width=1, min_file_size=100)
    b.set("d", "z" * 5000)
    prefix = b._prefix("d")
    _to_legacy(root, prefix, "B")

    def no_link(src, dst, **kwargs):
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(os, "link", no_link)
    a = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    a.consolidate(prefix, prune_writer_ids={"B"})

    assert len(list((root / prefix / "spill").glob("*.val"))) == 1  # copied in
    assert not (root / prefix / "B.spill").exists()  # namespace pruned
    assert (
        LogCache(root, writer_id="R", prefix_width=1, index_ttl=0).get("d")
        == "z" * 5000
    )


def test_failed_copy_fallback_keeps_namespace(tmp_path, monkeypatch, caplog):
    """A copy fallback failing with a transient fault (disk full) must clean
    up its tmp, keep the namespace and log for a retry, and never classify the
    fault as sync lag (which would prune the only copy)."""
    import errno

    import emboss._log_cache as m

    root = tmp_path / "c"
    b = LogCache(root, writer_id="B", prefix_width=1, min_file_size=100)
    b.set("d", "z" * 5000)
    prefix = b._prefix("d")
    legacy_dir, _ = _to_legacy(root, prefix, "B")

    def no_link(src, dst, **kwargs):
        raise OSError(errno.EXDEV, "cross-device link")

    def no_copy(src, dst):
        raise OSError(errno.ENOSPC, "no space left on device")

    monkeypatch.setattr(os, "link", no_link)
    monkeypatch.setattr(m.shutil, "copyfile", no_copy)
    a = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    with caplog.at_level("WARNING"):
        a.consolidate(prefix, prune_writer_ids={"B"})
    monkeypatch.undo()

    assert legacy_dir.exists()  # namespace kept for a retry
    assert (root / prefix / "B.log").exists()  # its log kept too
    assert not list((root / prefix / "spill").glob("*.tmp"))  # tmp cleaned up
    assert any("could not adopt" in r.message for r in caplog.records)
    a.consolidate(prefix, prune_writer_ids={"B"})  # fault cleared → retry prunes
    assert not legacy_dir.exists()
    assert (
        LogCache(root, writer_id="R", prefix_width=1, index_ttl=0).get("d")
        == "z" * 5000
    )


def test_record_wire_format_pinned():
    """`_frame` pickles a plain positional 6-tuple and `_parse_frame` rebuilds
    by position: changing the field count or order makes every EXISTING record
    parse as a torn frame — a silent full-cache invalidation. Pin the format."""
    assert _Record._fields == (
        "key",
        "value",
        "expire_time",
        "store_time",
        "deleted",
        "spill",
    )
    rec = _Record("k", "v", None, 1.0, False, None)
    parsed = _parse_frame(_frame(rec), 0)
    assert parsed is not None
    assert parsed[0] == rec


# ── single-writer safety (the 2026-07-10 fleet-cache clobber) ─────────────────
#
# A file syncer replicates every node's logs everywhere, so "another writer's
# log" is usually a REPLICA whose owner holds an un-synced tail. A replica can
# be byte-stable for a whole pass and still be incomplete — no local check can
# tell — so the only safe default is to never touch a peer's files at all.
# Pruning is an explicit, caller-asserted decision (`prune_writer_ids`): the
# caller names writers it KNOWS are dead/retired (a decommissioned host, a
# one-shot bulk importer), and only those logs are folded in and removed.


def test_consolidate_default_leaves_peer_logs_untouched(tmp_path):
    """Default consolidation must not rewrite, repair, or delete any peer log —
    byte-for-byte — while still compacting our own log and keeping reads whole."""
    root = tmp_path / "c"
    for w, k in zip("ABC", _SAME_PREFIX_KEYS):
        LogCache(root, writer_id=w, prefix_width=1).set(k, k * 3)
    prefix = LogCache(root, prefix_width=1)._prefix(_SAME_PREFIX_KEYS[0])
    peer_bytes = {p.name: p.read_bytes() for p in (root / prefix).glob("*.log")}

    gc = LogCache(root, writer_id="GC", prefix_width=1)
    gc.consolidate()

    for p in (root / prefix).glob("*.log"):
        assert peer_bytes[p.name] == p.read_bytes()  # untouched, byte-for-byte
    assert set(peer_bytes) == {p.name for p in (root / prefix).glob("*.log")}
    reader = LogCache(root, writer_id="R", prefix_width=1, index_ttl=0)
    for w, k in zip("ABC", _SAME_PREFIX_KEYS):
        assert reader.get(k) == k * 3


def test_consolidate_default_survives_torn_peer_replica(tmp_path):
    """The incident reproduction: a peer log that is a torn mid-sync replica must
    not be 'repaired' into our log and deleted — the owner's complete original
    (invisible to us) would be destroyed when the deletion syncs back."""
    root = tmp_path / "c"
    peer = LogCache(root, writer_id="PEER", prefix_width=1)
    for k in _SAME_PREFIX_KEYS:
        peer.set(k, k * 2)
    prefix = peer._prefix(_SAME_PREFIX_KEYS[0])
    peer_log = root / prefix / "PEER.log"
    # Tear the replica mid-frame, as a syncer shipping a mid-append snapshot does.
    whole = peer_log.read_bytes()
    peer_log.write_bytes(whole[: len(whole) - len(whole) // 3])
    torn = peer_log.read_bytes()

    gc = LogCache(root, writer_id="GC", prefix_width=1)
    gc.consolidate()

    assert peer_log.exists()  # never deleted
    assert peer_log.read_bytes() == torn  # never rewritten either
    # our log carries none of the replica's records (nothing was folded in)
    target = gc._log_path(prefix)
    assert not target.exists() or all(
        r.key not in _SAME_PREFIX_KEYS for r in _iter_records(target)
    )


def test_consolidate_prunes_only_allowlisted_writers(tmp_path):
    """`prune_writer_ids` folds exactly the named (caller-asserted-dead) writers'
    logs into ours and deletes them; every other peer stays byte-identical."""
    root = tmp_path / "c"
    k_dead, k_live, k_mine = _SAME_PREFIX_KEYS[:3]
    LogCache(root, writer_id="DEAD", prefix_width=1).set(k_dead, "dead-v")
    LogCache(root, writer_id="LIVE", prefix_width=1).set(k_live, "live-v")
    gc = LogCache(root, writer_id="GC", prefix_width=1)
    gc.set(k_mine, "mine-v")
    prefix = gc._prefix(k_dead)
    live_bytes = (root / prefix / "LIVE.log").read_bytes()

    gc.consolidate(prefix, prune_writer_ids={"DEAD"})

    assert not (root / prefix / "DEAD.log").exists()  # folded in and removed
    assert (root / prefix / "LIVE.log").read_bytes() == live_bytes
    target_keys = {r.key for r in _iter_records(gc._log_path(prefix))}
    assert target_keys == {k_dead, k_mine}  # DEAD's record adopted, LIVE's not
    reader = LogCache(root, writer_id="R", prefix_width=1, index_ttl=0)
    assert reader.get(k_dead) == "dead-v"
    assert reader.get(k_live) == "live-v"
    assert reader.get(k_mine) == "mine-v"


def test_clear_removes_only_own_writer_files(tmp_path):
    """clear() drops THIS writer's contributions; a peer's logs and their spilled
    values must survive byte-for-byte (under sync they are replicas of files the
    peer owns — deleting them here deletes them on the peer)."""
    root = tmp_path / "c"
    a = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    b = LogCache(root, writer_id="B", prefix_width=1, min_file_size=100)
    k_a, k_b, k_a2 = _SAME_PREFIX_KEYS[:3]
    a.set(k_a, "a-small")
    a.set(k_a2, "a" * 5000)  # spills to the shared pool
    b.set(k_b, "b" * 5000)  # spills to the shared pool
    prefix = a._prefix(k_a)
    b_bytes = (root / prefix / "B.log").read_bytes()
    b_spill = {r.spill for r in _iter_records(root / prefix / "B.log") if r.spill}

    dropped = a.clear()

    assert dropped == 2  # A's two live records
    assert not (root / prefix / "A.log").exists()
    assert (root / prefix / "B.log").read_bytes() == b_bytes
    for rel in b_spill:
        assert (root / rel).exists()  # B's spilled value untouched
    reader = LogCache(root, writer_id="R", prefix_width=1, index_ttl=0)
    assert reader.get(k_b) == "b" * 5000
    assert reader.get(k_a) is None
    assert reader.get(k_a2) is None


def test_no_auto_consolidation_on_writer_count(tmp_path):
    """Writer-count growth must never trigger an implicit cross-writer fold —
    the old `max_writers_per_prefix` trigger pruned replicas mid-sync. Many
    writers accumulate; every log stays."""
    root = tmp_path / "c"
    writers = [f"w{i}" for i in range(9)]  # exceeds the old default bound of 8
    for w in writers:
        LogCache(root, writer_id=w, prefix_width=1).set("k", w)
    prefix = LogCache(root, prefix_width=1)._prefix("k")
    me = LogCache(root, writer_id="ME", prefix_width=1, index_ttl=0)
    me.get("k")  # build the index so the write path sees every peer log
    me.set("k", "ME")
    logs = {p.name for p in (root / prefix).glob("*.log")}
    assert logs == {f"{w}.log" for w in writers} | {"ME.log"}
