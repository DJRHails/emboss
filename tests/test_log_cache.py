"""Tests for the `LogCache` backend — per-writer, prefix-sharded append logs."""

from __future__ import annotations

import shutil
import time

import pytest

from emboss import Cache, LogCache, cached
from emboss._log_cache import _iter_records


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
        c.set("k", "x" * 50)  # repeatedly overwrite -> log would grow without compaction
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
    return list(root.glob("**/*.spill/*.val"))


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


def test_spill_removed_on_overwrite(tmp_path):
    root = tmp_path / "c"
    c = LogCache(root, writer_id="A", min_file_size=100)
    c.set("k", "a" * 5000)
    assert len(_spills(root)) == 1
    c.set("k", "b" * 5000)  # supersedes -> old spill removed, new written
    assert len(_spills(root)) == 1
    assert c.get("k") == "b" * 5000


def test_spill_removed_on_delete(tmp_path):
    root = tmp_path / "c"
    c = LogCache(root, writer_id="A", min_file_size=100)
    c.set("k", "a" * 5000)
    c.delete("k")
    assert _spills(root) == []


def test_compaction_removes_expired_spills(tmp_path):
    root = tmp_path / "c"
    c = LogCache(root, writer_id="A", min_file_size=100)
    c.set("gone", "x" * 5000, expire=0.01)
    c.set("keep", "y" * 5000)
    time.sleep(0.03)
    assert len(_spills(root)) == 2  # expiry is lazy — both spill files still present
    c.compact()
    assert len(_spills(root)) == 1  # compaction dropped the expired one's spill
    assert c.get("keep") == "y" * 5000


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

    def append_then_prune(self, pdir, snapshot, target_name):
        peer.set(k_p2, 2)  # a concurrent peer append lands mid-consolidation
        return real_prune(self, pdir, snapshot, target_name)

    monkey = LogCache(root, writer_id="A", prefix_width=1)
    object.__setattr__(monkey, "_prune_consolidated_sources", append_then_prune.__get__(monkey))
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

    gc._prune_consolidated_sources(pdir, snapshot, "GC.log")
    assert not stable.exists()  # unchanged → fully represented → deleted
    assert changed.exists()  # changed since snapshot → kept (sync-safe)


def test_consolidate_foreign_spill_reread_byte_correct(tmp_path):
    root = tmp_path / "c"
    big = "z" * 5000
    b = LogCache(root, writer_id="B", prefix_width=1, min_file_size=100)
    b.set("big", big)
    prefix = b._prefix("big")
    assert (root / prefix / "B.spill").is_dir()  # B spilled the large value

    a = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    a.consolidate(prefix)

    assert LogCache(root, writer_id="R", prefix_width=1, index_ttl=0).get("big") == big
    assert not (root / prefix / "B.spill").exists()  # B's source spill pruned
    assert (root / prefix / "A.spill").is_dir()  # re-spilled under our namespace
    assert {p.name for p in (root / prefix).glob("*.log")} == {"A.log"}


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
    shutil.rmtree(root / prefix / "B.spill")

    a = LogCache(root, writer_id="A", prefix_width=1, min_file_size=100)
    a.consolidate(prefix)

    target = a._log_path(prefix)
    keys = [r.key for r in _iter_records(target)]
    assert big_k not in keys  # record genuinely DROPPED, not kept with a dead ref
    assert all(r.spill is None for r in _iter_records(target))  # no dangling spill ref
    reader = LogCache(root, writer_id="R", prefix_width=1, index_ttl=0)
    assert reader.get(big_k) is None  # spilled value gone (recomputes on next read)
    assert reader.get(keep_k) == "ok"  # the other live entry survived
    assert not (root / prefix / "A.spill").exists()  # we wrote no spill at all


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
        LogCache(root, writer_id=w, prefix_width=1, max_writers_per_prefix=0).set("k", w)
    prefix = LogCache(root, prefix_width=1)._prefix("k")
    assert len({p.name for p in (root / prefix).glob("*.log")}) == 3

    # our writer with a small bound: its set sees 3 peers + itself > 2 → consolidate
    writer = LogCache(root, writer_id="ME", prefix_width=1, max_writers_per_prefix=2, index_ttl=0)
    writer.get("k")  # build the index so _sig[prefix] is populated with peer logs
    writer.set("k", "ME")  # this write trips the auto-consolidation
    assert {p.name for p in (root / prefix).glob("*.log")} == {"ME.log"}  # back to one
    assert LogCache(root, writer_id="R", prefix_width=1, index_ttl=0).get("k") == "ME"


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
    assert dst.get("blob") == "q" * 5000  # spilled value survives transfer post-consolidate
    dst.close()
