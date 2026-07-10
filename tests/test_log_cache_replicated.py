"""Replication-safe consolidation — the `.replicated` marker (issue #22).

On 2026-07-10 concurrent consolidations on three nodes of a Syncthing-replicated
tree rewrote each other's writer logs; the syncer resolved the divergent
same-named files last-writer-wins and silently discarded records (>=4015 Opus
calls re-billed). These tests reproduce that incident against LEGACY (exclusive)
consolidation, then pin the replicated-mode invariants that prevent it.

No real syncer runs: `_ReplicaPair` simulates a file-level replicator between
two tree replicas — full-state convergence with either pure last-writer-wins
(the no-versioning fabric that lost the data) or conflict-copy semantics (the
losing side of a concurrent change survives as a `*.sync-conflict-*` file).
"""

from __future__ import annotations

import os
import pickle
import time
from hashlib import sha256
from pathlib import Path, PurePosixPath

import pytest

from emboss import LogCache
from emboss._log_cache import _HEADER, _frame, _iter_records, _Record

# All single-char keys below hash (md5, width 1) into the shard "8", so every
# writer's records land in ONE prefix dir — the collision consolidation must
# handle. Same set the exclusive-mode tests use.
_KEYS = ["d", "f", "i", "k", "p"]
_WIDTH1 = {"prefix_width": 1, "index_ttl": 0}


def _rec(key: str, value: str, store_time: float | None = None) -> _Record:
    return _Record(key, value, None, store_time or time.time(), False, None)


def _write_log(path: Path, *records: _Record) -> list[int]:
    """Write raw framed records to a fresh log; return each record's byte offset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets, buf = [], bytearray()
    for rec in records:
        offsets.append(len(buf))
        buf += _frame(rec)
    path.write_bytes(bytes(buf))
    return offsets


def _corrupt_middle_frame(path: Path, offsets: list[int]) -> None:
    """Corrupt the second frame's blob in place (length header intact) — a
    genuine mid-log tear with valid frames after it."""
    raw = bytearray(path.read_bytes())
    blob_start = offsets[1] + _HEADER.size
    raw[blob_start : blob_start + 3] = b"\xff\xff\xff"
    path.write_bytes(bytes(raw))


def _reader(root: Path) -> LogCache:
    return LogCache(root, writer_id="reader-probe", prefix_width=1, index_ttl=0)


def _file_state(path: Path) -> tuple[bytes, int]:
    return path.read_bytes(), path.stat().st_mtime_ns


def _tree_state(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(p.relative_to(root)): _file_state(p) for p in root.rglob("*") if p.is_file()
    }


def _age(path: Path, seconds: float) -> None:
    old = time.time() - seconds
    os.utime(path, (old, old))


# ── the simulated sync fabric ─────────────────────────────────────────────────


class _ReplicaPair:
    """Two directory-tree replicas plus the state each path had at the last
    convergence, so a sync round can three-way-classify every path (unchanged /
    one-sided change / true concurrent conflict) the way a file syncer does.
    Change detection is by content; mtimes are preserved when propagating (as
    real syncers do), so mtime-based staleness carries across nodes."""

    def __init__(self, a: Path, b: Path) -> None:
        self.a = a
        self.b = b
        for root in (a, b):
            root.mkdir(parents=True, exist_ok=True)
        self._base: dict[str, bytes | None] = {}
        self._conflict_seq = 0

    def converge(self, *, conflict_copies: bool) -> None:
        """One full sync round. `conflict_copies=False` is pure last-writer-wins
        with deletes winning a delete-vs-modify race — the no-versioning fabric
        of the 2026-07-10 incident. `conflict_copies=True` preserves the losing
        side of any concurrent conflict as a `*.sync-conflict-*` file."""
        rels = set(self._base) | self._files(self.a) | self._files(self.b)
        for rel in sorted(rels):
            self._sync_path(rel, conflict_copies)

    @staticmethod
    def _files(root: Path) -> set[str]:
        return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}

    @staticmethod
    def _read(root: Path, rel: str) -> bytes | None:
        path = root / rel
        return path.read_bytes() if path.is_file() else None

    def _sync_path(self, rel: str, conflict_copies: bool) -> None:
        base = self._base.get(rel)
        a_val = self._read(self.a, rel)
        b_val = self._read(self.b, rel)
        if a_val == b_val:
            self._base[rel] = a_val
            return
        a_changed = a_val != base
        b_changed = b_val != base
        if a_changed and not b_changed:
            self._propagate(rel, a_val, src=self.a)
        elif b_changed and not a_changed:
            self._propagate(rel, b_val, src=self.b)
        else:
            self._resolve_conflict(rel, a_val, b_val, conflict_copies)

    def _propagate(self, rel: str, val: bytes | None, src: Path) -> None:
        for root in (self.a, self.b):
            dst = root / rel
            if val is None:
                dst.unlink(missing_ok=True)
            elif root != src:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(val)
                st = (src / rel).stat()
                os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns))
        self._base[rel] = val

    def _resolve_conflict(
        self, rel: str, a_val: bytes | None, b_val: bytes | None, conflict_copies: bool
    ) -> None:
        """Both sides changed since the last convergence. Delete vs modify: the
        delete wins the canonical path (the incident's worst case); with
        `conflict_copies` the modified content survives as a sync-conflict file.
        Modify vs modify: newer mtime wins the canonical path; the loser is
        preserved as a conflict copy or (pure LWW) discarded."""
        if a_val is None or b_val is None:
            loser = a_val if b_val is None else b_val
            if conflict_copies and loser is not None:
                self._write_conflict_copy(rel, loser)
            self._propagate(rel, None, src=self.a)
            return
        a_newer = (self.a / rel).stat().st_mtime_ns >= (self.b / rel).stat().st_mtime_ns
        winner = self.a if a_newer else self.b
        loser_val = b_val if a_newer else a_val
        if conflict_copies:
            self._write_conflict_copy(rel, loser_val)
        self._propagate(rel, self._read(winner, rel), src=winner)

    def _write_conflict_copy(self, rel: str, content: bytes) -> None:
        p = PurePosixPath(rel)
        self._conflict_seq += 1
        tag = f"sync-conflict-20260710-{self._conflict_seq:06d}-TESTDEV"
        crel = str(p.with_name(f"{p.stem}.{tag}{p.suffix}"))
        for root in (self.a, self.b):
            dst = root / crel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(content)
        self._base[crel] = content


def _seeded_pair(tmp_path: Path, *, replicated: bool) -> _ReplicaPair:
    """Two replicas of one tree: writer `alpha` on replica A holds "d"/"f",
    writer `beta` on replica B holds "i"/"k" — converged, so both replicas carry
    both writers' logs (all in the one width-1 prefix shard)."""
    pair = _ReplicaPair(tmp_path / "node-a", tmp_path / "node-b")
    if replicated:
        (pair.a / ".replicated").touch()
        (pair.b / ".replicated").touch()
    alpha = LogCache(pair.a, writer_id="alpha", **_WIDTH1)
    beta = LogCache(pair.b, writer_id="beta", **_WIDTH1)
    alpha.set("d", "alpha-d")
    alpha.set("f", "alpha-f")
    beta.set("i", "beta-i")
    beta.set("k", "beta-k")
    pair.converge(conflict_copies=True)  # the seed sync raises no conflicts either way
    return pair


def _readable_everywhere(pair: _ReplicaPair, expected: dict[str, str]) -> list[str]:
    """Keys from `expected` NOT readable on BOTH replicas (empty == zero loss)."""
    missing = []
    for key, value in expected.items():
        if _reader(pair.a).get(key) != value or _reader(pair.b).get(key) != value:
            missing.append(key)
    return missing


_SEEDED = {"d": "alpha-d", "f": "alpha-f", "i": "beta-i", "k": "beta-k"}


# ── the incident (legacy mode) ────────────────────────────────────────────────


def test_incident_legacy_concurrent_consolidation_loses_records(tmp_path):
    """Reproduces the 2026-07-10 data loss (issue #22) — LEGACY exclusive mode.

    Both nodes run today's cross-writer consolidation concurrently, each on its
    own replica: each folds the OTHER writer's log into its own and deletes the
    other's file. The two rewritten/deleted same-named files then converge
    last-writer-wins with no versioning — and records vanish from BOTH replicas.
    This test documents the bug the `.replicated` marker exists to prevent."""
    pair = _seeded_pair(tmp_path, replicated=False)
    assert _readable_everywhere(pair, _SEEDED) == []  # fully replicated before the race

    LogCache(pair.a, writer_id="alpha", **_WIDTH1).consolidate()
    LogCache(pair.b, writer_id="beta", **_WIDTH1).consolidate()
    pair.converge(conflict_copies=False)  # last-writer-wins, no versioning

    lost = [
        k
        for k in _SEEDED
        if _reader(pair.a).get(k) is None and _reader(pair.b).get(k) is None
    ]
    assert lost, "legacy consolidation under LWW sync must lose records (issue #22)"


# ── the fix: replicated mode ──────────────────────────────────────────────────


def test_replicated_concurrent_passes_on_fresh_tree_are_noops(tmp_path):
    """With the marker, a pass over a tree of FRESH logs mutates nothing at all:
    the own-log rewrite is skipped when byte-identical (no mtime ripple through
    the syncer) and foreign logs are invisible. Zero loss, zero conflicts."""
    pair = _seeded_pair(tmp_path, replicated=True)
    before_a, before_b = _tree_state(pair.a), _tree_state(pair.b)

    LogCache(pair.a, writer_id="alpha", **_WIDTH1).consolidate()
    LogCache(pair.b, writer_id="beta", **_WIDTH1).consolidate()

    assert _tree_state(pair.a) == before_a  # byte-for-byte no-op, mtimes included
    assert _tree_state(pair.b) == before_b
    pair.converge(conflict_copies=True)
    assert _readable_everywhere(pair, _SEEDED) == []
    assert not [p for p in pair.a.rglob("*sync-conflict*")]


def test_replicated_concurrent_cross_folds_converge_without_loss(tmp_path):
    """Both nodes fold each other's (stale) log CONCURRENTLY. The delete-vs-
    rewrite races surface as sync-conflict copies; the next pass folds those by
    name. Every record stays readable on both replicas at every stage, with no
    coordination between the nodes — and each pass is idempotent."""
    pair = _seeded_pair(tmp_path, replicated=True)
    stale = {"replicated_stale_ttl": 0.05, **_WIDTH1}
    time.sleep(0.1)  # both writers idle past the (tiny) staleness TTL

    a_gc = LogCache(pair.a, writer_id="alpha", **stale)
    b_gc = LogCache(pair.b, writer_id="beta", **stale)
    a_gc.consolidate()
    b_gc.consolidate()
    # each node folded the other's log into its own and deleted its input
    assert {p.name for p in pair.a.rglob("*.log")} == {"alpha.log"}
    assert {p.name for p in pair.b.rglob("*.log")} == {"beta.log"}

    # idempotency: an immediate re-run on the same node changes NOTHING on disk
    before = _tree_state(pair.a)
    a_gc.consolidate()
    assert _tree_state(pair.a) == before

    pair.converge(conflict_copies=True)
    # the concurrent delete-vs-rewrite on each log surfaced as conflict copies;
    # every record is still readable everywhere (a conflict copy IS a log)
    assert _readable_everywhere(pair, _SEEDED) == []
    conflicts = {p.name for p in pair.a.rglob("*sync-conflict*.log")}
    assert len(conflicts) == 2

    # the next concurrent passes fold the conflict copies (any age) and converge
    LogCache(pair.a, writer_id="alpha", **stale).consolidate()
    LogCache(pair.b, writer_id="beta", **stale).consolidate()
    pair.converge(conflict_copies=True)
    assert _readable_everywhere(pair, _SEEDED) == []
    assert not [p for p in pair.a.rglob("*sync-conflict*")]  # self-healed
    assert not [p for p in pair.b.rglob("*sync-conflict*")]


def test_ttl_misfire_folds_conflict_copy_and_leaves_fresh_canonical(tmp_path):
    """A node folds a foreign log its replica shows as stale while the owner is
    in fact still appending (sync lag / clock skew). The delete-vs-modify race
    becomes a sync-conflict copy; the next pass folds it — zero loss — and the
    owner's NEW canonical log stays untouched."""
    pair = _seeded_pair(tmp_path, replicated=True)
    prefix_dir_a = next(p.parent for p in pair.a.rglob("beta.log"))
    _age(prefix_dir_a / "beta.log", 13 * 3600)  # A's replica shows beta.log stale
    beta = LogCache(pair.b, writer_id="beta", **_WIDTH1)
    beta.set("p", "beta-p")  # ...but the owner keeps appending on ITS replica

    LogCache(pair.a, writer_id="alpha", **_WIDTH1).consolidate()  # the misfire
    assert not (prefix_dir_a / "beta.log").exists()  # A folded + deleted its copy
    pair.converge(conflict_copies=True)  # delete-vs-modify -> conflict copy

    expected = {**_SEEDED, "p": "beta-p"}
    assert _readable_everywhere(pair, expected) == []  # nothing lost anywhere

    # the owner writes on: its next append recreates a FRESH canonical beta.log
    LogCache(pair.b, writer_id="beta", **_WIDTH1).set("x", "beta-x")
    pair.converge(conflict_copies=True)
    canonical_a = next(pair.a.rglob("**/beta.log"))
    before = _file_state(canonical_a)

    LogCache(pair.a, writer_id="alpha", **_WIDTH1).consolidate()
    assert not [p for p in pair.a.rglob("*sync-conflict*")]  # conflict copy folded
    assert _file_state(canonical_a) == before  # fresh canonical log untouched
    pair.converge(conflict_copies=True)
    assert _readable_everywhere(pair, {**expected, "x": "beta-x"}) == []


# ── fold-eligibility rules (single tree, no fabric) ───────────────────────────


def test_sync_conflict_log_folded_fresh_foreign_log_invisible(tmp_path):
    """A `*.sync-conflict-*` log is folded regardless of age (its unique name is
    sealed — nobody appends to it); a fresh foreign log is not read-blocked but
    is never mutated or deleted, and its records are not duplicated into ours."""
    root = tmp_path / "cache"
    cache = LogCache(root, writer_id="alpha", **_WIDTH1)
    (root / ".replicated").touch()
    cache.set("d", "mine")
    prefix = cache._prefix("d")
    conflict = root / prefix / "beta.sync-conflict-20260710-094512-ABCDEF7.log"
    _write_log(conflict, _rec("f", "from-conflict"))  # minted moments ago: FRESH
    foreign = root / prefix / "gamma.log"
    _write_log(foreign, _rec("i", "from-gamma"))
    foreign_before = _file_state(foreign)

    cache.consolidate(prefix)

    assert not conflict.exists()  # folded by name, despite being fresh
    assert _file_state(foreign) == foreign_before  # fresh foreign: untouched
    own = {r.key for r in _iter_records(root / prefix / "alpha.log")}
    assert {"d", "f"} <= own
    assert "i" not in own  # fresh foreign records are not copied around
    reader = _reader(root)
    assert reader.get("f") == "from-conflict"
    assert reader.get("i") == "from-gamma"  # invisible to consolidation, not to reads


def test_stale_foreign_log_folded_lock_left_alone(tmp_path):
    """A foreign log idle past `replicated_stale_ttl` is folded and deleted —
    but its lock file survives (deleting it would replicate to a live owner and
    split its same-node flock across two inodes). Idempotent on re-run."""
    root = tmp_path / "cache"
    cache = LogCache(root, writer_id="alpha", **_WIDTH1)
    (root / ".replicated").touch()
    cache.set("d", "mine")
    prefix = cache._prefix("d")
    stale_log = root / prefix / "beta.log"
    _write_log(stale_log, _rec("f", "from-beta"))
    _age(stale_log, 13 * 3600)  # idle past the default 12 h TTL
    (root / prefix / "beta.lock").touch()

    cache.consolidate(prefix)

    assert not stale_log.exists()  # folded input deleted whole-file
    assert (root / prefix / "beta.lock").exists()  # never a foreign lock
    assert _reader(root).get("f") == "from-beta"
    before = _tree_state(root)
    cache.consolidate(prefix)
    assert _tree_state(root) == before  # second pass: byte-for-byte no-op


def test_replicated_folds_own_superseded_records(tmp_path):
    """The local writer's own log is always fold-eligible: superseded records
    are dropped exactly as in exclusive mode, foreign files untouched."""
    root = tmp_path / "cache"
    cache = LogCache(root, writer_id="alpha", **_WIDTH1)
    (root / ".replicated").touch()
    for i in range(50):
        cache.set("d", f"v{i}")
    prefix = cache._prefix("d")
    foreign = root / prefix / "gamma.log"
    _write_log(foreign, _rec("i", "from-gamma"))
    foreign_before = _file_state(foreign)

    cache.consolidate(prefix)

    own = list(_iter_records(root / prefix / "alpha.log"))
    assert [(r.key, r.value) for r in own] == [("d", "v49")]  # 49 dead records dropped
    assert _file_state(foreign) == foreign_before


def test_replicated_tombstone_retained_while_fresh_foreign_holds_older_record(tmp_path):
    """Dropping a tombstone while a non-folded fresh foreign log still holds an
    OLDER record for the key would resurrect that record on read. The tombstone
    is retained until the foreign log ages out and folds — then both collapse."""
    root = tmp_path / "cache"
    cache = LogCache(root, writer_id="alpha", **_WIDTH1)
    (root / ".replicated").touch()
    prefix = cache._prefix("d")
    foreign = root / prefix / "beta.log"
    _write_log(foreign, _rec("d", "old-foreign", store_time=time.time() - 100))
    assert cache.get("d") == "old-foreign"
    assert cache.delete("d") is True  # newer tombstone, in OUR log

    cache.consolidate(prefix)

    own = list(_iter_records(root / prefix / "alpha.log"))
    assert [r.key for r in own if r.deleted] == ["d"]  # tombstone survived the fold
    assert _reader(root).get("d") is None  # ...so the old record stays suppressed

    _age(foreign, 13 * 3600)  # the foreign log ages out
    cache.consolidate(prefix)
    assert not foreign.exists()  # folded + deleted
    assert not (root / prefix / "alpha.log").exists()  # nothing live remains at all
    assert _reader(root).get("d") is None


def test_torn_fresh_foreign_log_recovered_but_never_rewritten(tmp_path, caplog):
    """A TORN fresh foreign log has its readable records recovered into OUR log
    for durability, but the file itself is left byte-for-byte untouched — in
    replicated mode tear repair never rewrites a foreign file in place."""
    root = tmp_path / "cache"
    cache = LogCache(root, writer_id="alpha", **_WIDTH1)
    (root / ".replicated").touch()
    cache.set("d", "mine")
    prefix = cache._prefix("d")
    torn = root / prefix / "beta.log"
    offsets = _write_log(torn, _rec("f", "F"), _rec("i", "I"), _rec("k", "K"))
    _corrupt_middle_frame(torn, offsets)  # "i" torn; "f" and "k" readable
    torn_before = _file_state(torn)

    with caplog.at_level("WARNING"):
        cache.consolidate(prefix)

    assert _file_state(torn) == torn_before  # untouched: no in-place repair
    own = {r.key: r.value for r in _iter_records(root / prefix / "alpha.log")}
    assert own == {"d": "mine", "f": "F", "k": "K"}  # readable records recovered
    assert any("left untouched" in r.message for r in caplog.records)


def test_auto_consolidation_respects_marker(tmp_path):
    """The auto-trigger in `set()` (past `max_writers_per_prefix`) routes through
    the same replicated path: fresh foreign logs survive byte-for-byte."""
    root = tmp_path / "cache"
    for w in ("w0", "w1", "w2"):
        LogCache(root, writer_id=w, prefix_width=1, max_writers_per_prefix=0).set("k", w)
    (root / ".replicated").touch()
    prefix = LogCache(root, prefix_width=1)._prefix("k")
    foreign_before = {
        p.name: _file_state(p) for p in (root / prefix).glob("*.log")
    }

    me = LogCache(root, writer_id="ME", prefix_width=1, max_writers_per_prefix=2, index_ttl=0)
    me.get("k")  # build the index so _sig counts the peer logs
    me.set("k", "ME")  # trips auto-consolidation (4 logs > bound of 2)

    logs = {p.name for p in (root / prefix).glob("*.log")}
    assert logs == {"w0.log", "w1.log", "w2.log", "ME.log"}  # nothing foreign folded
    for name, state in foreign_before.items():
        assert _file_state(root / prefix / name) == state
    assert _reader(root).get("k") == "ME"


# ── pool GC in replicated mode ────────────────────────────────────────────────


def test_replicated_pool_grace_raised(tmp_path):
    """An unreferenced pool file older than the exclusive-mode grace (1 h) but
    younger than `replicated_spill_grace` (24 h) survives a replicated pass —
    a spill can replicate ahead of the log that references it."""
    root = tmp_path / "cache"
    cache = LogCache(root, writer_id="alpha", min_file_size=100, **_WIDTH1)
    (root / ".replicated").touch()
    cache.set("d", "v" * 5000)  # a referenced spill keeps the log non-empty
    prefix = cache._prefix("d")
    orphan = root / prefix / "spill" / "orphan.val"
    orphan.write_bytes(b"replicated ahead of its log")
    _age(orphan, 2 * 3600)  # past 1 h (exclusive grace), inside 24 h

    cache.consolidate(prefix)
    assert orphan.exists()  # protected by the raised replicated grace

    tight = LogCache(
        root, writer_id="alpha", min_file_size=100, replicated_spill_grace=3600.0, **_WIDTH1
    )
    tight.consolidate(prefix)
    assert not orphan.exists()  # past the (tightened) grace + unreferenced -> swept
    assert cache.get("d") == "v" * 5000  # the referenced spill was never at risk


def test_pool_file_referenced_only_by_fresh_foreign_log_survives(tmp_path):
    """GC references derive from ALL logs present — including fresh foreign logs
    that the pass never folds. Age alone must not doom a referenced spill."""
    root = tmp_path / "cache"
    cache = LogCache(root, writer_id="alpha", min_file_size=100, **_WIDTH1)
    (root / ".replicated").touch()
    cache.set("d", "mine")
    prefix = cache._prefix("d")
    blob = pickle.dumps("foreign-value", protocol=pickle.HIGHEST_PROTOCOL)
    pool_file = root / prefix / "spill" / f"{sha256(blob).hexdigest()}.val"
    pool_file.parent.mkdir(parents=True, exist_ok=True)
    pool_file.write_bytes(blob)
    _age(pool_file, 25 * 3600)  # ancient — well past every grace window
    rel = f"{prefix}/spill/{pool_file.name}"
    _write_log(
        root / prefix / "beta.log",
        _Record("f", None, None, time.time(), False, rel),  # fresh foreign reference
    )

    cache.consolidate(prefix)

    assert pool_file.exists()  # pinned by the fresh foreign log's reference
    assert _reader(root).get("f") == "foreign-value"


# ── the unread guard (issue #22, from the local side) ─────────────────────────


def test_replicated_own_log_unreadable_aborts_pass(tmp_path, monkeypatch, caplog):
    """If our OWN log is transiently unreadable, a replicated pass must NOT write
    the peer-only merge over it (silently destroying this node's records) — it
    aborts, leaving our log and the stale peer log intact."""
    import emboss._log_cache as m

    root = tmp_path / "cache"
    LogCache(root, writer_id="alpha", **_WIDTH1).set("d", "mine")
    (root / ".replicated").touch()
    prefix = LogCache(root, prefix_width=1)._prefix("d")
    stale = root / prefix / "beta.log"
    _write_log(stale, _rec("f", "from-beta"))
    _age(stale, 13 * 3600)  # stale → fold-eligible, so it would be pruned but for the abort
    real_read = m._read_records

    def flaky(path):
        if path.name == "alpha.log":  # our own log fails transiently
            raise OSError("transient read failure on our own log")
        return real_read(path)

    monkeypatch.setattr(m, "_read_records", flaky)
    with caplog.at_level("WARNING"):
        LogCache(root, writer_id="alpha", **_WIDTH1).consolidate(prefix)
    monkeypatch.undo()

    assert (root / prefix / "alpha.log").exists()  # never overwritten
    assert stale.exists()  # stale peer not pruned under the abort
    reader = _reader(root)
    assert reader.get("d") == "mine"  # our own record survived
    assert reader.get("f") == "from-beta"
    assert any("could not be read" in r.message for r in caplog.records)


def test_replicated_unread_peer_suppresses_prune_and_sweeps(tmp_path, monkeypatch):
    """An unreadable PEER source (not our target) is never pruned, and because its
    references are unknown it also suppresses dead-record dropping and the pool
    sweep — the module's 'never lose what we couldn't read' invariant, in
    replicated mode."""
    import emboss._log_cache as m

    root = tmp_path / "cache"
    cache = LogCache(root, writer_id="alpha", **_WIDTH1)
    (root / ".replicated").touch()
    cache.set("d", "x")
    assert cache.delete("d") is True  # a tombstone in OUR log that a clean pass would drop
    prefix = cache._prefix("d")
    stale = root / prefix / "beta.log"
    _write_log(stale, _rec("f", "from-beta"))
    _age(stale, 13 * 3600)  # stale → fold-eligible (would be read + pruned)
    orphan = root / prefix / "spill" / "orphan.val"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"unreferenced")
    _age(orphan, 25 * 3600)  # past the 24 h replicated grace → would be swept
    real_read = m._read_records

    def flaky(path):
        if path.name == "beta.log":
            raise OSError("transient read failure")
        return real_read(path)

    monkeypatch.setattr(m, "_read_records", flaky)
    cache.consolidate(prefix)
    monkeypatch.undo()

    assert stale.exists()  # unread source never pruned
    assert orphan.exists()  # sweep suppressed — references unknown while a log is unread
    own = list(_iter_records(root / prefix / "alpha.log"))
    assert [r.key for r in own if r.deleted] == ["d"]  # tombstone retained (any_unread)


# ── dead-record retention: the expired arm ────────────────────────────────────


def test_replicated_expired_retained_while_fresh_foreign_holds_older(tmp_path):
    """The expired arm of the retention rule (mirrors the tombstone test): an
    EXPIRED record is kept while a fresh foreign log holds an OLDER record for the
    key — dropping it would resurrect the older value on read — and both collapse
    once the foreign log ages out and folds."""
    root = tmp_path / "cache"
    cache = LogCache(root, writer_id="alpha", **_WIDTH1)
    (root / ".replicated").touch()
    prefix = cache._prefix("d")
    now = time.time()
    foreign = root / prefix / "beta.log"
    _write_log(foreign, _Record("d", "old-foreign", None, now - 100, False, None))
    # our own log: a NEWER record for "d" that has already expired
    _write_log(root / prefix / "alpha.log", _Record("d", "mine", now - 1, now, False, None))

    cache.consolidate(prefix)

    own = list(_iter_records(root / prefix / "alpha.log"))
    assert [(r.key, r.expire_time is not None) for r in own] == [("d", True)]  # retained
    assert _reader(root).get("d") is None  # expired record suppresses the older foreign one

    _age(foreign, 13 * 3600)  # the foreign log ages out
    cache.consolidate(prefix)
    assert not foreign.exists()  # folded + deleted
    assert not (root / prefix / "alpha.log").exists()  # nothing live remains
    assert _reader(root).get("d") is None


# ── size_limit interaction ────────────────────────────────────────────────────


def test_replicated_size_limit_bounds_local_without_losing_foreign(tmp_path):
    """`size_limit` trims THIS node's own live records in a replicated merge, but
    never touches a fresh foreign log — no foreign record is lost — and the pass
    stays idempotent."""
    root = tmp_path / "cache"
    cache = LogCache(root, writer_id="alpha", size_limit=250, **_WIDTH1)
    (root / ".replicated").touch()
    prefix = cache._prefix("d")
    foreign = root / prefix / "beta.log"
    _write_log(foreign, _rec("i", "from-beta"))  # a fresh foreign record
    foreign_before = _file_state(foreign)
    for key in ("d", "f", "k", "p"):  # our own, oldest → newest; over the byte budget
        cache.set(key, key * 100)

    cache.consolidate(prefix)

    assert _file_state(foreign) == foreign_before  # foreign never touched by trimming
    reader = _reader(root)
    assert reader.get("i") == "from-beta"  # foreign record never at risk
    assert reader.get("p") == "p" * 100  # our newest survives the trim
    before = _tree_state(root)
    cache.consolidate(prefix)
    assert _tree_state(root) == before  # trimmed merge is idempotent


# ── knobs and append-path behaviour ───────────────────────────────────────────


def test_replicated_knobs_resolve_from_env_args_win(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBOSS_REPLICATED_STALE_TTL", "60")
    monkeypatch.setenv("EMBOSS_REPLICATED_SPILL_GRACE", "120")
    cache = LogCache(tmp_path / "c", writer_id="w")
    assert cache.replicated_stale_ttl == 60.0
    assert cache.replicated_spill_grace == 120.0
    explicit = LogCache(
        tmp_path / "c", writer_id="w", replicated_stale_ttl=5.0, replicated_spill_grace=7.0
    )
    assert explicit.replicated_stale_ttl == 5.0  # an explicit arg beats the env
    assert explicit.replicated_spill_grace == 7.0

    monkeypatch.setenv("EMBOSS_REPLICATED_STALE_TTL", "not-a-number")
    with pytest.raises(ValueError, match="EMBOSS_REPLICATED_STALE_TTL"):
        LogCache(tmp_path / "c2", writer_id="w")


def test_append_after_replicated_delete_recreates_log(tmp_path):
    """The append path opens the log fresh per write (no held fd), so a
    replicated deletion landing between appends is recovered by construction:
    the next append recreates the file — no fstat/st_nlink guard needed."""
    root = tmp_path / "cache"
    cache = LogCache(root, writer_id="alpha", **_WIDTH1)
    (root / ".replicated").touch()
    cache.set("d", "first")
    log = root / cache._prefix("d") / "alpha.log"
    log.unlink()  # a peer's fold of our (stale-looking) log replicated in

    cache.set("f", "second")  # no exception: the append recreates the log

    assert log.exists()
    assert cache.get("f") == "second"
    assert cache.get("d") is None  # folded away elsewhere -> a recomputable miss
