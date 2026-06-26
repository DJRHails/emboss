"""Tests for `transfer` — copy entries between cache backends."""

from __future__ import annotations

from emboss import FanoutCache, FileCache, LogCache, SqliteCache, cached, transfer


def test_transfer_from_filecache(tmp_path):
    src = FileCache(tmp_path / "src")
    src.set("a", 1)
    src.set("b", {"x": 2})
    dst = SqliteCache(tmp_path / "dst")
    assert transfer(src, dst) == 2
    assert dst.get("a") == 1
    assert dst.get("b") == {"x": 2}
    dst.close()


def test_transfer_log_to_log(tmp_path):
    src = LogCache(tmp_path / "src", writer_id="a")
    dst = LogCache(tmp_path / "dst", writer_id="b")
    for i in range(10):
        src.set(f"k{i}", i)
    assert transfer(src, dst) == 10
    assert all(dst.get(f"k{i}") == i for i in range(10))


def test_transfer_across_backends(tmp_path):
    src = SqliteCache(tmp_path / "src")
    dst = LogCache(tmp_path / "dst", writer_id="a")
    src.set("a", {"x": 1})
    src.set("b", [1, 2, 3])
    n = transfer(src, dst)
    assert n == 2
    assert dst.get("a") == {"x": 1}
    assert dst.get("b") == [1, 2, 3]
    src.close()


def test_transfer_clears_source(tmp_path):
    src = LogCache(tmp_path / "src", writer_id="a")
    dst = SqliteCache(tmp_path / "dst")
    src.set("k", "v")
    assert transfer(src, dst, clear_source=True) == 1
    assert len(src) == 0
    assert dst.get("k") == "v"
    dst.close()


def test_transfer_fanout_to_sqlite(tmp_path):
    src = FanoutCache(tmp_path / "src", shards=4)
    dst = SqliteCache(tmp_path / "dst")
    for i in range(50):
        src.set(f"k{i}", i)
    assert transfer(src, dst) == 50
    assert dst.get("k25") == 25
    src.close()
    dst.close()


def test_transfer_copies_keys_and_values_verbatim(tmp_path):
    """A cache populated by @cached transfers key-for-key and byte-for-byte, so
    the same @cached function would hit on the destination (keys are preserved,
    values copied as their stored encoding)."""
    src = SqliteCache(tmp_path / "src")

    @cached(src)
    def f(x: int) -> dict:
        return {"doubled": x * 2}

    f(21)
    src_items = {k: src.get(k) for k in src}
    assert src_items  # @cached wrote at least one entry

    dst = LogCache(tmp_path / "dst", writer_id="a")
    transfer(src, dst)

    assert {k: dst.get(k) for k in dst} == src_items  # identical keys + values
    src.close()


def test_transfer_empty_source(tmp_path):
    src = LogCache(tmp_path / "src", writer_id="a")
    dst = LogCache(tmp_path / "dst", writer_id="b")
    assert transfer(src, dst) == 0
