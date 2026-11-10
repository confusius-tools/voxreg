"""Benchmark-owned input pages are evicted without changing source contents."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "benchmark_cache",
    Path(__file__).resolve().parents[1] / "benchmarks/_benchmark_cache.py",
)
assert SPEC is not None and SPEC.loader is not None
CACHE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CACHE)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux page-cache inspection")
def test_evict_owned_inputs_and_leave_other_files_cached(benchmark_disk_dir):
    inputs = benchmark_disk_dir / "inputs"
    inputs.mkdir()
    contents = b"cold input payload" * 8192
    source = inputs / "chunk"
    source.write_bytes(contents)
    other = benchmark_disk_dir / "other"
    other.write_bytes(contents)
    with other.open("rb") as untouched:
        assert CACHE.count_resident_pages(untouched.fileno()) > 0
        result = CACHE.evict_input_cache(inputs)
        assert result["verified"]
        assert result["files"] == 1
        assert result["bytes"] == len(contents)
        assert result["resident_pages_before"] > 0
        assert result["resident_pages_after"] == 0
        assert CACHE.count_resident_pages(untouched.fileno()) > 0
    assert source.read_bytes() == contents
    again = CACHE.evict_input_cache(inputs)
    assert again["verified"] and again["resident_pages_after"] == 0


@pytest.mark.skipif(sys.platform != "linux", reason="Linux page-cache inspection")
def test_advice_is_not_mistaken_for_verified_eviction(benchmark_disk_dir, monkeypatch):
    (benchmark_disk_dir / "chunk").write_bytes(b"x" * 65536)
    monkeypatch.setattr(CACHE.os, "posix_fadvise", lambda *args: None)
    result = CACHE.evict_input_cache(benchmark_disk_dir)
    assert not result["verified"]
    assert result["resident_pages_after"] > 0


@pytest.mark.skipif(sys.platform != "linux", reason="Linux page-cache inspection")
def test_cache_inspection_handles_empty_files_and_rejects_empty_store(
    benchmark_disk_dir,
):
    with pytest.raises(ValueError, match="no files"):
        CACHE.evict_input_cache(benchmark_disk_dir)
    (benchmark_disk_dir / "empty").touch()
    result = CACHE.evict_input_cache(benchmark_disk_dir)
    assert result["verified"] and result["bytes"] == 0


@pytest.mark.skipif(sys.platform != "linux", reason="Linux page-cache inspection")
def test_cache_eviction_rejects_links_and_unowned_files(
    benchmark_disk_dir, monkeypatch
):
    target = benchmark_disk_dir / "payload"
    target.write_bytes(b"unchanged")
    link = benchmark_disk_dir / "link"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        CACHE.evict_input_cache(link)
    with pytest.raises(ValueError, match="symlink"):
        CACHE.evict_input_cache(benchmark_disk_dir)
    link.unlink()
    monkeypatch.setattr(CACHE.os, "geteuid", lambda: os.stat(target).st_uid + 1)
    with pytest.raises(ValueError, match="owned"):
        CACHE.evict_input_cache(benchmark_disk_dir)
    assert target.read_bytes() == b"unchanged"


def test_cache_eviction_rejects_unsupported_system(benchmark_disk_dir, monkeypatch):
    monkeypatch.setattr(CACHE.sys, "platform", "unsupported")
    with pytest.raises(OSError, match="require Linux"):
        CACHE.evict_input_cache(benchmark_disk_dir)
