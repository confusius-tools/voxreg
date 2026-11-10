"""Verified Linux input-page eviction for benchmark-owned disk files."""

import ctypes
import mmap
import os
import stat
import sys
from pathlib import Path
from time import perf_counter
from typing import TypedDict


class CacheEviction(TypedDict):
    """Evidence for one targeted input-cache eviction attempt."""

    method: str
    files: int
    bytes: int
    page_size: int
    resident_pages_before: int
    resident_pages_after: int
    seconds: float
    verified: bool


_MINCORE = (
    getattr(ctypes.CDLL(None, use_errno=True), "mincore", None)
    if sys.platform == "linux"
    else None
)
"""Linux residency syscall; absent on unsupported platforms."""
if _MINCORE is not None:
    _MINCORE.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    _MINCORE.restype = ctypes.c_int


def count_resident_pages(descriptor: int) -> int:
    """Count cached file pages without reading their contents.

    Parameters
    ----------
    descriptor : int
        Open descriptor for a file owned by the current user. Linux can obscure
        residency for files the caller does not own.

    Returns
    -------
    int
        Number of resident pages. Empty files have no pages.

    Raises
    ------
    OSError
        If Linux residency inspection is unavailable or fails.
    """
    if _MINCORE is None:
        raise OSError("Cold-input verification requires Linux mincore.")
    size = os.fstat(descriptor).st_size
    if not size:
        return 0
    with mmap.mmap(descriptor, size, access=mmap.ACCESS_COPY) as mapping:
        # A private writable mapping permits address inspection without touching data.
        address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        pages = (ctypes.c_ubyte * ((size + mmap.PAGESIZE - 1) // mmap.PAGESIZE))()
        if _MINCORE(address, size, pages) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        return sum(page & 1 for page in pages)


def evict_input_cache(directory: Path) -> CacheEviction:
    """Flush and evict owned input files, then verify page-cache residency.

    Parameters
    ----------
    directory : pathlib.Path
        Benchmark-owned input tree. Symlinks and files owned by another user
        are rejected. No global cache controls or privileged commands are used.

    Returns
    -------
    CacheEviction
        Eviction evidence. The caller must reject a pass when `verified` is
        false, as advice alone does not guarantee eviction (notably on tmpfs).

    Raises
    ------
    OSError
        If Linux fadvise/mincore, syncing or file access is unavailable or fails.
    ValueError
        If the tree is empty, contains symlinks or contains unowned/nonregular files.
    """
    if sys.platform != "linux" or not hasattr(os, "posix_fadvise") or _MINCORE is None:
        raise OSError("Cold-input benchmarks require Linux fadvise and mincore.")
    if directory.is_symlink():
        raise ValueError("Cold-input stores must not be symlinks.")
    started = perf_counter()
    files = []
    byte_count = before = after = 0
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Cold-input stores must not contain symlinks: {path}")
        if path.is_dir():
            continue
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
                raise ValueError(
                    f"Cold-input files must be regular and owned by the caller: {path}"
                )
            before += count_resident_pages(descriptor)
            byte_count += info.st_size
            os.fsync(descriptor)
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
            files.append(path)
        finally:
            os.close(descriptor)
    if not files:
        raise ValueError("Cold-input store contains no files.")
    # Verify after all eviction requests; inspection maps pages but does not read them.
    for path in files:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            after += count_resident_pages(descriptor)
        finally:
            os.close(descriptor)
    return {
        "method": "linux-fsync-fadvise-dontneed-mincore",
        "files": len(files),
        "bytes": byte_count,
        "page_size": mmap.PAGESIZE,
        "resident_pages_before": before,
        "resident_pages_after": after,
        "seconds": perf_counter() - started,
        "verified": after == 0,
    }
