"""Run native ANTs registration and verify fixed outer iteration counts."""

import os
import re
import sys
from pathlib import Path


def read_iteration_count(path: Path, expected: int) -> int:
    """Validate a single-level native registration iteration log.

    Parameters
    ----------
    path : pathlib.Path
        Native verbose output for one frame and one registration stage.
    expected : int
        Required number of outer updates.

    Returns
    -------
    int
        Verified update count.

    Raises
    ------
    ValueError
        If the log does not contain exactly the requested sequence of updates.
    """
    observed = [
        int(match[1])
        for match in re.finditer(
            r"^\s*\d*DIAGNOSTIC,\s*(\d+),", path.read_text(), re.MULTILINE
        )
    ]
    if expected < 1 or observed != list(range(1, expected + 1)):
        raise ValueError(
            f"ANTs iteration count mismatch in {path}: expected 1..{expected}, "
            f"observed {len(observed)} rows ending at {observed[-1:]}."
        )
    return len(observed)


def run_ants_registration(
    arguments: list[str], iterations: int, log: Path | None = None
) -> int | None:
    """Call the ANTsPy argument-list API without verbose timing overhead.

    Parameters
    ----------
    arguments : list of str
        Single-stage native registration arguments, including the fixed budget.
    iterations : int
        Expected outer updates when capturing validation output.
    log : pathlib.Path, optional
        Native stdout destination for an untimed accuracy call. If not provided,
        verbose output and count instrumentation are omitted.

    Returns
    -------
    int or None
        Observed updates for a validation call; otherwise no observed count.

    Raises
    ------
    RuntimeError
        If native registration fails.
    ValueError
        If a captured log does not verify the required updates.
    """
    import ants

    if log is None:
        ants.registration(fixed=arguments, moving=None)
        return None
    # Python stdout redirection alone does not capture the native C++ logger.
    sys.stdout.flush()
    descriptor = 1
    saved = os.dup(descriptor)
    try:
        with log.open("w") as stream:
            os.dup2(stream.fileno(), descriptor)
            ants.registration(fixed=[*arguments, "--verbose", "1"], moving=None)
            sys.stdout.flush()
    finally:
        os.dup2(saved, descriptor)
        os.close(saved)
    return read_iteration_count(log, iterations)
