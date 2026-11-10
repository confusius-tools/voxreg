"""Rich fitting progress for eager loops and local Dask schedulers."""

import re
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from typing import Any

from dask.callbacks import Callback
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

_PROGRESS_KEY = re.compile(r"voxregframes_([a-f0-9]+)_(\d+)_(\d+)_end")
"""Frame-range identities embedded in local lazy fitting tasks."""


@contextmanager
def progress_bar(
    description: str, total: int, *, show: bool
) -> Generator[Callable[[int], None], None, None]:
    """Yield an increment-based frame progress reporter.

    Parameters
    ----------
    description : str
        Label displayed beside the bar.
    total : int
        Total work units.
    show : bool
        Whether to render a Rich display.

    Yields
    ------
    callable
        Advance completed work by the supplied amount.
    """
    if not show:
        yield lambda amount: None
        return
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        "<",
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(description, total=total)
        yield lambda amount: progress.advance(task, amount)


def _find_ranges(value: Any) -> dict[str, int]:
    """Find fitting identities, including tasks nested by Dask fusion.

    Parameters
    ----------
    value : object
        Scheduler key or task specification.

    Returns
    -------
    dict of str to int
        Unique source-task identities and frame counts.
    """
    found = {}
    if isinstance(value, Mapping):
        parts = value.values()
    elif isinstance(value, (tuple, list)):
        parts = value
    elif callable(getattr(value, "func", None)):
        # Read actual nested tasks, not truncated names or references to cached fits.
        match = _PROGRESS_KEY.fullmatch(str(value.key))
        if match is not None:
            found[match.group(0)] = int(match[3]) - int(match[2])
        parts = (getattr(value, "args", ()), getattr(value, "kwargs", {}))
    else:
        return found
    for part in parts:
        if part is not value:
            found.update(_find_ranges(part))
    return found


class _FrameProgress(Callback):
    """One scheduler-owned display for completed lazy fitting frame ranges."""

    def _start_state(self, dsk: dict, state: dict) -> None:
        """Open a bar for reachable fits, excluding optimized orphan tasks.

        Parameters
        ----------
        dsk : dict
            Optimized local execution graph.
        state : dict
            Scheduler state identifying tasks required by the requested outputs.
        """
        self._ranges = {}
        total = {}
        for key in state["dependencies"]:
            if key in state["cache"]:
                continue
            ranges = _find_ranges(dsk.get(key))
            if ranges:
                self._ranges[key] = ranges
                total.update(ranges)
        self._completed = set()
        self._display = progress_bar(
            "Registering frames", sum(total.values()), show=bool(total)
        )
        self._advance = self._display.__enter__()

    def _posttask(
        self, key: Any, result: Any, dsk: dict, state: dict, worker_id: Any
    ) -> None:
        """Count each completed fitting chunk once, even through aliases.

        Parameters
        ----------
        key : object
            Completed scheduler task.
        result : object
            Unused task result.
        dsk, state : dict
            Unused scheduler graph and state.
        worker_id : object
            Unused worker identity.
        """
        for identity, count in self._ranges.get(key, {}).items():
            if identity not in self._completed:
                self._advance(count)
                self._completed.add(identity)

    def _finish(self, dsk: dict, state: dict, failed: bool) -> None:
        """Close the display without falsely completing failed work.

        Parameters
        ----------
        dsk, state : dict
            Unused scheduler graph and state.
        failed : bool
            Whether execution failed.
        """
        display = getattr(self, "_display", None)
        if display is not None:
            display.__exit__(None, None, None)
            self._ranges.clear()
            self._completed.clear()
            self._display = None


_LOCAL_PROGRESS = _FrameProgress()
"""Reusable local scheduler callback; no signal arrays or results are retained."""


def enable_lazy_progress() -> None:
    """Register automatic local Dask frame progress without starting a display."""
    _LOCAL_PROGRESS.register()
