"""Local Dask execution for streaming writes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from dask import config
from dask.core import istask
from dask.graph_manipulation import chunks
from dask.optimization import cull
from dask.order import order
from dask.threaded import get

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dask.typing import Key


def bounded_threads(
    graph: Any,
    keys: Key | Sequence[Key],
    *,
    num_workers: int | None = None,
    **kwargs: Any,
) -> Any:
    """Compute a local Dask write with limited threaded read-ahead.

    Parameters
    ----------
    graph : object
        Optimized graph supplied by Dask.
    keys : object
        Outputs requested by Dask.
    num_workers : int, optional
        Maximum number of runnable tasks admitted together. If not provided,
        use the local Dask worker count from configuration, or two by default.
    **kwargs : object
        Additional options for Dask's local threaded scheduler.

    Returns
    -------
    object
        Computed outputs in the order requested by Dask.

    Raises
    ------
    ValueError
        If the worker count is less than one.

    Notes
    -----
    Use `write.compute(scheduler=bounded_threads, num_workers=4)` on a
    deferred VoxelData write. Each group follows Dask's serial task order.
    A shared input stays live as long as its consumers need it. Source chunk
    sizes and dependencies still determine how much memory each task needs.
    """
    if num_workers is None:
        configured = config.get("num_workers", None)
        num_workers = 2 if configured is None else configured
    if num_workers < 1:
        raise ValueError("num_workers must be at least one.")
    if not isinstance(graph, Mapping):
        graph = graph.__dask_graph__()
    graph, dependencies = cull(graph, keys)
    priorities = order(graph)
    ordered = [
        key
        for key in sorted(priorities, key=priorities.__getitem__)
        if istask(graph[key])
    ]
    token = uuid4().hex
    previous_exits = set()
    for start in range(0, len(ordered), num_workers):
        group = set(ordered[start : start + num_workers])
        if previous_exits:
            # Each entry waits on a single completion signal from the previous group.
            ready = ("voxreg-write-window", token, start)
            graph[ready] = (chunks.checkpoint, *previous_exits)
            for key in group:
                if group.isdisjoint(dependencies[key]):
                    graph[key] = (chunks.bind, graph[key], ready)
        # An exit has no consumer within this group.
        previous_exits = group.difference(
            dep for key in group for dep in dependencies[key]
        )
    return get(graph, keys, num_workers=num_workers, **kwargs)
