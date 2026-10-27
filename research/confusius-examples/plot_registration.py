"""Shared research plotting; never selects a backend or imports an experiment runner."""

from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
from scipy.ndimage import map_coordinates

from voxreg import position_to_world


def sample(matrix, fixed, moving):
    af, am = position_to_world(fixed), position_to_world(moving)
    points = af @ np.vstack(
        (np.indices(fixed.shape).reshape(3, -1), np.ones((1, fixed.size)))
    )
    coordinates = (np.linalg.inv(am) @ matrix @ points)[:3]
    valid = (
        (coordinates >= -1e-5)
        & (coordinates <= np.array(moving.shape)[:, None] - 1 + 1e-5)
    ).all(axis=0)
    values = map_coordinates(
        moving.values, coordinates, order=1, mode="nearest", prefilter=False
    )
    return values.reshape(fixed.shape), valid.reshape(fixed.shape), coordinates


def display_sample(matrix, fixed, moving):
    """Never present nearest-edge extrapolation as an observed anatomical match."""
    values, valid, coordinates = sample(matrix, fixed, moving)
    return np.where(valid, values, np.nan), valid, coordinates


def composite_rgb(reference, image, display_range=(-30, 0)):
    """Reference cyan, sampled image red; invalid pixels gray, without extrapolation."""
    low, high = display_range
    cyan = np.clip((np.asarray(reference).astype(float) - low) / (high - low), 0, 1)
    red = np.clip((np.asarray(image).astype(float) - low) / (high - low), 0, 1)
    valid = np.isfinite(reference) & np.isfinite(image)
    rgb = np.stack((red, cyan, cyan), axis=-1)
    rgb[~valid] = 0.25
    return rgb


def supported_composites(reference, images, title, *, reference_is_red=False):
    import matplotlib.pyplot as plt

    assert not reference_is_red
    fig, axes = plt.subplots(
        1, len(images), figsize=(5 * len(images), 5), constrained_layout=True
    )
    for ax, (name, values) in zip(np.atleast_1d(axes), images.items(), strict=True):
        ax.imshow(
            composite_rgb(np.asarray(reference)[0], np.asarray(values)[0]),
            interpolation="nearest",
        )
        ax.set_title(name)
        ax.set_axis_off()
    fig.suptitle(
        title
        + " — template red / recording cyan; unsupported pixels gray; shared −30…0dB"
    )


@contextmanager
def observe_optimizer(on_sample):
    """Scoped research observer; not for concurrent fits in the same Python process."""
    from voxreg import rigid as engine

    original_optimize, original_sample = engine._optimize, engine.sample_rigid

    def optimize(images, target, grid, *args, **options):
        def observe(images, parameters, geometry, **sampling_options):
            on_sample(images, target, parameters, geometry)
            return original_sample(images, parameters, geometry, **sampling_options)

        with patch.object(engine, "sample_rigid", observe):
            return original_optimize(images, target, grid, *args, **options)

    with patch.object(engine, "_optimize", optimize):
        yield


def register_with_plot(
    moving, fixed, *, show=True, redraw_interval=0.25, display_range=(-30, 0), **kwargs
):
    """Register one pair with a live composite/metric figure; return (result, figure).

    Research-only scoped observation: no engine/API change. Select a single frame
    before calling. A fixed slice is displayed directly; a moving slice is displayed
    with the inverse pull; two volumes show the fixed volume's central k slice.
    Display uses supplied intensities, defaulting to -30..0 dB. Pass display_range
    for other units. show=False collects the trace and renders only the final pose.
    metric_interval controls observation cadence; redraw_interval throttles GUI work.
    The raw trial curve can worsen; the final image/line use the returned best pose.
    """
    from time import monotonic

    import matplotlib.pyplot as plt

    from voxreg import rigid as engine

    if not np.isfinite(redraw_interval) or redraw_interval < 0:
        raise ValueError("redraw_interval must be finite and nonnegative.")
    limits = np.asarray(display_range, dtype=float)
    if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
        raise ValueError("display_range must contain two increasing finite limits.")
    callback = kwargs.pop("metric_callback", None)
    if callback is not None and not callable(callback):
        raise TypeError("metric_callback must be callable or None.")

    def single(data):
        if "time" not in data.dims:
            return data
        if data.sizes["time"] != 1:
            raise ValueError("Plot one image pair; select a moving/fixed frame first.")
        return data.isel(time=0, drop=True)

    source, target = single(moving), single(fixed)
    inverse = source.sizes["k"] == 1 and target.sizes["k"] > 1
    reference, other = (source, target) if inverse else (target, source)
    central = reference.sizes["k"] > 1
    if central:
        middle = reference.sizes["k"] // 2
        reference = reference.isel(k=slice(middle, middle + 1))
    fig, (image_ax, metric_ax) = plt.subplots(
        1, 2, figsize=(10, 5), constrained_layout=True
    )
    basis = position_to_world(reference)[:3, :3]
    unsupported = np.full((*reference.shape[1:], 3), 0.25)
    artist = image_ax.imshow(
        unsupported,
        interpolation="nearest",
        aspect=np.linalg.norm(basis[:, 1]) / np.linalg.norm(basis[:, 2]),
    )
    image_ax.set_axis_off()
    (line,) = metric_ax.plot([], [], label="Raw trial")
    metric_ax.set(xlabel="Optimizer updates", ylabel="Negative NCC² (lower is better)")
    fig.suptitle(
        "Reference cyan / sampled image red; unsupported pixels gray"
        + (" — fixed central k slice" if central else "")
    )
    clocks, metrics = [], []
    state = None
    last_redraw = -float("inf")

    def redraw(matrix, title):
        if not plt.fignum_exists(fig.number):
            return
        rgb = unsupported
        if np.isfinite(matrix).all():
            try:
                pull = np.linalg.inv(matrix) if inverse else matrix
                values, _, _ = display_sample(pull, reference, other)
            except np.linalg.LinAlgError:
                pass  # A rejected singular trial need not be displayable.
            else:
                rgb = composite_rgb(reference.values[0], values[0], limits)
        artist.set_data(rgb)
        image_ax.set_title(title)
        line.set_data(clocks, metrics)
        metric_ax.relim()
        metric_ax.autoscale_view()
        fig.canvas.draw_idle()
        if show:
            fig.canvas.flush_events()

    def observe(images, target, parameters, grid):
        nonlocal state
        state = (parameters.detach(), grid)

    def report(indices, steps, objective):
        nonlocal last_redraw
        clocks.append(int(steps[0]))
        metrics.append(float(objective[0]))
        now = monotonic()
        if show and state is not None and now - last_redraw >= redraw_interval:
            parameters, grid = state
            matrix = engine.world_matrices(parameters.cpu().numpy(), grid)[0]
            redraw(matrix, f"Raw trial at update {clocks[-1]}")
            last_redraw = now
        if callback is not None:
            callback(indices, steps, objective)

    try:
        if show:
            plt.show(block=False)
        with observe_optimizer(observe):
            result = engine.register(moving, fixed, metric_callback=report, **kwargs)
        value = result.diagnostics.objective.item()
        if np.isfinite(value):
            metric_ax.axhline(
                value, linestyle="--", color="black", label="Returned best"
            )
        metric_ax.legend()
        matrix = np.asarray(result.transforms).reshape(-1, 4, 4)[0]
        redraw(
            matrix,
            "Returned best pose"
            if np.isfinite(value)
            else "Returned pose (unusable metric)",
        )
    except BaseException:
        plt.close(fig)
        raise
    return result, fig
