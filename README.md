[![PyPI version](https://img.shields.io/pypi/v/voxreg?style=flat-square&color=d93a54&label=PyPI)](https://pypi.org/project/voxreg/)
[![Python versions](https://img.shields.io/pypi/pyversions/voxreg?style=flat-square&color=0099e5&logo=python&logoColor=white)](https://pypi.org/project/voxreg/)
[![License](https://img.shields.io/github/license/confusius-tools/voxreg?style=flat-square&color=3ad9a4)](https://github.com/confusius-tools/voxreg/blob/main/LICENSE)
[![Discord](https://img.shields.io/badge/Discord-join-5865F2?style=flat-square&logo=discord&logoColor=white)](https://discord.gg/mZd87tgmy2)

# voxreg

> [!NOTE]
> **Research prototype.** Anatomical registration quality is not yet validated.

Batched physical-space image registration with **PyTorch** and
[ConfUSIus](https://github.com/confusius-tools/confusius) VoxelData.

- Translation, rigid, affine and symmetric nonlinear (SyN) registration.
- Single image pairs or motion correction of recordings, on CPU or CUDA.
- Independent fixed/moving grids, with original intensities and dtype retained.

## Installation

Requires **Python 3.11–3.14**.

```bash
pip install voxreg
```

For development, from a source checkout:

```bash
git clone https://github.com/confusius-tools/voxreg.git
cd voxreg
uv sync
```

ConfUSIus provides the VoxelData model; registration itself runs in PyTorch. CUDA
requires a compatible GPU and PyTorch installation.

## Quick start

```python
import confusius as cf
import voxreg

recording = cf.load("recording.nii.gz").compute()
result = voxreg.register(recording)

corrected = result.registered
valid = result.valid
print(result.diagnostics)
```

For a recording, the first frame is the default reference and is preserved exactly. For
a single image pair, pass `voxreg.register(moving, fixed)`. Select `device="cuda"` to
use a GPU.

## `voxreg.register`

```python
result = voxreg.register(
    moving,
    fixed=None,
    transform="rigid",
    intensity_scaling="db",
    device="cpu",
    return_transforms=False,
)
```

| Argument | Purpose |
| --- | --- |
| `moving`, `fixed` | NumPy-backed VoxelData in mm: `(k, j, i)` images or `(time, k, j, i)` recordings. `fixed` is spatial-only and required for a single image. |
| `transform` | `"translation"`, `"rigid"` (default), `"affine"` or `"syn"`. Slices use `k=1`; volumes have at least two samples on each axis. |
| `initialization` | Optional `"center_geometry"`, `"center_moments"`, `"align_planes"` or a fixed→moving world matrix. Defaults to identity. |
| `intensity_scaling` | `"db"` (default) for nonnegative power, or `"none"` for already-prepared/signed data. Affects fitting only. |
| `batch_size` | Maximum frames fitted together; automatic by default. Full returned images still occupy CPU memory. |
| `max_iterations` | Per-frame update limit at each level: an integer (default `100`) or one integer per level. Inspect diagnostics rather than assuming convergence. |
| `shrink_factors`, `smoothing_sigmas` | Explicit coarse→fine downsampling factors and Gaussian image smoothing standard deviations in mm. Defaults `(1,)` and `(0.0,)` retain native-resolution fitting. |
| `syn_cc_radius` | SyN local NCC radius in voxels; default `2` gives a rectangular window of width `5`. |
| `syn_update_sigma`, `syn_total_sigma` | SyN Gaussian smoothing standard deviations in voxels, before update composition and on the accumulated field respectively; defaults `1.5` and `0.5`. Zero disables the corresponding smoothing. |
| `device`, `progress` | `"cpu"` or `"cuda"`; progress defaults to `"auto"`, or use `False` to silence it. |
| `return_transforms` | Opt into forward and inverse maps: linear matrices or SyN displacement fields. Dense fields can be much larger than the images. |

The result contains **`registered`** intensities and a boolean **`valid`** support
mask on the fixed grid, plus **`diagnostics`** with per-frame status and objective.
Inputs are not modified; output retains the moving dtype and time labels.
Use `valid` to exclude out-of-bounds padding from analysis.

The `syn_` options apply only to `transform="syn"`; nondefault values with linear
models raise an error. Smoothing values are standard deviations, not ANTs field
variances. Local CC uses a uniformly weighted rectangular window.

With `return_transforms=True`, `forward_transforms` are physical pulls:
**`moving_world = matrix @ fixed_world`**, in **z/y/x, mm**. Linear models return
matrices and their inverses; SyN returns complete forward displacements on the
fixed grid and inverse displacements on the moving grid. Both attributes are
`None` by default. `transforms` remains a compatibility alias for `forward_transforms`.

Pass original linear intensities when using `center_moments`; logarithms and other
nonlinear preprocessing change its centroid weights. Noncoplanar slice pairs
need compatible initialization. SyN does not support mixed slice/volume inputs;
Dask-backed inputs are not supported.

For all options, including optimizer controls and callbacks, use
`help(voxreg.register)` or read the [API docstring](src/voxreg/rigid.py).

## Multi-resolution fitting

All models accept explicit coarse-to-fine schedules:

```python
result = voxreg.register(
    moving, fixed,
    transform="syn",
    shrink_factors=(4, 2, 1),
    smoothing_sigmas=(2.0, 1.0, 0.0),  # Gaussian standard deviations in mm
    max_iterations=(100, 50, 25),
)
```

Choose smoothing for your physical voxel spacing and image content; this example
is not a recommended preset. A scalar `max_iterations` applies at every level.
Both schedules must have the same length. Shrink factors are positive integers
in nonincreasing order; sigmas are finite and nonnegative.

Each level smooths the prepared native fitting images before striding their
original grids. Singleton axes remain singleton, and active axes retain at least
two samples. Trailing samples can be omitted by striding. Linear world matrices
and both SyN halfway fields carry between levels, with fresh optimizer and
convergence state. Original intensities are sampled once for the final output
on the original fixed grid, even if the final shrink factor exceeds one.

`diagnostics.level_n_iterations`, `level_status` and `level_objective` describe
each level; `shrink_factor`, `smoothing_sigma` (mm) and `iteration_budget` describe
the schedule. The usual `n_iterations` sums actual updates across levels;
`objective` describes the final level, and `status` includes final safety checks.
Failed or converged frames are retried at the next level. Metric callback clocks
restart at each level, with reports ordered coarse to fine within each batch;
losses from different levels must not be compared.

Image smoothing in mm is separate from `syn_update_sigma` and `syn_total_sigma`,
which remain field smoothing standard deviations in **current-level voxels**.

## Apply transforms offline

`voxreg.resample` applies exported maps without fitting, on CPU or CUDA. For a
spatial image pair:

```python
result = voxreg.register(moving, fixed, return_transforms=True)
resampled, valid = voxreg.resample(moving, fixed, result.forward_transforms, device="cuda")
reverse, reverse_valid = voxreg.resample(fixed, moving, result.inverse_transforms, device="cuda")
```

The reference must be spatial-only; for a recording use a spatial frame as the
reference. Displacement fields must match that grid exactly. Original dtype and
time labels are retained; use `interpolation="nearest"` for labels or masks.
See `help(voxreg.resample)` for broadcasting, batching and support conventions.

## Development and benchmarks

```bash
uv run pytest
uv run prek run --all-files

# List configurations; see BENCHMARK.md before starting the full matrix.
uv run --extra examples python examples/benchmark_register.py --list
uv run --extra examples python examples/benchmark_resample.py --list
```

- [Benchmark guide](BENCHMARK.md): registration/resampling runtime, accuracy and peak memory.
- [Design](DESIGN.md): physical conventions and implementation details.
- [Validation](VALIDATION.md): correctness checks and known accuracy limits.
- [Motion-correction example](examples/notebook_motion_correction.py).
