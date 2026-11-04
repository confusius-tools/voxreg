# voxreg

> [!NOTE]
> **Research prototype.** Anatomical registration quality is not yet validated.

Batched physical-space image registration with **PyTorch** and
[ConfUSIus](https://github.com/confusius-tools/confusius) VoxelData.

- Translation, rigid, affine and symmetric nonlinear (SyN) registration.
- Single image pairs or motion correction of recordings, on CPU or CUDA.
- Independent fixed/moving grids, with original intensities and dtype retained.

## Installation

Requires **Python 3.11–3.14**. From a source checkout:

```bash
git clone https://github.com/sdiebolt/voxreg.git
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
| `max_iterations` | Per-frame update limit; default `100`. Inspect diagnostics rather than assuming convergence. |
| `device`, `progress` | `"cpu"` or `"cuda"`; progress defaults to `"auto"`, or use `False` to silence it. |
| `return_transforms` | Opt into forward and inverse maps: linear matrices or SyN displacement fields. Dense fields can be much larger than the images. |

The result contains **`registered`** intensities and a boolean **`valid`** support
mask on the fixed grid, plus **`diagnostics`** with per-frame status and objective.
Inputs are not modified; output retains the moving dtype and time labels.
Use `valid` to exclude out-of-bounds padding from analysis.

With `return_transforms=True`, `forward_transforms` are physical pulls:
**`moving_world = matrix @ fixed_world`**, in **z/y/x, mm**. Linear models return
matrices and their inverses; SyN returns complete forward displacements on the
fixed grid and inverse displacements on the moving grid. Both attributes are
`None` by default. `transforms` remains a compatibility alias for `forward_transforms`.

Pass original linear intensities when using `center_moments`; logarithms and other
nonlinear preprocessing change its centroid weights. Noncoplanar slice pairs
need compatible initialization. SyN does not support mixed slice/volume inputs;
Dask-backed inputs and multiresolution fitting are not supported.

For all options, including optimizer controls and callbacks, use
`help(voxreg.register)` or read the [API docstring](src/voxreg/rigid.py).

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
