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
| `return_transforms` | Opt into linear matrices or SyN forward/inverse displacement fields. Dense fields can be much larger than the images. |

The result contains **`registered`** intensities and a boolean **`valid`** support
mask on the fixed grid, plus **`diagnostics`** with per-frame status and objective.
Inputs are not modified; output retains the moving dtype and time labels.
Use `valid` to exclude out-of-bounds padding from analysis.

With `return_transforms=True`, linear `transforms` are physical pulls:
**`moving_world = matrix @ fixed_world`**, in **z/y/x, mm**. SyN returns complete
displacements on the fixed grid and `inverse_transforms` on the moving grid.
Both transform attributes are `None` by default.

Pass original linear intensities when using `center_moments`; logarithms and other
nonlinear preprocessing change its centroid weights. Noncoplanar slice pairs
need compatible initialization. SyN does not support mixed slice/volume inputs;
Dask-backed inputs and multiresolution fitting are not supported.

For all options, including optimizer controls and callbacks, use
`help(voxreg.register)` or read the [API docstring](src/voxreg/rigid.py).

## Development and benchmarks

```bash
uv run pytest
uv run prek run --all-files

# List configurations; see BENCHMARK.md before starting the full matrix.
uv run --extra examples python examples/benchmark.py --list
```

- [Benchmark guide](BENCHMARK.md): one entry point, runtime, truth RMS and peak memory.
- [Design](DESIGN.md): physical conventions and implementation details.
- [Validation](VALIDATION.md): correctness checks and known accuracy limits.
- [Motion-correction example](examples/notebook_motion_correction.py).
