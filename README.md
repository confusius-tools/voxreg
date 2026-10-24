# voxreg

A standalone proof of concept for **batched physical-space rigid registration**
using PyTorch and xarray-based VoxelData. No ConfUSIus runtime dependency.

The initial engine corrects in-plane rotation and translation in single-slice
recordings with dimensions `(time, k=1, j, i)`. CPU and CUDA are supported. This
is an eager correctness baseline, not yet a tuned high-throughput engine.

See [DESIGN.md](DESIGN.md) for conventions and scope, and
[VALIDATION.md](VALIDATION.md) for the measured quality checks and remaining gates.

## Install and test

Requires Python **3.14**. CUDA execution has been checked on Linux with an RTX
5080 Laptop GPU. The default PyTorch distribution includes large CUDA runtime
packages; a CPU-only package source is not configured here.

```bash
uv sync
uv run pytest
uv run ruff check src tests examples
uv run ruff format --check src tests examples

# Also exercise the optional NIfTI/validation example.
uv sync --extra examples
uv run --extra examples pytest
```

The CUDA test runs when CUDA is available and otherwise skips. Examples use
Matplotlib, nibabel and SciPy; the engine itself depends only on NumPy, xarray
and PyTorch. ITK and a pinned upstream FireANTs archive are development-only
comparison dependencies; `uv sync --no-dev` keeps both out of a runtime-only environment.

## Use

```python
import numpy as np
import voxreg

# values: floating-point NumPy array (time, 1, j, i).
# Affine rows/columns are world z/y/x and native voxel k/j/i, respectively.
recording = voxreg.create_voxeldata(
    values,
    dims=("time", "k", "j", "i"),
    coords={"time": time_seconds},
    voxel_to_world=np.diag([0.4, 0.1, 0.1, 1.0]),
    units="mm",
)

result = voxreg.register_rigid(recording, device="cuda", batch_size=32)
registered = result.registered     # VoxelData, original intensities, float32.
valid = result.valid               # VoxelData, boolean sample-support mask.
matrices = result.transforms       # (time, world_out, world_in), float64.
print(result.diagnostics)
```

Registration defaults to the first frame, which is preserved exactly after the
explicit float32 conversion. An explicit spatial-only `fixed=` image on the same
grid is also accepted. Defaults are 100 iterations, learning rate 0.1 in scaled
physical parameters, and dB fitting intensities. Use `intensity_scaling="none"`
for already-prepared fitting intensities. dB mode requires nonnegative power.

**Transform direction:** `moving_world = matrix @ fixed_world`, using homogeneous
**z/y/x** vectors. These are physical pull mappings, not normalized Torch grids.
Matrix inversion gives inverse rigid transforms. Reported `translation_i/j` are
physical displacements of the rotation center along an orthonormal plane basis;
they are not the homogeneous matrix's translation column.

Spatial cropping/striding and oblique/anisotropic geometry are handled through
native coordinates and a lazy `VoxelToWorldIndex`. Use
`voxreg.position_to_world(data)` when a zero-based array-position affine is
needed. Assigning new native coordinates without rebuilding the world index is
an error, not a supported way to relocate an image.

The index is an independent implementation of the VoxelData model. It is **not**
class-identical to ConfUSIus's private index. Exchange arrays plus geometry at an
I/O boundary, or use the NIfTI example; do not assume the two index classes are
interchangeable. World-label lookup is not implemented; slice native `k/j/i`
coordinates instead.

## Reproduce the real motion-correction example

The example uses Cybis Pereira 2026, rat75, session 20220523, slice32, frames
**220:340**. Supply the downloaded NIfTI and its BIDS JSON sidecar; there is no
built-in downloader. The sidecar supplies the acquisition's irregular time labels.

From the voxreg directory:

```bash
INPUT="$HOME/.cache/confusius/cybis-pereira-2026-bids/1.0.0/sub-rat75/ses-20220523/fusi/sub-rat75_ses-20220523_task-openfield_acq-slice32_pwd.nii.gz"

uv run --extra examples python examples/notebook_motion_correction.py \
    "$INPUT" --device cuda --output artifacts/notebook
```

This produces original-power NIfTI recordings, validity and difference volumes,
exact timing sidecars, transforms/diagnostics in NPZ and CSV, and motion, voxel
trace and raster PNGs. Open the NIfTI outputs in napari with a NIfTI reader.
Artifacts stay local and are ignored by Git.

### Optional frozen ITK comparison

The reference exporter uses **an existing ConfUSIus environment**, solely to run
its original public registration functions. ConfUSIus is not installed into
voxreg; ITK is now available separately as a development dependency. With the neighboring checkout layout used during development:

```bash
# Run with ConfUSIus's interpreter; quality export only, no timing repetitions.
uv run --directory ../../confusius-tools/confusius python \
    "$PWD/examples/freeze_itk_reference.py" "$INPUT" \
    "$PWD/artifacts/notebook/itk_reference.npz"

# Back in the standalone environment: no ConfUSIus imports in this command.
uv run --extra examples python examples/notebook_motion_correction.py \
    "$INPUT" --device cuda --output artifacts/notebook \
    --reference artifacts/notebook/itk_reference.npz
```

The frozen reference records source hashes, resolved ITK registration settings,
input arrays/coordinates, pull matrices and original-power registered outputs.
The comparison uses one independent SciPy bilinear sampler on dB inputs and
**common valid support**, not each implementation's internal optimizer score.
The reference retains the source's float64 intensities; voxreg explicitly uses
float32 samples with float64 NCC statistics.

Timing repetitions default to **zero**. Only on an otherwise idle machine, add
`--repeats 3` to the respective scripts. The reference exporter measures the warm
public pipeline at `n_jobs=1` and `--jobs 8`, using its default joblib backend.
Run each configuration separately; do not infer speedups from contested timings.

## Standalone time and memory comparisons

The motion-correction example also writes `artifacts/notebook/input.npz`. This
neutral input artifact lets the ITK/FireANTs benchmark run entirely in voxreg's
Python environment, without ConfUSIus:

```bash
# Practical subset; run only on an otherwise idle machine.
uv run python examples/benchmark_registration.py \
    --cases itk-1-t8 itk-8 cpu-32-t8 cuda-120 \
            fireants-cpu-32-t8 fireants-cuda-120

# Omitting --cases also includes deliberately slow unbatched controls.
# The full suite can take more than ten minutes.
```

Each configuration uses fresh interpreters, with **separate time and memory
passes**. Warm timings include preprocessing, original-power resampling and
return to host. Memory reporting includes process-tree RSS/PSS, exact PyTorch
CUDA allocator peaks, and sampled per-process GPU VRAM including context memory.
Process-parallel ITK includes scheduling and result-transfer overhead. Startup
costs and loaded-input/warm memory baselines are recorded separately.

See [BENCHMARK.md](BENCHMARK.md) for measured timings, memory, quality and caveats.
The current eager voxreg CPU path is slower than ITK and FireANTs. Native
float32 FireANTs is also faster on full-batch CUDA; these are not precision-matched
solvers. Both comparisons use direct 2D APIs, not the ConfUSIus pipeline or its
earlier frozen-reference optimizer configuration. See the report for the pinned
FireANTs archive and its Python 3.14 packaging rationale.

[PERFORMANCE.md](PERFORMANCE.md) diagnoses the hot path with fresh single-thread
controls and profiler evidence. An isolated float64 scalar-affine sampler prototype
is about 2× faster on threaded CPU and 5× on CUDA while passing the current tests;
that optimization has **not** been applied to the engine yet.

## Current limits

- Single-slice rigid only; no full 3D, affine, deformable or multi-pose registration.
- Identical fixed/moving grids, uniform native coordinates, mm units and finite
  NumPy-backed floating-point input; no masks, Dask or multiresolution.
- A frame losing more than 10% overlap is flagged and retains its best valid
  transform. Constant/invalid fitting metrics are diagnosed separately.
- Output fill is each original frame's minimum. Use `result.valid` to exclude
  padding from downstream analyses.
- Per-frame status distinguishes convergence, iteration limit, invalid metric,
  insufficient overlap and the fixed reference. An iteration limit is **not** a
  convergence claim. The real recording currently reaches the 100-step limit.
- Interpolation/output are float32; physical geometry, optimizer state, NCC
  statistics and world matrices are float64 for batch stability. This is a
  correctness-first precision policy, not a GPU throughput optimum. Cross-device
  results are numerically, not bitwise, equal.
- The lazy index builds on xarray's experimental coordinate-transform index API.

No compiled engine, speedup claim, affine/deformable extension or compatibility
layer is included in this first implementation.
