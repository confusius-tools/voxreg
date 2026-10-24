# voxreg

A proof of concept for **batched physical-space rigid registration** using
PyTorch and ConfUSIus's xarray-based VoxelData implementation.

The initial engine corrects in-plane rotation and translation in single-slice
recordings with dimensions `(time, k=1, j, i)`. CPU and CUDA are supported. This
is an eager single-slice engine with validated throughput on the documented
recording, not a general-purpose registration replacement.

See [DESIGN.md](DESIGN.md) for conventions and scope, and
[VALIDATION.md](VALIDATION.md) for the measured quality checks and remaining gates.

## Install and test

Requires Python **3.14** (`>=3.14,<3.15`, matching the temporary ConfUSIus dependency). CUDA execution has been checked on Linux with an RTX
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

The CUDA test runs when CUDA is available and otherwise skips. The registration
engine uses NumPy, xarray and PyTorch, with **ConfUSIus temporarily required for
VoxelData** (`confusius>=0.8.0`). `uv sync` installs the released package; no
sibling checkout is required. ConfUSIus brings its full dependency stack,
including ITK/napari and Rich. The NumPy minimum matches ConfUSIus (2.3.4) to
keep its platform dependency constraints resolvable; this Linux environment
uses NumPy 2.5.4 and xarray 2026.9.0.

FireANTs remains a pinned development-only comparison dependency. Matplotlib,
nibabel and SciPy are listed in the `examples` extra, although ConfUSIus also
installs them transitively. No registration is delegated to ConfUSIus or ITK.

## Use

```python
import numpy as np
import voxreg

# values: real NumPy array (time, 1, j, i); its dtype is retained.
# Affine rows/columns are world z/y/x and native voxel k/j/i, respectively.
recording = voxreg.create_voxeldata(
    values,
    dims=("time", "k", "j", "i"),
    coords={"time": time_seconds},
    voxel_to_world=np.diag([0.4, 0.1, 0.1, 1.0]),
    units="mm",
)

result = voxreg.register_rigid(recording, device="cuda", batch_size=32)
registered = result.registered     # VoxelData, original intensities and dtype.
valid = result.valid               # VoxelData, boolean sample-support mask.
matrices = result.transforms       # (time, world_out, world_in), float64.
print(result.diagnostics)
```

Registration defaults to the first frame, which is preserved exactly, including
its original dtype. An explicit spatial-only `fixed=` image on the same
grid is also accepted. Defaults are 100 iterations, learning rate 0.1 in scaled
physical parameters, and dB fitting intensities. Use `intensity_scaling="none"`
for already-prepared fitting intensities or signed integer images. dB mode
requires nonnegative power.

### Image dtype and final interpolation

Input arrays are never modified. Working fitting images are float32; geometry,
Adam state, NCC and world matrices remain float64. Integer/float64 preprocessing
runs in float64 and uses NCC's intensity-offset/scale invariance to retain
low-contrast detail before conversion to fitting tensors.

Final resampling uses the **original values**, once, independently of fitting:

- Float64 images are interpolated in float64 and returned as float64—not sampled
  in float32 and merely cast back. Float32 images stay float32; float16 output is
  retained after float32 interpolation.
- Integer images are interpolated in float64, rounded to nearest-even, clipped
  to their type range and returned in their original dtype. This necessarily
  quantizes fractional intensities. Integer values outside `[-2**53, 2**53]` are
  rejected rather than silently losing precision on conversion.
- `interpolation="linear"` is the default; `"nearest"` changes **final output only**.
  Fitting always uses differentiable linear interpolation. Nearest is appropriate
  for categorical labels, not automatically for integer-valued intensity images.
- Boolean/complex/extended-precision floating intensity arrays are not supported.

There is no public compute-precision or compiler switch. Float32 fitting and
float32-only NCC failed the existing CPU/CUDA agreement gates; see
[the dtype validation report](research/performance/DTYPE.md).

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

### Using ConfUSIus VoxelData directly

There is now one index implementation: ConfUSIus's. No conversion is needed:

```python
import confusius as cf
import voxreg

pwd = cf.load(path).compute()
result = voxreg.register_rigid(pwd, device="cuda", batch_size=120)
```

Inputs and registered/valid outputs use the same geometry class, including
ConfUSIus's world-coordinate lookup. Voxreg still enforces its registration subset:
NumPy-backed arrays, integer native voxel indices, uniform spatial steps, mm,
identical grids and a single slice. It does not modify source values or geometry.
Start with `pwd.isel(time=slice(0, 120))` before processing a long recording.
The default device is CPU; select CUDA explicitly when available.

`voxreg.create_voxeldata(..., coords=...)` remains a thin compatibility constructor
that attaches ConfUSIus's index; it does not implement another index/transform.
Use `cf.create_voxeldata` for the full acquisition-metadata constructor. Existing
floating native-coordinate arrays must use integer voxel indices; no rounding or
coordinate rebasing is performed automatically. Restart an existing Python kernel
after `uv sync` to discard the previous duplicated index implementation.

### CPU and CUDA progress

**Rich progress is enabled by default** on both CPU and CUDA, reusing
ConfUSIus's progress-bar style: percentage, frame counts, elapsed time and ETA.
No callback is needed:

```python
result = voxreg.register_rigid(pwd, device="cuda", batch_size=120)
# Quiet execution: progress=False (or None).
result = voxreg.register_rigid(pwd, device="cpu", progress=False)
```

A callback overrides the built-in display and receives
`(completed_frames, total_frames)` after each completed batch:

```python
def report_progress(completed, total):
    print(f"Registered {completed:,}/{total:,} frames")

result = voxreg.register_rigid(pwd, progress=report_progress)
```

Progress counts processed frames, including reference/failed frames, not optimizer
updates. Updates happen after the existing CPU downloads, with no additional CUDA
synchronization. Preparation and final output assembly do not emit events; smaller
batches give more frequent updates but may reduce throughput. Reporting updates
are outside stage timings; callback exceptions propagate and the built-in display
is closed on errors. Timed/profiling harnesses explicitly use `progress=False`.
No extra dependency or per-iteration GPU polling is added.

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
its original public registration functions. ConfUSIus is also installed into
voxreg for shared VoxelData; the reference still runs a separate numerical pipeline. With the neighboring checkout layout used during development:

```bash
# Run with ConfUSIus's interpreter; quality export only, no timing repetitions.
uv run --directory ../../confusius-tools/confusius python \
    "$PWD/examples/freeze_itk_reference.py" "$INPUT" \
    "$PWD/artifacts/notebook/itk_reference.npz"

# Back in voxreg's environment: registration still uses voxreg's PyTorch engine.
uv run --extra examples python examples/notebook_motion_correction.py \
    "$INPUT" --device cuda --output artifacts/notebook \
    --reference artifacts/notebook/itk_reference.npz
```

The frozen reference records source hashes, resolved ITK registration settings,
input arrays/coordinates, pull matrices and original-power registered outputs.
The comparison uses one independent SciPy bilinear sampler on dB inputs and
**common valid support**, not each implementation's internal optimizer score.
Both outputs retain the source's decoded intensity dtype. Voxreg uses float32
working fitting images with float64 NCC statistics, then resamples the original
float64 intensities for this NIfTI example.

Timing repetitions default to **zero**. Only on an otherwise idle machine, add
`--repeats 3` to the respective scripts. The reference exporter measures the warm
public pipeline at `n_jobs=1` and `--jobs 8`, using its default joblib backend.
Run each configuration separately; do not infer speedups from contested timings.

## Standalone time and memory comparisons

The motion-correction example also writes `artifacts/notebook/input.npz`. This
neutral input artifact lets the ITK/FireANTs benchmark run entirely in voxreg's
Python environment, without calling ConfUSIus's registration pipeline:

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
The latest float32-input benchmark takes **0.219 s on full-batch CUDA** and
**2.128 s on eight-thread CPU**. Fresh FireANTs controls take 0.355 s and 0.963 s
respectively: CUDA is competitive, while CPU remains slower than native float32
FireANTs. These are not precision-
matched solvers or universal speedup guarantees. Comparisons use direct 2D APIs,
not the ConfUSIus pipeline or its earlier frozen-reference optimizer configuration.
See the report for memory, quality, source hashes and the pinned FireANTs archive.

[PERFORMANCE.md](PERFORMANCE.md) preserves the profiler investigation that led to
this rewrite. Float64 physical geometry/state/NCC and independent frame policies
are unchanged; only dense coordinate evaluation was reorganized.

## Current limits

- Single-slice rigid only; no full 3D, affine, deformable or multi-pose registration.
- Identical fixed/moving grids, uniform native coordinates, mm units and finite
  NumPy-backed real input with supported dtype/range; no masks, Dask or multiresolution.
- A frame losing more than 10% overlap is flagged and retains its best valid
  transform. Constant/invalid fitting metrics are diagnosed separately.
- Output fill is each original frame's minimum. Use `result.valid` to exclude
  padding from downstream analyses.
- Per-frame status distinguishes convergence, iteration limit, invalid metric,
  insufficient overlap and the fixed reference. An iteration limit is **not** a
  convergence claim. The real recording currently reaches the 100-step limit.
- Fitting interpolation is float32; physical geometry, optimizer state, NCC
  statistics and world matrices are float64 for stability. Final interpolation
  preserves original image precision as described above. Cross-device transforms
  are numerically, not bitwise, equal; output dtype is not a transform-accuracy guarantee.
- The lazy index builds on xarray's experimental coordinate-transform index API.

No compiled engine, affine/deformable extension or compatibility layer is included.
Performance claims are limited to the documented single-recording comparisons.
