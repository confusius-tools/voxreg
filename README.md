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
a single image pair, pass `voxreg.register(moving, fixed)`. Both `register` and
`resample` default to `device="auto"`: CUDA when available, otherwise CPU. Use
`device="cpu"` or `device="cuda"` to select explicitly. CUDA execution errors
propagate, including out-of-memory errors; there is no CPU retry. Results remain
NumPy-backed on CPU for eager calls; Dask inputs retain lazy outputs.
Eager registration records the device in `result.diagnostics.attrs["device"]`;
lazy diagnostics expose worker-resolved `execution_device` per frame.

## `voxreg.register`

```python
result = voxreg.register(
    moving,
    fixed=None,
    transform="rigid",
    moving_intensity_scaling=None,
    fixed_intensity_scaling=None,
    device="auto",
    return_transforms=False,
)
```

| Argument | Purpose |
| --- | --- |
| `moving`, `fixed` | VoxelData with physical length units: eager `(k, j, i)` pairs or NumPy/Dask `(time, k, j, i)` recordings. `fixed` is spatial-only and required for a single image. |
| `transform` | `"translation"`, `"rigid"` (default), `"affine"`, `"syn"`, a `Stage`, or a sequence of model names/`Stage` specifications. Slices use `k=1`; volumes have at least two samples on each axis. |
| `initialization` | Optional `"center_geometry"`, `"center_moments"`, `"align_planes"` or a fixed→moving world matrix. Defaults to identity. |
| `fixed_mask`, `moving_mask` | Optional boolean ROI VoxelData in their own physical grids. Fixed mask is spatial-only; moving mask can be shared or per-frame with matching time labels. Masks restrict fitting at every stage and level. |
| `moving_intensity_scaling`, `fixed_intensity_scaling` | Independent fitting-only transforms: `None` (default), `"log1p"`, `"sqrt"` or a positive finite power exponent. Fixed scaling applies only to an explicit fixed image. |
| `moving_winsorize_percentiles`, `fixed_winsorize_percentiles` | Optional lower/upper percentile pairs on the 0–100 scale, e.g. `(1, 99)`. Disabled by default; clip each image independently before intensity transforms. |
| `metric` | Default `None` retains global squared NCC for linear models and local squared NCC for SyN. Explicit `"ncc"` selects global NCC, `"local_ncc"` selects SyN local NCC, and `"mi"` selects mutual information. |
| `mi_bins`, `mi_sigma` | MI histogram bins (default `32`, range `2`–`256`) and Gaussian standard deviation in bin widths (default `0.5`, range `0.1`–`10`). |
| `resample_interpolation` | Final resampling of original intensities: `"linear"` (default), `"nearest"` or cubic `"bspline"`. Fitting always uses linear interpolation. |
| `batch_size` | Eager fitting batch limit, chosen automatically inside workers for lazy calls. Lazy task sizes follow temporal source chunks independently; an explicit limit can be smaller than a chunk. |
| `max_iterations` | Per-frame update limit at each level: an integer (default `100`) or one integer per level. Inspect diagnostics rather than assuming convergence. |
| `shrink_factors`, `smoothing_sigmas` | Explicit coarse→fine downsampling factors and Gaussian image smoothing standard deviations in mm. Defaults `(1,)` and `(0.0,)` retain native-resolution fitting. |
| `syn_cc_radius` | SyN local NCC radius in voxels; default `2` gives a rectangular window of width `5`. |
| `syn_update_sigma`, `syn_total_sigma` | SyN Gaussian smoothing standard deviations in voxels, before update composition and on the accumulated field respectively; defaults `1.5` and `0.5`. Zero disables the corresponding smoothing. |
| `device`, `progress` | Device defaults to `"auto"` (CUDA when available, otherwise CPU); explicit `"cpu"`/`"cuda"` are supported. Progress defaults to `"auto"`, or use `False` to silence it. |
| `return_transforms` | Opt into forward and inverse maps: linear matrices or SyN displacement fields. Dense fields can be much larger than the images. |

The result contains **`registered`** intensities and a boolean **`valid`** support
mask on the fixed grid, plus **`diagnostics`** with per-frame status and objective.
Inputs are not modified; output retains the moving dtype and time labels.
Use `valid` to exclude out-of-bounds padding from analysis.

The `syn_` options apply only to `transform="syn"`; nondefault values with linear
models raise an error. In a pipeline containing SyN, shared `syn_` options apply
only to its SyN stage. Smoothing values are standard deviations, not ANTs field
variances. Local CC uses a uniformly weighted rectangular window.

With `return_transforms=True`, `forward_transforms` are physical pulls:
**`moving_world = matrix @ fixed_world`**, in **z/y/x, mm**. Linear models return
matrices and their inverses; SyN returns complete forward displacements on the
fixed grid and inverse displacements on the moving grid. Both attributes are
`None` by default. `transforms` remains a compatibility alias for `forward_transforms`.

Pass original linear intensities when using `center_moments`; logarithms and other
nonlinear preprocessing change its centroid weights. Noncoplanar slice pairs
need compatible initialization. SyN does not support mixed slice/volume inputs.
See [lazy execution](#lazy-recordings-and-resampling) for Dask contracts.

For all options, including optimizer controls and callbacks, use
`help(voxreg.register)` or read the [API docstring](src/voxreg/registration.py).

## Fitting winsorization

Optionally clip fitting outliers without modifying input images or output
intensities:

```python
result = voxreg.register(
    moving, fixed,
    moving_winsorize_percentiles=(1, 99),
    fixed_winsorize_percentiles=(1, 99),
)
```

Both controls default independently to `None` (disabled). Supply two finite,
non-boolean bounds with `0 <= lower < upper <= 100`. Bounds are computed with
`numpy.percentile(method="linear")` over each complete native image/frame,
including zeros and values outside masks. Clip signed intensities **before**
log1p/square root/power transforms, metric ranges and pyramid smoothing. Clipping
is not a magnitude operation and does not normalize across time chunks.

The implicit first-frame reference uses moving clipping; non-`None` fixed bounds
require an explicit fixed image. Settings apply to all stages, each starting
from originals. Native moment weights, output dtype/intensities and geometric
support remain unchanged. Constant clipped moving frames use the usual
invalid-metric status; constant clipped fixed images raise an error. Nonfinite
percentile interpolation bounds raise a descriptive error. No percentile preset
is enabled or anatomically qualified; clipping may suppress genuine structures.

## Fitting intensity scaling

Both scaling arguments default to `None`: no nonlinear transform, preserving
signed values before numerical metric conditioning. The old shared
`intensity_scaling` argument and strings `"none"`/`"db"` are removed. An omitted
fixed transform does not inherit the moving transform.

`"log1p"` computes `ln(1 + abs(x))`, preserving zeros without a floor or global
reference. Its compression depends on the absolute intensity scale: values much
smaller than 1 barely change, while large values behave like natural log.
`"sqrt"`/numeric powers compute `abs(x)**exponent`, following ConfUSIus's magnitude
convention. Exponents must be positive and finite; booleans are invalid. Signed
inputs are allowed but magnitude transforms discard their signs. All-zero
recordings and constant fixed fitting images remain invalid.

Transforms are evaluated in float64 before metric-range preparation/pyramids and
float32 fitting. Power overflow raises a descriptive error; underflow rounds to
zero and may erase contrast. Final resampling always uses original intensities
and dtype, and moment initialization uses original nonnegative native weights.
Masks restrict fitting samples, not transforms or their reductions.

There is no recording-wide intensity normalization. NCC bounds each image/frame
independently by its magnitude maximum; double/integer or transformed inputs also
retain per-image offset/contrast conditioning before float32 conversion. MI uses
independent per-image min/max ranges. A bright unrelated frame cannot change
another frame's conditioning, and temporal chunks need no shared intensity scan.
Masks do not limit these per-image reductions.

The implicit reference reuses the transformed first moving frame and its output
is preserved exactly. Non-`None` fixed scaling without an explicit fixed image is
an error. Stages transform originals afresh, never a previous stage's fitting
image. Historical benchmark sources/artifacts retain their original API/settings;
new comparisons must disclose the log1p and per-image-conditioning changes.

## Physical units

World coordinates may use ConfUSIus length units: `m`, `cm`, `mm`, `um`/`µm`
or `nm`. Fixed images, moving images and masks may each use different units.
Voxreg converts their geometry to **mm internally**, without changing inputs.
Image outputs retain the reference/fixed grid's native coordinates and length
unit; intensity units and dtype still come from the moving image.

Exported matrices and displacement components, physical motion diagnostics,
bare matrix seeds and `smoothing_sigmas` remain in **mm**. CC radii and SyN
deformation smoothing remain in **voxels**. Labelled matrix seeds and resampling
transforms may declare other supported length units through `spatial_units`;
only matrix translations are scaled, since linear coefficients are dimensionless.
Matrices map world coordinates expressed in one common declared length unit,
not separately in each input image's native unit.
Displacement fields must declare `spatial_units` for their values; their grid's
coordinate unit is independent. A field must match the physical reference grid
without implicit regridding, even when the grid units differ.

`position_to_world(data)` returns the array-position affine in mm. Request a
different output unit explicitly, for example `position_to_world(data, units="um")`.
These are spatial coordinate units, separate from intensity `attrs["units"]`.
Unknown or non-length units are rejected rather than assigned an assumed scale.

## Mutual information

Use MI for an explicit alternative to intensity correlation, for example when
the images have different contrast:

```python
result = voxreg.register(
    moving,
    fixed,
    transform=(voxreg.Stage("rigid", metric="mi"), "syn"),
    moving_intensity_scaling=None,
    fixed_intensity_scaling=None,
)
```

This fits rigid with MI and SyN with its default local NCC. Set shared
`metric="mi"` to use MI at every stage, or override `metric`, `mi_bins` and
`mi_sigma` in individual `Stage` specifications. Explicit `"local_ncc"` is
available only for SyN and uses `syn_cc_radius`.

MI uses Gaussian Parzen histograms with unit-sum memberships per sample. Each
native fitting image is independently mapped from its min/max to `[0, 1]` after
each image's intensity transform, before any pyramid smoothing. These ranges stay fixed during
fitting and include intensities outside ROIs; outliers can compress useful contrast.
Only supported ROI samples contribute to the joint histogram. Both SyN halfway
images are differentiated. Spatial chunking and recomputation during backward
bound bin-weight workspace; this does not bound full image/output storage.

`diagnostics.objective` and metric callbacks report **negative MI in nats**;
lower is better. The diagnostics record bins, kernel width and range policy.
SyN's `ncc_squared` remains a separate endpoint NCC check, regardless of the
fitting metric. MI values are not comparable to NCC or across stages/levels.
The estimator uses Gaussian kernels; it does not reproduce ANTs' Mattes values.
Convergence settings and useful bins/bandwidths still need real-data qualification.

## ROI masks

```python
result = voxreg.register(
    moving, fixed,
    fixed_mask=fixed_brain_mask,
    moving_mask=moving_brain_mask,
    transform=("rigid", "affine", "syn"),
)
```

Masks must be NumPy- or Dask-backed boolean VoxelData with supported physical length units.
Convert binary integer masks explicitly with `mask.astype(bool)`. Their shape,
spacing, origin and orientation may differ from the images. The fixed mask is
spatial-only; a spatial moving mask broadcasts across recording frames, while
per-frame masks must have exactly matching time labels. Either mask can be omitted.

Following ITK's physical-mask model, each candidate pull samples the original
mask's binary voxel cells, with half-index ties rounded upwards. Both supplied
ROIs and geometric image support must include a point for it to contribute.
For SyN, both masks follow their own halfway maps, including the affine seed on
the moving side. Masks are evaluated directly at each pyramid level and are never
Gaussian-smoothed or repeatedly resized into soft weights. Local NCC excludes
masked neighbours from its statistics as well as excluding masked centres.

Image interpolation, image-pyramid smoothing, intensity scaling and automatic
initialization keep their existing behaviour. Pixels outside an ROI can therefore
influence image values near its boundary through interpolation or smoothing.
Masks do not zero or crop the input images, clip the output, or force deformation
to vanish outside the ROI. Shared masks apply to every stage and level; this
differs from ANTsPy's default final-stage-only mask policy.

`result.valid` and `diagnostics.valid_fraction` still describe geometric output
support. With masks, `diagnostics.roi_valid_fraction` separately reports the
fraction of native fixed-grid centres inside both ROIs and image support at the
returned endpoint map. For SyN this is an endpoint diagnostic, separate from its
halfway fitting objective. Empty/tiny intersections can fail explicitly with
insufficient overlap or invalid statistics; coverage is not registration confidence.

## Staged registration

Pass a sequence for a pipeline using the existing model defaults:

```python
result = voxreg.register(
    recording,
    transform=("rigid", "affine"),
    shrink_factors=(2, 1),
    smoothing_sigmas=(0.1, 0.0),
)
```

Shared fitting arguments apply to each stage. Use `Stage` to override only the
options that differ:

```python
result = voxreg.register(
    moving, fixed,
    transform=(
        "rigid",
        "affine",
        voxreg.Stage("syn", max_iterations=(50, 25)),
    ),
    shrink_factors=(2, 1),
    smoothing_sigmas=(0.1, 0.0),  # image sigma in mm, chosen for these images
    max_iterations=(100, 50),
    return_transforms=True,
)
```

Settings follow **model defaults → shared arguments → explicit stage overrides**.
`Stage` accepts iteration, pyramid, learning-rate, convergence and SyN controls;
device, batching, intensity scaling, output interpolation and callbacks are shared.
Unset stage options inherit the shared value. There are no named workflow presets.

Stages must follow translation→rigid→affine→SyN, skipping models as needed;
repeated/reversed stages are rejected and SyN must be last. Initialization runs
once, and complete transforms carry forward with a fixed physical pivot. Every
stage fits the original images. Original intensities are sampled once for final
output; exported transforms describe the complete final mapping.

Each recording frame has its own stage outcomes. Convergence and reaching an
iteration limit permit continuation. Registration failure stops that frame's
pipeline and retains its best valid transform; later stages mark it `skipped`.
Other frames continue, and an implicit first-frame reference is preserved exactly.

`result.stage_diagnostics` is a tuple with one dataset per stage, including that
stage's pyramid diagnostics and zero-based `stage`/`transform` attributes.
`result.diagnostics` describes the final model, preserves any earlier failure
status and sums updates across all stages. Its level variables describe only the
final stage; consult `stage_diagnostics` for earlier levels. Phase timings sum
stage timings; `stage_overhead_seconds` records validation/orchestration overhead.
Convergence windows restart at every stage and level. A summary `converged`
status describes the final stage/level; it does not assert that every earlier
stage converged. Objectives from different metrics or levels are not combined.

Progress callbacks count completed stage/frame work units, with a total of
`number_of_frames * number_of_stages`, including skipped frames. Metric callbacks
retain their three arguments; update clocks reset per stage and level, and losses
from different stages/levels must not be compared. Reports run stage by stage.

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

## Final image interpolation

Use `resample_interpolation="bspline"` in `register` for cubic B-spline
resampling of the original intensities, in either 2D or 3D:

```python
result = voxreg.register(moving, fixed, resample_interpolation="bspline")
```

Fitting stays linear, and the default final interpolation is also linear.
Offline `resample` uses its `interpolation` argument and accepts the same three
methods.

Cubic interpolation prepares spline coefficients once per source batch, then
samples the original image once. This coefficient prefilter is not Gaussian
smoothing. Coefficients and evaluation use float64 on the selected CPU/CUDA
device. Whole-sample mirror extension supplies stencil neighbours at boundaries;
the existing geometric validity mask and per-frame minimum outside fill remain
unchanged. A wider stencil does not shrink valid support.

Floating outputs retain spline overshoot, including negative values from
nonnegative inputs; values outside the original floating dtype's finite range
raise an error. Integer outputs use nearest-even rounding and dtype clipping.
Cubic half-integer ties within float64 roundoff are stabilized before rounding.
Outputs retain the moving image dtype. Cubic uses additional coefficient memory
and is not performance-qualified yet. Use nearest for categorical labels;
label-aware interpolation remains deferred.

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

## Lazy recordings and resampling

Dask-backed recordings, fixed images, masks or seeds make `register` return lazy
images, validity, diagnostics and optional maps. Spatial registration pairs remain
eager: compute those images first. `resample` also supports lazy spatial images,
shared/per-frame matrices and dense fields, including eager/lazy combinations.
A lazy reference image alone does not force lazy resampling: its intensities are
unused. Geometry, option types and time labels are checked without reading pixels;
finite values, integer ranges, moments and numeric transform checks occur in tasks
and may fail at compute/write time.

```python
import dask
import xarray as xr

result = voxreg.register(
    recording.chunk(time=16), fixed,
    device="auto", return_transforms=True,
)
images, valid, diagnostics = dask.compute(
    result.registered, result.valid, result.diagnostics,
)
```

Each task handles one temporal source chunk and runs its full stage/pyramid
pipeline eagerly. The worker chooses a CPU/GPU fitting batch size using the
existing memory policy. For example, a 16-frame input chunk can be fitted in
batches of 4; `batch_size=4` forces that fitting limit without rechunking input.
The fixed/reference image is a shared dependency;
per-image conditioning and winsorization do not depend on time chunks. Optional
maps and every stage/level diagnostic project the same fit. Compute related
outputs **jointly** to avoid repeating registration; separate `.compute()` or
writes rerun it unless you persist results. `.compute()` of a whole image still
requires RAM for that complete output; incremental writes do not.

Task boundaries follow moving temporal chunks, or time-chunked maps/masks/seeds
when the moving input has none. Spatial-image broadcasting follows map chunks,
not a synthetic full-recording chunk. Without any temporal Dask source, task sizes
fall back to an explicit `batch_size` or one frame. Auxiliary time chunks larger
than the largest primary task chunk are rejected; align them at the data source.
Spatial chunks are assembled into complete volumes within each task. **Rechunking
cannot avoid reading a larger underlying storage/loader block**. The complete
source chunk and task outputs must fit worker RAM; only the fitting batch and its
working state must fit VRAM. Input dependencies can be prefetched by the scheduler; bound
worker threads, workers per host and storage blocks as well as task size. Dense
SyN fields remain per-task, not accumulated in one global eager result.

Synchronous, threaded, multiprocessing and distributed Dask execution are supported.
Device auto selection occurs in workers; there is no CPU retry after CUDA errors.
Fitting/resampling concurrency is conservatively one task per host CPU or physical
GPU. Local schedulers coordinate via file leases under `~/.cache/voxreg/execution`;
distributed workers use scheduler semaphore leases. Use one worker per GPU and
bounded worker memory; separate schedulers/jobs are not a shared resource manager.
CPU task concurrency is deliberately conservative rather than an automatic tuner.

Metric/progress callbacks run in tasks, with global frame indices/work offsets;
distributed callbacks execute on workers, not the client. They must be serializable.
Execution order can make progress offsets nonmonotonic; retries/recomputation repeat
notifications. Default `progress="auto"` is silent for lazy graphs: use Dask progress
tools, explicit task progress or callbacks. `diagnostics.execution_device`,
`execution_task_size`, `execution_batch_size` and `execution_batch_start` record
actual execution. The last field is the task's global start; task size counts
loaded frames while batch size reports each frame's fitting group, including a
smaller final group. Lazy
`batch_*_seconds` timings exclude input loading/scheduler waits and repeat per frame;
deduplicate by batch start rather than summing them over frames. Data-dependent
centers are lazy `rotation_center_zyx_mm` variables instead of eager attributes.
All-zero moving task batches against a nonconstant fixed image produce per-frame
invalid-metric statuses rather than aborting otherwise independent frames.

For incremental Zarr writes, store affine metadata and restore VoxelData with
`confusius.io.load`; plain `xarray.open_zarr` does not rebuild custom indexes:

```python
def storage_view(image):
    index = image.xindexes["z"]
    return image.drop_vars(["z", "y", "x"]).assign_attrs(
        voxel_to_world=index.voxel_to_world.tolist(),
        voxel_to_world_units=index.units,
    )

tree = xr.DataTree.from_dict({
    "images": storage_view(result.registered).to_dataset(name="image"),
    "validity": storage_view(result.valid).to_dataset(name="valid"),
    "diagnostics": result.diagnostics,
})
tree.to_zarr("registration.zarr", compute=False).compute()
# confusius.io.load("registration.zarr", group="images", variable="image").
```

Use **one DataTree/Dataset write** for related outputs. Separate
`to_zarr(compute=False)` calls can independently optimize/clone the fitting graphs,
even when later passed together to `dask.compute`; that can repeat registration.
Add map/stage datasets as sibling DataTree groups to retain different spatial grids.

Small CPU/CUDA and distributed tests cover bounded source loading and Zarr round
trips. They are not a measured full-recording peak-RAM/VRAM qualification on data
larger than physical RAM; benchmark that workload before production deployment.
Complex IQ and distributed fitting of one spatial volume remain unsupported.

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
