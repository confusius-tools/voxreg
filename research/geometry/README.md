# Independent physical grids and initial 3D rigid qualification

This extends the slice engine at `ed9cb21`; it is **numerical qualification, not
an ITK/FireANTs 3D performance comparison**. Exact settings, source hashes and
measurements are in [results.json](results.json).

## Geometry and API

The fixed grid owns the metric domain and output spatial coordinates. The moving
grid owns the sampled values, support and interpolation normalization:

```
moving_array_position = inverse(A_moving) @ T @ A_fixed @ fixed_array_position
```

Both affines come from `position_to_world`, including native coordinate offsets,
strides and reversals. Known grid differences do not become fitted motion:
identity **world** motion can require non-identity voxel resampling.

Tiny physical matrices are composed before broadcast coordinate evaluation. Dense
world-coordinate GEMMs are not used. The mapping is expressed relative to the
fixed center and the difference between grid origins, avoiding subtraction of a
large common world origin at every sample.

- `register_rigid(moving, fixed)` accepts spatial-only `(k, j, i)` inputs. Outputs
  are spatial-only, with a `(4, 4)` world pull matrix and scalar diagnostics.
- `(time, k, j, i)` uses the same optimizer and final-resampling path in batches.
  Omitting fixed uses the first frame, copied exactly. Otherwise fixed has an
  independent shape/spacing/origin/orientation. Source time labels and time-indexed
  auxiliary coordinates are retained; spatial coordinates come from fixed.
- Full volumes require all axes >= 2 and fit six parameters: a world z/y/x
  rotation vector, followed by world z/y/x center displacement in mm. The vector
  is axis-angle with conventional right-handed XYZ handedness, **not Euler angles**.
- Coplanar `k=1` slice pairs fit fixed-plane rotation and intrinsic i/j translation.
  Moving and fixed in-plane grids may differ. The already-qualified identical-grid
  slice kernel is retained to preserve its arithmetic exactly. This is kernel
  dispatch, not a second optimizer or registration pipeline.
- Singletons in other configurations, non-coplanar slice pairs and slice-to-volume
  registration are rejected. Geometry-based plane initialization is not inferred.

Translations are displacements of the fixed physical center, not homogeneous
matrix offsets. For a centered rotation `R` and displacement `t`, the exported
matrix offset is `t - (R - I) @ center`. Volume parameter scales use minimum
fixed physical spacing and per-world-axis RMS lever arms. Off-axis variances are
summed directly to avoid cancellation for elongated grids. This is not ITK's
physical-shift scales estimator.

## Numerical evidence

`tests/test_grids.py` adds 31 tests in this CUDA-enabled environment:

- Identity and rigid sampling versus independent SciPy interpolation on unequal
  anisotropic, oblique, sheared grids, with offset, strided and reversed native
  coordinates; linear and nearest interpolation on CPU/CUDA.
- Float64 gradient checks including zero rotation; the sampler check avoids
  interpolation knots, where a unique derivative does not exist. Rotation-vector
  values/handedness are checked separately against SciPy and differentiated at zero.
- Fixed-grid output geometry, original float64/integer intensities, rounding,
  input immutability and canonical ConfUSIus validation.
- Analytically generated asymmetric 3D phantoms with known physical motion, rather
  than generating the moving input with the sampler under test.
- Batch independence, CPU/CUDA pointwise agreement, reference identity and
  float64 detail varying only through k. Common large origins and highly
  elongated-grid scaling are covered.
- Explicit unsupported-plane errors, zero overlap, best-valid identity fallback,
  invalid objective diagnostics and original-minimum output fill.

The single analytic volume pair uses fixed `(17, 19, 21)` and moving
`(29, 31, 33)` arrays with different orientations, sheared anisotropic affines,
and native-coordinate strides/reversals. Minimum fixed spacing is 0.4 mm.
The true rotation vector is `[0.02, -0.015, 0.01]` radians in z/y/x and the center
translation is `[0.11, -0.08, 0.09]` mm. Fitting is unscaled global NCC, with a
180-update budget and the existing 50/1e-6 convergence policy.

| Check | CPU | CUDA |
|---|---:|---:|
| RMS physical truth error / minimum spacing | 0.010373 pixels | 0.010373 pixels |
| Maximum pointwise truth error / minimum spacing | 0.021529 pixels | 0.021529 pixels |
| Independent SciPy NCC² on common support | 0.995404515 | 0.995404515 |
| Updates / status | 81 / converged | 81 / converged |

The unchanged truth gate is RMS < 0.25 minimum-spacing pixels. Maximum CPU/CUDA
pointwise disagreement is **1.07e-6 pixels** (rounded upward), below the unchanged
0.001-pixel comparison gate. Common valid support across identity, truth and both
estimates is 99.602%; independent NCC² changes from 0.977054935 at identity to
0.995404515. The known transform scores 0.995401110: sampling/interpolation bias
can favor a slightly different transform, so NCC alone is not truth.

A separate fresh-process comparison against the parent commit uses the existing
float32 rat75 120-frame neutral input: CPU batch32 and CUDA batch120. Registered
intensities, validity, transforms, objective, valid fraction, update counts,
statuses and all slice motion parameters are **bitwise identical per device**.
The input hash and individual equality checks are stored in results.json. Timing
attrs are deliberately excluded. Raw arrays/logs are under the ignored
`artifacts/independent-grids/` directory; this is not a retiming of old benchmarks.

## Reproduce the acceptance checks

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run pytest -q tests/test_grids.py

# Include the existing slice, convergence, dtype and backend-regression gates.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run pytest -q
```

The case generator and independent coordinate evaluator are in
`tests/test_grids.py::volume_phantom` and `reference_coordinates`. The JSON's
physical truth matrix, source hashes and settings identify the numerical case;
its NCC comparisons use original float32 inputs converted to float64 for SciPy
linear interpolation and the intersection of all four support masks.

## Deliberate remaining limits

Initialization remains identity in world space. No initial-transform parameter,
center alignment, multiresolution, masks or alternate metrics were added. The
existing minimum 90% fixed-domain overlap is still a conservative motion-correction
policy: a smaller moving field of view can be diagnosed as insufficient overlap,
not searched into alignment. Large-angle capture range is not established.

The existing convergence threshold was calibrated on slices, not general 3D;
one small-motion synthetic volume is not broad qualification. Real 3D data,
representative volume sizes, ITK/FireANTs single-pair latency and separate peak
memory measurements are the next acceptance boundary. For 3D series use small
batches; the default batch32 was chosen for slices, not memory-qualified volumes. No such performance or
biological-accuracy claims follow from this change.
