# Registration tests

Run from the repository root with `uv run pytest`. CUDA cases run when CUDA is
available; a CPU-only run does not qualify CUDA. Tests use one PyTorch CPU thread
and restore the previous setting after the session.

## Shared data and oracles

Reusable image data belongs in the package-wide `conftest.py`:

- `recording_factory`: asymmetric 2D movies with known physical rigid motion,
  including oblique, reversed and strided coordinates.
- `linear_pair_factory`: translation, genuinely rotating rigid, or scale/shear
  affine pairs in 2D/3D, with unequal oblique grids and an off-origin center.
- `slice_volume_pair_factory`: native singleton/volume pairs with a finite slice
  voxel cell. Truth is built with NumPy/SciPy, not voxreg's parameter decoder.

Factories are function-scoped and create fresh arrays/RNGs. Do not cache mutable
VoxelData, set global NumPy seeds, import fixtures from another test module, or
add another phantom generator when one of these already covers the contract.
Specialized local data may remain where it tests a genuinely different case.

`oracles.py` holds shared pure construction/interpolation helpers. Its SciPy
mixed-grid pull uses float64 interpolation and half-open finite-voxel support.
Existing native ITK/SimpleITK point and metric oracles remain independent checks.

## What deserves a test

Assert a meaningful contract: known physical point error, model constraints,
independent interpolation, gradients, frame failure isolation, original dtype,
fixed geometry, metadata, input immutability or CPU/CUDA agreement. Parameterize
an existing contract before adding another test body. Do not use voxreg's own
transform decoder to manufacture the expected transform for a recovery test.
Mock only the boundary/control behavior being tested, not the optimizer in an
end-to-end recovery or output-contract check.

Sibling syntx, ANTs, FireANTs and ITK tests are useful references for known-pose
recovery, distinct transform families, dimensionality and point/center semantics.
Their optimizer trajectories, image baselines and tolerances are not our oracle.
No new runtime dependency or external data download is needed for these fixtures.
Keep cached real examples and heavyweight comparisons separate from unit tests.
