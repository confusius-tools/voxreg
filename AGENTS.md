# voxreg Agent Guidelines

This file guides AI coding agents working in this repository.

## What is voxreg

voxreg is a Python library for batched physical-space image registration of VoxelData
arrays, including translation, rigid, affine, and symmetric normalization (SyN)
registration, resampling, and motion correction. Dask recordings execute as bounded
eager time batches. Registration fitting runs in PyTorch on
CPU or CUDA; spatial data and geometry use VoxelData.

This is a **research prototype**. Anatomical registration quality is not yet validated.
Backward compatibility is not a concern — feel free to make breaking API changes when
they improve the design.

## Data Model: VoxelData

**VoxelData** is a canonical `xarray.DataArray` model for spatially referenced voxel
data. voxreg accepts NumPy-backed VoxelData images with dimensions `(k, j, i)` and
NumPy- or Dask-backed recordings with dimensions `(time, k, j, i)`. Lazy resampling
also supports spatial images and labelled maps. A VoxelData array has native voxel indices
`k`/`j`/`i` last and a `VoxelToWorldIndex` attached to `z`/`y`/`x` that derives world
coordinates from voxel indices and the voxel-to-world affine. Geometry and transform
values in voxreg use **z/y/x, mm**.

Registration inputs, masks, resampling inputs, and spatial outputs must follow the
VoxelData model. Use voxreg `validate_voxeldata`/`ensure_voxeldata` to validate
VoxelData and `create_voxeldata` to construct it. Do not add plain-coordinate fallbacks
or code that silently accepts a spatial DataArray without its `VoxelToWorldIndex`.
Invalid spatial inputs should raise a descriptive exception.

VoxelData images must have supported physical length units (m, cm, mm, um/µm or nm). Internal geometry is normalized to
mm while spatial outputs retain reference-grid coordinate units. Registration supports spatial images `(k, j, i)` and recordings
`(time, k, j, i)`; masks follow the shapes and time-label rules documented by
`voxreg.register`. Output images and masks remain VoxelData on their documented output
grids. Keep geometry conversions at explicit boundaries and preserve coordinate, affine,
and time-label semantics. Lazy tasks use complete eager volumes, per-image
conditioning and task-time callbacks. Tasks follow temporal source chunks, which
must fit worker RAM; fitting batches are selected independently inside workers. Preserve shared computation across lazy images, masks, maps and diagnostics.

## Commands

Uses [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just).

```bash
uv sync              # install dependencies
just test            # run tests
just test-verbose    # run tests with verbose output
just pre-commit      # ruff check/format, ty, codespell, numpydoc-validation
```

Run a single test: `uv run pytest path/to/test_file.py::TestClass::test_method`.

Benchmark recipes are `just benchmark-resample OUTPUT ...`,
`just benchmark-register OUTPUT ...`, and
`just benchmark-report OUTPUT RESULT...`. The first two install/reuse an isolated
Python 3.13 environment with ANTsPy for reference comparisons.
`just benchmark-lazy OUTPUT --operation register|resample ...` measures streaming
read/compute/Zarr writes without ANTs; large files require explicit `--size-gib`. Research scripts,
notebooks, and benchmark notes live in `research/` and `examples/`.

## Code Architecture

```
src/voxreg/
├── convergence.py  # per-frame optimizer convergence tracking
├── fields.py       # displacement-field sampling, composition, smoothing, inversion
├── geometry.py     # physical geometry validation and position conversion
├── masks.py        # physical ROI validation and sampling
├── metrics.py      # global/local NCC and mutual information
├── pyramid.py      # physical multi-resolution image and field levels
├── resampling.py   # public physical-space resampling
├── registration.py # public registration API, validation, preparation and dispatch
├── lazy.py         # shared bounded Dask tasks, projections and concurrency limits
├── preparation.py  # shared settings, prepared inputs, conditioning and batch policy
├── results.py      # result containers, shared statuses and native output assembly
├── linear.py       # translation/rigid/affine fitting and pyramid refinement
├── interpolation.py # final-image cubic coefficients, evaluation, dtype restoration
├── sampling.py     # physical grids, interpolation, model sampling
├── stages.py       # staged registration specifications and orchestration
├── syn.py          # symmetric nonlinear registration
└── transforms.py   # linear transform parameterization and diagnostics
```

### Module organization

- Keep a module focused on one registration concern. Put a helper shared by multiple
  files in an appropriate topic module; do not collect unrelated helpers in one
  general-purpose module.
- Module-private helpers belong in the module that owns their behavior. Do not import
  a `_name` across module boundaries. If another module needs that behavior, either
  inline it, expose it as a public function in its owning module, or move it into a
  clearly named shared module.
- Keep public API exports in `voxreg/__init__.py` intentional and minimal. Internal
  optimizer, tensor, and geometry details should not become public API accidentally.
- Keep benchmark/reference code in `examples/` or `research/`; production registration
  code belongs in `src/voxreg/`.

## Coding Conventions

- **Imports**: absolute only (`from voxreg.metrics import compute_global_ncc`); group
  stdlib/third-party/local; use `TYPE_CHECKING` for type-only imports.
- **Naming**: `snake_case` functions/variables (prefer imperative verb phrases, e.g.
  `get_source_data`, not noun phrases), `PascalCase` classes, `UPPER_CASE`
  constants, leading `_` for private.
- **Types**: comprehensive hints incl. `numpy.typing.NDArray`; `Literal` for string
  literals; `TypedDict` for structured dicts; `TypeAlias` for complex types; `py.typed`
  is enabled.
- **Errors**: specific exceptions (`ValueError`, `TypeError`, `FileNotFoundError`);
  `warnings.warn()` for non-critical issues; validate inputs early with descriptive
  messages.
- **Docstrings**: NumPy format for *all* functions/methods, public and private
  (private helpers still need full Parameters/Returns/Raises).
  - Optional params: `arg : type, optional` (default `None`) or
    `arg : type, default: value` — never `arg : type or None, default: None`.
  - Fallback behavior: "If not provided, ..." — never "If `None`, ...".
  - Booleans: start the description with "Whether to ..." — never "If `True`/`False`, ...".
  - Multiple return values: one `name : type` block per value in `Returns` — never
    `tuple[type1, type2]` as the documented return type.
  - Inline code: single backticks. Full package names in the `type` line
    (`xarray.DataArray`); no `xarray.` prefix/backticks in prose (`DataArray`).
  - Array shapes: `(X, Y, Z) numpy.ndarray` / `(X, Y, Z) xarray.DataArray`.
  - Cross-references: use numpydoc cross-reference syntax for public objects included
    in the API docs. Private objects (`_name`) should be referenced as plain
    `` `name` `` rather than linked.
  - Module-level constants get a triple-quoted docstring immediately after them.
- **Comments**:
  1. Comments should not duplicate code.
  2. Good comments do not excuse unclear code.
  3. If you can't write a clear comment, there may be a problem with the code.
  4. Comments should dispel confusion, not cause it.
  5. Explain unidiomatic code in comments.
  6. Provide links to the original source of copied code.
  7. Include links to external references where they will be most helpful.
  8. Add comments when fixing bugs.
  9. Use `TODO:` prefix for incomplete implementations.
  10. All comments end with a period.
  - Code adapted from another project needs a module-level `NOTICE` file reference:
    "Portions of this file are derived from [Project], which is licensed under the
    [License]. See `NOTICE` file for details."

## Test Conventions

- **No useless tests**: must fail if the function returns garbage — avoid tests that
  only check shape preservation or "output differs from input".
- **Test public API only**: don't test `_private` functions directly; they're covered
  via the public functions that use them.
- **No `# pragma: no cover`**, and don't force tests for branches that are unreachable
  due to an upstream library invariant.
- Cover: edge cases (empty inputs, boundaries), error validation (`pytest.raises`),
  and comparisons against reference implementations (scipy, naive implementations)
  where one exists. Reach for property-based tests only when no reference
  implementation exists (mathematical properties: idempotence, commutativity, ...).
- Use fixtures from `conftest.py` before creating new test data;
  `numpy.testing.assert_allclose`/`assert_array_equal`; seeded RNGs; small arrays.
- For registration and resampling, check physical-coordinate behavior, interpolation,
  support masks, transform direction, and diagnostics where relevant. Prefer small
  synthetic VoxelData inputs and independent reference calculations. Use ANTsPy
  comparisons only where the behavior is intended to match its documented semantics.

## Git

Commit messages follow [Commitizen](https://commitizen.github.io/cz-cli/):

```
<type>(<scope>): <short summary>

<body>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`.

Use a concise scope when it clarifies the affected area, such as `syn`, `metrics`,
`rigid`, `stages`, `masks`, `pyramid`, `fields`, `sampling`, `geometry`,
`resampling`, `tests`, `examples`, or `research`. Omit the scope when the change spans
the package broadly.
