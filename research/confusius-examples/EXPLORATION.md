# Native-pair exploration

Current entry point: `explore_atlas.py`. It uses the original cached recording
and cropped template: no slab expansion, hidden role reversal, or downloads.
ITK, FireANTs and syntx are sources of ideas, not output-equivalence requirements.
Higher correlation or agreement with CF alone does not establish anatomical truth.

## Run

From the repository root, with an interactive Matplotlib backend available:

```sh
uv run python research/confusius-examples/explore_atlas.py \
  --moving recording --transform affine --device cuda \
  --output artifacts/exploration/recording-affine
```

The destination must not exist. A live composite and raw metric curve update
during fitting; the final image and dashed metric line show the **returned best**
result, not necessarily the last trial. The window stays open after completion.
Use `--moving template` to fit the opposite caller-fixed objective, or
`--transform rigid` for a genuine rigid fit. These are independent fits, not
expected to return inverse-equivalent poses.

For headless export, add `--headless`. Outputs are `registration.png`,
`arrays.npz` (registered image, validity, matrix, observed metric trace), and
`results.json` (settings, status and input/source hashes). Failed/unusable metric
values remain NaN in the arrays and null in JSON; they are not successful fits.
Plot-enabled runs are not performance benchmarks.

## Reuse the small helper

`plot_registration.py` owns the shared display sampling, RGB composite and scoped
pose observer. It does not choose a backend or import historical experiment runners.
From a script in this directory:

```python
from plot_registration import register_with_plot

# These images are already in dB; no second intensity conversion.
result, figure = register_with_plot(
    moving_db,
    fixed_db,
    transform="affine",
    initialization=seed,
    intensity_scaling="none",
    device="cuda",
    metric_interval=10,
    redraw_interval=0.25,
)
figure.savefig("registration.png")
```

- One image pair only; explicitly select a frame from a movie. A singleton time
  axis is accepted. Batched registration itself remains available through voxreg.
- Default display limits are −30…0 dB, using the supplied image intensities.
  For other units, pass `display_range=(low, high)`. Preview interpolation is
  linear; output interpolation still follows the ordinary registration API.
- The reference is cyan and the sampled image red; unsupported pixels are gray.
  There are no support-percentage annotations. For mixed pairs, display is on the
  recording plane regardless of fitting direction; **only display** may invert
  the pull. Two volumes show the fixed volume's central k slice.
- `metric_interval` controls callbacks; `redraw_interval` throttles GUI work.
  `show=False` captures the metric curve and renders only the final pose.
  Existing user `metric_callback` functions are forwarded after plot observation.
- Observation temporarily patches private engine functions and restores them on
  success or exception. **Do not run concurrent fits in the same Python process**
  while observing. No runtime API or optimizer default changed.
- Raw trial scores are not an acceptance history and can include rejected poses.
  The final horizontal line identifies the returned best valid score; no exact
  best-iteration clock is inferred from a subsampled trace.

## Manual rigid→affine staging

Already supported, without a new orchestration API:

```python
rigid = voxreg.register(moving, fixed, transform="rigid", initialization=seed)
affine = voxreg.register(
    moving,
    fixed,
    transform="affine",
    initialization=rigid.transforms.values,
)
```

Use the original images again, not `rigid.registered`. The affine seed is the
complete physical map; the optimizer can refine all permitted affine parameters.
Keep intensity scaling and other options appropriate to the images in both calls.

## What is historical

The other atlas runners and reports preserve investigations and failure evidence;
some deliberately implement retired sampling policies. Start with this entry point
for exploration and `atlas_native_fixed.py` for the qualified eight-fit report.
Older source hashes describe measured snapshots, not a promise that today's
refactored working-tree files have identical bytes. Checkpoint `ebe1349` preserves
the pre-cleanup working tree; generated artifacts and frozen controls remain local.
Multistart (especially 2D→3D) and line search are remembered in [ROADMAP](../../ROADMAP.md),
not implemented or enabled by this plotting work.
