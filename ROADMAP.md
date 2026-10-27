# Scope and remembered candidates

Goal: the simplest correct, general GPU-accelerated registration engine for
ConfUSIus, with room for other modalities. ITK, FireANTs and syntx provide ideas
and comparators, not a requirement for identical trajectories or outputs.

## Remember, do not enable by default

- **Multistart:** try several physical initial placements, especially for2D→3D
  section/depth ambiguity. Needs a defensible selection criterion: maximum NCC
  alone can select tiny-overlap or anatomically incorrect fits. No search policy
  or implementation chosen yet.
- **Line search:** test a proposed direction at different step lengths before
  accepting it. Investigate measured overshoot, validity and extra evaluation cost
  first. ITK/SimpleITK offer gradient-descent and conjugate-gradient line-search
  optimizers; the plain gradient descent used in our CF comparisons did not.
- **Automatic rigid→affine staging:** manual staging already works. Pass the full
  returned rigid matrix as the affine initialization, using the original moving
  and fixed images again. Do not feed resampled rigid output into affine or freeze
  the initial rigid stage as a separate transform chain. Automate only if useful.

## Exploration, not runtime

A small shared Matplotlib helper should show a composite and raw metric history,
mark the best state, throttle redraws, and work with headless figure export.
Intermediate-pose observation belongs in research instrumentation for now. No
plotting dependency, viewer framework or public plotting option in voxreg.

## Cleanup boundary

Checkpoint validated work before deleting anything. Remove dead runtime paths and
centralize actively reused research helpers. Preserve frozen controls, minimal
failure evidence, source snapshots and historical results; clearly identify the
current entry point instead of treating every old experiment as current guidance.
Do not loosen numerical gates, change defaults or claim anatomical success from
higher correlation or agreement with a single library.
