# NCC calibration and whole-batch termination

Baseline: `d27a368`. This follow-up retains the reference-tested ITK convergence
**calculation** and the 50-value window. It changes the default minimum from 1e-8
to **1e-6**, and stops evaluating a resident batch once all its frames have stopped.
No fitting precision, Adam learning rate, transform-error gate or failure guard changed.

## Qualification, not biological truth

Seven explicit policies were compared on four 120-frame windows, on CPU batch32
and CUDA batch120. This was a quality pass, not a timing pass. Controls run exactly
100 or 400 updates per moving frame, with the history window set beyond the budget.
Every policy is scored by independent SciPy interpolation on one joint support
per input. Disagreement with a longer run is not ground-truth registration error.

Inputs (archive hashes and dtypes are in `calibration-results.json`):

- rat75 slice32 frames 220:340, neutral float32.
- rat75 slice32 frames 800:920, native decoded float64.
- rat75 slice37 frames 220:340, native decoded float64.
- ALD001 awake/ref24 run-2 frames 0:120, float32.

The screening budgets were max full-grid RMS mapping disagreement <0.05 minimum-
spacing pixels, worst per-frame NCC² drop <1e-4, and median per-frame NCC² change
at least -1e-5, against **both** controls. These are conservative regression screens,
not new truth/accuracy gates. The existing synthetic recovery and strict CPU/CUDA
pointwise agreement tests remain unchanged.

The 50/1e-6 candidate passed all screens on both devices:

| Window | Updates min/median/max | Converged /119 | Worst RMS vs400, pixels | Worst NCC² drop vs400 |
|---|---:|---:|---:|---:|
| rat75 slice32 |51/79/95|119|.004991|1.45e-6|
| rat75 late slice32 |50/78/98|119|.004971|1.83e-6|
| rat75 slice37 |49/79/100|118|.006785|6.87e-6|
| ALD001 |68/78/88|119|.005271|9.73e-7|

CPU/CUDA update percentiles and convergence counts agree. The worst RMS across
either control/device is below 0.007 pixels. One frame still reaches the limit;
we do not relabel it converged. The complete pre-change 167-test suite also passed
with only the candidate default injected, before production code was changed.

Shorter windows were rejected: 20/1e-6 loses up to 2.13e-4 NCC² and differs by 0.054
pixels RMS on slice37; 10/1e-6 loses substantially more on every window. 50/1e-7
barely reduces the typical 100-update limit. 50/1e-6 is the smallest tested change
that provides useful stopping without those losses. Literal ITK gradient descent
settings remain available explicitly with `convergence_minimum_value=1e-8`.

## Terminal check and synchronization cost

Frame histories, normalization, Adam clocks/moments and stopping decisions remain
independent. Completed/failed/reference frames are frozen, not removed from tensors.
After a full window exists:

- CPU checks whole-batch completion after every iteration.
- CUDA checks every ten iterations, aligned to the first full window. Each check
  transfers **one batch-wide boolean** and synchronizes that stream. With the
  defaults there are at most five such checks, not one read per frame/iteration.
- There is no pointless poll on the final budget iteration.
- A window beyond the budget disables both the monitor and these checks, preserving
  the historical fixed-budget benchmark/precision protocol.

CUDA may compute up to nine extra frozen iterations. No active frame is stopped
by a batch companion: only an entirely inactive batch can leave the loop. Mixed
failures, frozen states, polling cadence and disabled controls are covered by tests.
Against the same threshold with the full loop retained, **all four real windows
produce identical registered/valid arrays, matrices, diagnostics and update counts**
on CPU/CUDA. Timing outputs also match bitwise for CUDA batch32/120.

## Separate fresh-process timings

Quiet whole-call timing, eight Torch CPU threads, one cold and three warm calls,
no profiler or memory sampler. Full-loop baseline uses the **same calibrated
threshold** with only terminal checks disabled. Thus this isolates execution policy;
it is not a fresh end-to-end ITK/FireANTs comparison or an old-release speedup.

| Input / device / batch | Full loop median s | Terminal median s |
|---|---:|---:|
| rat75 CPU32 |2.442|2.322|
| ALD001 CPU32 |2.090|1.787|
| rat75 CUDA32 |1.146|.909|
| ALD001 CUDA32 |1.178|1.049|
| rat75 CUDA120 |.305|.314|
| ALD001 CUDA120 |.329|.275|

This is a tradeoff, **not a universal speedup**: rat75's slowest full-batch frame
needs 95 updates (96 metric evaluations), missing the last useful poll at 90. It
saves no iterations at that batch size and pays about 3% in observed median polling
cost. ALD001 finishes its CUDA batch at 90 instead of 100. Smaller resident batches
can finish independently and showed larger gains here. CUDA120 remains faster in
absolute terms than CUDA32 on these inputs.

CPU ranges overlap substantially; hardware clocks/affinity were not fixed and
other desktop activity remained. Do not attribute every timing difference to this
change. Raw three-sample times and source hashes are in `calibration-results.json`;
prototype interval1/5/10 trials remain ignored under `artifacts/convergence-calibration/`.
No new memory pass or full-recording comparison was made.

## Reproduce

Quality commands are deliberately separate from timing commands:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run python research/convergence/calibrate.py artifacts/notebook/input.npz \
    --device cuda --batch-size 120 --output artifacts/calibration/quality.json

for mode in full terminal; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    uv run python research/convergence/benchmark.py artifacts/notebook/input.npz \
      --device cuda --batch-size 120 --mode "$mode" \
      --output "artifacts/calibration/$mode.json"
done
```

Other windows use the same NPZ fields: `values`, `voxel_to_world`, `time`, `k`, `j`,
`i`. Preserve original dtype and integer native coordinates. The CLI guards reject
nonfinite common-support scores rather than reporting a successful qualification.
Both CLIs reject output paths that could overwrite the input; the timing CLI also
rejects collisions between its JSON report and NPZ result archive.
