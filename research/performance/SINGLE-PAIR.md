# Why a single small pair is slower on CUDA

**The fit dispatches hundreds of tiny PyTorch operations per iteration. For one
image, CUDA spends much of the instrumented interval waiting for host-side work,
not doing enough arithmetic to amortize it.** This is measured evidence, not just
an assumption that GPU startup or input transfer must dominate.

Scope: actual original-float64 same-subject pair used by the public-workflow
comparison, about 112×128 pixels; rigid, geometry seed, float32 fitting/float64
geometry and NCC unchanged. Fixed100 with window101 disables convergence checks,
so CUDA stopping polls are **not necessary to reproduce** the slowdown. This does
not separately measure their contribution to native-convergence timing.

[profile_pair.py](profile_pair.py) runs CPU8 then CUDA1 in sequential fresh
processes, one warmup API call and three warm samples each, synchronized on CUDA.
It separately measures resident optimization and profiles five steady steps after
20 waiting and two warmup steps. `trace_summary` has an overlap-union unit check.
No library optimization was enabled or precision policy changed.

## Observations

Bounded raw samples, stages, source hashes and operator summaries:
[single-pair-results.json](single-pair-results.json).

| Warm diagnostic median | CPU8 | CUDA1 |
| --- | ---: | ---: |
| Entire register call | 73.56 ms | 202.99 ms |
| Fitting stage | 69.73 ms | 198.47 ms |
| Image upload stage | 0.020 ms | 0.065 ms |
| Final sampling/download stage | 0.515 ms | 0.718 ms |
| Resident optimizer only | 69.19 ms | 197.90 ms |

Input upload/download and initial CUDA setup do not explain this warm gap.

The instrumented five-step CUDA trace contains **1,045 kernel launches**, or
**209 per step**, with median kernel duration **1.312 µs**. Kernel execution totals
**1.961 ms** within a **17.825 ms** first-to-last-kernel interval: about **11%**
compute-kernel busy fraction of that instrumented timeline. Launch API calls alone
occupy 2.981 ms of host time; the remaining gaps also include interpreter,
dispatcher/autograd and profiler work. One synchronization call occupies 0.0066 ms
in that window. The trace also records small internal copy events; avoiding bulk
image transfers alone would not remove the dispatch pipeline.

**Profiler adds overhead**: this 11% is not an uninstrumented utilization reading
or a precise native-run idle percentage. It establishes the many-tiny-kernels
pattern; unprofiled resident timings and the following intervention support the
host-overhead diagnosis.

## Controlled diagnostic, not a shipped feature

Capture and replay the same fixed100 resident optimizer with a static CUDA graph,
excluding graph construction cost. Replayed parameters, iteration counts and
statuses are **bitwise equal** to ordinary execution. Median replay takes
**42.20 ms**, versus **197.90 ms** for ordinary resident execution in this run.
This removes repeated host operation/tape construction and launch dispatch while
retaining the optimizer's numerical work. It is not an end-to-end API speedup,
production graph implementation or evidence that dynamic stopping can simply be
captured without redesign and qualification.

Candidate directions are batching independent fits (already supported), reducing
small dispatches, or a qualified graph/fusion path. None was added here. For this
small single-pair workload, explicit CPU remains the faster ordinary API choice;
that is not a rule for volumes or multi-frame batches.

## Conditions and limits

Measurements: 2026-10-26 11:17:37–11:17:50 UTC, AC online, battery Full, platform
performance. No external test/benchmark workers appeared in the process snapshot;
the earlier external ConfUSIus job had finished and was not stopped. Desktop
processes were untouched: initial GPU utilization was 19%, final 0%. Consequently
these are **diagnostic samples**, not a replacement for the controlled public API
benchmark or a universal speed ratio. Full traces remain ignored under
`artifacts/atlas-and-single-pair/profile-qualified-{cpu,cuda}/trace.json`.
