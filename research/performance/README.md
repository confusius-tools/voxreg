# Historical performance investigations

Only **[examples/benchmark.py](../../examples/benchmark.py)** is the current
benchmark entry point; see [BENCHMARK.md](../../BENCHMARK.md). The exploratory
profilers, sampler substitutions and thread/batch sweeps have been removed.
[Research archive scope](../README.md) explains source/evidence preservation.

These reports retain their original settings, limits and failures. Their timings
are neither current five-run medians nor precision-matched backend rankings.

- [CPU-VOLUME.md](CPU-VOLUME.md): cached float64 coordinate layout, zero-copy
  single-volume sampling partitions, full-size 100-update truth/parity checks,
  8/24-thread attribution and separately qualified FireANTs linear batch 32.
- [BATCH-SIZE.md](BATCH-SIZE.md), [batch-size-results.json](batch-size-results.json):
  historical fixed-budget ref20/3D batch sweep, transform export off.
- [CPU.md](CPU.md), [CPU-current.md](CPU-current.md): CPU diagnosis snapshots.
- [DTYPE.md](DTYPE.md), [dtype-results.json](dtype-results.json): precision evidence,
  not permission to relax production float64 contracts.
- [NCC.md](NCC.md): objective/operator investigation.
- [SINGLE-PAIR.md](SINGLE-PAIR.md): full-pair versus recording attribution.
- [results.json](results.json), [sampler-results.json](sampler-results.json): compact
  pre/post scalar-affine evidence with hashes, versions and original test counts.

The baseline behind the early scalar-affine improvement is commit `83b6dd6`, not
the current engine. Comparing current code with an old elementwise prototype
would not reproduce the historical improvement. Raw recordings, traces, matrices
and frozen sources remain in ignored local artifacts, not in these JSON summaries.
