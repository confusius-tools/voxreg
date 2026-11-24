# Roadmap

Work one item at a time, then stop for review and a separate commit. Start with the two failures below that affect whether a fit succeeds and whether its saved transform reproduces the result.

## Registration and transform replay

- [x] **Make SyN recognize low-contrast images on a bright background.** SyN now subtracts each image’s minimum before scaling it for local correlation. A regression test checks that adding 10,000 to a float32 recording preserves the fitted motion and the score, while a constant frame still reports an unusable metric.
- [x] **Use the same image boundary for registration and replay.** Both use the same tolerance at the outermost sample centers. Regression tests check that replay keeps the same edge pixels, fills the same out-of-bounds row, and returns the same mask for all three interpolation methods.
- [x] **Accept valid starting poses together in a recording.** Each slice frame can now start upright or flipped. Tests compare both frame orders with individual fits and check that motion correction preserves the first frame when it is the reference.
- [x] **Allow registration inside PyTorch’s `no_grad()` context.** The optimizers enable gradients during fitting and restore the caller’s setting afterward. Regression tests compare rigid and SyN transforms, images, validity masks, and scores with ordinary registration.

## Resampling accuracy

- [x] **Make nearest-neighbor ties predictable for labels.** `resample` rounds the source voxel positions directly. Tests cover exact ties and the closest representable position on either side for both matrices and displacement fields.
- [x] **Preserve exact voxel values under an identity transform.** Linear resampling now preserves values at voxel centers, including on oblique grids. Tests cover identity matrices, zero displacement fields, and a pull that lands halfway between voxels in every other column.
- [x] **Round cubic interpolation consistently for integer images.** Cubic sampling rounds before adding the background back. Tests check shifted ramps with positive and negative backgrounds, exact half-integer ties, and values on either side.
- [x] **Keep small translations at large world coordinates.** Matrix resampling now preserves a half-voxel shift when both grids move to a distant origin. Tests check sampled values and edge validity for nearest, linear, and cubic interpolation.
- [ ] **Support reversed displacement fields.** Reversing an axis of both a saved SyN field and its reference grid produced a valid pair that PyTorch could not read because of the array’s memory layout. Convert each field batch to a supported layout at upload. Its output should match the original resampling result reversed along the same axis.

## Memory and runtime

- [ ] **Stop fitting when every frame in a batch has stopped.** A reference-only batch reported zero updates but still issued 51 metric callbacks in both rigid and SyN fitting with a 100-iteration budget and reporting every iteration. Exit the loop once all frames are done. Check reference-only, failed, and skipped batches, plus a mixed batch where one frame still needs updates.
- [ ] **Send only reference-grid metadata to lazy resampling workers.** A 96³ reference produced a roughly 902 KB serialized Dask task graph for a 1 × 2 × 2 moving image. A dummy reference array expands when Dask packages it. Construct the placeholder on the worker and verify that graph size follows the coordinate metadata as reference volume grows.
- [ ] **Release intermediate validity masks after counting valid voxels.** Earlier linear stages only return transforms and diagnostics, yet the code collects and concatenates a mask for the entire recording. Compute the overlap fraction per frame within each batch. Compare peak memory and verify that staged transforms, overlap fractions, and final masks agree. This cost was identified by code inspection.

## Validation after the fixes

- **3D SyN accuracy.** Complete the 200-update comparison for volume pairs and recordings on CPU and CUDA. Keep the existing 0.30-voxel error limit. The CPU pair pilot passed for voxreg and ANTsPy. FireANTs measured 0.303709 voxels. [Fixed-work pilot](PERFORMANCE.md#fixed-work-pilot)
- **Runtime and memory.** Profile the full SyN call to distinguish time spent fitting from time spent checking and exporting transforms. Repeat the cold-input memory benchmarks after the lazy changes, keeping input chunks and fitting batches fixed. Extend coverage to lazy SyN. [Performance baseline](PERFORMANCE.md#coruscant-baseline)
- **Anatomical accuracy.** Build on the first real-data motion check with image-pair, recording, and atlas cases on their native grids. Set quality limits for each workflow, then measure how intensity preprocessing, mutual information, and coarse-to-fine fitting affect accuracy and runtime. [Real-data evidence and limits](VALIDATION.md#first-real-data-motion-quality-case)
