# Roadmap

Work one item at a time, then stop for review and a separate commit. Start with the two failures below that affect whether a fit succeeds and whether its saved transform reproduces the result.

## Registration and transform replay

- [x] **Make SyN recognize low-contrast images on a bright background.** SyN now subtracts each image’s minimum before scaling it for local correlation. A regression test checks that adding 10,000 to a float32 recording preserves the fitted motion and the score, while a constant frame still reports an unusable metric.
- [x] **Use the same image boundary for registration and replay.** Both use the same tolerance at the outermost sample centers. Regression tests check that replay keeps the same edge pixels, fills the same out-of-bounds row, and returns the same mask for all three interpolation methods.
- [x] **Accept valid starting poses together in a recording.** Each slice frame can now start upright or flipped. Tests compare both frame orders with individual fits and check that motion correction preserves the first frame when it is the reference.
- [x] **Allow registration inside PyTorch’s `no_grad()` context.** The optimizers enable gradients during fitting and restore the caller’s setting afterward. Regression tests compare rigid and SyN transforms, images, validity masks, and scores with ordinary registration.

## Resampling accuracy

- [x] **Make nearest-neighbor ties predictable for labels.** `resample` rounds the source voxel positions directly. Tests cover exact ties and the closest representable position on either side for both matrices and displacement fields.
- [ ] **Preserve exact voxel values under an identity transform.** Linear resampling changed 418 values in an integer image containing only 0 and 2⁵², even though every output point was an original voxel center. Avoid mixing neighbors at exact centers. Require exact equality for identity pulls on high-contrast integer data.
- [ ] **Round cubic interpolation consistently for integer images.** Adding a background of 2⁴⁴ to an integer ramp changed which way fractional samples rounded. On the same 0.58-voxel shift, removing that background afterward gave 2, 4, 4, 6 where the original gave 3, 4, 5, 6. Correct the rounding tolerance and test both shifted ramps and genuine half-integer ties.
- [ ] **Keep small translations at large world coordinates.** A half-voxel shift disappeared when both grids had the same origin at 2⁵² mm. Subtract the common origin before adding the small translation. Verify that changing the shared origin preserves interpolated values and edge validity.
- [ ] **Support reversed displacement fields.** Reversing an axis of both a saved SyN field and its reference grid produced a valid pair that PyTorch could not read because of the array’s memory layout. Convert each field batch to a supported layout at upload. Its output should match the original resampling result reversed along the same axis.

## Memory and runtime

- [ ] **Stop fitting when every frame in a batch has stopped.** A reference-only batch reported zero updates but still issued 51 metric callbacks in both rigid and SyN fitting with a 100-iteration budget and reporting every iteration. Exit the loop once all frames are done. Check reference-only, failed, and skipped batches, plus a mixed batch where one frame still needs updates.
- [ ] **Send only reference-grid metadata to lazy resampling workers.** A 96³ reference produced a roughly 902 KB serialized Dask task graph for a 1 × 2 × 2 moving image. A dummy reference array expands when Dask packages it. Construct the placeholder on the worker and verify that graph size follows the coordinate metadata as reference volume grows.
- [ ] **Release intermediate validity masks after counting valid voxels.** Earlier linear stages only return transforms and diagnostics, yet the code collects and concatenates a mask for the entire recording. Compute the overlap fraction per frame within each batch. Compare peak memory and verify that staged transforms, overlap fractions, and final masks agree. This cost was identified by code inspection.

## Validation after the fixes

- **3D SyN accuracy.** Complete the 200-update comparison for volume pairs and recordings on CPU and CUDA. Keep the existing 0.30-voxel error limit. The CPU pair pilot passed for voxreg and ANTsPy. FireANTs measured 0.303709 voxels. [Fixed-work pilot](PERFORMANCE.md#fixed-work-pilot)
- **Runtime and memory.** Profile the full SyN call to distinguish time spent fitting from time spent checking and exporting transforms. Repeat the cold-input memory benchmarks after the lazy changes, keeping input chunks and fitting batches fixed. Extend coverage to lazy SyN. [Performance baseline](PERFORMANCE.md#coruscant-baseline)
- **Anatomical accuracy.** Build on the first real-data motion check with image-pair, recording, and atlas cases on their native grids. Set quality limits for each workflow, then measure how intensity preprocessing, mutual information, and coarse-to-fine fitting affect accuracy and runtime. [Real-data evidence and limits](VALIDATION.md#first-real-data-motion-quality-case)
