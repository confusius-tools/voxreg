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
- [x] **Support reversed displacement fields.** Flipping a saved SyN field and its reference grid now flips the resampled image and validity mask together. Tests cover both shared fields and one field per frame.

## Memory and runtime

- [x] **Stop fitting when every frame in a batch has stopped.** Linear and SyN fitting check for active frames before each update. A reference-only batch reports its score once. Tests cover failed and skipped frames and verify that active frames in a mixed batch keep fitting.
- [x] **Send only reference-grid metadata to lazy resampling workers.** Workers now build the placeholder from the grid coordinates. For a 96³ reference and a 1 × 2 × 2 source, the serialized image task graph fell from about 905 KB to 15 KB. A regression test checks how graph size grows with the reference axes.
- [x] **Release intermediate validity masks after counting valid voxels.** Earlier linear stages keep one count per frame. Only the final stage retains voxel masks. For 128 slices fitted to a 16 × 96 × 96 volume one frame at a time, median peak process memory for the intermediate translation stage fell from 358 to 337 MiB across three CPU runs per version. Tests compare staged transforms, overlap fractions, and final masks with standalone fits.

## Validation after the fixes

- **3D SyN accuracy.** Complete the 200-update comparison for volume pairs and recordings on CPU and CUDA. Keep the existing 0.30-voxel error limit. The CPU pair pilot passed for voxreg and ANTsPy. FireANTs measured 0.303709 voxels. [Fixed-work pilot](PERFORMANCE.md#fixed-work-pilot)
- **Runtime and memory.** Profile the full SyN call to distinguish time spent fitting from time spent checking and exporting transforms. Repeat the cold-input memory benchmarks after the lazy changes, keeping input chunks and fitting batches fixed. Extend coverage to lazy SyN. [Performance baseline](PERFORMANCE.md#coruscant-baseline)
- **Anatomical accuracy.** Build on the first real-data motion check with image-pair, recording, and atlas cases on their native grids. Set quality limits for each workflow, then measure how intensity preprocessing, mutual information, and coarse-to-fine fitting affect accuracy and runtime. [Real-data evidence and limits](VALIDATION.md#first-real-data-motion-quality-case)
