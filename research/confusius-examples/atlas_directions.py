"""Compare affine atlas fits in both supplied argument directions, without tuning."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from render_notebooks import ROOT, SETUP, Notebook, page


def canonical_pull(matrix, *, moving_is_slice):
    """Normalize a returned pull to template-world→recording-world."""
    matrix = np.asarray(matrix, dtype=float)
    return matrix.copy() if moving_is_slice else np.linalg.inv(matrix)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    book = Notebook("Atlas affine registration: both directions", "directions", output)
    book.text(
        "<h2>Both argument directions, same physical starting placement</h2><p>Affine-only, original cached slab/recording and preprocessing. All fits max500/window50; CF learning_rate1, voxreg0.1. CF8 threads; voxreg CPU8/CUDA1 host thread. No tuning, engine changes, downloads or timing claims. This avoids the separate CF matrix-seeded requested-rigid transform-family issue. Voxreg currently canonicalizes both fitting directions internally; CF retains the supplied fixed grid. This experiment measures their actual behavior.</p>"
    )
    book.code(SETUP)
    book.code("""
from atlas_directions import canonical_pull
from voxreg import position_to_world
fixed,moving,placement,paths,recording_shape = inputs()
records, matrices, outputs = {}, {}, {}
backends = ["CF","voxreg cpu"] + (["voxreg cuda"] if torch.cuda.is_available() else [])
for direction,moving_is_slice in [("slice_to_volume",True),("volume_to_slice",False)]:
    source,target,seed = (moving,fixed,np.linalg.inv(placement)) if moving_is_slice else (fixed,moving,placement)
    for backend in backends:
        initial = seed.copy()
        if backend == "CF":
            registered,matrix,diagnostics = cf.registration.register_volume(source,target,transform_type="affine",metric="correlation",learning_rate=1.,number_of_iterations=500,convergence_window_size=50,initialization=initial.copy(),sitk_threads=8,show_progress=False,plot_metric=False,plot_composite=False)
            meta = {"iterations":int(diagnostics.n_iterations),"status":diagnostics.stop_condition,"metric_history":diagnostics.metric_values.tolist()}
        else:
            device = backend.split()[-1]
            torch.set_num_threads(8 if device=="cpu" else 1)
            result = voxreg.register(source,target,transform="affine",initialization=initial.copy(),intensity_scaling="none",learning_rate=.1,max_iterations=500,convergence_window_size=50,device=device,progress=False)
            registered,matrix = result.registered,result.transforms.values
            meta = {"iterations":int(result.diagnostics.n_iterations),"status":str(result.diagnostics.status.values),"metric_domain":result.diagnostics.attrs["metric_domain"],"metric_valid_fraction":float(result.diagnostics.valid_fraction),"output_valid_fraction":float(result.valid.values.mean())}
        assert np.array_equal(seed,initial)
        assert registered.shape == target.shape and registered.dtype == source.dtype
        key = direction+" / "+backend
        matrices[key] = canonical_pull(matrix,moving_is_slice=moving_is_slice)
        outputs[key] = registered.values
        records[key] = {"direction":direction,"backend":backend,"initialization":initial.tolist(),"returned_pull":np.asarray(matrix).tolist(),"canonical_pull":matrices[key].tolist(),"output_dtype":str(registered.dtype),"output_shape":list(registered.shape),**meta}
        print(key,meta["iterations"],meta["status"])
""")
    book.text(
        "<h2>Common native-slice comparison</h2><p>Every matrix below is normalized to template→recording pull. For display/scoring, its inverse samples the template on the original recording grid using the same SciPy helper for all fits. NCC² uses joint support across all arms. Placement errors use all10000 native recording centers, not only supported centers. Each voxreg arm is compared with CF in the same supplied direction; the original-direction CF reference is reported separately, not silently replaced.</p>"
    )
    book.code("""
samples,masks,rows = {},{},[]
for key,matrix in matrices.items():samples[key],masks[key],_ = sample(np.linalg.inv(matrix),moving,fixed)
common = np.logical_and.reduce(list(masks.values()))
original_reference = matrices["slice_to_volume / CF"]
for key,matrix in matrices.items():
    record = records[key]
    same_direction = matrices[record["direction"]+" / CF"]
    error = point_error(matrix,same_direction,moving)
    original_error = point_error(matrix,original_reference,moving)
    rows.append({"arm":key,"iterations":record["iterations"],"common_NCC2":ncc(moving.values.astype(float),samples[key],common),"RMS_px_vs_CF_same_direction":error["rms_native_pixels"],"gate_same_direction":error["rms_native_pixels"]<=.25,"RMS_px_vs_CF_original_direction":original_error["rms_native_pixels"]})
display(pd.DataFrame(rows));print("Joint native support:",float(common.mean()))
direction_errors = {backend:point_error(matrices["volume_to_slice / "+backend],matrices["slice_to_volume / "+backend],moving) for backend in backends}
print("Effect of reversing supplied arguments (native pixel errors):")
display(pd.DataFrame(direction_errors).T)
precision = {}
if "voxreg cuda" in backends:
    for direction in ["slice_to_volume","volume_to_slice"]:
        precision[direction] = point_error(matrices[direction+" / voxreg cuda"],matrices[direction+" / voxreg cpu"],moving)
        assert precision[direction]["max_native_pixels"] < .001
for direction in ["slice_to_volume","volume_to_slice"]:
    composites(moving.values,{key.split(" / ")[1]:samples[key] for key in matrices if key.startswith(direction)},direction+" — fitted poses",reference_is_red=False)
""")
    book.text(
        "<h2>Official Allen-colored labels on the recording</h2><p>Same annotation data, palette, template-world→Allen transform and CF nearest-neighbor resampler for every fitted pose. Thus label differences reflect poses, not different label-interpolation implementations. Both CF results are retained; reversing direction is not itself proof that the new result matches the original designated reference.</p>"
    )
    book.code("""
annotation,annotation_provenance = cached_annotation()
color_path = Path.home()/".brainglobe/brainglobe-atlasapi/terminologies/allen_mouse-terminology/3_1/terminology.csv"
colors = allen_colors(color_path)
world_to_allen = np.asarray(fixed.attrs["affines"]["world_to_sform"])
labels = {key:cf.registration.resample_like(annotation,moving,world_to_allen@np.linalg.inv(matrix),interpolation="nearest",fill_value=0,sitk_threads=8).values for key,matrix in matrices.items()}
label_rows = []
for direction in ["slice_to_volume","volume_to_slice"]:
    atlas_panels(moving.values,{key.split(" / ")[1]:labels[key] for key in labels if key.startswith(direction)},colors,direction)
    for backend in backends:
        if backend=="CF":continue
        score=label_agreement(labels[direction+" / "+backend],labels[direction+" / CF"],np.ones(moving.shape,dtype=bool))
        label_rows.append({"direction":direction,"backend":backend,**{k:score[k] for k in ["agreement_fraction","foreground_label_agreement","macro_region_dice"]}})
display(pd.DataFrame(label_rows))
np.savez_compressed(OUT/"arrays.npz",**{k.replace(" / ","__").replace(" ","_")+"__pull":v for k,v in matrices.items()},**{k.replace(" / ","__").replace(" ","_")+"__native_template":v for k,v in samples.items()},**{k.replace(" / ","__").replace(" ","_")+"__labels":v for k,v in labels.items()},**{k.replace(" / ","__").replace(" ","_")+"__output":v for k,v in outputs.items()})
""")
    cf_root = Path(book.namespace["cf"].__file__).parent
    sources = [
        Path(__file__).resolve(),
        ROOT / "research/confusius-examples/render_notebooks.py",
        ROOT / "research/confusius-examples/compare_atlas.py",
        ROOT / "research/confusius-examples/compare_same_subject.py",
        ROOT / "research/confusius-examples/rigid_affine_atlas.py",
        *sorted((ROOT / "src/voxreg").glob("*.py")),
    ]
    record = {
        "records": book.namespace["records"],
        "comparison": book.namespace["rows"],
        "direction_errors": book.namespace["direction_errors"],
        "precision": book.namespace["precision"],
        "label_agreement": book.namespace["label_rows"],
        "joint_support": float(book.namespace["common"].mean()),
        "annotation": book.namespace["annotation_provenance"],
        "inputs": [
            {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in [*book.namespace["paths"], book.namespace["color_path"]]
        ],
        "versions": {
            "confusius": book.namespace["cf"].__version__,
            "torch": book.namespace["torch"].__version__,
        },
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "confusius_sources": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((cf_root / "registration").glob("*.py"))
        },
    }
    (output / "results.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    (output / "index.html").write_text(
        page(
            "Atlas direction comparison",
            '<h1>Atlas direction comparison</h1><p><a href="directions.html">Executed comparison</a> · <a href="directions.ipynb">Notebook</a></p>',
        )
    )
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
