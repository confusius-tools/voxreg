"""CF direction curves, correctly paired best poses, and a bounded rate ablation."""

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import confusius as cf
import numpy as np
import SimpleITK as sitk
from atlas_alignment_investigation import score, sitk_pair
from confusius.registration.affines import sitk_linear_transform_to_affine
from render_notebooks import ROOT, SETUP, Notebook, page


def fit_curve(source, target, seed, *, learning_rate=1.0, window=50):
    """Observe CF itself without changing its optimizer or installed sources.

    This build emits IterationEvent with a pre-step metric and post-step position.
    Pair each metric with the previous position, then verify against MetricEvaluate.
    """
    original = sitk.ImageRegistrationMethod
    states = [np.asarray(seed).copy()]
    losses = []

    def factory():
        registration = original()

        def observe():
            transform = sitk.AffineTransform(registration.GetInitialTransform())
            transform.SetParameters(registration.GetOptimizerPosition())
            losses.append(float(registration.GetMetricValue()))
            states.append(sitk_linear_transform_to_affine(transform))

        registration.AddCommand(sitk.sitkIterationEvent, observe)
        return registration

    with patch.object(sitk, "ImageRegistrationMethod", factory):
        _, pull, diagnostics = cf.registration.register_volume(
            source,
            target,
            transform_type="affine",
            metric="correlation",
            initialization=np.asarray(seed).copy(),
            learning_rate=learning_rate,
            number_of_iterations=500,
            convergence_window_size=window,
            sitk_threads=8,
            show_progress=False,
            plot_metric=False,
            plot_composite=False,
        )
    assert sitk.ImageRegistrationMethod is original
    np.testing.assert_array_equal(losses, diagnostics.metric_values)
    np.testing.assert_array_equal(states[-1], pull)
    for index in {0, len(losses) - 1, int(np.argmin(losses))}:
        assert (
            abs(-score(source, target, states[index], expand=True) - losses[index])
            < 1e-10
        )
    final_loss = -score(source, target, pull, expand=True)
    all_losses = np.r_[losses, final_loss]
    best = int(np.argmin(all_losses))
    meta = dataclasses.asdict(diagnostics)
    meta["metric_values"] = np.asarray(losses).tolist()
    return {
        "learning_rate": learning_rate,
        "window": window,
        "diagnostics": meta,
        "raw_metric": all_losses.tolist(),
        "final_pull": np.asarray(pull).tolist(),
        "best_pull": np.asarray(states[best]).tolist(),
        "best_evaluation": best,
        "best_metric": float(all_losses[best]),
        "final_metric": final_loss,
        "event_pairing_verified": True,
    }, np.stack(states)


def first_step_probe(source, target, states, index=0):
    """Shrink an actual update; track changes to the hard moving-support mask."""
    moving, fixed = sitk_pair(source, target, True)

    def affine(image):
        matrix = np.eye(4)
        matrix[:3, :3] = np.asarray(image.GetDirection()).reshape(3, 3) @ np.diag(
            image.GetSpacing()
        )
        matrix[:3, 3] = image.GetOrigin()
        return matrix

    points = affine(fixed) @ np.vstack(
        (
            np.indices(fixed.GetSize()).reshape(3, -1),
            np.ones((1, np.prod(fixed.GetSize()))),
        )
    )
    inverse = np.linalg.inv(affine(moving))
    upper = np.asarray(moving.GetSize())[:, None] - 0.5
    result = []
    initial_valid = None
    for fraction in [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0]:
        pull = states[index] + fraction * (states[index + 1] - states[index])
        mapped = (inverse @ pull @ points)[:3]
        valid = ((mapped >= -0.5) & (mapped < upper)).all(axis=0)
        if initial_valid is None:
            initial_valid = valid
        result.append(
            {
                "fraction_of_selected_update": fraction,
                "objective": -score(source, target, pull, expand=True),
                "valid_points": int(valid.sum()),
                "support_changes_from_start": int((valid != initial_valid).sum()),
            }
        )
    return result


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    book = Notebook(
        "CF direction comparison: metric, best pose and learning rate", "rates", output
    )
    book.text(
        "<h2>Actual CF fits: left/right curves and lower learning rates</h2><p>Template moving/recording fixed is the right-hand case. Same cached images/manual seed, affine/correlation, rate1 baseline, max500 updates/window50. Lower rates0.3,0.1,0.03 change only rate. Separate window501 arms disable stopping to distinguish rate from premature termination. Plane depth/tilt remain free in all right-hand arms. No product edits or speed claims.</p>"
    )
    book.code(SETUP)
    book.code("""
from atlas_direction_learning_rate import fit_curve,first_step_probe
from atlas_alignment_investigation import plane_pose
fixed,moving,placement,paths,_=inputs()
reference_path=ROOT/"artifacts/atlas-directions/affine/results.json"
reference=json.loads(reference_path.read_text())
for path,expected in reference["confusius_sources"].items():assert digest(Path(path))==expected
assert [digest(path) for path in paths]==[row["sha256"] for row in reference["inputs"][:2]]
engine_sources={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/"src/voxreg").glob("*.py"))}
records,states={},{}
configs=[("left_rate1",True,1.,50),("right_rate1",False,1.,50)]
configs += [("right_rate"+str(rate),False,rate,50) for rate in [.3,.1,.03]]
configs += [("right_fixed_rate"+str(rate),False,rate,501) for rate in [1.,.3,.1,.03]]
for name,moving_is_slice,rate,window in configs:
    source,target,seed=(moving,fixed,np.linalg.inv(placement)) if moving_is_slice else (fixed,moving,placement)
    record,state=fit_curve(source,target,seed,learning_rate=rate,window=window)
    record["moving_is_slice"]=moving_is_slice
    for label in ["final","best"]:
        pull=np.asarray(record[label+"_pull"])
        canonical=pull if moving_is_slice else np.linalg.inv(pull)
        record[label+"_canonical_pull"]=canonical.tolist()
        record[label+"_plane"]=plane_pose(canonical,moving)
    records[name],states[name]=record,state
    print(name,"updates",record["diagnostics"]["n_iterations"],"best",record["best_evaluation"],record["best_metric"],"returned",record["final_metric"],flush=True)
for name,key in [("left_rate1","slice_to_volume / CF"),("right_rate1","volume_to_slice / CF")]:
    assert np.array_equal(records[name]["final_pull"],reference["records"][key]["returned_pull"])
    assert np.array_equal(records[name]["diagnostics"]["metric_values"],reference["records"][key]["metric_history"])
first_steps={"left":first_step_probe(moving,fixed,states["left_rate1"]),"right":first_step_probe(fixed,moving,states["right_rate1"])}
peak_steps={}
for name in ["left_rate1","right_rate1","right_fixed_rate0.3"]:
    index=records[name]["best_evaluation"]
    source,target=(moving,fixed) if name=="left_rate1" else (fixed,moving)
    peak_steps[name]={"evaluation":index,"probe":first_step_probe(source,target,states[name],index)}
display(pd.DataFrame(first_steps["right"]));display(pd.DataFrame(peak_steps["right_rate1"]["probe"]))
""")
    book.text(
        "<h2>Left versus right metric histories</h2><p>Different fixed sampling domains: absolute losses across the two panels are not an anatomical ranking. Lower negative squared correlation is better. The final returned-pose metric is evaluated separately and included as the last point. Green marks the actual best-scoring pose, not the post-update pose accidentally paired with a pre-update metric.</p>"
    )
    book.code("""
fig,axes=plt.subplots(1,2,figsize=(14,4),constrained_layout=True)
for ax,name,title in zip(axes,["left_rate1","right_rate1"],["Left: recording moving → template fixed","Right: template moving → recording fixed"],strict=True):
    row=records[name];loss=np.asarray(row["raw_metric"]);best=row["best_evaluation"]
    ax.plot(loss,lw=1.4);ax.plot(best,loss[best],"go",label=f"Best evaluation {best}");ax.plot(len(loss)-1,loss[-1],"rs",label="Returned pose")
    ax.axvspan(max(0,len(loss)-51),len(loss)-1,color="gray",alpha=.1)
    ax.set(title=title,xlabel="Evaluated parameter state (0 = seed)",ylabel="Negative NCC² — lower is better");ax.legend();ax.grid(alpha=.2)
fig,axes=plt.subplots(1,2,figsize=(14,4),constrained_layout=True)
for ax,prefix,title in zip(axes,["right_rate","right_fixed_rate"],["Right-hand fit: original stopping rule","Right-hand fit: stopping disabled, 500 updates"],strict=True):
    for rate in [1.,.3,.1,.03]:
        name="right_rate1" if prefix=="right_rate" and rate==1. else prefix+str(rate)
        row=records[name];ax.plot(row["raw_metric"],label=f"rate {rate:g}");ax.plot(row["best_evaluation"],row["best_metric"],"o",ms=4)
    ax.set(title=title,xlabel="Evaluated parameter state",ylabel="Negative NCC² — lower is better");ax.legend();ax.grid(alpha=.2)
""")
    book.text(
        "<h2>Check the images, not just the loss</h2><p>Same original recording grid, same SciPy template sampler and red/cyan display for every pose. The good left-hand returned pose remains the reference. Best metric means best under the stated optimizer objective, not verified anatomy. No best-pose restoration is added to CF or voxreg.</p>"
    )
    book.code("""
reference_pull=np.asarray(records["left_rate1"]["final_canonical_pull"])
samples={};rows=[]
samples["manual"],_,_=sample(placement,moving,fixed)
for name,row in records.items():
    for label in ["best","final"]:
        pull=np.asarray(row[label+"_canonical_pull"])
        samples[name+"__"+label],_,_=sample(np.linalg.inv(pull),moving,fixed)
        row[label+"_reference_error"]=point_error(pull,reference_pull,moving)
    rows.append({"run":name,"updates":row["diagnostics"]["n_iterations"],"best_at":row["best_evaluation"],"best_NCC2":-row["best_metric"],"returned_NCC2":-row["final_metric"],"best_RMS_px_vs_good_left":row["best_reference_error"]["rms_native_pixels"],"returned_RMS_px_vs_good_left":row["final_reference_error"]["rms_native_pixels"],"best_depth_mm":row["best_plane"]["center_zyx_mm"][0],"returned_depth_mm":row["final_plane"]["center_zyx_mm"][0]})
display(pd.DataFrame(rows))
composites(moving.values,{"Initial placement":samples["manual"],"Left best metric":samples["left_rate1__best"],"Left returned (good reference)":samples["left_rate1__final"]},"Check the good left-hand alignment",reference_is_red=False)
composites(moving.values,{"Good left reference":samples["left_rate1__final"],"Right best metric":samples["right_rate1__best"],"Right returned":samples["right_rate1__final"]},"Does the right-hand metric peak fix the anatomy?",reference_is_red=False)
composites(moving.values,{"Good left reference":samples["left_rate1__final"],**{f"Right rate {rate:g}, best of 500":samples["right_fixed_rate"+str(rate)+"__best"] for rate in [1.,.3,.1,.03]}},"Learning rate controls: best metric poses, not automatic successes",reference_is_red=False)
np.savez_compressed(OUT/"arrays.npz",**{name+"__states":value for name,value in states.items()},**{name+"__native_template":value for name,value in samples.items()})
for path,expected in engine_sources.items():assert digest(ROOT/path)==expected
""")
    sources = [
        Path(__file__).resolve(),
        ROOT / "research/confusius-examples/atlas_alignment_investigation.py",
        ROOT / "research/confusius-examples/compare_atlas.py",
        ROOT / "research/confusius-examples/compare_same_subject.py",
        ROOT / "research/confusius-examples/render_notebooks.py",
        *sorted((ROOT / "src/voxreg").glob("*.py")),
    ]
    record = {
        "records": book.namespace["records"],
        "summary": book.namespace["rows"],
        "first_step_probe": book.namespace["first_steps"],
        "peak_step_probe": book.namespace["peak_steps"],
        "reference_sha256": hashlib.sha256(
            book.namespace["reference_path"].read_bytes()
        ).hexdigest(),
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "inputs": [
            {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in book.namespace["paths"]
        ],
        "confusius_sources": book.namespace["reference"]["confusius_sources"],
        "versions": {
            "confusius": cf.__version__,
            "SimpleITK": sitk.Version_VersionString(),
        },
        "production_sources_unchanged": True,
    }
    (output / "results.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    (output / "index.html").write_text(
        page(
            "Direction metric and learning rate",
            '<h1>CF curves, best poses and learning rates</h1><a href="rates.html">Executed comparison</a>',
        )
    )
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
