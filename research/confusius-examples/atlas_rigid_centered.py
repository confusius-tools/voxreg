"""True centered CF Euler3D controls: tilt free, all scale/shear forbidden."""

import argparse
import dataclasses
import hashlib
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import confusius as cf
import numpy as np
import SimpleITK as sitk
from atlas_alignment_investigation import score
from atlas_centering import centered_affine, physical_center
from confusius.registration.affines import sitk_linear_transform_to_affine
from render_notebooks import ROOT, SETUP, Notebook, page


def centered_rigid(matrix, center):
    """Preserve a proper rigid seed with an actual Euler3D, not CF's affine conversion."""
    matrix = np.asarray(matrix, dtype=float)
    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0
    ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10, rtol=0):
        raise ValueError(
            "A genuine rigid seed must have a proper orthogonal linear part."
        )
    center = np.asarray(center, dtype=float)
    transform = sitk.Euler3DTransform()
    transform.SetCenter(tuple(center))
    transform.SetMatrix(tuple(rotation.ravel()))
    transform.SetTranslation(tuple(matrix[:3, 3] + (rotation - np.eye(3)) @ center))
    np.testing.assert_allclose(
        sitk_linear_transform_to_affine(transform), matrix, atol=1e-12, rtol=0
    )
    return transform


def linear_matrix(transform):
    """CF's exporter does not yet accept ComposeScaleSkewVersor3DTransform."""
    # This SimpleITK build omits ComposeScaleSkewVersor from Downcast's lookup.
    transform = (
        sitk.ComposeScaleSkewVersor3DTransform(transform)
        if transform.GetTransformEnum() == sitk.sitkComposeScaleSkewVersor
        else transform.Downcast()
    )
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(transform.GetMatrix()).reshape(3, 3)
    matrix[:3, 3] = (
        -matrix[:3, :3] @ np.array(transform.GetCenter())
        + np.array(transform.GetCenter())
        + np.array(transform.GetTranslation())
    )
    return matrix


def centered_unit_normal(matrix, center):
    """Input normal is world z: R @ diag(1,sy,sx) @ in-plane shear."""
    rigid = centered_rigid(matrix, center)
    versor = sitk.VersorRigid3DTransform()
    versor.SetMatrix(rigid.GetMatrix())
    transform = sitk.ComposeScaleSkewVersor3DTransform()
    transform.SetCenter(rigid.GetCenter())
    transform.SetRotation(versor.GetVersor())
    transform.SetTranslation(rigid.GetTranslation())
    np.testing.assert_allclose(linear_matrix(transform), matrix, atol=1e-12, rtol=0)
    return transform


def fit_control(
    source,
    target,
    seed,
    *,
    model="rigid",
    learning_rate=1.0,
    max_iterations=500,
    window=50,
):
    """Actual CF fitting with verified transform class, center and paired state history."""
    if model not in {"rigid", "unit_normal_affine", "affine"}:
        raise ValueError("Unknown control model")
    rigid = model == "rigid"
    if model == "affine":
        builder, transform_class = centered_affine, sitk.AffineTransform
    else:
        builder = centered_rigid if rigid else centered_unit_normal
        transform_class = (
            sitk.Euler3DTransform if rigid else sitk.ComposeScaleSkewVersor3DTransform
        )
    expected_type = (
        "AffineTransform" if model == "affine" else transform_class().GetName()
    )
    count = 6 if rigid else 12
    weights = (
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
        if model == "unit_normal_affine"
        else None
    )
    center = physical_center(target)
    volume = importlib.import_module("confusius.registration.volume")
    original_converter = volume.affine_to_sitk_linear_transform
    original_exporter = volume.sitk_linear_transform_to_affine
    original_method = sitk.ImageRegistrationMethod
    original_downcast = sitk.Transform.Downcast

    def downcast(transform):
        if transform.GetTransformEnum() == sitk.sitkComposeScaleSkewVersor:
            return sitk.ComposeScaleSkewVersor3DTransform(transform)
        return original_downcast(transform)

    states = [np.asarray(seed).copy()]
    losses = []
    seed_types = []
    observed_types = []
    parameter_states = []

    def convert(matrix):
        transform = builder(matrix, center)
        seed_types.append(transform.GetName())
        parameter_states.append(list(transform.GetParameters()))
        return transform

    def factory():
        registration = original_method()

        def observe():
            current = transform_class(registration.GetInitialTransform())
            assert current.GetName() == expected_type
            assert len(registration.GetOptimizerPosition()) == count
            transform = transform_class(current)
            transform.SetParameters(registration.GetOptimizerPosition())
            observed_types.append(transform.GetName())
            parameter_states.append(list(registration.GetOptimizerPosition()))
            losses.append(float(registration.GetMetricValue()))
            states.append(linear_matrix(transform))

        registration.AddCommand(sitk.sitkIterationEvent, observe)
        return registration

    with (
        patch.object(volume, "affine_to_sitk_linear_transform", convert),
        patch.object(volume, "sitk_linear_transform_to_affine", linear_matrix),
        patch.object(sitk, "ImageRegistrationMethod", factory),
        patch.object(sitk.Transform, "Downcast", downcast),
    ):
        _, pull, diagnostics = cf.registration.register_volume(
            source,
            target,
            transform_type="rigid" if rigid else "affine",
            optimizer_weights=weights,
            metric="correlation",
            initialization=np.asarray(seed).copy(),
            learning_rate=learning_rate,
            number_of_iterations=max_iterations,
            convergence_window_size=window,
            sitk_threads=8,
            show_progress=False,
            plot_metric=False,
            plot_composite=False,
        )
    assert volume.affine_to_sitk_linear_transform is original_converter
    assert volume.sitk_linear_transform_to_affine is original_exporter
    assert sitk.ImageRegistrationMethod is original_method
    assert sitk.Transform.Downcast is original_downcast
    assert seed_types == [expected_type] and observed_types
    np.testing.assert_array_equal(losses, diagnostics.metric_values)
    np.testing.assert_array_equal(states[-1], pull)
    for index in {0, int(np.argmin(losses)), len(losses) - 1}:
        assert (
            abs(-score(source, target, states[index], expand=True) - losses[index])
            < 1e-10
        )
    all_losses = np.r_[losses, -score(source, target, pull, expand=True)]
    states = np.stack(states)
    singular = np.linalg.svd(states[:, :3, :3], compute_uv=False)
    error = float(np.max(np.abs(singular - 1)))
    normal = states[:, :3, 0]
    normal_error = float(np.max(np.abs(np.linalg.norm(normal, axis=1) - 1)))
    orthogonality_error = float(
        np.max(np.abs(np.einsum("ni,nij->nj", normal, states[:, :3, 1:3])))
    )
    if model != "affine":
        assert normal_error < 1e-10 and orthogonality_error < 1e-10
        assert np.all(np.linalg.det(states[:, :3, :3]) > 0)
    if rigid:
        assert error < 1e-10
        np.testing.assert_allclose(
            np.linalg.det(states[:, :3, :3]), 1.0, atol=1e-10, rtol=0
        )
    best = int(np.argmin(all_losses))
    meta = dataclasses.asdict(diagnostics)
    meta["metric_values"] = list(losses)
    return {
        "diagnostics": meta,
        "raw_metric": all_losses.tolist(),
        "final_pull": np.asarray(pull).tolist(),
        "best_pull": states[best].tolist(),
        "best_evaluation": best,
        "best_metric": float(all_losses[best]),
        "final_metric": float(all_losses[-1]),
        "center_fixed_world_zyx_mm": center.tolist(),
        "optimized_transform": expected_type,
        "optimized_parameters": count,
        "free_parameters": 6 if rigid else (9 if model == "unit_normal_affine" else 12),
        "model": model,
        "settings": {
            "learning_rate": learning_rate,
            "max_iterations": max_iterations,
            "window": window,
        },
        "parameter_states": parameter_states,
        "all_states_positive_determinant": bool(
            np.all(np.linalg.det(states[:, :3, :3]) > 0)
        ),
        "optimizer_weights": weights,
        "max_unit_singular_value_error_all_states": error,
        "max_normal_scale_error_all_states": normal_error,
        "max_normal_tangent_dot_all_states": orthogonality_error,
        "event_pairing_verified": True,
    }, states


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    book = Notebook("Centered CF: true rigid and unit-normal affine", "rigid", output)
    book.text(
        "<h2>Tilt remains free; no scaling or shear</h2><p>Actual CF calls with Euler3D transforms and6 verified optimizer parameters, not the affine transform otherwise substituted by its matrix-seed path. Preserve the manual world-space seed and center on each fixed image. Same cached images, expansion, correlation, rate1, max500/window50 and8threads as the centered affine controls. No rate tuning or new defaults. Rigid forbids all scale/shear. An additional recording-fixed arm uses ComposeScaleSkewVersor3DTransform with9 free parameters: rotation, translation,2 in-plane scales and in-plane shear. Hold normal scale=1 and two normal-mixing shears=0 in its rotating local frame, NOT in the fixed world affine matrix. Tilt remains free. No unmeasured through-plane shear is estimated. This native-recording-input constraint is only tested in the recording-fixed direction.</p>"
    )
    book.code(SETUP)
    book.code("""
from atlas_rigid_centered import fit_control
from atlas_alignment_investigation import plane_pose
fixed,moving,placement,paths,_=inputs()
reference_path=ROOT/"artifacts/atlas-centering/comparison/results.json"
reference=json.loads(reference_path.read_text())
for path,expected in reference["confusius_sources"].items():assert digest(Path(path))==expected
assert [digest(path) for path in paths]==[row["sha256"] for row in reference["inputs"]]
engine_sources={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/"src/voxreg").glob("*.py"))}
records,states={},{}
for direction,moving_is_slice,model in [("recording_moving",True,"rigid"),("template_moving",False,"rigid"),("template_moving_unit_normal_affine",False,"unit_normal_affine")]:
    source,target,seed=(moving,fixed,np.linalg.inv(placement)) if moving_is_slice else (fixed,moving,placement)
    row,state=fit_control(source,target,seed,model=model)
    for label in ["best","final"]:
        pull=np.asarray(row[label+"_pull"])
        canonical=pull if moving_is_slice else np.linalg.inv(pull)
        row[label+"_canonical_pull"]=canonical.tolist()
        row[label+"_plane"]=plane_pose(canonical,moving)
    records[direction],states[direction]=row,state
    affine=reference["records"][("recording_moving" if moving_is_slice else "template_moving")+"__centered"]
    np.testing.assert_allclose(row["raw_metric"][0],affine["raw_metric"][0],atol=1e-12,rtol=0)
    print(direction,"updates",row["diagnostics"]["n_iterations"],"best",row["best_metric"],"returned",row["final_metric"],"unit-scale error",row["max_unit_singular_value_error_all_states"],flush=True)
""")
    book.code(r"""
fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
for ax,direction,title in zip(axes,["recording_moving","template_moving"],["Recording moving → template fixed","Template moving → recording fixed"],strict=True):
    curves=[(reference["records"][direction+"__centered"],"Centered affine"),(records[direction],"Centered true rigid")]
    if direction=="template_moving":curves.append((records["template_moving_unit_normal_affine"],"Tilt + in-plane affine; unit normal"))
    for row,label in curves:
        loss=np.asarray(row["raw_metric"]);line,=ax.plot(loss,label=label,lw=1.4)
        ax.plot(row["best_evaluation"],row["best_metric"],"o",color=line.get_color(),ms=4)
        ax.plot(len(loss)-1,loss[-1],"s",color=line.get_color(),ms=4)
    ax.set(title=title,xlabel="Evaluated parameter state",ylabel="Negative NCC² — lower is better");ax.grid(alpha=.2);ax.legend()
fig.suptitle("Same center, seed, rate and stopping; only model changes\nCircle = best metric, square = returned pose; separate domains across panels",fontsize=11)
""")
    book.code(r"""
good=np.asarray(reference["records"]["recording_moving__origin"]["final_canonical_pull"])
samples={"good_original_affine":sample(np.linalg.inv(good),moving,fixed)[0]};rows=[]
for direction,row in records.items():
    for label in ["best","final"]:
        pull=np.asarray(row[label+"_canonical_pull"])
        samples[direction+"__"+label]=sample(np.linalg.inv(pull),moving,fixed)[0]
        row[label+"_reference_error"]=point_error(pull,good,moving)
    rows.append({"run":direction,"updates":row["diagnostics"]["n_iterations"],"best_at":row["best_evaluation"],"best_NCC2":-row["best_metric"],"returned_NCC2":-row["final_metric"],"returned_depth_mm":row["final_plane"]["center_zyx_mm"][0],"returned_tilt_degrees":row["final_plane"]["tilt_degrees"],"returned_RMS_px_vs_original_good_affine":row["final_reference_error"]["rms_native_pixels"],"best_RMS_px_vs_original_good_affine":row["best_reference_error"]["rms_native_pixels"],"free_parameters":row["free_parameters"],"max_normal_scale_error":row["max_normal_scale_error_all_states"]})
display(pd.DataFrame(rows))
direction_error=point_error(np.asarray(records["template_moving"]["final_canonical_pull"]),np.asarray(records["recording_moving"]["final_canonical_pull"]),moving)
print("Rigid direction discrepancy:",direction_error)
for label in ["final","best"]:
    composites(moving.values,{"Original good affine reference":samples["good_original_affine"],"Rigid: recording moving":samples["recording_moving__"+label],"Rigid: template moving":samples["template_moving__"+label],"Tilt + in-plane affine: template moving":samples["template_moving_unit_normal_affine__"+label]},"Returned poses" if label=="final" else "Best-metric poses",reference_is_red=False)
np.savez_compressed(OUT/"arrays.npz",**{name+"__states":value for name,value in states.items()},**{name+"__native_template":value for name,value in samples.items()})
for path,expected in engine_sources.items():assert digest(ROOT/path)==expected
""")
    sources = [
        Path(__file__).resolve(),
        ROOT / "research/confusius-examples/atlas_centering.py",
        ROOT / "research/confusius-examples/atlas_alignment_investigation.py",
        ROOT / "research/confusius-examples/compare_atlas.py",
        ROOT / "research/confusius-examples/compare_same_subject.py",
        ROOT / "research/confusius-examples/render_notebooks.py",
        *sorted((ROOT / "src/voxreg").glob("*.py")),
    ]
    result = {
        "records": book.namespace["records"],
        "summary": book.namespace["rows"],
        "direction_error": book.namespace["direction_error"],
        "reference_sha256": hashlib.sha256(
            book.namespace["reference_path"].read_bytes()
        ).hexdigest(),
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "confusius_sources": book.namespace["reference"]["confusius_sources"],
        "inputs": book.namespace["reference"]["inputs"],
        "versions": {
            "confusius": cf.__version__,
            "SimpleITK": sitk.Version_VersionString(),
        },
        "production_sources_unchanged": True,
        "centering_issue": "https://github.com/confusius-tools/confusius/issues/537",
    }
    (output / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (output / "index.html").write_text(
        page(
            "CF true rigid controls",
            '<h1>CF true rigid, both directions</h1><a href="rigid.html">Executed traces and overlays</a>',
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
