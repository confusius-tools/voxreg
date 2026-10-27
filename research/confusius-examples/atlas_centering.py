"""One-factor CF affine-center comparison in both supplied image directions."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from atlas_direction_learning_rate import fit_curve
from confusius.registration.affines import (
    affine_to_sitk_linear_transform,
    sitk_linear_transform_to_affine,
)
from render_notebooks import ROOT, SETUP, Notebook, page

from voxreg import position_to_world


def physical_center(data):
    """Center of voxel centers in fixed-image world coordinates, not array indices."""
    return (position_to_world(data) @ np.r_[(np.asarray(data.shape) - 1) / 2, 1])[:3]


def centered_affine(matrix, center):
    """Change the affine parameterization, never the supplied world mapping."""
    matrix = np.asarray(matrix, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    transform = affine_to_sitk_linear_transform(matrix)
    transform.SetCenter(tuple(center))
    transform.SetTranslation(
        tuple(matrix[:3, 3] + (matrix[:3, :3] - np.eye(3)) @ center)
    )
    np.testing.assert_allclose(
        sitk_linear_transform_to_affine(transform), matrix, atol=1e-12, rtol=0
    )
    return transform


def fit_centered(source, target, seed, center):
    """Scoped research patch of seed construction; same actual CF fitter/settings."""
    module = importlib.import_module("confusius.registration.volume")
    original = module.affine_to_sitk_linear_transform
    calls = []

    def convert(matrix):
        transform = centered_affine(matrix, center)
        calls.append(tuple(transform.GetCenter()))
        return transform

    with patch.object(module, "affine_to_sitk_linear_transform", convert):
        record, states = fit_curve(source, target, seed)
    assert module.affine_to_sitk_linear_transform is original and len(calls) == 1
    np.testing.assert_array_equal(calls[0], center)
    return record, states


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    book = Notebook("CF centering: both registration directions", "centering", output)
    book.text(
        "<h2>Only the affine center changes</h2><p>Compare origin-pivot versus fixed-image-center pivot in each direction. Left: fixed template slab center. Right: fixed recording center. This is ITK geometric centering, not a claim that this pivot is uniquely optimal. Preserve the exact original manual world-space seed. All four fits use CF affine/correlation, learning_rate1, max500/window50,8threads and the same expanded-slice handling. No production/source/default changes.</p>"
    )
    book.code(SETUP)
    book.code("""
from atlas_centering import physical_center,fit_centered
from atlas_direction_learning_rate import fit_curve
from atlas_alignment_investigation import plane_pose
fixed,moving,placement,paths,_=inputs()
reference_path=ROOT/"artifacts/atlas-directions/affine/results.json"
reference=json.loads(reference_path.read_text())
for path,expected in reference["confusius_sources"].items():assert digest(Path(path))==expected
assert [digest(path) for path in paths]==[row["sha256"] for row in reference["inputs"][:2]]
engine_sources={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/"src/voxreg").glob("*.py"))}
records,states={},{}
for direction,moving_is_slice in [("recording_moving",True),("template_moving",False)]:
    source,target,seed=(moving,fixed,np.linalg.inv(placement)) if moving_is_slice else (fixed,moving,placement)
    for centered in [False,True]:
        name=direction+("__centered" if centered else "__origin")
        center=physical_center(target) if centered else np.zeros(3)
        record,state=fit_centered(source,target,seed,center) if centered else fit_curve(source,target,seed)
        record.update(direction=direction,center_fixed_world_zyx_mm=center.tolist(),initial_pull=seed.tolist())
        for label in ["best","final"]:
            pull=np.asarray(record[label+"_pull"])
            canonical=pull if moving_is_slice else np.linalg.inv(pull)
            record[label+"_canonical_pull"]=canonical.tolist()
            record[label+"_plane"]=plane_pose(canonical,moving)
            singular=np.linalg.svd(pull[:3,:3],compute_uv=False)
            record[label+"_linear_singular_values"]=singular.tolist()
            record[label+"_linear_determinant"]=float(np.linalg.det(pull[:3,:3]))
        records[name],states[name]=record,state
        print(name,"center",center,"updates",record["diagnostics"]["n_iterations"],"best",record["best_metric"],"returned",record["final_metric"],flush=True)
    np.testing.assert_allclose(records[direction+"__centered"]["raw_metric"][0],records[direction+"__origin"]["raw_metric"][0],atol=1e-12,rtol=0)
for name,key in [("recording_moving__origin","slice_to_volume / CF"),("template_moving__origin","volume_to_slice / CF")]:
    np.testing.assert_array_equal(records[name]["final_pull"],reference["records"][key]["returned_pull"])
    np.testing.assert_array_equal(records[name]["diagnostics"]["metric_values"],reference["records"][key]["metric_history"])
""")
    book.code(r"""
fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
for ax,direction,title in zip(axes,["recording_moving","template_moving"],["Recording moving → template fixed","Template moving → recording fixed"],strict=True):
    for suffix,label in [("origin","World-origin center"),("centered","Fixed-image center")]:
        row=records[direction+"__"+suffix];loss=np.asarray(row["raw_metric"])
        line,=ax.plot(loss,label=label,lw=1.5)
        ax.plot(row["best_evaluation"],row["best_metric"],"o",color=line.get_color(),ms=5)
        ax.plot(len(loss)-1,loss[-1],"s",color=line.get_color(),ms=5)
    ax.set(title=title,xlabel="Evaluated parameter state (0 = manual seed)",ylabel="Negative NCC² — lower is better");ax.grid(alpha=.2);ax.legend()
fig.suptitle("Only the pivot changes; circle = best metric, square = returned pose\nDifferent sampling domains across panels: absolute losses are not an anatomical ranking",fontsize=11)
""")
    book.code(r"""
reference_pull=np.asarray(records["recording_moving__origin"]["final_canonical_pull"])
samples,rows={},[]
for name,row in records.items():
    for label in ["best","final"]:
        pull=np.asarray(row[label+"_canonical_pull"])
        samples[name+"__"+label],_,_=sample(np.linalg.inv(pull),moving,fixed)
        row[label+"_reference_error"]=point_error(pull,reference_pull,moving)
    rows.append({"run":name,"updates":row["diagnostics"]["n_iterations"],"best_at":row["best_evaluation"],"returned_NCC2":-row["final_metric"],"best_NCC2":-row["best_metric"],"returned_RMS_px_vs_original_good_left":row["final_reference_error"]["rms_native_pixels"],"best_RMS_px_vs_original_good_left":row["best_reference_error"]["rms_native_pixels"],"returned_min_linear_singular_value":min(row["final_linear_singular_values"]),"best_min_linear_singular_value":min(row["best_linear_singular_values"])})
display(pd.DataFrame(rows))
recording_gray=np.clip((moving.values[0].astype(float)+30)/30,0,1)
for label,title in [("final","Returned CF poses"),("best","Best-metric poses — not automatic anatomical successes")]:
    fig,axes=plt.subplots(2,2,figsize=(8,10),constrained_layout=True)
    for r,direction in enumerate(["recording_moving","template_moving"]):
        for c,suffix in enumerate(["origin","centered"]):
            name=direction+"__"+suffix;red=np.clip((samples[name+"__"+label][0].astype(float)+30)/30,0,1)
            axes[r,c].imshow(np.stack((red,recording_gray,recording_gray),axis=-1),interpolation="nearest")
            axes[r,c].set_title(("Recording moving" if r==0 else "Template moving")+"\n"+("World-origin center" if c==0 else "Fixed-image center"),fontsize=11)
            axes[r,c].set_axis_off()
    fig.suptitle(title+"\nTemplate red / recording cyan; same native grid and −30…0dB display",fontsize=12)
np.savez_compressed(OUT/"arrays.npz",**{name+"__states":value for name,value in states.items()},**{name+"__native_template":value for name,value in samples.items()})
for path,expected in engine_sources.items():assert digest(ROOT/path)==expected
""")
    sources = [
        Path(__file__).resolve(),
        ROOT / "research/confusius-examples/atlas_direction_learning_rate.py",
        ROOT / "research/confusius-examples/atlas_alignment_investigation.py",
        ROOT / "research/confusius-examples/render_notebooks.py",
        ROOT / "research/confusius-examples/compare_atlas.py",
        ROOT / "research/confusius-examples/compare_same_subject.py",
        *sorted((ROOT / "src/voxreg").glob("*.py")),
    ]
    result = {
        "records": book.namespace["records"],
        "summary": book.namespace["rows"],
        "inputs": [
            {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in book.namespace["paths"]
        ],
        "reference_sha256": hashlib.sha256(
            book.namespace["reference_path"].read_bytes()
        ).hexdigest(),
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "confusius_sources": book.namespace["reference"]["confusius_sources"],
        "production_sources_unchanged": True,
    }
    (output / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (output / "index.html").write_text(
        page(
            "CF centering comparison",
            '<h1>CF centering: both directions</h1><a href="centering.html">Executed curves and overlays</a>',
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
