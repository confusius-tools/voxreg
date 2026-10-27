"""Compare atlas resampling at the literal manual seed, with zero optimizer updates.

Research-only optimizer bypass exercises register's seed decoding, both output
paths, dtype conversion and matrix export. It is not a new public no-fit API.
"""

import argparse
import hashlib
import importlib
import json
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from render_notebooks import ROOT, SETUP, Notebook, page


def held_register(moving, fixed, pull, *, device="cpu"):
    """Return the production pipeline's seeded output without invoking its optimizer."""
    engine = importlib.import_module("voxreg.rigid")
    before = np.asarray(pull).copy()
    calls = []

    def hold(
        images, _fixed, _plane, _reference, *_args, initial_parameters=None, **_kwargs
    ):
        assert initial_parameters is not None
        calls.append(initial_parameters.detach().cpu().numpy())
        return (
            initial_parameters.detach().clone(),
            torch.zeros(len(images), dtype=torch.int64, device=images.device),
            torch.full((len(images),), 5, dtype=torch.int64, device=images.device),
        )

    with (
        patch.object(engine, "_optimize", hold),
        warnings.catch_warnings(record=True) as caught,
    ):
        result = engine.register(
            moving,
            fixed,
            transform="affine",
            initialization=pull,
            intensity_scaling="none",
            max_iterations=1,
            device=device,
            progress=False,
        )
    assert len(calls) == 1
    assert int(result.diagnostics.n_iterations) == 0
    np.testing.assert_array_equal(pull, before)
    assert result.registered.dtype == moving.dtype
    return result, [str(item.message) for item in caught]


def compare_arrays(cf_values, cf_valid, vr_values, vr_valid):
    common = cf_valid & vr_valid
    difference = vr_values.astype(float) - cf_values.astype(float)
    return {
        "common_centers": int(common.sum()),
        "cf_valid_centers": int(cf_valid.sum()),
        "voxreg_valid_centers": int(vr_valid.sum()),
        "support_disagreement_centers": int((cf_valid != vr_valid).sum()),
        "common_max_abs": float(np.abs(difference[common]).max())
        if common.any()
        else None,
        "common_rms": float(np.sqrt(np.mean(difference[common] ** 2)))
        if common.any()
        else None,
        "whole_output_max_abs": float(np.abs(difference).max()),
    }


def mapping_error(actual, expected, target, source):
    from voxreg import position_to_world

    points = position_to_world(target) @ np.vstack(
        (np.indices(target.shape).reshape(3, -1), np.ones((1, target.size)))
    )
    delta = ((actual - expected) @ points)[:3]
    source_delta = np.linalg.solve(position_to_world(source)[:3, :3], delta)
    return {
        "max_world_mapping_error_mm": float(np.linalg.norm(delta, axis=0).max()),
        "max_source_pixel_mapping_error": float(
            np.linalg.norm(source_delta, axis=0).max()
        ),
        "matrix_max_abs": float(np.abs(actual - expected).max()),
    }


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    book = Notebook(
        "Atlas manual placement: zero-update control", "initial-placement", output
    )
    book.text(
        "<h2>The supplied affine alone — no registration</h2><p>Same original template slab, recording mean, qform/dB preprocessing and manual placement. CF uses its public resample_like. Voxreg has no public no-fit resampler: a research-only, scoped optimizer bypass returns the initial physical parameters unchanged while exercising production seed decoding, final interpolation and matrix export. Zero updates are asserted. No package/engine files are patched; diagnostic status is artificial and must not be interpreted as convergence.</p>"
    )
    book.code(SETUP)
    book.code("""
from atlas_seed_control import held_register, compare_arrays, mapping_error
from voxreg import position_to_world
fixed,moving,placement,paths,recording_shape = inputs()
seed = np.linalg.inv(placement)
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
records, saved, native, original = [], {}, {}, {}
cases = [("template_on_native_recording",fixed,moving,placement,native),
         ("recording_on_template_grid",moving,fixed,seed,original)]
for case,source,target,pull,outputs in cases:
    cf_values = cf.registration.resample_like(source,target,pull,interpolation="linear",fill_value=float(source.min()),sitk_threads=8).values
    ones = source.copy(data=np.ones(source.shape,dtype=np.float64))
    cf_valid = cf.registration.resample_like(ones,target,pull,interpolation="linear",fill_value=0.,sitk_threads=8).values > .5
    oracle,_,coordinates = sample(pull,target,source)
    sizes = np.array(source.shape)[:,None]
    within = (coordinates >= -1e-5) & (coordinates <= sizes-1+1e-5)
    oracle_valid = np.where(sizes==1,(coordinates>=-.5)&(coordinates<.5),within).all(axis=0).reshape(target.shape)
    outputs["CF"] = cf_values
    saved[case+"__cf"] = cf_values
    saved[case+"__cf_valid"] = cf_valid
    for device in DEVICES:
        result,caught = held_register(source,target,pull,device=device)
        errors = mapping_error(result.transforms.values,pull,target,source)
        assert errors["max_source_pixel_mapping_error"] < .001
        assert np.array_equal(result.valid.values,oracle_valid)
        values = result.registered.values
        shared = result.valid.values & cf_valid
        oracle_error = float(np.abs(values[shared]-oracle[shared]).max()) if shared.any() else None
        record = {"case":case,"device":device,"optimizer_updates":int(result.diagnostics.n_iterations),
            "source_dtype":str(source.dtype),"output_dtype":str(result.registered.dtype),
            **errors,**compare_arrays(cf_values,cf_valid,values,result.valid.values),
            "voxreg_vs_scipy_common_max_abs":oracle_error,"warnings":caught}
        records.append(record);outputs["voxreg "+device] = values
        saved[case+"__"+device] = values
        saved[case+"__"+device+"_valid"] = result.valid.values
print("Original supplied template→recording pull:\\n",seed)
print("Inverse pull used for template-on-recording view:\\n",placement)
display(pd.DataFrame(records).drop(columns=["warnings"]))
for record in records:
    for message in record["warnings"]:print(record["case"],record["device"],message)
""")
    book.text(
        "<h2>Native recording view: identical initial placement?</h2><p>Volume sampled on the native recording grid using the inverse original pull. All panels use exactly the same transform, not a fitted result. Template red / recording cyan; shared −30…0dB range. The anatomy need not align yet; this tests whether both libraries show the same starting point.</p>"
    )
    book.code("""
composites(moving.values,native,"Manual seed only — zero updates",reference_is_red=False)
common = saved["template_on_native_recording__cf_valid"] & saved["template_on_native_recording__cpu_valid"]
fig,axes = plt.subplots(1,len(DEVICES),figsize=(6*len(DEVICES),5),constrained_layout=True)
for ax,device in zip(np.atleast_1d(axes),DEVICES,strict=True):
    delta = np.ma.array((native["voxreg "+device]-native["CF"])[0],mask=~common[0])
    image = ax.imshow(delta,cmap="coolwarm",vmin=-1e-8,vmax=1e-8)
    ax.set_title(device+" − CF; common centers");ax.set_axis_off()
fig.colorbar(image,ax=np.atleast_1d(axes),label="dB difference; ±1e-8 display range",shrink=.7)
fig,axes = plt.subplots(1,2,figsize=(10,5),constrained_layout=True)
for ax,name in zip(axes,["cf_valid","cpu_valid"],strict=True):
    ax.imshow(saved["template_on_native_recording__"+name][0],vmin=0,vmax=1,cmap="gray");ax.set_title(name);ax.set_axis_off()
""")
    book.text(
        "<h2>Original recording→template-grid output</h2><p>These use the original pull, not its inverse. Both libraries give the singleton voxel finite affine-defined support. Voxreg retains its existing sample-center bounds on active axes, whereas CF includes half-voxel margins there too. Compare the nearest template plane and coverage by plane; any remaining edge-support differences are separate from slice thickness.</p>"
    )
    book.code("""
af,am = position_to_world(fixed),position_to_world(moving)
plane_z = float((placement @ np.r_[am[:3,3],1.])[0])
z = (af @ np.vstack((np.arange(fixed.sizes["k"]),np.zeros((2,fixed.sizes["k"])),np.ones(fixed.sizes["k"]))))[0]
closest = int(np.abs(z-plane_z).argmin())
print("Placed recording plane z (mm):",plane_z)
print("Closest template plane z (mm):",float(z[closest]),"offset (mm):",float(z[closest]-plane_z))
fig,axes = plt.subplots(1,len(original),figsize=(5*len(original),5),constrained_layout=True)
for ax,(name,values) in zip(np.atleast_1d(axes),original.items(),strict=True):
    image=ax.imshow(values[closest],cmap="gray",vmin=-30,vmax=0);ax.set_title(name+" — nearest template plane");ax.set_axis_off()
fig.colorbar(image,ax=np.atleast_1d(axes),label="Recording dB (same limits)",shrink=.7)
fig,ax = plt.subplots(figsize=(10,3),constrained_layout=True)
for name in ["cf_valid",*[device+"_valid" for device in DEVICES]]:
    ax.plot(z,saved["recording_on_template_grid__"+name].mean(axis=(1,2)),marker=".",label=name)
ax.axvline(plane_z,color="black",ls="--",label="Exact recording plane")
ax.set(xlabel="Template plane z (mm)",ylabel="Valid fraction",title="Same transform; different single-slice support policies");ax.legend()
np.savez_compressed(OUT/"arrays.npz",seed=seed,placement=placement,**saved)
""")
    source_files = [
        Path(__file__).resolve(),
        ROOT / "research/confusius-examples/render_notebooks.py",
        ROOT / "research/confusius-examples/compare_atlas.py",
        ROOT / "research/confusius-examples/compare_same_subject.py",
        *sorted((ROOT / "src/voxreg").glob("*.py")),
    ]
    cf_root = Path(book.namespace["cf"].__file__).parent
    result = {
        "records": book.namespace["records"],
        "inputs": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in book.namespace["paths"]
        ],
        "versions": {
            "confusius": book.namespace["cf"].__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "placed_plane_z_mm": book.namespace["plane_z"],
        "nearest_template_z_mm": float(book.namespace["z"][book.namespace["closest"]]),
        "sources": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
        "confusius_sources": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((cf_root / "registration").glob("*.py"))
        },
    }
    (output / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (output / "index.html").write_text(
        page(
            "Atlas initial-placement control",
            '<h1>Atlas initial-placement control</h1><p><a href="initial-placement.html">Open executed comparison</a> · <a href="initial-placement.ipynb">Executed notebook</a></p>',
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
