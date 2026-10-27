"""Bounded example01 rate comparison; unchanged100-step budget/native stopping."""

import argparse
import hashlib
import json
from pathlib import Path

from render_notebooks import ROOT, SETUP, Notebook


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    book = Notebook(
        "Same-subject rigid: larger learning rates", "learning-rate", output
    )
    book.text(
        "<h2>Change learning rate only</h2><p>Original full rat75/qform/dB pair. Fresh CF rigid reference, and voxreg rates0.1,0.3,1.0. All voxreg runs retain geometry initialization,100 maximum updates, window50 and the same device. No timing claim, engine edit or default change. CF placement remains the reference, independent of NCC.</p>"
    )
    book.code(SETUP)
    book.code("""
fixed,moving,INPUTS = load_inputs()
cf_image,cf_matrix,cf_diag = cf.registration.register_volume(
    moving=moving,fixed=fixed,transform_type="rigid",sitk_threads=8,
    show_progress=False,plot_metric=False,plot_composite=False)
fits = {}
for rate in [.1,.3,1.]:
    fits[rate] = voxreg.register(moving,fixed,transform="rigid",
        initialization="center_geometry",intensity_scaling="none",
        max_iterations=100,convergence_window_size=50,learning_rate=rate,
        device=DEVICE,progress=False)
assert set(fits) == {.1,.3,1.}
for result in fits.values():
    assert result.registered.dtype == moving.dtype
    assert int(result.diagnostics.n_iterations) <= 100
    linear = result.transforms.values[:3,:3]
    assert np.isfinite(result.transforms.values).all()
    assert np.allclose(linear.T@linear,np.eye(3),atol=1e-10,rtol=0)
""")
    book.code("""
samples,masks = {},{}
samples["CF"],masks["CF"],_ = sample(cf_matrix,fixed,moving)
for rate,result in fits.items():
    name = f"voxreg LR {rate}"
    samples[name],masks[name],_ = sample(result.transforms.values,fixed,moving)
common = np.logical_and.reduce(list(masks.values()))
rows = [{"method":"CF","iterations":int(cf_diag.n_iterations),"stop":cf_diag.stop_condition,
         "common_NCC2":ncc(fixed.values,samples["CF"],common),"reference_RMS_px":0.,"reference_max_px":0.,"reference_gate_pass":True}]
for rate,result in fits.items():
    name = f"voxreg LR {rate}"
    error = point_error(result.transforms.values,cf_matrix,moving)
    rows.append({"method":name,"iterations":int(result.diagnostics.n_iterations),
        "stop":str(result.diagnostics.status.values),"common_NCC2":ncc(fixed.values,samples[name],common),
        "reference_RMS_px":error["rms_native_pixels"],"reference_max_px":error["max_native_pixels"],
        "reference_gate_pass":error["rms_native_pixels"]<=.25})
display(pd.DataFrame(rows))
print("Shared fixed-grid support:",float(common.mean()))
print("Pose error: inverse pulls on all moving slice pixel centers; native minimum pixel spacing. Gate:0.25px RMS.")
composites(fixed.values,{"CF rigid":cf_image.values,**{f"voxreg LR {rate}":result.registered.values for rate,result in fits.items()}},"100-step cap; only voxreg learning rate changes")
fig,axes = plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
for ax,(rate,result) in zip(axes,fits.items(),strict=True):
    difference = np.ma.array((result.registered.values-cf_image.values)[0],mask=~common[0])
    image=ax.imshow(difference,cmap="coolwarm",vmin=-10,vmax=10)
    ax.set_title(f"LR {rate}: voxreg − CF");ax.set_axis_off()
fig.colorbar(image,ax=axes,label="dB; shared support/range",shrink=.7)
np.savez_compressed(OUT/"arrays.npz",fixed=fixed.values,moving=moving.values,cf=cf_image.values,cf_matrix=cf_matrix,
    **{f"voxreg_{rate}":result.registered.values for rate,result in fits.items()},
    **{f"matrix_{rate}":result.transforms.values for rate,result in fits.items()})
""")
    record = {
        "rows": book.namespace["rows"],
        "common_support": float(book.namespace["common"].mean()),
        "inputs": book.namespace["INPUTS"],
        "device": book.namespace["DEVICE"],
        "versions": {
            "confusius": book.namespace["cf"].__version__,
            "torch": book.namespace["torch"].__version__,
        },
        "sources": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__).resolve(),
                ROOT / "research/confusius-examples/render_notebooks.py",
                ROOT / "research/confusius-examples/compare_same_subject.py",
                ROOT / "research/confusius-examples/rigid_affine_atlas.py",
                *sorted((ROOT / "src/voxreg").glob("*.py")),
            ]
        },
    }
    (output / "results.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
