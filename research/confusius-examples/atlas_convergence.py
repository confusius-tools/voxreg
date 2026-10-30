"""Trace atlas convergence and test longer unchanged-rate runs; no tuning shipped."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from render_notebooks import ROOT, SETUP, Notebook, page


def rigid_residual(candidate_pull, reference_pull, native_slice):
    """Separate a best rigid pose correction from remaining affine disagreement."""
    from voxreg import position_to_world

    points = position_to_world(native_slice) @ np.vstack(
        (np.indices(native_slice.shape).reshape(3, -1), np.ones((1, native_slice.size)))
    )
    candidate = (np.linalg.inv(candidate_pull) @ points)[:3].T
    reference = (np.linalg.inv(reference_pull) @ points)[:3].T
    center_a, center_b = candidate.mean(axis=0), reference.mean(axis=0)
    a, b = candidate - center_a, reference - center_b
    u, _, vt = np.linalg.svd(a.T @ b)
    correction = vt.T @ np.diag([1.0, 1.0, np.linalg.det(vt.T @ u.T)]) @ u.T
    before = float(np.sqrt(np.mean(np.sum((candidate - reference) ** 2, axis=1))))
    after = float(np.sqrt(np.mean(np.sum((a @ correction.T - b) ** 2, axis=1))))
    return {
        "rigid_correction_angle_degrees": float(
            np.degrees(np.arccos(np.clip((np.trace(correction) - 1) / 2, -1, 1)))
        ),
        "center_shift_zyx_mm": (center_b - center_a).tolist(),
        "rms_before_mm": before,
        "rms_after_rigid_correction_mm": after,
        "fraction_squared_error_explained": 1 - after**2 / before**2
        if before > 0
        else None,
    }


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    book = Notebook(
        "Atlas convergence: metric traces and longer runs", "convergence", output
    )
    book.text(
        "<h2>Did the native affine fit really settle?</h2><p>Same original images/manual seed, learning_rate0.1 and affine model. Compare native max500/window50 with fixed500 and fixed1000 (early stopping disabled). No optimizer tuning or default change. Metric callback interval1 deliberately synchronizes CUDA on every evaluation; these are quality diagnostics, not timings. CF reference/curve are reused from the source-hashed both-directions experiment.</p>"
    )
    book.code(SETUP)
    book.code("""
from atlas_convergence import rigid_residual
from voxreg.convergence import WindowConvergence
from voxreg import position_to_world
fixed,moving,placement,paths,recording_shape = inputs()
reference_path = ROOT/"artifacts/atlas-directions/affine/results.json"
reference_record = json.loads(reference_path.read_text())
for path,expected in reference_record["confusius_sources"].items():assert digest(Path(path))==expected
assert [digest(path) for path in paths]==[row["sha256"] for row in reference_record["inputs"][:2]]
cf_pull = np.asarray(reference_record["records"]["slice_to_volume / CF"]["canonical_pull"])
cf_history = reference_record["records"]["slice_to_volume / CF"]["metric_history"]
trace_device = "cuda" if torch.cuda.is_available() else "cpu"
configs = [("native_cpu","cpu",500,50),("native_"+trace_device,trace_device,500,50)] if trace_device=="cuda" else [("native_cpu","cpu",500,50)]
configs += [("fixed500",trace_device,500,501),("fixed1000",trace_device,1000,1001)]
results,traces,records = {},{},{}
for name,device,budget,window in configs:
    torch.set_num_threads(1 if device=="cuda" else 8)
    history = []
    def report(frames,steps,metric):
        assert frames.tolist()==[0]
        history.append({"updates":int(steps[0]),"objective":float(metric[0])})
        if int(steps[0])%100==0:print(name,"updates",int(steps[0]),"NCC²",float(-metric[0]),flush=True)
    result = voxreg.register(moving,fixed,return_transforms=True,transform="affine",initialization=np.linalg.inv(placement),intensity_scaling="none",learning_rate=.1,max_iterations=budget,convergence_window_size=window,device=device,progress=False,metric_callback=report,metric_interval=1)
    results[name],traces[name] = result,history
    best = min(history,key=lambda row:row["objective"])
    assert np.isclose(best["objective"],float(result.diagnostics.objective),atol=1e-12,rtol=0)
    monitor = WindowConvergence(1,50,torch.device("cpu"))
    crossing = None
    for event in history[:-1]:
        value = float(monitor.update(torch.tensor([event["objective"]],dtype=torch.float64),torch.tensor([True]))[0])
        if crossing is None and value<=1e-6:crossing={"updates":event["updates"],"value":value}
    records[name] = {"device":device,"budget":budget,"window":window,"updates":int(result.diagnostics.n_iterations),"status":str(result.diagnostics.status.values),"objective":float(result.diagnostics.objective),"metric_valid_fraction":float(result.diagnostics.valid_fraction),"output_valid_fraction":float(result.valid.values.mean()),"best_trace_update":best["updates"],"raw_final_objective":history[-1]["objective"],"first_window50_crossing_replayed_cpu":crossing,"pull":result.transforms.values.tolist(),"rigid_residual_vs_CF":rigid_residual(result.transforms.values,cf_pull,moving)}
    if name.startswith("native"):
        baseline = np.asarray(reference_record["records"]["slice_to_volume / voxreg "+device]["canonical_pull"])
        assert np.array_equal(result.transforms.values,baseline)
        records[name]["callback_pose_bitwise_matches_prior_no_callback"] = True
native_name = "native_"+trace_device
prefix,seen = [],set()
for event in traces[native_name]:
    if event["updates"] not in seen:
        prefix.append(event);seen.add(event["updates"])
for name in ["fixed500","fixed1000"]:assert traces[name][:len(prefix)]==prefix
print("Native trace is an exact prefix of both no-stopping runs; callback poses match prior unobserved fits.")
""")
    book.text(
        "<h2>Actual metric histories</h2><p>Voxreg traces are raw trial negative NCC²; stored best is the running minimum. Final returned pose can be an earlier best trial. Repeated final clocks are allowed. CF is on a separate panel because its finite-slab metric domain differs; numerical loss values are not an anatomical ranking.</p>"
    )
    book.code("""
fig,axes = plt.subplots(1,2,figsize=(14,4),constrained_layout=True)
for name in [native_name,"fixed500","fixed1000"]:
    x=np.array([v["updates"] for v in traces[name]]);y=np.array([v["objective"] for v in traces[name]])
    axes[0].plot(x,-y,alpha=.55,lw=.8,label=name+" raw")
    axes[0].plot(x,-np.minimum.accumulate(y),lw=1.3,label=name+" retained best")
axes[0].axvline(records[native_name]["updates"],color="black",ls="--",label="Native stop")
axes[0].set(xlabel="Actual parameter updates",ylabel="voxreg native-slice NCC²",title="Does the metric keep improving after native stopping?");axes[0].legend(fontsize=8)
axes[1].plot(-np.asarray(cf_history));axes[1].set(xlabel="CF metric evaluation",ylabel="CF finite-slab NCC²",title="Saved CF reference history — different metric domain")
fig,ax = plt.subplots(figsize=(12,4),constrained_layout=True)
for name in [native_name,"fixed500","fixed1000"]:
    ax.plot([e["updates"] for e in traces[name]],[e["objective"] for e in traces[name]],label=name,lw=.9)
ax.set(xlim=(50,200),xlabel="Updates",ylabel="Raw negative NCC²",title="Zoom around native stopping; lower is better");ax.legend()
""")
    book.code("""
samples,masks = {},{}
samples["CF"],masks["CF"],_ = sample(np.linalg.inv(cf_pull),moving,fixed)
for name,result in results.items():samples[name],masks[name],_ = sample(np.linalg.inv(result.transforms.values),moving,fixed)
common = np.logical_and.reduce(list(masks.values()))
rows = []
for name,result in results.items():
    error = point_error(result.transforms.values,cf_pull,moving)
    records[name]["reference_error"] = error
    records[name]["common_NCC2"] = ncc(moving.values.astype(float),samples[name],common)
    rows.append({"run":name,"updates":records[name]["updates"],"stop":records[name]["status"],"best_NCC2_internal":-records[name]["objective"],"best_at_update":records[name]["best_trace_update"],"common_NCC2":records[name]["common_NCC2"],"CF_reference_RMS_px":error["rms_native_pixels"],**records[name]["rigid_residual_vs_CF"]})
display(pd.DataFrame(rows))
print("Common native support:",float(common.mean()))
composites(moving.values,{name:samples[name] for name in ["CF",native_name,"fixed500","fixed1000"]},"Identical learning rate, different stopping budgets",reference_is_red=False)
np.savez_compressed(OUT/"arrays.npz",**{name+"__pull":result.transforms.values for name,result in results.items()},**{name+"__native_template":value for name,value in samples.items()})
""")
    sources = [
        Path(__file__).resolve(),
        ROOT / "research/confusius-examples/render_notebooks.py",
        ROOT / "research/confusius-examples/compare_atlas.py",
        ROOT / "research/confusius-examples/compare_same_subject.py",
        *sorted((ROOT / "src/voxreg").glob("*.py")),
    ]
    record = {
        "records": book.namespace["records"],
        "traces": book.namespace["traces"],
        "joint_support": float(book.namespace["common"].mean()),
        "reference_results_sha256": hashlib.sha256(
            book.namespace["reference_path"].read_bytes()
        ).hexdigest(),
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
    }
    (output / "results.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    (output / "index.html").write_text(
        page(
            "Atlas convergence",
            '<h1>Atlas convergence</h1><p><a href="convergence.html">Executed comparison</a> · <a href="convergence.ipynb">Notebook</a></p>',
        )
    )
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
