"""Validate caller-fixed atlas fits on original singleton/volume images, without expansion."""

import argparse
import hashlib
import json
from pathlib import Path

from render_notebooks import ROOT, SETUP, Notebook, page


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    book = Notebook("Native caller-fixed atlas registration", "native-fixed", output)
    book.text(
        "<h2>No replicated slab, no internal role reversal</h2><p>Original singleton recording and template crop; manual inverse-equivalent seeds. Both rigid and constrained affine, CPU/CUDA, rate0.1, budget500/window50. Same optimizer; fixed-image geometric pivots. Old good CF pose is a frozen comparison, not a newly run expanded-image fit or anatomical truth. Unsupported template pixels are gray. Metric support uses half-voxel cells; display support uses native centers. No timing comparison.</p>"
    )
    book.code(SETUP)
    book.code("""
from atlas_centered_followup import vox_trace,display_sample,supported_composites
fixed,moving,placement,paths,_=inputs()
assert moving.shape==(1,125,80) and fixed.shape==(20,91,116)
prior=json.loads((ROOT/'artifacts/atlas-centering/comparison/results.json').read_text())
good=np.asarray(prior['records']['recording_moving__origin']['final_canonical_pull'])
records,states={},{}
for model in ['rigid','affine']:
    for direction,moving_is_slice in [('recording_moving',True),('template_moving',False)]:
        source,target,seed=(moving,fixed,np.linalg.inv(placement)) if moving_is_slice else (fixed,moving,placement)
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            name=model+'__'+direction+'__'+device
            row,state=vox_trace(source,target,seed,model=model,device=device)
            assert row['sampled_domains']==[{'moving':list(source.shape),'fixed':list(target.shape)}]
            row['reference_error']=point_error(np.asarray(row['canonical_pull']),good,moving)
            records[name]=row;states[name]=state
            (OUT/(name+'.json')).write_text(json.dumps(row,indent=2,allow_nan=False)+'\\n')
            np.savez_compressed(OUT/(name+'.npz'),recording_to_template_states=state)
            print(name,row['n_iterations'],row['status'],row['reference_error']['rms_native_pixels'],flush=True)
""")
    book.code("""
rows=[];precision={}
for name,row in records.items():
    rows.append({'arm':name,'steps':row['n_iterations'],'best_clock':row['best_clock'],'fixed_metric_support':row['valid_fraction'],'reference_RMS_px':row['reference_error']['rms_native_pixels'],'reference_gate':row['reference_error']['rms_native_pixels']<=.25,'normal_length_error':row['normal_scale_max_error'],'normal_tangent_dot':row['normal_tangent_max_dot']})
    if name.endswith('__cuda'):
        error=point_error(np.asarray(row['canonical_pull']),np.asarray(records[name[:-4]+'cpu']['canonical_pull']),moving)
        precision[name]={'point_error':error,'passed':error['max_native_pixels']<.001}
display(pd.DataFrame(rows));print('Unchanged CPU/CUDA precision gate:');display(pd.DataFrame({k:v['point_error'] for k,v in precision.items()}).T)
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for ax,(model,direction) in zip(axes.ravel(),[(m,d) for m in ['rigid','affine'] for d in ['recording_moving','template_moving']],strict=True):
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        row=records[model+'__'+direction+'__'+device];ax.plot(row['clocks'],row['raw_metric'],label=device);ax.scatter(row['best_clock'],row['best_metric'],s=25)
    ax.set_title(model+' / '+direction);ax.set_xlabel('updates');ax.set_ylabel('own negative squared NCC');ax.legend()
""")
    book.code("""
for model in ['rigid','affine']:
    samples={'Original good CF reference':display_sample(np.linalg.inv(good),moving,fixed)[0]}
    for direction in ['recording_moving','template_moving']:
        row=records[model+'__'+direction+'__cpu']
        samples[direction]=display_sample(np.linalg.inv(np.asarray(row['canonical_pull'])),moving,fixed)[0]
    supported_composites(moving.values,samples,model+' — native inputs, caller-fixed sampling',reference_is_red=False)
""")
    sources = [
        Path(__file__).resolve(),
        ROOT / "research/confusius-examples/atlas_centered_followup.py",
        ROOT / "research/confusius-examples/rigid_affine_atlas.py",
        ROOT / "research/confusius-examples/compare_same_subject.py",
        *sorted((ROOT / "src/voxreg").glob("*.py")),
    ]
    result = {
        "records": book.namespace["records"],
        "comparison": book.namespace["rows"],
        "precision": book.namespace["precision"],
        "inputs": [
            {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in book.namespace["paths"]
        ],
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "versions": {
            "torch": book.namespace["torch"].__version__,
            "confusius": book.namespace["cf"].__version__,
        },
        "reference_sha256": hashlib.sha256(
            (ROOT / "artifacts/atlas-centering/comparison/results.json").read_bytes()
        ).hexdigest(),
        "expanded_data": False,
    }
    (output / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (output / "index.html").write_text(
        page(
            "Native fixed-grid atlas",
            '<a href="native-fixed.html">Executed comparison</a> · <a href="native-fixed.ipynb">Notebook</a>',
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output.resolve())
