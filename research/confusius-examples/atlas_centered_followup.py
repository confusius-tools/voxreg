"""Continue CF past late worsening and audit both already-centered voxreg directions."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk
import torch
from atlas_alignment_investigation import score, sitk_pair
from atlas_centering import physical_center
from atlas_rigid_centered import centered_rigid, centered_unit_normal, linear_matrix
from compare_same_subject import sample as native_sample
from render_notebooks import ROOT, SETUP, Notebook, page

import voxreg


def display_sample(matrix, fixed, moving):
    """Never present nearest-edge extrapolation as an observed anatomical match."""
    values, valid, coordinates = native_sample(matrix, fixed, moving)
    return np.where(valid, values, np.nan), valid, coordinates


def supported_composites(reference, images, title, *, reference_is_red=False):
    import matplotlib.pyplot as plt

    assert not reference_is_red
    fig, axes = plt.subplots(
        1, len(images), figsize=(5 * len(images), 5), constrained_layout=True
    )
    cyan = np.clip((np.asarray(reference)[0].astype(float) + 30) / 30, 0, 1)
    for ax, (name, values) in zip(np.atleast_1d(axes), images.items(), strict=True):
        red = np.clip((np.asarray(values)[0].astype(float) + 30) / 30, 0, 1)
        valid = np.isfinite(red) & np.isfinite(cyan)
        rgb = np.stack((red, cyan, cyan), axis=-1)
        rgb[~valid] = 0.25
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(name)
        ax.set_axis_off()
    fig.suptitle(
        title
        + " — template red / recording cyan; unsupported pixels gray; shared −30…0dB"
    )


def metric_geometry(source, target):
    """Expanded ITK half-voxel support geometry, distinct from native-center support."""
    moving, fixed = sitk_pair(source, target, True)

    def affine(image):
        a = np.eye(4)
        a[:3, :3] = np.array(image.GetDirection()).reshape(3, 3) @ np.diag(
            image.GetSpacing()
        )
        a[:3, 3] = image.GetOrigin()
        return a

    points = affine(fixed) @ np.vstack(
        (
            np.indices(fixed.GetSize()).reshape(3, -1),
            np.ones((1, np.prod(fixed.GetSize()))),
        )
    )
    return (
        np.linalg.inv(affine(moving)),
        points,
        np.asarray(moving.GetSize())[:, None] - 0.5,
    )


def metric_support(source, target, matrix):
    inverse, points, upper = metric_geometry(source, target)
    mapped = (inverse @ matrix @ points)[:3]
    return float(((mapped >= -0.5) & (mapped < upper)).all(axis=0).mean())


def step_probe(source, target, record, index):
    """Interpolate actual optimizer parameters, not a matrix chord leaving the model."""
    model = record["model"]
    seed = np.asarray(record["parameter_states"][0])
    center = record["center_fixed_world_zyx_mm"]
    if model == "rigid":
        transform = centered_rigid(np.eye(4), center)
    elif model == "unit_normal_affine":
        transform = centered_unit_normal(np.eye(4), center)
    else:
        transform = sitk.AffineTransform(3)
        transform.SetCenter(center)
    transform.SetParameters(seed)
    parameters = np.asarray(record["parameter_states"])
    inverse, points, upper = metric_geometry(source, target)
    rows = []
    base_valid = None
    for fraction in [0.0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0]:
        transform.SetParameters(
            parameters[index] + fraction * (parameters[index + 1] - parameters[index])
        )
        matrix = linear_matrix(transform)
        mapped = (inverse @ matrix @ points)[:3]
        valid = ((mapped >= -0.5) & (mapped < upper)).all(axis=0)
        if base_valid is None:
            base_valid = valid
        rows.append(
            {
                "fraction": fraction,
                "metric": -score(source, target, matrix, expand=True),
                "valid_points": int(valid.sum()),
                "changed_support_points": int((valid != base_valid).sum()),
            }
        )
    return {
        "state": index,
        "cached_metric_before": record["raw_metric"][index],
        "cached_metric_after": record["raw_metric"][index + 1],
        "fractions": rows,
    }


def vox_trace(
    source, target, seed, *, model="affine", device="cpu", budget=500, window=50
):
    """Observe without changing the engine; public metrics and physical-state snapshots."""
    engine = importlib.import_module("voxreg.rigid")
    original = engine._optimize
    original_sampler = engine.sample_rigid
    states = []
    losses = []
    clocks = []
    pivots = []
    decoded_seeds = []
    domains = []

    def report(indices, iterations, objective):
        np.testing.assert_array_equal(indices, [0])
        clocks.append(int(iterations[0]))
        losses.append(float(objective[0]))

    def optimize(images, fixed, grid, *args, **kwargs):
        domains.append({"moving": list(images.shape[1:]), "fixed": list(fixed.shape)})
        pivots.append(grid.center.tolist())
        decoded_seeds.append(
            engine.world_matrices(
                kwargs["initial_parameters"].detach().cpu().numpy(), grid
            )[0].tolist()
        )

        def sample(images, parameters, geometry):
            states.append(
                engine.world_matrices(parameters.detach().cpu().numpy(), geometry)[
                    0
                ].copy()
            )
            return original_sampler(images, parameters, geometry)

        with patch.object(engine, "sample_rigid", sample):
            return original(images, fixed, grid, *args, **kwargs)

    torch.set_num_threads(8 if device == "cpu" else 1)
    with patch.object(engine, "_optimize", optimize):
        result = voxreg.register(
            source,
            target,
            transform=model,
            initialization=np.asarray(seed).copy(),
            intensity_scaling="none",
            learning_rate=0.1,
            max_iterations=budget,
            convergence_window_size=window,
            device=device,
            progress=False,
            metric_callback=report,
            metric_interval=1,
        )
    assert engine._optimize is original and engine.sample_rigid is original_sampler
    assert len(states) == len(losses) == len(clocks) and len(pivots) == 1
    np.testing.assert_array_equal(pivots[0], physical_center(target))
    moving_is_slice = source.shape[0] == 1
    np.testing.assert_allclose(decoded_seeds[0], seed, atol=1e-12, rtol=0)
    states = np.stack(states)
    # Normalize captured public pulls to recording-to-volume for geometric diagnostics.
    if moving_is_slice:
        states = np.linalg.inv(states)
    best = int(np.nanargmin(losses))
    public = np.asarray(result.transforms).reshape(-1, 4, 4)[0]
    canonical = public if moving_is_slice else np.linalg.inv(public)
    np.testing.assert_allclose(
        canonical, np.linalg.inv(states[best]), atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        result.diagnostics.objective.item(), losses[best], atol=1e-9, rtol=0
    )
    # Geometry checks use the recording-to-volume map, not necessarily the fitted pull.
    normal = states[:, :3, 0]
    tangents = states[:, :3, 1:3]
    normal_error = float(np.max(np.abs(np.linalg.norm(normal, axis=1) - 1)))
    normal_dot = float(np.max(np.abs(np.einsum("ni,nij->nj", normal, tangents))))
    assert normal_error < 1e-10 and normal_dot < 1e-10
    return {
        "model": model,
        "device": device,
        "settings": {"budget": budget, "window": window, "learning_rate": 0.1},
        "sampled_domains": domains,
        "metric_domain": result.diagnostics.attrs["metric_domain"],
        "valid_fraction": float(result.diagnostics.valid_fraction.item()),
        "raw_metric": losses,
        "clocks": clocks,
        "best_evaluation": best,
        "best_clock": clocks[best],
        "best_metric": losses[best],
        "raw_final_metric": losses[-1],
        "returned_metric": float(result.diagnostics.objective.item()),
        "public_pull": public.tolist(),
        "canonical_pull": canonical.tolist(),
        "last_canonical_pull": np.linalg.inv(states[-1]).tolist(),
        "center_zyx_mm": pivots[0],
        "decoded_seed": decoded_seeds[0],
        "n_iterations": int(result.diagnostics.n_iterations.item()),
        "status": str(result.diagnostics.status.item()),
        "normal_scale_max_error": normal_error,
        "normal_tangent_max_dot": normal_dot,
        "optimized_parameters": int(result.diagnostics.attrs["optimized_parameters"]),
    }, states


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    book = Notebook(
        "Late CF overshoot and the already-centered voxreg fit", "followup", output
    )
    book.text(
        "<h2>Continue, do not assume divergence</h2><p>Three recording-fixed CF models run1000 updates with convergence disabled (window1001), rate1 unchanged. Compare exact prefixes with the previous stopped controls, inspect actual-parameter fractional steps, and show best as well as final results. Recheck voxreg rigid/affine in both public directions on CPU/CUDA at their unchanged rate0.1; voxreg already centers mixed fits at the recording center and fits only native pixel centers. CF expands the singleton slice, so reported fitting losses have different domains. No product code or defaults are changed.</p>"
    )
    book.code(SETUP)
    book.code("""
from atlas_rigid_centered import fit_control
from atlas_centered_followup import step_probe,vox_trace
from atlas_alignment_investigation import plane_pose
fixed,moving,placement,paths,_=inputs()
rigid_path=ROOT/"artifacts/atlas-rigid-centered/final/results.json";center_path=ROOT/"artifacts/atlas-centering/comparison/results.json"
reference=json.loads(rigid_path.read_text());centers=json.loads(center_path.read_text())
for path,expected in reference['confusius_sources'].items():assert digest(Path(path))==expected
assert [digest(path) for path in paths]==[x['sha256'] for x in reference['inputs']]
engine_sources={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'src/voxreg').glob('*.py'))}
cf_records={};cf_states={};probes={}
for model,old in [('affine',centers['records']['template_moving__centered']),('rigid',reference['records']['template_moving']),('unit_normal_affine',reference['records']['template_moving_unit_normal_affine'])]:
    row,state=fit_control(fixed,moving,placement,model=model,max_iterations=1000,window=1001)
    n=len(old['raw_metric'])-1
    np.testing.assert_array_equal(row['raw_metric'][:n],old['raw_metric'][:-1])
    row['stopped_control']=old;row['prefix_verified_events']=n
    peak=old['best_evaluation'];jump=peak+int(np.argmax(np.diff(row['raw_metric'][peak:n+1])))
    probes[model]={'native_peak_step':step_probe(fixed,moving,row,peak),'largest_late_native_jump':step_probe(fixed,moving,row,jump)}
    cf_records[model],cf_states[model]=row,state
    print('CF continued',model,'best',row['best_evaluation'],row['best_metric'],'last',row['final_metric'],flush=True)
(OUT/'cf-checkpoint.json').write_text(json.dumps({'records':cf_records,'probes':probes},indent=2))
np.savez_compressed(OUT/'cf-checkpoint.npz',**cf_states)
""")
    book.code("""
vox_records={};vox_states={}
for model in ['rigid','affine']:
    for device in ['cpu','cuda'] if torch.cuda.is_available() else ['cpu']:
        for direction,is_slice in [('recording_moving',True),('template_moving',False)]:
            source,target,seed=(moving,fixed,np.linalg.inv(placement)) if is_slice else (fixed,moving,placement)
            name=model+'__'+device+'__'+direction
            row,state=vox_trace(source,target,seed,model=model,device=device)
            vox_records[name],vox_states[name]=row,state
            print('voxreg',name,'pivot',row['center_zyx_mm'],'steps',row['n_iterations'],'best',row['best_clock'],row['returned_metric'],flush=True)
checks={}
for model in ['rigid','affine']:
    left=vox_records[model+'__cpu__recording_moving'];right=vox_records[model+'__cpu__template_moving']
    np.testing.assert_array_equal(left['canonical_pull'],right['canonical_pull']);np.testing.assert_array_equal(left['raw_metric'],right['raw_metric'])
    checks[model]={'directions_bitwise_equal':True}
    if torch.cuda.is_available():
        gpu=vox_records[model+'__cuda__template_moving']
        error=point_error(np.asarray(gpu['canonical_pull']),np.asarray(right['canonical_pull']),moving)
        assert error['max_native_pixels']<=.001
        checks[model]['cpu_cuda_point_error']=error
old_vox=json.loads((ROOT/'artifacts/atlas-directions/affine/results.json').read_text())
for device in ['cpu','cuda'] if torch.cuda.is_available() else ['cpu']:
    for direction,old_direction in [('recording_moving','slice_to_volume'),('template_moving','volume_to_slice')]:
        np.testing.assert_array_equal(vox_records['affine__'+device+'__'+direction]['canonical_pull'],old_vox['records'][old_direction+' / voxreg '+device]['canonical_pull'])
checks['affine']['matches_previous_default_bitwise']=True
""")
    book.code(r"""
fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
for col,model in enumerate(['affine','rigid','unit_normal_affine']):
    row=cf_records[model];old=row['stopped_control'];loss=np.asarray(row['raw_metric'])
    for ax in axes[:,col]:
        ax.plot(loss,label='No stopping:1000 updates',lw=1.)
        ax.plot(old['raw_metric'],label='Original window50',lw=1.2)
        ax.plot(row['best_evaluation'],row['best_metric'],'o',label='Best continued pose',ms=4)
        ax.axvline(len(old['raw_metric'])-1,color='gray',ls=':',lw=1)
        ax.set(xlabel='Evaluated parameter state',ylabel='Negative NCC² — lower is better');ax.grid(alpha=.2)
    axes[0,col].set_title(model);axes[0,col].legend(fontsize=7)
    axes[1,col].set_xlim(0,200)
fig.suptitle('CF template moving: continue the same trajectory, not a restart; lower row zooms the first200 states')
fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
for ax,model in zip(axes,['rigid','affine'],strict=True):
    for direction in ['recording_moving','template_moving']:
        row=vox_records[model+'__cpu__'+direction]
        ax.plot(row['clocks'],row['raw_metric'],label='voxreg CPU '+direction,lw=1.3)
    match='rigid' if model=='rigid' else 'unit_normal_affine'
    cfrow=cf_records[match]['stopped_control'];ax.plot(cfrow['raw_metric'],'--',label='CF template moving (expanded domain)',lw=1)
    ax.set(title=model,xlabel='Optimizer updates',ylabel='Negative NCC²');ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.suptitle('voxreg is already centered; its two directions coincide. CF and voxreg objectives sample different domains.')
""")
    book.code(r"""
from atlas_centered_followup import display_sample as sample,supported_composites as composites,metric_support
good=np.asarray(centers['records']['recording_moving__origin']['final_canonical_pull'])
rows=[];samples={}
def summarize(name,pull,metric,kind):
    pull=np.asarray(pull);error=point_error(pull,good,moving);pose=plane_pose(pull,moving)
    values,valid,_=sample(np.linalg.inv(pull),moving,fixed)
    rows.append({'run':name,'pose':kind,'own_NCC2':-metric,'RMS_px_vs_original_good_CF':error['rms_native_pixels'],'depth_mm':pose['center_zyx_mm'][0],'tilt_degrees':pose['tilt_degrees'],'native_center_support':float(valid.mean()),'CF_expanded_metric_support':metric_support(fixed,moving,np.linalg.inv(pull)) if name.startswith('CF') else None})
    samples[name+'__'+kind]=values
for model,row in cf_records.items():
    for label in ['best','final']:summarize('CF1000_'+model,np.linalg.inv(row[label+'_pull']),row[label+'_metric'],label)
    old=row['stopped_control'];summarize('CFstopped_'+model,np.linalg.inv(old['best_pull']),old['best_metric'],'best');summarize('CFstopped_'+model,np.linalg.inv(old['final_pull']),old['final_metric'],'final')
for name,row in vox_records.items():
    summarize('vox_'+name,row['canonical_pull'],row['returned_metric'],'returned_best')
    summarize('vox_'+name,row['last_canonical_pull'],row['raw_final_metric'],'raw_last')
display(pd.DataFrame(rows))
matched_CF={}
for model in ['rigid','affine']:
    for direction in ['recording_moving','template_moving']:
        v=vox_records[model+'__cpu__'+direction]
        if model=='rigid':c=np.asarray(reference['records'][direction]['final_canonical_pull'])
        elif direction=='recording_moving':c=np.asarray(centers['records']['recording_moving__centered']['final_canonical_pull'])
        else:c=np.linalg.inv(cf_records['unit_normal_affine']['stopped_control']['final_pull'])
        matched_CF[model+'__'+direction]=point_error(np.asarray(v['canonical_pull']),c,moving)
print('Same-direction CF disagreements:',matched_CF)
composites(moving.values,{'CF rigid: recording moving':sample(np.linalg.inv(np.asarray(reference['records']['recording_moving']['final_canonical_pull'])),moving,fixed)[0],'CF rigid: template moving':samples['CFstopped_rigid__final'],'voxreg rigid: recording moving':samples['vox_rigid__cpu__recording_moving__returned_best'],'voxreg rigid: template moving':samples['vox_rigid__cpu__template_moving__returned_best']},'Returned rigid fits; no model can scale or shear',reference_is_red=False)
composites(moving.values,{'Original good CF affine':sample(np.linalg.inv(good),moving,fixed)[0],'CF unit-normal: best before stop':samples['CFstopped_unit_normal_affine__best'],'voxreg affine: recording moving':samples['vox_affine__cpu__recording_moving__returned_best'],'voxreg affine: template moving':samples['vox_affine__cpu__template_moving__returned_best']},'Tilt and in-plane affine: native-grid comparison',reference_is_red=False)
composites(moving.values,{'CF unit-normal: best before stop':samples['CFstopped_unit_normal_affine__best'],'CF unit-normal: best over1000':samples['CF1000_unit_normal_affine__best'],'CF unit-normal: state1000':samples['CF1000_unit_normal_affine__final']},'Does continuing recover the alignment?',reference_is_red=False)
np.savez_compressed(OUT/'arrays.npz',**{'CF__'+k+'__states':v for k,v in cf_states.items()},**{'vox__'+k+'__states':v for k,v in vox_states.items()},**samples)
for path,expected in engine_sources.items():assert digest(ROOT/path)==expected
""")
    sources = [
        Path(__file__).resolve(),
        ROOT / "research/confusius-examples/atlas_rigid_centered.py",
        ROOT / "research/confusius-examples/atlas_centering.py",
        ROOT / "research/confusius-examples/atlas_alignment_investigation.py",
        ROOT / "research/confusius-examples/render_notebooks.py",
        ROOT / "research/confusius-examples/rigid_affine_atlas.py",
        ROOT / "research/confusius-examples/compare_atlas.py",
        ROOT / "research/confusius-examples/compare_same_subject.py",
        *sorted((ROOT / "src/voxreg").glob("*.py")),
    ]
    result = {
        "cf_records": book.namespace["cf_records"],
        "vox_records": book.namespace["vox_records"],
        "probes": book.namespace["probes"],
        "checks": book.namespace["checks"],
        "matched_CF": book.namespace["matched_CF"],
        "summary": book.namespace["rows"],
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "inputs": book.namespace["reference"]["inputs"],
        "confusius_sources": book.namespace["reference"]["confusius_sources"],
        "reference_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [book.namespace["rigid_path"], book.namespace["center_path"]]
        },
        "production_sources_unchanged": True,
    }
    (output / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (output / "index.html").write_text(
        page(
            "Centered follow-up",
            '<h1>Late overshoot and voxreg centering</h1><a href="followup.html">Executed curves and comparisons</a>',
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
