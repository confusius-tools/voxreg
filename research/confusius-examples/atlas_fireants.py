"""FireANTs atlas controls on CF's expanded images, plus fixed-grid voxreg ablation."""

import argparse
import hashlib
import inspect
import json
from importlib.metadata import version
from pathlib import Path

import confusius as cf
import numpy as np
import SimpleITK as sitk
import torch
from confusius.registration.volume import expand_thin_dims, voxeldata_to_sitk_image
from render_notebooks import ROOT, SETUP, Notebook, page


def sitk_input(data, *, expanded=True, channels=False):
    image = voxeldata_to_sitk_image(data)
    if expanded:
        image = expand_thin_dims(image)
    if channels:
        values = sitk.GetArrayFromImage(image)
        vector = sitk.GetImageFromArray(
            np.stack((values, np.ones_like(values)), axis=-1), isVector=True
        )
        vector.CopyInformation(image)
        image = vector
    return image


def expanded_voxeldata(data):
    image = sitk_input(data)
    affine = np.eye(4)
    affine[:3, :3] = np.array(image.GetDirection()).reshape(3, 3) @ np.diag(
        image.GetSpacing()
    )
    affine[:3, 3] = image.GetOrigin()
    return cf.create_voxeldata(
        sitk.GetArrayFromImage(image).T.copy(),
        dims=("k", "j", "i"),
        voxel_to_world=affine,
    )


def masked_global_ncc(moved, target):
    """Same custom float32 masked global NCC as the existing FireANTs benchmark."""
    axes = tuple(range(1, moved.ndim - 1))
    weights = (moved[:, 1] >= 1 - 1e-5).to(moved.dtype)
    count = weights.sum(dim=axes, keepdim=True).clamp_min(1)
    x, y = moved[:, 0], target[:, 0]
    x = (x - (x * weights).sum(dim=axes, keepdim=True) / count) * weights
    y = (y - (y * weights).sum(dim=axes, keepdim=True) / count) * weights
    vx, vy = x.square().sum(dim=axes), y.square().sum(dim=axes)
    denominator = vx * vy
    good = (
        (count.reshape(len(moved)) > 1)
        & (vx > 0)
        & (vy > 0)
        & torch.isfinite(denominator)
        & (denominator > torch.finfo(moved.dtype).tiny)
    )
    safe = torch.where(good, denominator, torch.ones_like(denominator))
    return torch.where(good, -(x * y).sum(dim=axes).square() / safe, 0).sum()


def fit_fireants(
    source,
    target,
    seed,
    *,
    model="affine",
    metric="local_cc",
    device="cuda",
    budget=500,
):
    from fireants.interpolator import fireants_interpolator
    from fireants.io.image import BatchedImages, Image
    from fireants.registration.affine import AffineRegistration
    from fireants.registration.rigid import RigidRegistration

    if model not in {"rigid", "affine"} or metric not in {"local_cc", "global_ncc"}:
        raise ValueError("Unknown model or metric")
    channels = metric == "global_ncc"
    moving = BatchedImages(
        [Image(sitk_input(source, channels=channels), device=device)]
    )
    fixed = BatchedImages([Image(sitk_input(target, channels=channels), device=device)])
    tensor = torch.tensor(np.asarray(seed)[None], dtype=torch.float32, device=device)
    common = {
        "scales": [1],
        "iterations": [budget],
        "fixed_images": fixed,
        "moving_images": moving,
        "optimizer": "Adam",
        "optimizer_lr": 0.03,
        "around_center": True,
        "blur": False,
        "progress_bar": False,
        "max_tolerance_iters": budget + 1,
        "tolerance": 1e-6,
        "loss_type": "custom" if channels else "cc",
    }
    if channels:
        common["custom_loss"] = masked_global_ncc
    if model == "rigid":
        np.testing.assert_allclose(
            np.asarray(seed)[:3, :3].T @ np.asarray(seed)[:3, :3],
            np.eye(3),
            atol=1e-10,
            rtol=0,
        )
        registration = RigidRegistration(
            **common,
            scaling=False,
            init_translation=tensor[:, :3, 3].clone(),
            init_moment=tensor[:, :3, :3].clone(),
        )
        matrix = registration.get_rigid_matrix
    else:
        registration = AffineRegistration(**common, init_rigid=tensor.clone())
        matrix = registration.get_affine_matrix
    initial = matrix().detach().cpu().numpy()[0]
    np.testing.assert_allclose(initial, seed, atol=2e-6, rtol=0)
    # CF's transpose convention makes SITK physical x/y/z numerically world z/y/x.
    coordinates = (
        registration.get_warped_coordinates(fixed, moving).detach().cpu().numpy()[0]
    )
    source_image = sitk_input(source)
    target_image = sitk_input(target)

    def affine(image):
        a = np.eye(4)
        a[:3, :3] = np.array(image.GetDirection()).reshape(3, 3) @ np.diag(
            image.GetSpacing()
        )
        a[:3, 3] = image.GetOrigin()
        return a

    shape = coordinates.shape[:-1]
    voxel = np.stack(
        np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1
    )[..., ::-1]
    points = np.c_[voxel.reshape(-1, 3), np.ones(np.prod(shape))].T
    expected = (
        np.linalg.inv(affine(source_image)) @ initial @ affine(target_image) @ points
    )[:3].T
    normalized = expected * 2 / (np.array(source_image.GetSize()) - 1) - 1
    grid_error = float(np.max(np.abs(normalized - coordinates.reshape(-1, 3))))
    assert grid_error < 1e-4
    original_loss = registration.loss_fn
    states = []
    losses = []

    def observe(moved, target):
        value = original_loss(moved, target)
        states.append(matrix().detach().cpu().numpy()[0].copy())
        losses.append(float(value.detach().cpu()))
        return value

    registration.loss_fn = observe
    registration.optimize()
    assert len(losses) == budget
    states.append(matrix().detach().cpu().numpy()[0].copy())

    def evaluate(pull):
        p = torch.tensor(pull[None], device=device, dtype=torch.float32)
        theta = (moving.get_phy2torch() @ p @ fixed.get_torch2phy())[:, :3].contiguous()
        values = fireants_interpolator(
            moving(),
            affine=theta,
            out_shape=fixed().shape,
            mode="bilinear",
            align_corners=True,
        )
        return float(original_loss(values, fixed()).detach().cpu())

    with torch.no_grad():
        losses.append(evaluate(states[-1]))
        best = int(np.argmin(losses))
        assert abs(evaluate(states[best]) - losses[best]) < 1e-5
    return {
        "model": model,
        "metric": metric,
        "raw_metric": losses,
        "best_evaluation": best,
        "best_metric": losses[best],
        "final_metric": losses[-1],
        "best_pull": states[best].tolist(),
        "final_pull": states[-1].tolist(),
        "initial_pull": initial.tolist(),
        "center_world_zyx_mm": registration.center.detach().cpu().numpy()[0].tolist(),
        "initial_grid_max_normalized_error": grid_error,
        "device": device,
        "settings": {
            "budget": budget,
            "learning_rate": 0.03,
            "optimizer": "Adam",
            "scales": [1],
            "blur": False,
            "around_center": True,
            "early_stopping": False,
        },
        "source_shape": list(moving().shape),
        "target_shape": list(fixed().shape),
        "fused_interpolator": bool(fireants_interpolator.use_ffo),
    }, np.stack(states)


def run(output, *, replay_from=None):
    output.mkdir(parents=True, exist_ok=False)
    book = Notebook(
        "FireANTs atlas: native local CC and masked global NCC", "fireants", output
    )
    book.text(
        "<h2>Actual FireANTs, both supplied directions</h2><p>Same cached dB images/manual placement as CF. First test its native singleton-image handling; then use exactly CF’s four replicated recording slices (no new measured depth information). Single-scale500 Adam updates, native API rate0.03, around_center=True, no blur or early stopping. Test both default local CC (kernel3) and the existing benchmark’s custom masked global NCC. These losses/support rules are not identical to ITK. Track matching best and final poses, coverage and normal thickness, not only score. No installed source or product default is changed.</p>"
    )
    book.code(SETUP)
    book.namespace["REPLAY"] = replay_from
    if replay_from is not None:
        book.text(
            "<p>Rendering the saved fitting checkpoints from "
            + str(replay_from)
            + ". No optimization is repeated; the earlier fitting/source snapshot is retained.</p>"
        )
    book.code("""
from atlas_fireants import fit_fireants,sitk_input,expanded_voxeldata
from fireants.io.image import Image
fixed,moving,placement,paths,_=inputs()
previous_path=ROOT/'artifacts/atlas-centered-followup/validated/results.json'
previous=json.loads(previous_path.read_text())
assert [digest(p) for p in paths]==[x['sha256'] for x in previous['inputs']]
for p,h in previous['confusius_sources'].items():assert digest(Path(p))==h
engine_sources={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'src/voxreg').glob('*.py'))}
try:
    native=Image(sitk_input(moving,expanded=False),device=DEVICE)
    native_probe={'supported':True,'shape':list(native.array.shape)}
except (RuntimeError,ValueError) as error:
    native_probe={'supported':False,'error_type':type(error).__name__,'error':str(error)}
print('Native singleton FireANTs Image:',native_probe)
records={};states={}
for model in ['rigid','affine']:
    for metric in ['local_cc','global_ncc']:
        for direction,is_slice in [('recording_moving',True),('template_moving',False)]:
            source,target,seed=(moving,fixed,np.linalg.inv(placement)) if is_slice else (fixed,moving,placement)
            name=model+'__'+metric+'__'+direction
            if REPLAY is None:row,state=fit_fireants(source,target,seed,model=model,metric=metric,device=DEVICE)
            else:
                row=json.loads((REPLAY/(name+'.json')).read_text())
                with np.load(REPLAY/(name+'.npz')) as saved:state=saved['states']
                assert row['model']==model and row['metric']==metric and row['direction']==direction and row['settings']['budget']==500
            row['direction']=direction
            for label in ['best','final']:
                p=np.asarray(row[label+'_pull']);row[label+'_canonical_pull']=(p if is_slice else np.linalg.inv(p)).tolist()
            records[name],states[name]=row,state
            (OUT/(name+'.json')).write_text(json.dumps(row,indent=2))
            np.savez_compressed(OUT/(name+'.npz'),states=state)
            print(name,'best',row['best_evaluation'],row['best_metric'],'last',row['final_metric'],flush=True)
""")
    book.code("""
# An explicit voxreg full-grid/slab control bypasses its native-slice reversal.
expanded=expanded_voxeldata(moving);vox_expanded={}
print('Expanded recording shape:',expanded.shape,'affine:',voxreg.position_to_world(expanded))
for model in ['rigid','affine']:
    for direction,is_slice in [('recording_moving',True),('template_moving',False)]:
        source,target,seed=(expanded,fixed,np.linalg.inv(placement)) if is_slice else (fixed,expanded,placement)
        if REPLAY is None:
            trace=[]
            def report(indices,iterations,objective):trace.append([int(iterations[0]),float(objective[0])])
            result=voxreg.register(source,target,return_transforms=True,transform=model,initialization=seed,intensity_scaling='none',max_iterations=500,convergence_window_size=50,learning_rate=.1,device=DEVICE,progress=False,metric_callback=report,metric_interval=1)
            p=np.asarray(result.transforms);canonical=p if is_slice else np.linalg.inv(p)
            row={'model':model,'direction':direction,'canonical_pull':canonical.tolist(),'metric':float(result.diagnostics.objective),'steps':int(result.diagnostics.n_iterations),'trace':trace,'pivot':result.diagnostics.attrs.get('rotation_center_zyx_mm',result.diagnostics.attrs.get('center_zyx_mm')),'status':str(result.diagnostics.status.values),'optimized_parameters':6 if model=='rigid' else 12}
        else:
            row=json.loads((REPLAY/('vox_expanded__'+model+'__'+direction+'.json')).read_text())
            assert row['model']==model and row['direction']==direction
        vox_expanded[model+'__'+direction]=row
        (OUT/('vox_expanded__'+model+'__'+direction+'.json')).write_text(json.dumps(row,indent=2))
        print('vox expanded',model,direction,row['steps'],row['metric'],flush=True)
""")
    book.code(r"""
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for col,model in enumerate(['rigid','affine']):
    for r,metric in enumerate(['local_cc','global_ncc']):
        ax=axes[r,col]
        for direction in ['recording_moving','template_moving']:
            row=records[model+'__'+metric+'__'+direction]
            line,=ax.plot(row['raw_metric'],label=direction,lw=1.)
            ax.plot(row['best_evaluation'],row['best_metric'],'o',color=line.get_color(),ms=4)
        ax.set(title=model+' / '+metric,xlabel='Evaluated parameter state',ylabel='Own loss (not cross-metric comparable)');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('FireANTs fixed-grid fitting, native rate0.03; circles mark best, endpoints are library returns')
""")
    book.code(r"""
from atlas_centered_followup import display_sample as sample,supported_composites as composites
from atlas_alignment_investigation import plane_pose
good=np.asarray(json.loads((ROOT/'artifacts/atlas-centering/comparison/results.json').read_text())['records']['recording_moving__origin']['final_canonical_pull'])
rows=[];samples={};masks={}
for name,row in records.items():
    for label in ['best','final']:
        p=np.asarray(row[label+'_canonical_pull']);native=np.linalg.inv(p);u,s,vt=np.linalg.svd(native[:3,:3]);normal=np.cross(native[:3,1],native[:3,2]);normal/=np.linalg.norm(normal)
        values,valid,_=sample(native,moving,fixed);samples[name+'__'+label]=values;masks[name+'__'+label]=valid
        pose=plane_pose(p,moving);error=point_error(p,good,moving)
        rows.append({'run':name,'pose':label,'state':row['best_evaluation'] if label=='best' else 500,'own_loss':row[label+'_metric'],'native_support':float(valid.mean()),'normal_thickness_factor':float(abs(normal@native[:3,0])),'min_singular_value':float(s.min()),'max_singular_value':float(s.max()),'native_map_determinant':float(np.linalg.det(native[:3,:3])),'depth_mm':pose['center_zyx_mm'][0],'tilt_degrees':pose['tilt_degrees'],'RMS_px_vs_good_CF':error['rms_native_pixels']})
for name,row in vox_expanded.items():
    p=np.asarray(row['canonical_pull']);values,valid,_=sample(np.linalg.inv(p),moving,fixed);samples['vox_expanded__'+name]=values;masks['vox_expanded__'+name]=valid
    row['reference_error']=point_error(p,good,moving);row['native_support']=float(valid.mean())
common=np.logical_and.reduce(list(masks.values()))
for row in rows:row['joint_native_NCC2']=ncc(moving.values.astype(float),samples[row['run']+'__'+row['pose']],common)
print('Joint native support across all FireANTs and expanded-vox poses:',float(common.mean()))
display(pd.DataFrame(rows))
for metric in ['local_cc','global_ncc']:
    for label in ['final','best']:
        composites(moving.values,{model+' / '+direction:samples[model+'__'+metric+'__'+direction+'__'+label] for model in ['rigid','affine'] for direction in ['recording_moving','template_moving']},'FireANTs '+metric+' — '+label,reference_is_red=False)
composites(moving.values,{'Original good CF':sample(np.linalg.inv(good),moving,fixed)[0],**{k:samples['vox_expanded__'+k] for k in vox_expanded}},'Voxreg expanded-slab ablation: caller-fixed sampling, not native-slice fitting',reference_is_red=False)
np.savez_compressed(OUT/'arrays.npz',**{k+'__states':v for k,v in states.items()},**samples)
for p,h in engine_sources.items():assert digest(ROOT/p)==h
""")
    sources = [
        Path(__file__).resolve(),
        ROOT / "research/confusius-examples/atlas_centered_followup.py",
        ROOT / "research/confusius-examples/render_notebooks.py",
        ROOT / "research/confusius-examples/rigid_affine_atlas.py",
        ROOT / "research/confusius-examples/compare_same_subject.py",
        *sorted((ROOT / "src/voxreg").glob("*.py")),
    ]
    import fireants.io.image as fa_image
    import fireants.registration.abstract as fa_abstract
    import fireants.registration.affine as fa_affine
    import fireants.registration.rigid as fa_rigid

    external = [
        Path(inspect.getsourcefile(module))
        for module in [fa_affine, fa_rigid, fa_abstract, fa_image]
    ]
    result = {
        "records": book.namespace["records"],
        "summary": book.namespace["rows"],
        "vox_expanded": book.namespace["vox_expanded"],
        "native_singleton_probe": book.namespace["native_probe"],
        "joint_support": float(book.namespace["common"].mean()),
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "fireants_sources": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in external
        },
        "confusius_sources": book.namespace["previous"]["confusius_sources"],
        "inputs": book.namespace["previous"]["inputs"],
        "versions": {
            name: version(name) for name in ["fireants", "torch", "numpy", "confusius"]
        },
        "replay_from": str(replay_from) if replay_from is not None else None,
        "replay_checkpoint_hashes": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(replay_from.glob("*"))
            if p.suffix in {".json", ".npz"} or p.name == "measured-source.py"
        }
        if replay_from is not None
        else {},
        "reference_sha256": hashlib.sha256(
            book.namespace["previous_path"].read_bytes()
        ).hexdigest(),
        "production_sources_unchanged": True,
    }
    (output / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (output / "index.html").write_text(
        page(
            "FireANTs atlas",
            '<h1>FireANTs and caller-fixed sampling controls</h1><a href="fireants.html">Executed results and overlays</a>',
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--replay-from",
        type=Path,
        default=None,
        help="Render saved per-fit checkpoints without optimizing again",
    )
    args = parser.parse_args()
    run(
        args.output.resolve(),
        replay_from=args.replay_from.resolve() if args.replay_from else None,
    )
