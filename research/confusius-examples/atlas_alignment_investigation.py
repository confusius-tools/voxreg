"""Diagnose atlas objective, through-plane freedoms and optimizer differences.

Research controls only: no product edits, downloads or changes to reference gates.
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk
import torch
from confusius.registration._utils import (
    expand_thin_dims,
    set_sitk_thread_count,
    voxeldata_to_sitk_image,
)
from confusius.registration.affines import (
    affine_to_sitk_linear_transform,
    sitk_linear_transform_to_affine,
)
from render_notebooks import ROOT, SETUP, Notebook, page

from voxreg import position_to_world


def project_native_pull(pull, native):
    """Complete the slice normal without changing any native-center placement."""
    original = np.linalg.inv(pull)
    affine = position_to_world(native)
    basis, _ = np.linalg.qr(affine[:3, 1:3])
    normal = np.cross(basis[:, 0], basis[:, 1])
    tangents = original[:3, :3] @ basis
    mapped_normal = np.cross(tangents[:, 0], tangents[:, 1])
    mapped_normal /= np.linalg.norm(mapped_normal)
    projected = original.copy()
    projected[:3, :3] = tangents @ basis.T + np.outer(mapped_normal, normal)
    projected[:3, 3] += (original[:3, :3] - projected[:3, :3]) @ affine[:3, 3]
    points = affine @ np.vstack(
        (np.indices(native.shape).reshape(3, -1), np.ones((1, native.size)))
    )
    assert np.max(np.abs((original - projected) @ points)) < 1e-11
    return np.linalg.inv(projected)


def plane_pose(pull, native):
    """The first world axis is the normal of this experiment's axial recording."""
    affine = position_to_world(native)
    assert np.allclose(affine[:3, 0] / np.linalg.norm(affine[:3, 0]), [1, 0, 0])
    center = affine @ np.r_[(np.array(native.shape) - 1) / 2, 1]
    normal = np.asarray(pull)[0, :3]
    return {
        "center_zyx_mm": (np.linalg.inv(pull) @ center)[:3].tolist(),
        "tilt_degrees": float(
            np.degrees(
                np.arccos(np.clip(abs(normal[0]) / np.linalg.norm(normal), 0, 1))
            )
        ),
    }


def sitk_pair(source, target, expand):
    moving, fixed = voxeldata_to_sitk_image(source), voxeldata_to_sitk_image(target)
    if expand:
        moving, fixed = expand_thin_dims(moving), expand_thin_dims(fixed)
    return sitk.Cast(moving, fixed.GetPixelID()), fixed


def score(source, target, pull, *, expand, mask=None):
    moving, fixed = sitk_pair(source, target, expand)
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsCorrelation()
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetMetricUseFixedImageGradientFilter(False)
    registration.SetMetricUseMovingImageGradientFilter(False)
    registration.SetInitialTransform(affine_to_sitk_linear_transform(pull))
    if mask is not None:
        registration.SetMetricFixedMask(
            voxeldata_to_sitk_image(target.copy(data=mask.astype(np.uint8)))
        )
    with set_sitk_thread_count(8):
        return -float(registration.MetricEvaluate(fixed, moving))


def direct_sitk(source, target, pull, *, expand=True, moving_gradient=True):
    """Reproduce the actual CF optimizer without calling its registration wrapper."""
    moving, fixed = sitk_pair(source, target, expand)
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsCorrelation()
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetMetricUseFixedImageGradientFilter(expand)
    registration.SetMetricUseMovingImageGradientFilter(moving_gradient)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=500,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=50,
        estimateLearningRate=registration.Never,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([1])
    registration.SetSmoothingSigmasPerLevel([0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    transform = affine_to_sitk_linear_transform(pull)
    registration.SetInitialTransform(transform, inPlace=True)
    history = []
    registration.AddCommand(
        sitk.sitkIterationEvent,
        lambda: history.append(float(registration.GetMetricValue())),
    )
    with set_sitk_thread_count(8):
        registration.Execute(fixed, moving)
    return sitk_linear_transform_to_affine(transform), {
        "updates": int(registration.GetOptimizerIteration()),
        "stop": registration.GetOptimizerStopConditionDescription(),
        "history": history,
    }


def native_itk_probe(source, target, pull, *, expand):
    """Independent native ITKv4 value/derivative, not the SimpleITK wrapper."""
    import itk

    moving_sitk, fixed_sitk = sitk_pair(source, target, expand)

    def image(sitk_image):
        result = itk.image_from_array(
            sitk.GetArrayFromImage(sitk_image).astype(np.float64)
        )
        result.SetSpacing(sitk_image.GetSpacing())
        result.SetOrigin(sitk_image.GetOrigin())
        result.SetDirection(
            itk.matrix_from_array(np.array(sitk_image.GetDirection()).reshape(3, 3))
        )
        return result

    moving, fixed = image(moving_sitk), image(fixed_sitk)
    metric = itk.CorrelationImageToImageMetricv4[type(fixed), type(moving)].New()
    transform = itk.AffineTransform[itk.D, 3].New()
    transform.SetMatrix(itk.matrix_from_array(np.asarray(pull)[:3, :3].copy()))
    transform.SetTranslation(tuple(np.asarray(pull)[:3, 3]))
    metric.SetFixedImage(fixed)
    metric.SetMovingImage(moving)
    metric.SetMovingTransform(transform)
    metric.SetVirtualDomainFromImage(fixed)
    metric.SetUseFixedImageGradientFilter(expand)
    itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(8)
    metric.Initialize()
    derivative = itk.Array[itk.D](12)
    metric.GetDerivative(derivative)
    derivative = np.array(derivative)
    return {
        "NCC2": -float(metric.GetValue()),
        "valid_points": int(metric.GetNumberOfValidPoints()),
        "derivative": derivative.tolist(),
        "through_slice_parameter_derivative_max": float(
            np.abs(derivative[[0, 1, 2, 9]]).max()
        ),
        "all_parameter_derivative_max": float(np.abs(derivative).max()),
    }


def native_itk_optimize(source, target, pull):
    """Reproduce the F64 CF reference with Python ITKv4's optimizer directly."""
    import itk

    moving_sitk, fixed_sitk = sitk_pair(source, target, True)
    assert fixed_sitk.GetPixelID() == sitk.sitkFloat64

    def image(value):
        result = itk.image_from_array(sitk.GetArrayFromImage(value).astype(np.float64))
        result.SetSpacing(value.GetSpacing())
        result.SetOrigin(value.GetOrigin())
        result.SetDirection(
            itk.matrix_from_array(np.array(value.GetDirection()).reshape(3, 3))
        )
        return result

    moving, fixed = image(moving_sitk), image(fixed_sitk)
    itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(8)
    metric = itk.CorrelationImageToImageMetricv4[type(fixed), type(moving)].New()
    transform = itk.AffineTransform[itk.D, 3].New()
    transform.SetMatrix(itk.matrix_from_array(np.asarray(pull)[:3, :3].copy()))
    transform.SetTranslation(tuple(np.asarray(pull)[:3, 3]))
    optimizer = itk.GradientDescentOptimizerv4Template[itk.D].New()
    optimizer.SetLearningRate(1.0)
    optimizer.SetNumberOfIterations(500)
    optimizer.SetMinimumConvergenceValue(1e-6)
    optimizer.SetConvergenceWindowSize(50)
    optimizer.SetDoEstimateLearningRateOnce(False)
    optimizer.SetDoEstimateLearningRateAtEachIteration(False)
    optimizer.SetReturnBestParametersAndValue(False)
    scales = itk.RegistrationParameterScalesFromPhysicalShift[type(metric)].New()
    scales.SetMetric(metric)
    scales.SetTransformForward(True)
    optimizer.SetScalesEstimator(scales)
    registration = itk.ImageRegistrationMethodv4[type(fixed), type(moving)].New()
    registration.SetFixedImage(fixed)
    registration.SetMovingImage(moving)
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    registration.SetInitialTransform(transform)
    registration.InPlaceOn()
    registration.SetNumberOfLevels(1)
    registration.SetShrinkFactorsPerLevel([1])
    registration.SetSmoothingSigmasPerLevel([0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    history = []
    optimizer.AddObserver(
        itk.IterationEvent(), lambda: history.append(float(optimizer.GetValue()))
    )
    registration.Update()
    matrix = np.eye(4)
    matrix[:3, :3] = itk.array_from_matrix(transform.GetMatrix())
    matrix[:3, 3] = tuple(transform.GetOffset())
    return matrix, {
        "updates": int(optimizer.GetCurrentIteration()),
        "stop": optimizer.GetStopConditionDescription(),
        "history": history,
    }


def vox_control(source, target, pull, *, frozen=False, window=50):
    """Research-only gradient mask fixes depth/tilt, without changing product code."""
    engine = importlib.import_module("voxreg.rigid")
    original = engine.sample_rigid

    def sampler(images, parameters, grid):
        if frozen and parameters.requires_grad:
            assert parameters.shape == (1, 9)
            assert np.allclose(grid.slice_basis[0], 0)
            assert np.allclose(grid.slice_basis[1:] @ grid.slice_basis[1:].T, np.eye(2))
            mask = torch.ones_like(parameters)
            mask[:, [0, 1, 6]] = 0
            parameters.register_hook(lambda gradient: gradient * mask)
        return original(images, parameters, grid)

    history = []
    torch.set_num_threads(8)
    with patch.object(engine, "sample_rigid", sampler):
        result = engine.register(
            source,
            target,
            transform="affine",
            initialization=pull,
            intensity_scaling="none",
            max_iterations=500,
            learning_rate=0.1,
            convergence_window_size=window,
            progress=False,
            metric_callback=lambda frames, steps, loss: history.append(
                {"updates": int(steps[0]), "objective": float(loss[0])}
            ),
            metric_interval=1,
        )
    assert engine.sample_rigid is original
    if frozen:
        np.testing.assert_allclose(
            result.transforms.values[0], np.asarray(pull)[0], atol=1e-12, rtol=0
        )
    return result.transforms.values, {
        "updates": int(result.diagnostics.n_iterations),
        "stop": str(result.diagnostics.status.values),
        "objective": float(result.diagnostics.objective),
        "history": history,
    }


def gradient_check(source, target, pull):
    """Check the production mixed gradient against double-interpolation differences."""
    engine = importlib.import_module("voxreg.rigid")
    report = {}

    def hold(
        images, fixed, grid, reference, *_args, initial_parameters=None, **_kwargs
    ):
        assert initial_parameters is not None and len(images) == 1
        scale = grid.parameter_scale
        seed = initial_parameters / scale

        def objective(state, dtype):
            sampled, valid = engine.sample_rigid(images.to(dtype), state * scale, grid)
            loss, meaningful = engine.global_ncc(sampled, fixed.to(dtype), valid)
            assert meaningful.all()
            return loss.sum(), valid

        gradients = []
        for dtype in [torch.float32, torch.float64]:
            state = seed.detach().clone().requires_grad_(True)
            loss, _ = objective(state, dtype)
            gradients.append(torch.autograd.grad(loss, state)[0].numpy().ravel())
        finite = []
        epsilon = 1e-6
        for column in range(seed.shape[1]):
            delta = torch.zeros_like(seed)
            delta[:, column] = epsilon
            plus, plus_valid = objective(seed + delta, torch.float64)
            minus, minus_valid = objective(seed - delta, torch.float64)
            assert torch.equal(plus_valid, minus_valid)
            finite.append(float((plus - minus) / (2 * epsilon)))
        mixed, double = gradients
        finite = np.asarray(finite)
        report.update(
            {
                "finite_difference_normalized_parameter_step": epsilon,
                "mixed_gradient": mixed.tolist(),
                "double_gradient": double.tolist(),
                "finite_difference_gradient": finite.tolist(),
                "double_FD_relative_L2_error": float(
                    np.linalg.norm(double - finite) / np.linalg.norm(finite)
                ),
                "mixed_double_relative_L2_error": float(
                    np.linalg.norm(mixed - double) / np.linalg.norm(double)
                ),
            }
        )
        return (
            initial_parameters.detach().clone(),
            torch.zeros(1, dtype=torch.int64),
            torch.full((1,), 5, dtype=torch.int64),
        )

    torch.set_num_threads(8)
    with patch.object(engine, "_optimize", hold):
        engine.register(
            source,
            target,
            transform="affine",
            initialization=pull,
            intensity_scaling="none",
            max_iterations=1,
            progress=False,
        )
    assert report["double_FD_relative_L2_error"] < 1e-5
    assert report["mixed_double_relative_L2_error"] < 1e-3
    return report


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    book = Notebook(
        "Why atlas poses differ: direct ITK and matched-freedom controls",
        "investigation",
        output,
    )
    book.text(
        "<h2>Causal controls, not a new fitting policy</h2><p>Same cached images/manual placement. Direct SimpleITK reproduces CF, native ITKv4 independently measures its derivatives, then scoped controls fix voxreg depth/tilt. Cross-score the exact same poses under all metric domains. No production edits, default changes, downloads or speed claims.</p>"
    )
    book.code(SETUP)
    book.code("""
from atlas_alignment_investigation import project_native_pull,plane_pose,score,direct_sitk,native_itk_probe,native_itk_optimize,vox_control,gradient_check
from atlas_seed_control import held_register
from voxreg import position_to_world
fixed,moving,placement,paths,_=inputs()
reference_path=ROOT/"artifacts/atlas-directions/affine/results.json"
reference=json.loads(reference_path.read_text())
convergence_path=ROOT/"artifacts/atlas-convergence/qualified/results.json"
convergence=json.loads(convergence_path.read_text())
for path,expected in reference["confusius_sources"].items():assert digest(Path(path))==expected
assert [digest(path) for path in paths]==[row["sha256"] for row in reference["inputs"][:2]]
engine_sources={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/"src/voxreg").glob("*.py"))}
poses={"manual":np.linalg.inv(placement),"CF":np.asarray(reference["records"]["slice_to_volume / CF"]["canonical_pull"]),"CF_reverse":np.asarray(reference["records"]["volume_to_slice / CF"]["canonical_pull"]),"vox_native":np.asarray(convergence["records"]["native_cpu"]["pull"]),"vox_long":np.asarray(convergence["records"]["fixed500"]["pull"])}
for name in ["CF","CF_reverse"]:poses[name+"_projected"]=project_native_pull(poses[name],moving)
pose_scores={}
for name,pull in poses.items():
    values,mask,_=sample(np.linalg.inv(pull),moving,fixed)
    row={"CF_original_NCC2":score(moving,fixed,pull,expand=True),"CF_reverse_NCC2":score(fixed,moving,np.linalg.inv(pull),expand=True),"native_ITK_half_voxel_bounds_NCC2":score(fixed,moving,np.linalg.inv(pull),expand=False),"native_ITK_center_mask_NCC2":score(fixed,moving,np.linalg.inv(pull),expand=False,mask=mask),"native_SciPy_NCC2":ncc(moving.values.astype(float),values,mask),"center_support":float(mask.mean()),**plane_pose(pull,moving)}
    if name not in ["CF","CF_reverse"]:
        held,_=held_register(moving,fixed,pull)
        row["native_vox_NCC2"]=-float(held.diagnostics.objective)
        assert abs(row["native_vox_NCC2"]-row["native_ITK_center_mask_NCC2"])<1e-6
    pose_scores[name]=row
assert pose_scores["vox_native"]["CF_original_NCC2"]>pose_scores["CF"]["CF_original_NCC2"]
assert pose_scores["vox_long"]["CF_original_NCC2"]>pose_scores["CF"]["CF_original_NCC2"]
display(pd.DataFrame(pose_scores).T)
""")
    book.text(
        "<h2>Independent native ITKv4 gradient probe</h2><p>ITK affine parameters are9 row-major linear coefficients followed by3 translations. Indices0,1,2,9 control the first physical coordinate (world z here). Their derivatives vanish when the moving image is a replicated recording; they do not vanish when the moving image is the measured3D template.</p>"
    )
    book.code("""
derivatives={"CF_original_manual":native_itk_probe(moving,fixed,poses["manual"],expand=True),"CF_original_vox_pose":native_itk_probe(moving,fixed,poses["vox_native"],expand=True),"native_slice_to_template":native_itk_probe(fixed,moving,placement,expand=False)}
assert derivatives["CF_original_manual"]["through_slice_parameter_derivative_max"]<1e-12
assert derivatives["CF_original_vox_pose"]["through_slice_parameter_derivative_max"]<1e-12
assert derivatives["native_slice_to_template"]["through_slice_parameter_derivative_max"]>1e-6
assert abs(derivatives["CF_original_manual"]["NCC2"]-pose_scores["manual"]["CF_original_NCC2"])<1e-10
display(pd.DataFrame(derivatives).T)
gradient_checks={name:gradient_check(moving,fixed,poses[name]) for name in ["manual","CF_projected"]}
display(pd.DataFrame(gradient_checks).T)
""")
    book.code("""
controls={}
for name,source,target,seed,expand,gradient in [
    ("direct_CF",moving,fixed,poses["manual"],True,True),
    ("direct_CF_from_vox",moving,fixed,poses["vox_native"],True,True),
    ("direct_CF_unsmoothed",moving,fixed,poses["manual"],True,False),
    ("native_ITK_GD_smoothed",fixed,moving,placement,False,True),
    ("native_ITK_GD_unsmoothed",fixed,moving,placement,False,False)]:
    matrix,record=direct_sitk(source,target,seed,expand=expand,moving_gradient=gradient)
    pull=matrix if name.startswith("direct_CF") else np.linalg.inv(matrix)
    if name.startswith("direct_CF"):np.testing.assert_allclose(pull[0],seed[0],atol=1e-12,rtol=0)
    controls[name]={**record,"pull":pull.tolist(),"reference_error":point_error(pull,poses["CF"],moving),"plane":plane_pose(pull,moving)}
    print(name,record["updates"],controls[name]["reference_error"]["rms_native_pixels"])
assert np.array_equal(np.asarray(controls["direct_CF"]["pull"]),poses["CF"])
matrix,record=native_itk_optimize(moving,fixed,poses["manual"])
controls["direct_native_ITK"]={**record,"pull":matrix.tolist(),"reference_error":point_error(matrix,poses["CF"],moving),"plane":plane_pose(matrix,moving)}
assert controls["direct_native_ITK"]["reference_error"]["max_native_pixels"]<1e-10
for name,seed,frozen,window in [("vox_observed",poses["manual"],False,50),("vox_from_CF",poses["CF_projected"],False,50),("vox_frozen",poses["manual"],True,50),("vox_frozen_fixed500",poses["manual"],True,501),("vox_frozen_from_CF",poses["CF_projected"],True,50)]:
    pull,record=vox_control(moving,fixed,seed,frozen=frozen,window=window)
    controls[name]={**record,"pull":pull.tolist(),"reference_error":point_error(pull,poses["CF"],moving),"plane":plane_pose(pull,moving)}
    print(name,record["updates"],controls[name]["reference_error"]["rms_native_pixels"])
assert np.array_equal(np.asarray(controls["vox_observed"]["pull"]),poses["vox_native"])
display(pd.DataFrame({name:{"updates":row["updates"],"CF_RMS_px":row["reference_error"]["rms_native_pixels"],"tilt_degrees":row["plane"]["tilt_degrees"],"center_z_mm":row["plane"]["center_zyx_mm"][0]} for name,row in controls.items()}).T)
""")
    book.text(
        "<h2>Metric curves and identical native display sampler</h2><p>Raw objective histories below are separated by implementation/domain; values across panels are not a fair anatomical ranking. All overlays use the same SciPy sampler. Frozen controls change only3 native affine parameter gradients in a scoped research hook; they are not a new public transform model or a quality fix.</p>"
    )
    book.code("""
fig,axes=plt.subplots(1,2,figsize=(14,4),constrained_layout=True)
for name in ["direct_CF","direct_native_ITK","direct_CF_unsmoothed","direct_CF_from_vox"]:
    axes[0].plot(-np.asarray(controls[name]["history"]),label=name)
axes[0].set(title="Direct ITK: same slab/domain, different start or gradient",xlabel="Metric evaluation",ylabel="CF-domain NCC²");axes[0].legend(fontsize=8)
for name in ["vox_observed","vox_from_CF","vox_frozen","vox_frozen_fixed500","vox_frozen_from_CF"]:
    history=controls[name]["history"];axes[1].plot([r["updates"] for r in history],[-r["objective"] for r in history],label=name,lw=.9)
axes[1].set(title="voxreg: native domain, matched/freed plane",xlabel="Actual updates",ylabel="Native-domain NCC²");axes[1].legend(fontsize=8)
samples={}
for name in ["CF","vox_native"]:samples[name],_,_=sample(np.linalg.inv(poses[name]),moving,fixed)
for name in ["vox_frozen","vox_frozen_fixed500","vox_from_CF","direct_CF_unsmoothed"]:samples[name],_,_=sample(np.linalg.inv(controls[name]["pull"]),moving,fixed)
composites(moving.values,{name:samples[name] for name in ["CF","vox_native","vox_frozen","vox_frozen_fixed500"]},"Freeing versus fixing plane depth/tilt",reference_is_red=False)
composites(moving.values,{name:samples[name] for name in ["CF","vox_from_CF","direct_CF_unsmoothed"]},"Warm start and ITK image-gradient control",reference_is_red=False)
np.savez_compressed(OUT/"arrays.npz",**{name+"__pull":np.asarray(row["pull"]) for name,row in controls.items()},**{name+"__native_template":value for name,value in samples.items()})
for path,expected in engine_sources.items():assert digest(ROOT/path)==expected
""")
    import confusius as cf
    import itk

    sources = [
        Path(__file__).resolve(),
        ROOT / "research/confusius-examples/compare_atlas.py",
        ROOT / "research/confusius-examples/atlas_seed_control.py",
        ROOT / "research/confusius-examples/compare_same_subject.py",
        ROOT / "research/confusius-examples/render_notebooks.py",
        *sorted((ROOT / "src/voxreg").glob("*.py")),
    ]
    result = {
        "pose_scores": book.namespace["pose_scores"],
        "poses": {k: v.tolist() for k, v in book.namespace["poses"].items()},
        "native_itk": book.namespace["derivatives"],
        "gradient_checks": book.namespace["gradient_checks"],
        "controls": book.namespace["controls"],
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "references": {
            str(book.namespace[name].relative_to(ROOT)): hashlib.sha256(
                book.namespace[name].read_bytes()
            ).hexdigest()
            for name in ["reference_path", "convergence_path"]
        },
        "inputs": [
            {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in book.namespace["paths"]
        ],
        "versions": {
            "ITK": itk.Version.GetITKVersion(),
            "SimpleITK": sitk.Version_VersionString(),
            "torch": torch.__version__,
            "confusius": cf.__version__,
        },
        "production_sources_unchanged": True,
        "fireants_executed": False,
    }
    (output / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (output / "index.html").write_text(
        page(
            "Atlas alignment investigation",
            '<h1>Atlas alignment investigation</h1><p><a href="investigation.html">Executed investigation</a> · <a href="investigation.ipynb">Notebook</a></p>',
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
