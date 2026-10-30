"""Execute the three cached ConfUSIus/voxreg comparisons; export offline HTML/ipynb.

Only reporting code: no notebook-server dependencies, downloads, engine changes,
or timing claims. Code cells really execute; their stdout, tables and figures
are stored as notebook outputs and embedded in standalone HTML.
"""

import argparse
import base64
import contextlib
import csv
import hashlib
import html
import io
import json
import platform
import sys
import textwrap
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class Notebook:
    def __init__(self, title, stem, output):
        self.title, self.stem, self.output = title, stem, output
        self.cells = []
        self.namespace = {"display": self.display, "OUT": output}
        self.displays = []
        self.figure_count = 0

    def display(self, value):
        data = {"text/plain": repr(value)}
        if hasattr(value, "_repr_html_"):
            data["text/html"] = value._repr_html_()
        self.displays.append(
            {"output_type": "display_data", "data": data, "metadata": {}}
        )

    def text(self, source):
        self.cells.append(
            {
                "cell_type": "markdown",
                "id": f"cell-{len(self.cells)}",
                "metadata": {},
                "source": source,
            }
        )

    def code(self, source):
        source = textwrap.dedent(source).strip()
        print(self.stem, len(self.cells), source.splitlines()[0], flush=True)
        stdout, stderr = io.StringIO(), io.StringIO()
        self.displays = []
        error = None
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                exec(compile(source, self.stem, "exec"), self.namespace)  # noqa: S102 -- trusted generated notebook cells
            except Exception as caught:  # noqa: BLE001 -- persist the cell error, then re-raise
                error = caught
                self.displays.append(
                    {
                        "output_type": "error",
                        "ename": type(caught).__name__,
                        "evalue": str(caught),
                        "traceback": traceback.format_exc().splitlines(),
                    }
                )
        outputs = []
        for name, stream in [("stdout", stdout), ("stderr", stderr)]:
            if stream.getvalue():
                outputs.append(
                    {"output_type": "stream", "name": name, "text": stream.getvalue()}
                )
        outputs.extend(self.displays)
        for number in plt.get_fignums():
            fig = plt.figure(number)
            data = io.BytesIO()
            fig.savefig(data, format="png", dpi=120, bbox_inches="tight")
            self.figure_count += 1
            (
                self.output / f"{self.stem}-figure-{self.figure_count:02}.png"
            ).write_bytes(data.getvalue())
            outputs.append(
                {
                    "output_type": "display_data",
                    "data": {"image/png": base64.b64encode(data.getvalue()).decode()},
                    "metadata": {},
                }
            )
            plt.close(fig)
        self.cells.append(
            {
                "cell_type": "code",
                "id": f"cell-{len(self.cells)}",
                "metadata": {},
                "source": source,
                "execution_count": sum(c["cell_type"] == "code" for c in self.cells)
                + 1,
                "outputs": outputs,
            }
        )
        self.write()
        if error is not None:
            raise error

    def write(self):
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python (voxreg environment)",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                    "version": platform.python_version(),
                },
                "title": self.title,
            },
            "cells": self.cells,
        }
        (self.output / (self.stem + ".ipynb")).write_text(
            json.dumps(notebook, indent=1, allow_nan=False) + "\n"
        )
        body = [
            f'<h1>{html.escape(self.title)}</h1><p><a href="index.html">All comparisons</a> · <a href="{self.stem}.ipynb">Executed notebook</a></p>'
        ]
        for cell in self.cells:
            if cell["cell_type"] == "markdown":
                body.append(cell["source"])
                continue
            body.append(
                "<details><summary>Executed Python source</summary><pre>"
                + html.escape(cell["source"])
                + "</pre></details>"
            )
            for item in cell["outputs"]:
                if item["output_type"] == "stream":
                    body.append("<pre>" + html.escape(item["text"]) + "</pre>")
                elif item["output_type"] == "error":
                    body.append(
                        '<pre class="failure">'
                        + html.escape("\n".join(item["traceback"]))
                        + "</pre>"
                    )
                elif "image/png" in item["data"]:
                    body.append(
                        '<img alt="Executed comparison plot" src="data:image/png;base64,'
                        + item["data"]["image/png"]
                        + '">'
                    )
                else:
                    body.append(
                        item["data"].get(
                            "text/html",
                            "<pre>"
                            + html.escape(item["data"]["text/plain"])
                            + "</pre>",
                        )
                    )
        (self.output / (self.stem + ".html")).write_text(
            page(self.title, "\n".join(body))
        )


def page(title, body):
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'
        + html.escape(title)
        + "</title><style>body{max-width:1300px;margin:35px auto;padding:0 22px;font:16px/1.5 system-ui;color:#18212b;background:#fff}h1,h2{line-height:1.2}h2{margin-top:45px;border-top:1px solid #ccd4dd;padding-top:22px}img{max-width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 monospace;background:#f2f5f8;padding:14px}table{border-collapse:collapse;font-size:14px;display:block;overflow:auto}td,th{padding:7px 10px;border-bottom:1px solid #ddd}details{margin:15px 0}summary{cursor:pointer;color:#245a91}.failure{background:#ffe8e8;color:#8c1717}.note{background:#fff4d6;padding:14px}a{color:#125896}</style><main>"
        + body
        + "</main></html>"
    )


def allen_colors(path):
    """Keep large Allen IDs as Python integers, not the failing UInt16 CSV dtype."""
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        int(row["annotation_value"]): (
            row["abbreviation"],
            row["name"],
            tuple(
                int(row["color_hex_triplet"].lstrip("#")[i : i + 2], 16) / 255
                for i in [0, 2, 4]
            ),
        )
        for row in rows
    }


def color_labels(labels, colors):
    ids, inverse = np.unique(labels, return_inverse=True)
    missing = [int(label) for label in ids if label != 0 and int(label) not in colors]
    if missing:
        raise ValueError(f"Allen colors missing for label IDs: {missing}")
    rgb = np.array([colors[int(label)][2] if label else (0, 0, 0) for label in ids])
    return rgb[inverse].reshape((*labels.shape, 3))


def atlas_panels(recording, labels, colors, title, alpha=0.45):
    gray = np.clip((np.asarray(recording)[0] + 30) / 30, 0, 1)
    fig, axes = plt.subplots(
        1, len(labels) + 1, figsize=(5 * (len(labels) + 1), 6), constrained_layout=True
    )
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("Recording (native grid)")
    for ax, (name, data) in zip(axes[1:], labels.items(), strict=True):
        rgb = color_labels(data[0], colors)
        image = np.repeat(gray[..., None], 3, axis=-1)
        mask = data[0] != 0
        image[mask] = (1 - alpha) * image[mask] + alpha * rgb[mask]
        ax.imshow(image, interpolation="nearest")
        ax.set_title(name)
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(title + f" — official Allen colors, alpha {alpha:.2f}")


def composites(reference, images, title, *, reference_is_red=True):
    fig, axes = plt.subplots(
        1, len(images), figsize=(5 * len(images), 5), constrained_layout=True
    )
    reference_values = np.clip((np.asarray(reference)[0].astype(float) + 30) / 30, 0, 1)
    for ax, (name, values) in zip(np.atleast_1d(axes), images.items(), strict=True):
        candidate = np.clip((np.asarray(values)[0].astype(float) + 30) / 30, 0, 1)
        red, cyan = (
            (reference_values, candidate)
            if reference_is_red
            else (candidate, reference_values)
        )
        ax.imshow(np.stack((red, cyan, cyan), axis=-1), interpolation="nearest")
        ax.set_title(name)
        ax.set_axis_off()
    roles = (
        "reference red / candidate cyan"
        if reference_is_red
        else "template red / recording cyan"
    )
    fig.suptitle(title + " — " + roles + "; shared −30…0 dB")


SETUP = """
from pathlib import Path
import sys, warnings, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import confusius as cf
import voxreg, torch
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "src/voxreg").is_dir())
OUT = Path(globals().get("OUT", Path.cwd()))
sys.path.insert(0, str(ROOT / "research/confusius-examples"))
from render_notebooks import composites, atlas_panels, allen_colors, color_labels
from compare_same_subject import sample, ncc, load_inputs, digest
from rigid_affine_atlas import inputs, cached_annotation, point_error, label_agreement, fit_chain
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(1 if DEVICE == "cuda" else 8)
print("Python:", sys.executable)
print("ConfUSIus:", cf.__version__, "voxreg device:", DEVICE)
"""


def pair_book(book):
    book.text(
        "<h2>01 — Same-subject sessions</h2><p>Recreates example01 using the original rat75 slice32 sessions, qform geometry and dB images. ConfUSIus and voxreg use their native rigid stopping policies. CPU resources are capped: CF8 threads; voxreg CUDA1 host thread (or CPU8). No timing comparison.</p>"
    )
    book.code(SETUP)
    book.code("""
fixed, moving, provenance = load_inputs()
INPUTS = provenance
display(pd.DataFrame([{"role":role,"shape":image.shape,"dtype":str(image.dtype),"range_dB":(float(image.min()),float(image.max()))} for role,image in [("fixed",fixed),("moving",moving)]]))
before, before_valid, _ = sample(np.eye(4), fixed, moving)
composites(fixed.values, {"Before registration":before}, "Initial alignment")
""")
    book.text(
        "<h2>Rigid registration: actual public calls</h2><p>CF uses its example defaults (100 updates/window10/automatic learning rate/geometry initialization). Voxreg uses its existing defaults (100/window50/learning rate0.1), with explicit geometry initialization to match the example’s starting pose. Different stopping policies are not presented as equal optimization work.</p>"
    )
    book.code("""
cf_rigid, cf_matrix, cf_diag = cf.registration.register_volume(
    moving=moving, fixed=fixed, transform_type="rigid",
    sitk_threads=8, show_progress=False, plot_metric=False, plot_composite=False,
)
vr = voxreg.register(moving, fixed, return_transforms=True, transform="rigid", initialization="center_geometry",
                    intensity_scaling="none", device=DEVICE, progress=False)
print("CF pull:\\n", cf_matrix)
print("voxreg pull:\\n", vr.transforms.values)
composites(fixed.values, {"Before":before,"ConfUSIus rigid":cf_rigid.values,"voxreg rigid":vr.registered.values}, "Rigid results")
""")
    book.code("""
cf_sample, cf_valid, _ = sample(cf_matrix, fixed, moving)
vr_sample, vr_valid, _ = sample(vr.transforms.values, fixed, moving)
common = before_valid & cf_valid & vr_valid
qc = pd.DataFrame([
    {"method":"Before","iterations":0,"common_NCC2":ncc(fixed.values.astype(float),before,common),"stop":"identity"},
    {"method":"ConfUSIus rigid","iterations":cf_diag.n_iterations,"common_NCC2":ncc(fixed.values.astype(float),cf_sample,common),"stop":cf_diag.stop_condition},
    {"method":"voxreg rigid","iterations":int(vr.diagnostics.n_iterations),"common_NCC2":ncc(fixed.values.astype(float),vr_sample,common),"stop":str(vr.diagnostics.status.values)},
])
print("Common fixed-grid support:", float(common.mean()))
display(qc)
fig, ax = plt.subplots(figsize=(9,3),constrained_layout=True)
ax.plot(cf_diag.metric_values,label="CF actual history")
ax.scatter([int(vr.diagnostics.n_iterations)],[float(vr.diagnostics.objective)],marker="x",s=70,label="voxreg final metric only")
ax.set(xlabel="Iteration",ylabel="Internal metric",title="Native diagnostics — voxreg exposes no per-iteration history")
ax.legend()
""")
    book.text(
        '<h2>Original B-spline refinement — ConfUSIus only</h2><p class="note">Voxreg does not implement B-splines. The CF stage is executed, not replaced by affine or omitted. Its output is shown next to both rigid outputs; there is no claimed voxreg nonrigid counterpart.</p>'
    )
    book.code("""
cf_bspline, bspline_transform, bspline_diag = cf.registration.register_volume(
    moving=moving, fixed=fixed, transform_type="bspline", mesh_size=(6,6,6),
    initialization=cf_matrix, sitk_threads=8, show_progress=False,
    plot_metric=False, plot_composite=False,
)
composites(fixed.values, {"CF rigid":cf_rigid.values,"CF rigid + B-spline":cf_bspline.values,"voxreg rigid (not B-spline)":vr.registered.values}, "Original nonrigid stage")
fig, ax = plt.subplots(figsize=(9,3),constrained_layout=True)
ax.plot(bspline_diag.metric_values);ax.set(xlabel="Iteration",ylabel="CF B-spline metric",title=bspline_diag.stop_condition)
""")
    book.text(
        "<h2>Combined and local B-spline displacement</h2><p>As in the source example, invert sampled pull fields approximately for display. Arrow lengths are auto-scaled per panel; colors report physical displacement in mm.</p>"
    )
    book.code("""
combined_field = cf.registration.sample_displacement_field_like(bspline_transform,fixed,sitk_threads=8)
local_transform = bspline_transform.copy()
local_transform.attrs = {**local_transform.attrs,"affines":{}}
local_field = cf.registration.sample_displacement_field_like(local_transform,fixed,sitk_threads=8)
inverse_fields = [cf.registration.invert_displacement_field(field,sitk_threads=8) for field in [combined_field,local_field]]
fig, axes = plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
for ax,field,title in zip(axes,inverse_fields,["Approximate inverse combined warp","Approximate inverse local warp"],strict=True):
    cf.plotting.plot_volume(fixed,axes=ax,show_colorbar=False)
    dx,dy = [field.sel(component=component).squeeze(drop=True) for component in ["i","j"]]
    step = max(1,min(dx.sizes["j"],dx.sizes["i"])//20)
    x,y = np.meshgrid(dx.x.isel(j=0).values[::step],dx.y.isel(i=0).values[::step])
    u,v = dx.values[::step,::step],dy.values[::step,::step]
    arrows = ax.quiver(x,y,u,v,np.hypot(u,v),angles="xy",cmap="autumn")
    fig.colorbar(arrows,ax=ax,label="Displacement (mm)",shrink=.7);ax.set_title(title)
np.savez_compressed(OUT/"01-arrays.npz",fixed=fixed.values,moving=moving.values,cf_rigid=cf_rigid.values,voxreg_rigid=vr.registered.values,cf_bspline=cf_bspline.values,cf_matrix=cf_matrix,voxreg_matrix=vr.transforms.values)
SUMMARY = {"rigid":qc.to_dict(orient="records"),"common_support":float(common.mean()),"cf_bspline_iterations":int(bspline_diag.n_iterations),"voxreg_bspline":"unsupported"}
""")


def motion_book(book):
    book.text(
        "<h2>02 — Volumewise motion correction</h2><p>Original rat75/openfield/slice32 window: frames220–339, all120 frames, first frame as reference. Fresh original power values are registered and returned; dB is used for fitting/display only. The current source example uses CF learning_rate=0.1, not the older benchmark’s1.0.</p>"
    )
    book.code(SETUP)
    book.code("""
path = Path.home()/".cache/confusius/cybis-pereira-2026-bids/1.0.0/sub-rat75/ses-20220523/fusi/sub-rat75_ses-20220523_task-openfield_acq-slice32_pwd.nii.gz"
if not path.is_file():raise FileNotFoundError("Cached recording required; no implicit download")
data = cf.load(path).isel(time=slice(220,340)).compute()
INPUTS = [{"path":str(path),"sha256":digest(path),"sidecar_sha256":digest(path.with_name(path.name.removesuffix(".nii.gz")+".json"))}]
print("Shape:", data.shape, "dtype:", data.dtype, "time range:", (float(data.time[0]),float(data.time[-1])))
cf_motion = cf.registration.register_volumewise(data,transform="rigid",moving_intensity_scaling="db",metric="correlation",learning_rate=.1,n_jobs=8,show_progress=False)
vr = voxreg.register(data,return_transforms=True,transform="rigid",initialization="center_geometry",intensity_scaling="db",batch_size=120,device=DEVICE,progress=False)
assert cf_motion.dtype == vr.registered.dtype == data.dtype
cf_df = cf_motion.attrs["motion_params"]
vr_df = cf.registration.create_motion_dataframe(list(vr.transforms.values),reference=data.isel(time=0,drop=True),time_coords=data.time.values)
vr_df["final_metric_value"] = vr.diagnostics.objective.values
vr_df["n_iterations"] = vr.diagnostics.n_iterations.values
print("CF motion table (first six rows):");display(cf_df.head(6))
print("voxreg poses through the same CF motion/FD helper:");display(vr_df.head(6))
""")
    book.text(
        "<h2>Motion diagnostics</h2><p>Both panels use ConfUSIus’s motion dataframe/FD conversion and plotting helper, so axes and definitions match. Fits retain native stopping: CF100/window10 versus voxreg100/window50. CF8 workers each use one native thread; voxreg uses one batch of120 frames.</p>"
    )
    book.code("""
cf_fig, _ = cf.plotting.plot_motion_diagnostics(cf_df);cf_fig.suptitle("ConfUSIus — original public workflow",y=1.02)
vr_fig, _ = cf.plotting.plot_motion_diagnostics(vr_df);vr_fig.suptitle("voxreg — same motion/FD definitions",y=1.02)
for left,right in zip(cf_fig.axes,vr_fig.axes,strict=True):
    limits = min(left.get_ylim()[0],right.get_ylim()[0]),max(left.get_ylim()[1],right.get_ylim()[1])
    left.set_ylim(limits);right.set_ylim(limits)
""")
    book.text(
        "<h2>Representative voxel and raster</h2><p>Choose the highest temporal-standard-deviation voxel from the <em>uncorrected</em> recording, exactly as the original example does. Use that same voxel/column for both methods. Power traces are not dB-converted; raster limits come from the original recording only.</p>"
    )
    book.code("""
k,j,i = np.unravel_index(np.std(data.values,axis=0).argmax(),data.shape[1:])
time = data.time.values
mean_db = data.mean("time").fusi.scale.db().values[0]
fig, axes = plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
axes[0].imshow(mean_db,cmap="gray",vmin=-30,vmax=0);axes[0].scatter(i,j,color="red");axes[0].set_title(f"Highest raw std voxel: k,j,i={k,j,i}")
for name,values in [("Before",data.values),("CF",cf_motion.values),("voxreg",vr.registered.values)]:axes[1].plot(time,values[:,k,j,i],label=name,lw=1)
axes[1].set(xlabel="Time (s)",ylabel="Original power Doppler intensity");axes[1].legend()
db_movies = {"Before":data.fusi.scale.db().values,"CF":cf_motion.fusi.scale.db().values,"voxreg":vr.registered.fusi.scale.db().values}
vmin,vmax = np.percentile(db_movies["Before"][:,k,:,i],[2,99.8])
fig, axes = plt.subplots(1,3,figsize=(15,5),sharey=True,constrained_layout=True)
for ax,(name,values) in zip(axes,db_movies.items(),strict=True):
    ax.imshow(values[:,k,:,i].T,aspect="auto",cmap="gray",vmin=vmin,vmax=vmax,extent=[time[0],time[-1],data.shape[-2]-1,0]);ax.set_title(name);ax.set_xlabel("Time (s)")
axes[0].set_ylabel("Native j position (pixels)")
""")
    book.text(
        "<h2>Frame overlays and quantitative QC</h2><p>Frames are chosen by initial alignment only: worst and median initial NCC². Final-image QC compares returned power images after dB conversion, on joint geometric support; it is not the internal fitting loss or a biological truth claim. Exclude the reference frame from summaries.</p>"
    )
    book.code("""
from benchmark_examples import matrices_from_motion
cf_matrices = matrices_from_motion(cf_df)
reference = data.isel(time=0,drop=True)
rows = []
for frame in range(len(data.time)):
    source = data.isel(time=frame,drop=True)
    _,a,_ = sample(cf_matrices[frame],reference,source)
    _,b,_ = sample(vr.transforms.values[frame],reference,source)
    valid = a & b
    row = {"frame":frame+220,"joint_support":float(valid.mean())}
    for name,values in db_movies.items():row[name] = ncc(db_movies["Before"][0].astype(float),values[frame].astype(float),valid)
    rows.append(row)
quality = pd.DataFrame(rows)
display(quality.iloc[1:].describe().loc[["min","50%","max"]])
order = np.argsort(quality["Before"].to_numpy()[1:])+1
chosen = [int(order[0]),int(order[len(order)//2])]
for frame in chosen:
    composites(db_movies["Before"][0],{name:values[frame] for name,values in db_movies.items()},f"Source frame {frame+220}, selected from initial alignment")
fig, ax = plt.subplots(figsize=(12,3),constrained_layout=True)
for name in db_movies:ax.plot(quality.frame,quality[name],label=name)
ax.set(xlabel="Source frame",ylabel="Joint-support final-image NCC²");ax.legend()
np.savez_compressed(OUT/"02-arrays.npz",before=data.values,confusius=cf_motion.values,voxreg=vr.registered.values,cf_matrices=cf_matrices,voxreg_matrices=vr.transforms.values,time=time)
cf_df.to_csv(OUT/"02-confusius-motion.csv");vr_df.to_csv(OUT/"02-voxreg-motion.csv");quality.to_csv(OUT/"02-quality.csv",index=False)
SUMMARY = {"frames":120,"window":[220,340],"selected_frames":[v+220 for v in chosen],"median_ncc2_excluding_reference":{name:float(quality[name].iloc[1:].median()) for name in db_movies},"cf_median_iterations":float(cf_df.n_iterations.median()),"voxreg_median_iterations":float(vr_df.n_iterations.median())}
""")


def atlas_book(book):
    book.text(
        "<h2>03 — Register to the Allen fUSI template</h2><p>The original example’s CR022755-frame mean, manual physical placement and3D ±1mm template slab are retained. Show both the original affine-only workflow and the subsequently requested rigid→affine workflow. ConfUSIus is the operator-designated ground truth; higher correlation cannot override placement disagreement.</p>"
    )
    book.code(SETUP)
    book.code("""
fixed,moving,placement,paths,shape = inputs()
seed = np.linalg.inv(placement)
print("Recording shape:", shape,"native slice:",moving.shape,"template slab:",fixed.shape)
initial,initial_valid,_ = sample(placement,moving,fixed)
composites(moving.values,{"Manual template placement":initial},"Native recording slice view",reference_is_red=False)
""")
    book.text(
        "<h2>Original affine-only workflow</h2><p>Both methods start from the original manual pull. CF learning_rate1, voxreg0.1; both max500/window50. Voxreg’s slice-as-moving3D output uses finite singleton-voxel support from the image affine, not zero thickness. Fitting still uses native slice centers. For comparison, sample the template on the native recording grid using the inverse fitted pull.</p>"
    )
    book.code("""
cf_direct,cf_direct_matrix,cf_direct_diag = cf.registration.register_volume(moving,fixed,transform_type="affine",metric="correlation",learning_rate=1.,number_of_iterations=500,convergence_window_size=50,initialization=seed,sitk_threads=8,show_progress=False,plot_metric=False,plot_composite=False)
with warnings.catch_warnings(record=True) as caught:
    vr_direct = voxreg.register(moving,fixed,return_transforms=True,transform="affine",initialization=seed,intensity_scaling="none",max_iterations=500,convergence_window_size=50,device=DEVICE,progress=False)
print("voxreg output coverage:",float(vr_direct.diagnostics.output_valid_fraction))
for warning in caught:print(str(warning.message))
matrices = {"CF original affine":cf_direct_matrix,"voxreg original affine":vr_direct.transforms.values}
""")
    book.text(
        "<h2>Rigid→affine comparison</h2><p>Run each backend’s own rigid fit, then pass its full returned matrix directly to affine. No frozen prefix, pre-resampling or extra composition. The helper calls are the same public APIs shown above, with rigid then affine; literal handoff equality is asserted.</p>"
    )
    book.code("""
cf_chain = fit_chain(moving,fixed,seed,"confusius")
vr_chain = fit_chain(moving,fixed,seed,"voxreg-"+DEVICE)
for name,chain in [("CF",cf_chain),("voxreg",vr_chain)]:
    assert chain["affine"]["initialization"] == chain["rigid"]["matrix"]
    for stage in ["rigid","affine"]:matrices[name+" chained "+stage] = np.asarray(chain[stage]["matrix"])
print("Exact rigid→affine matrix handoff verified for both libraries")
samples,masks = {},{}
for name,matrix in matrices.items():samples[name],masks[name],_ = sample(np.linalg.inv(matrix),moving,fixed)
common = np.logical_and.reduce(list(masks.values()))
qc_rows = []
for name,matrix in matrices.items():
    reference_name = "CF original affine" if "original" in name else ("CF chained rigid" if name.endswith("rigid") else "CF chained affine")
    error = point_error(matrix,matrices[reference_name],moving)
    qc_rows.append({"method":name,"common_NCC2":ncc(moving.values.astype(float),samples[name],common),"reference_RMS_mm":error["rms_mm"],"reference_RMS_px":error["rms_native_pixels"],"reference_gate_0.25px":error["rms_native_pixels"]<.25})
qc = pd.DataFrame(qc_rows);display(qc)
print("Common native support:",float(common.mean()))
for title,names in [("Original workflow",["CF original affine","voxreg original affine"]),("Chained rigid",["CF chained rigid","voxreg chained rigid"]),("Chained affine",["CF chained affine","voxreg chained affine"])]:
    composites(moving.values,{name:samples[name] for name in names},title,reference_is_red=False)
""")
    book.text(
        '<h2>Audit: requested rigid versus returned matrix</h2><p class="note">With a matrix initializer, the installed CF implementation uses an AffineTransform as its optimized initial transform even when rigid is requested. The returned first-stage matrix contains scale/shear: this is not a genuine six-parameter rigid comparator. The CF outputs remain the designated reference; no CF code is patched and no result is replaced. The original affine-only workflow is unaffected by this naming caveat.</p>'
    )
    book.code("""
audit = []
for name in ["CF chained rigid","voxreg chained rigid"]:
    linear = matrices[name][:3,:3]
    audit.append({"requested_stage":name,"orthogonality_error":float(np.linalg.norm(linear.T@linear-np.eye(3))),"singular_values":np.linalg.svd(linear,compute_uv=False).tolist(),"rigid_matrix":bool(np.allclose(linear.T@linear,np.eye(3),atol=1e-6,rtol=0))})
display(pd.DataFrame(audit))
print("Matrix-seeded CF requested-rigid result is affine-valued; retain this distinction when interpreting the earlier chained comparison.")
""")
    book.text(
        "<h2>Actual Allen-colored annotation overlays</h2><p>Read cached100µm Allen v3.1 labels and the official terminology’s RGB colors. Large IDs stay as Python integers, avoiding the installed BrainGlobe/Pandas UInt16 structures-table error. No cache/package patch or download. Label geometry matches the current CF atlas builder (resolution-diagonal, zero-origin). Both fitted poses use the same CF nearest-neighbor resampler and the original template-world→Allen transform.</p>"
    )
    book.code("""
annotation,annotation_provenance = cached_annotation()
color_path = Path.home()/".brainglobe/brainglobe-atlasapi/terminologies/allen_mouse-terminology/3_1/terminology.csv"
colors = allen_colors(color_path)
INPUTS = [{"path":str(path),"sha256":digest(path)} for path in [*paths,color_path]]
world_to_allen = np.asarray(fixed.attrs["affines"]["world_to_sform"])
labels = {name:cf.registration.resample_like(annotation,moving,world_to_allen @ np.linalg.inv(matrix),interpolation="nearest",fill_value=0,sitk_threads=8).values for name,matrix in matrices.items()}
for title,names in [("Original affine-only",["CF original affine","voxreg original affine"]),("Rigid→affine",["CF chained affine","voxreg chained affine"])]:
    atlas_panels(moving.values,{name:labels[name] for name in names},colors,title)
    fig,axes = plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
    for ax,name in zip(axes,names,strict=True):ax.imshow(color_labels(labels[name][0],colors),interpolation="nearest");ax.set_title(name+" — labels only");ax.set_axis_off()
""")
    book.text(
        "<h2>Region legend and label agreement</h2><p>Official Allen names/colors for the twenty most represented CF chained-affine regions. Label agreement includes matching background and separately reports foreground union. The known CF-reference disagreement remains a failure, regardless of improved NCC.</p>"
    )
    book.code("""
ids,counts = np.unique(labels["CF chained affine"],return_counts=True)
legend = []
for index in np.argsort(counts)[::-1]:
    region = int(ids[index])
    if region==0:continue
    acronym,name,color = colors[region]
    legend.append({"id":region,"acronym":acronym,"name":name,"native_pixels":int(counts[index]),"color":"#"+"".join(f"{round(c*255):02x}" for c in color)})
legend = pd.DataFrame(legend[:20]);display(legend)
from matplotlib.patches import Patch
fig, ax = plt.subplots(figsize=(12,5),constrained_layout=True);ax.axis("off")
ax.legend(handles=[Patch(facecolor=row.color,label=row.acronym+" — "+row["name"]) for _,row in legend.iterrows()],ncols=2,loc="center",frameon=False,title="Official Allen palette")
label_rows = []
for name,reference_name in [("voxreg original affine","CF original affine"),("voxreg chained rigid","CF chained rigid"),("voxreg chained affine","CF chained affine")]:
    score=label_agreement(labels[name],labels[reference_name],np.ones(moving.shape,dtype=bool))
    label_rows.append({"method":name,**{k:score[k] for k in ["agreement_fraction","foreground_label_agreement","macro_region_dice"]}})
display(pd.DataFrame(label_rows))
np.savez_compressed(OUT/"03-arrays.npz",moving=moving.values,fixed=fixed.values,**{name.replace(" ","_")+"__labels":value for name,value in labels.items()},**{name.replace(" ","_")+"__matrix":value for name,value in matrices.items()})
SUMMARY = {"quality":qc.to_dict(orient="records"),"requested_rigid_audit":audit,"label_agreement":label_rows,"common_support":float(common.mean()),"annotation":annotation_provenance}
""")
    book.text(
        "<h2>Saving and limitations</h2><p>Native label maps and fitted matrices are saved alongside this report in03-arrays.npz. The source example’s complete Atlas save/reload with structures/meshes is not executed: the installed public BrainGlobe constructor fails while parsing structures. This report deliberately uses cached labels and CSV colors, not a patched CF installation or a pretend full Atlas export. All displayed overlays are actual outputs on the native recording grid.</p>"
    )


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "python": sys.version,
        "sources": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__).resolve(),
                ROOT / "research/confusius-examples/compare_atlas.py",
                ROOT / "research/confusius-examples/compare_same_subject.py",
                ROOT / "research/confusius-examples/rigid_affine_atlas.py",
                *sorted((ROOT / "src/voxreg").glob("*.py")),
            ]
        },
        "notebooks": [],
        "scope": "Executed quality/visual comparison, no timing claims. Unsupported voxreg B-spline and full Atlas export limitations are explicit.",
    }
    for stem, title, build in [
        (
            "01-same-subject",
            "Same-subject registration: ConfUSIus vs voxreg",
            pair_book,
        ),
        ("02-motion-correction", "Motion correction: ConfUSIus vs voxreg", motion_book),
        ("03-allen-atlas", "Allen atlas overlays: ConfUSIus vs voxreg", atlas_book),
    ]:
        book = Notebook(title, stem, output)
        book.namespace["OUT"] = output
        build(book)
        manifest["notebooks"].append(
            {
                "stem": stem,
                "title": title,
                "code_cells": sum(c["cell_type"] == "code" for c in book.cells),
                "figures": sum(
                    "image/png" in item.get("data", {})
                    for c in book.cells
                    for item in c.get("outputs", [])
                ),
                "summary": book.namespace["SUMMARY"],
                "inputs": book.namespace["INPUTS"],
                "device": book.namespace["DEVICE"],
                "versions": {
                    "confusius": book.namespace["cf"].__version__,
                    "torch": book.namespace["torch"].__version__,
                    "numpy": np.__version__,
                },
            }
        )
        cf_root = Path(book.namespace["cf"].__file__).parent
        manifest["confusius_source_sha256"] = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                *sorted((cf_root / "registration").glob("*.py")),
                cf_root / "datasets/_brainglobe.py",
            ]
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
    links = "".join(
        f'<h2>{html.escape(item["title"])}</h2><p><a href="{item["stem"]}.html">Open HTML comparison</a> · <a href="{item["stem"]}.ipynb">Executed notebook</a> · {item["figures"]} figures</p>'
        for item in manifest["notebooks"]
    )
    (output / "index.html").write_text(
        page(
            "ConfUSIus vs voxreg — all three examples",
            "<h1>ConfUSIus vs voxreg</h1><p>Executed from cached original inputs. Embedded plots work offline; expand code blocks to inspect actual calls. Native stopping policies are labeled; CF is the designated reference. No timing claims or engine changes.</p>"
            + links,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
