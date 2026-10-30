"""Offline executed reports must contain real cell outputs and official label RGB."""

import base64
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.fixture
def reports():
    path = (
        Path(__file__).resolve().parents[1]
        / "research/confusius-examples/render_notebooks.py"
    )
    spec = importlib.util.spec_from_file_location("example_reports", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_notebook_executes_and_embeds_outputs_offline(reports, tmp_path):
    book = reports.Notebook("Actual outputs", "example", tmp_path)
    book.text("<h2>Comparison</h2>")
    book.code("""
import matplotlib.pyplot as plt
answer = 6 * 7
print("<script>not markup</script>", answer)
plt.figure();plt.plot([0,1],[0,answer])
""")
    assert book.namespace["answer"] == 42
    document = json.loads((tmp_path / "example.ipynb").read_text())
    assert document["nbformat"] == 4
    cell = document["cells"][1]
    assert cell["execution_count"] == 1
    assert "42" in cell["outputs"][0]["text"]
    image = cell["outputs"][1]["data"]["image/png"]
    assert base64.b64decode(image).startswith(b"\x89PNG")
    assert (tmp_path / "example-figure-01.png").read_bytes() == base64.b64decode(image)
    page = (tmp_path / "example.html").read_text()
    assert "data:image/png;base64," in page and 'src="http' not in page
    assert "&lt;script&gt;not markup&lt;/script&gt;" in page


def test_failed_cell_is_preserved_and_raised(reports, tmp_path):
    book = reports.Notebook("Failure", "failure", tmp_path)
    with pytest.raises(ValueError, match="real failure"):
        book.code('raise ValueError("real failure")')
    document = json.loads((tmp_path / "failure.ipynb").read_text())
    assert document["cells"][0]["outputs"][0]["output_type"] == "error"
    assert "real failure" in (tmp_path / "failure.html").read_text()


def test_official_colors_keep_large_ids_and_zero_background(reports, tmp_path):
    path = tmp_path / "colors.csv"
    path.write_text(
        "identifier,annotation_value,abbreviation,name,color_hex_triplet\n614454277,614454277,ROI,Example region,#ff0080\n"
    )
    colors = reports.allen_colors(path)
    rgb = reports.color_labels(np.array([[0, 614454277]]), colors)
    assert_allclose(rgb[0, 0], [0, 0, 0])
    assert_allclose(rgb[0, 1], [1, 0, 128 / 255])
    with pytest.raises(ValueError, match="missing"):
        reports.color_labels(np.array([[42]]), colors)


def test_atlas_blends_only_labeled_pixels(reports):
    recording = np.full((1, 2, 2), -15.0)
    labels = {"CF": np.array([[[1, 0], [0, 1]]])}
    reports.atlas_panels(
        recording,
        labels,
        {1: ("ROI", "Region", (1, 0, 0))},
        "Actual overlay",
        alpha=0.45,
    )
    fig = reports.plt.gcf()
    image = np.asarray(fig.axes[1].images[0].get_array())
    assert_allclose(image[0, 0], [0.725, 0.275, 0.275])
    assert_allclose(image[0, 1], [0.5, 0.5, 0.5])
    reports.plt.close(fig)


def test_existing_destination_is_never_overwritten(reports, tmp_path):
    sentinel = tmp_path / "existing-report.html"
    sentinel.write_text("Keep this report")
    with pytest.raises(FileExistsError):
        reports.run(tmp_path)
    assert sentinel.read_text() == "Keep this report"


def test_composites_preserve_template_red_and_recording_cyan(reports):
    recording = np.full((1, 2, 2), -30.0)
    template = np.zeros((1, 2, 2))
    for reference_is_red, expected in [(True, [0, 1, 1]), (False, [1, 0, 0])]:
        reports.composites(
            recording,
            {"Template": template},
            "Actual colors",
            reference_is_red=reference_is_red,
        )
        fig = reports.plt.gcf()
        image = np.asarray(fig.axes[0].images[0].get_array())
        assert_allclose(image[0, 0], expected)
        reports.plt.close(fig)
