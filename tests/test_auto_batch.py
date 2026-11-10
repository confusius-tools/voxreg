"""VRAM-aware policy is deterministic and performs no optimizer probes/retries."""

from types import SimpleNamespace

import pytest

from voxreg import preparation, register, registration


def images(*, voxels=11600, planar=True, frames=500):
    moving = SimpleNamespace(
        sizes={"time": frames, "k": 1 if planar else 32}, size=frames * voxels
    )
    fixed = SimpleNamespace(sizes={"k": 1 if planar else 32}, size=voxels)
    return moving, fixed


def memory(monkeypatch, *, free=12 << 30, reserved=0, allocated=0):
    monkeypatch.setattr(
        preparation.torch.cuda, "mem_get_info", lambda: (free, 16 << 30)
    )
    monkeypatch.setattr(preparation.torch.cuda, "memory_reserved", lambda: reserved)
    monkeypatch.setattr(preparation.torch.cuda, "memory_allocated", lambda: allocated)


@pytest.mark.parametrize("model", ["translation", "rigid", "affine", "syn"])
@pytest.mark.parametrize("planar", [True, False])
def test_automatic_policy_caps(monkeypatch, model, planar):
    memory(monkeypatch)
    moving, fixed = images(planar=planar)
    expected = 256 if planar and model != "syn" else 32
    assert preparation.choose_batch_size(moving, fixed, model, "cuda") == expected


def test_vram_size_and_reusable_cache(monkeypatch):
    moving, fixed = images(voxels=65536, planar=False)
    memory(monkeypatch, free=128 << 20)
    assert preparation.choose_batch_size(moving, fixed, "affine", "cuda") == 2
    memory(monkeypatch, free=128 << 20, reserved=512 << 20, allocated=128 << 20)
    assert preparation.choose_batch_size(moving, fixed, "affine", "cuda") == 8
    memory(monkeypatch, free=1)
    assert preparation.choose_batch_size(moving, fixed, "syn", "cuda") == 1
    memory(monkeypatch)
    fixed.size *= 128
    assert preparation.choose_batch_size(moving, fixed, "syn", "cuda") == 1


def test_cpu_and_short_recording(monkeypatch):
    def unexpected():
        raise AssertionError("CPU selection must not touch CUDA")

    monkeypatch.setattr(preparation.torch.cuda, "mem_get_info", unexpected)
    moving, fixed = images()
    assert preparation.choose_batch_size(moving, fixed, "syn", "cpu") == 32
    moving, fixed = images(frames=3)
    assert preparation.choose_batch_size(moving, fixed, "affine", "cpu") == 3
    memory(monkeypatch)
    assert preparation.choose_batch_size(moving, fixed, "affine", "cuda") == 3


def test_public_default_and_explicit_override(recording_factory, monkeypatch):
    moving, _, _ = recording_factory()
    calls = []

    def choose(*args):
        calls.append(True)
        return 2

    monkeypatch.setattr(registration, "choose_batch_size", choose)
    result = register(moving, max_iterations=1, progress=False)
    assert calls == [True]
    assert result["diagnostics"].to_dataset().attrs["batch_size"] == 2
    result = register(moving, batch_size=1, max_iterations=1, progress=False)
    assert calls == [True]
    assert result["diagnostics"].to_dataset().attrs["batch_size"] == 1
