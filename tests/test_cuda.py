"""GPU tests. Skipped wholesale when no CUDA device is present."""

import numpy as np
import pytest
import torch

from poretorch.field import distance_field
from poretorch.psd import covering_sizes, pore_size_distribution

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _positions(cell, n_atoms=60, seed=61):
    rng = np.random.default_rng(seed)
    return rng.random((n_atoms, 3)) @ np.asarray(cell, dtype=float)


def test_cuda_and_cpu_agree_on_the_field(cell):
    positions = _positions(cell)
    on_gpu = distance_field(
        positions, cell, radii=1.7, grid_spacing=0.8, device="cuda", torch_dtype=torch.float64
    )
    on_cpu = distance_field(
        positions, cell, radii=1.7, grid_spacing=0.8, device="cpu", torch_dtype=torch.float64
    )
    assert torch.allclose(on_gpu.clearance.cpu(), on_cpu.clearance, atol=1e-12)
    assert on_gpu.n_void == on_cpu.n_void


def test_cuda_backends_agree_with_each_other(cell):
    positions = _positions(cell)
    fast = distance_field(positions, cell, radii=1.7, grid_spacing=0.8, method="cell_list", device="cuda")
    slow = distance_field(positions, cell, radii=1.7, grid_spacing=0.8, method="brute", device="cuda")
    assert torch.allclose(fast.clearance, slow.clearance, atol=1e-5)


def test_float32_tracks_float64_closely(cell):
    positions = _positions(cell)
    single = distance_field(positions, cell, radii=1.7, grid_spacing=0.8, device="cuda")
    double = distance_field(
        positions, cell, radii=1.7, grid_spacing=0.8, device="cuda", torch_dtype=torch.float64
    )
    difference = (single.clearance.double() - double.clearance).abs().max().item()
    assert difference < 1e-4


def test_covering_agrees_across_devices(cell):
    positions = _positions(cell, n_atoms=30)
    kwargs = dict(radii=1.7, grid_spacing=1.0, torch_dtype=torch.float64)
    on_gpu = covering_sizes(distance_field(positions, cell, device="cuda", **kwargs))
    on_cpu = covering_sizes(distance_field(positions, cell, device="cpu", **kwargs))
    assert torch.allclose(on_gpu.cpu(), on_cpu, atol=1e-10)


@pytest.mark.parametrize("method", ["all_void", "covering", "packed"])
def test_every_method_runs_on_gpu(cell, method):
    field = distance_field(_positions(cell, n_atoms=30), cell, radii=1.7, grid_spacing=1.0, device="cuda")
    result = pore_size_distribution(field, method=method, bin_width=0.3, min_radius=0.3)
    assert result.volume_A3.sum() > 0.0
    assert np.all(np.isfinite(result.volume_A3))


def test_a_small_batch_size_does_not_change_the_result(cell):
    positions = _positions(cell)
    kwargs = dict(radii=1.7, grid_spacing=0.8, device="cuda", torch_dtype=torch.float64)
    big = distance_field(positions, cell, batch_size=1_000_000, **kwargs)
    small = distance_field(positions, cell, batch_size=97, **kwargs)
    assert torch.allclose(big.clearance, small.clearance, atol=1e-12)
