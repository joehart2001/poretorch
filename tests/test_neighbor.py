import numpy as np
import pytest
import torch

from poretorch.cell import cell_tensor, minimum_image_setup, pbc_tensor
from poretorch.neighbor import (
    brute_min_surface_distance,
    build_atom_bins,
    query_min_surface_distance,
)


def _bins(cell, positions, radii, bin_width, pbc=(True, True, True)):
    cell_t = cell_tensor(cell, "cpu", torch.float64)
    mic = minimum_image_setup(cell_t, pbc_tensor(pbc, "cpu"))
    positions_t = torch.as_tensor(positions, dtype=torch.float64)
    radii_t = torch.as_tensor(np.broadcast_to(radii, (len(positions),)).copy(), dtype=torch.float64)
    return build_atom_bins(positions_t, radii_t, mic, bin_width), positions_t, radii_t, mic


def test_bin_table_holds_every_atom_exactly_once():
    cell = np.diag([12.0, 12.0, 12.0])
    rng = np.random.default_rng(51)
    positions = rng.random((50, 3)) @ cell
    bins, *_ = _bins(cell, positions, 1.7, bin_width=3.0)

    stored = bins.table[bins.table >= 0]
    assert stored.numel() == 50
    assert sorted(stored.tolist()) == list(range(50))


def test_atoms_outside_the_cell_are_wrapped_into_bins():
    cell = np.diag([10.0, 10.0, 10.0])
    positions = np.array([[-3.0, 12.0, 5.0], [25.0, -1.0, 0.5]])
    bins, positions_t, radii_t, mic = _bins(cell, positions, 1.0, bin_width=2.5)
    assert bins.table[bins.table >= 0].numel() == 2

    points = torch.as_tensor([[7.0, 2.0, 5.0]], dtype=torch.float64)
    value, resolved = query_min_surface_distance(points, bins, ring=1)
    reference = brute_min_surface_distance(points, positions_t, radii_t, mic)
    if bool(resolved.all()):
        assert torch.allclose(value, reference, atol=1e-12)


def test_coverage_radius_grows_with_the_ring_and_saturates():
    cell = np.diag([12.0, 12.0, 12.0])
    positions = np.zeros((1, 3))
    bins, *_ = _bins(cell, positions, 1.0, bin_width=2.0)  # 6 bins per axis

    assert bins.coverage_radius(1) == pytest.approx(2.0)
    assert bins.coverage_radius(2) == pytest.approx(4.0)
    # Ring 3 spans 7 bins across a 6-bin axis, so the search is exhaustive.
    assert np.isinf(bins.coverage_radius(3))


def test_resolved_points_always_match_brute_force(cell, random_structure):
    """Whenever the flag says resolved, the value must be the true minimum."""
    positions = random_structure(cell, n_atoms=40, seed=52)
    bins, positions_t, radii_t, mic = _bins(cell, positions, 1.7, bin_width=2.0)

    rng = np.random.default_rng(53)
    points = torch.as_tensor(rng.random((200, 3)) @ np.asarray(cell), dtype=torch.float64)

    values, resolved = query_min_surface_distance(points, bins, ring=1)
    reference = brute_min_surface_distance(points, positions_t, radii_t, mic)
    assert torch.allclose(values[resolved], reference[resolved], atol=1e-12)
    # Unresolved entries are upper bounds, never underestimates.
    assert bool((values[~resolved] >= reference[~resolved] - 1e-12).all())


def test_exhaustive_ring_needs_no_fallback(cell, random_structure):
    positions = random_structure(cell, n_atoms=30, seed=54)
    bins, positions_t, radii_t, mic = _bins(cell, positions, 1.7, bin_width=100.0)
    assert bins.n_bins == 1

    rng = np.random.default_rng(55)
    points = torch.as_tensor(rng.random((100, 3)) @ np.asarray(cell), dtype=torch.float64)
    values, resolved = query_min_surface_distance(points, bins, ring=1)
    reference = brute_min_surface_distance(points, positions_t, radii_t, mic)
    assert bool(resolved.all())
    assert torch.allclose(values, reference, atol=1e-12)


def test_empty_query_returns_empty_results():
    cell = np.diag([10.0, 10.0, 10.0])
    bins, *_ = _bins(cell, np.zeros((1, 3)), 1.0, bin_width=2.0)
    values, resolved = query_min_surface_distance(
        torch.empty((0, 3), dtype=torch.float64), bins, ring=1
    )
    assert values.numel() == 0 and resolved.numel() == 0


def test_bad_bin_width_is_rejected():
    cell = np.diag([10.0, 10.0, 10.0])
    with pytest.raises(ValueError, match="bin_width must be"):
        _bins(cell, np.zeros((1, 3)), 1.0, bin_width=0.0)


def test_brute_force_is_independent_of_chunking(cell, random_structure):
    positions = random_structure(cell, n_atoms=40, seed=56)
    cell_t = cell_tensor(cell, "cpu", torch.float64)
    mic = minimum_image_setup(cell_t, pbc_tensor(True, "cpu"))
    positions_t = torch.as_tensor(positions)
    radii_t = torch.full((40,), 1.7, dtype=torch.float64)

    rng = np.random.default_rng(57)
    points = torch.as_tensor(rng.random((50, 3)) @ np.asarray(cell))
    whole = brute_min_surface_distance(points, positions_t, radii_t, mic)
    split = brute_min_surface_distance(points, positions_t, radii_t, mic, atom_chunk=3)
    assert torch.allclose(whole, split, atol=1e-12)
