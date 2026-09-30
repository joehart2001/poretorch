import numpy as np
import pytest
import torch

from poretorch.cell import cell_tensor, mic_distance, minimum_image_setup, pbc_tensor
from poretorch.field import distance_field


def _field(positions, cell, **kwargs):
    kwargs.setdefault("device", "cpu")
    kwargs.setdefault("torch_dtype", torch.float64)
    kwargs.setdefault("grid_spacing", 0.8)
    return distance_field(positions, cell, **kwargs)


def test_single_atom_clearance_is_analytic(cell):
    """With one atom the clearance is exactly the minimum-image distance - R."""
    field = _field(np.zeros((1, 3)), cell, radii=1.0)
    cell_t = cell_tensor(cell, "cpu", torch.float64)
    mic = minimum_image_setup(cell_t, pbc_tensor(True, "cpu"))
    expected = mic_distance(field.grid.coordinates(), mic) - 1.0
    assert torch.allclose(field.clearance.reshape(-1), expected, atol=1e-12)


@pytest.mark.parametrize("n_atoms", [1, 5, 60])
def test_cell_list_matches_brute_force(cell, random_structure, n_atoms):
    """The two backends must be identical, not merely close."""
    positions = random_structure(cell, n_atoms=n_atoms, seed=3)
    fast = _field(positions, cell, radii=1.7, method="cell_list")
    slow = _field(positions, cell, radii=1.7, method="brute")
    assert torch.allclose(fast.clearance, slow.clearance, atol=1e-12)


def test_cell_list_matches_brute_force_with_per_atom_radii(cell, random_structure):
    """Pruning must account for a distant-but-larger atom winning the minimum."""
    positions = random_structure(cell, n_atoms=40, seed=4)
    rng = np.random.default_rng(5)
    radii = rng.uniform(0.5, 3.0, size=len(positions))
    fast = _field(positions, cell, radii=radii, method="cell_list")
    slow = _field(positions, cell, radii=radii, method="brute")
    assert torch.allclose(fast.clearance, slow.clearance, atol=1e-12)


def test_small_bin_width_still_gives_the_exact_answer(cell, random_structure):
    """Undersized bins force ring expansion. The result must not change."""
    positions = random_structure(cell, n_atoms=40, seed=6)
    reference = _field(positions, cell, radii=1.7, method="brute")
    tight = _field(positions, cell, radii=1.7, method="cell_list", bin_width=1.0)
    assert torch.allclose(tight.clearance, reference.clearance, atol=1e-12)


def test_max_ring_of_one_falls_back_and_stays_exact(cell, random_structure):
    """Capping the ring must trigger the brute-force fallback, not an error."""
    positions = random_structure(cell, n_atoms=40, seed=7)
    reference = _field(positions, cell, radii=1.7, method="brute")
    capped = _field(
        positions, cell, radii=1.7, method="cell_list", bin_width=1.0, max_ring=1
    )
    assert torch.allclose(capped.clearance, reference.clearance, atol=1e-12)


def test_probe_shrinks_the_void_but_not_the_clearance(cell, random_structure):
    """The probe sets accessibility. The reported pore size stays the full gap."""
    positions = random_structure(cell, n_atoms=40, seed=8)
    bare = _field(positions, cell, radii=1.7, probe_radius=0.0)
    probed = _field(positions, cell, radii=1.7, probe_radius=1.0)

    assert torch.allclose(bare.clearance, probed.clearance, atol=1e-12)
    assert probed.n_void <= bare.n_void
    assert bool((probed.void_clearance() >= 1.0).all())


def test_named_probe_matches_its_radius(cell, random_structure):
    positions = random_structure(cell, n_atoms=30, seed=9)
    named = _field(positions, cell, probe_radius="N2")
    numeric = _field(positions, cell, probe_radius=1.657)
    assert named.probe_radius == pytest.approx(1.657)
    assert named.n_void == numeric.n_void


def test_porosity_and_volumes_are_consistent(cell, random_structure):
    positions = random_structure(cell, n_atoms=40, seed=10)
    field = _field(positions, cell, radii=1.7)
    assert field.void_volume == pytest.approx(field.n_void * field.grid.voxel_volume)
    assert field.porosity == pytest.approx(field.void_volume / field.cell_volume)
    assert 0.0 <= field.porosity <= 1.0


def test_void_accessors_agree_with_the_grids(cell, random_structure):
    positions = random_structure(cell, n_atoms=30, seed=11)
    field = _field(positions, cell, radii=1.7)
    indices = field.void_indices()
    assert indices.numel() == field.n_void
    assert torch.allclose(
        field.void_clearance(), field.clearance.reshape(-1)[indices]
    )
    assert torch.allclose(field.void_coordinates(), field.grid.cartesian(indices))


def test_masses_give_specific_volume_and_are_validated(cell, random_structure):
    positions = random_structure(cell, n_atoms=20, seed=12)
    masses = np.full(20, 12.011)
    field = _field(positions, cell, radii=1.7, masses=masses)
    assert field.mass_g > 0.0
    assert field.specific_void_volume > 0.0

    without = _field(positions, cell, radii=1.7)
    assert without.mass_g is None
    assert without.specific_void_volume is None

    with pytest.raises(ValueError, match="masses must be"):
        _field(positions, cell, radii=1.7, masses=np.ones(3))


def test_open_boundaries_leave_more_void_than_periodic_ones(random_structure):
    """Without wrapping, atoms across the boundary stop shadowing the void."""
    cell = np.diag([12.0, 12.0, 12.0])
    positions = random_structure(cell, n_atoms=40, seed=13)
    periodic = _field(positions, cell, radii=1.7, pbc=(True, True, True))
    open_z = _field(positions, cell, radii=1.7, pbc=(True, True, False))
    assert open_z.n_void >= periodic.n_void


def test_uff_radii_come_from_the_symbols():
    positions = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    cell = np.diag([10.0, 10.0, 10.0])
    field = _field(positions, cell, radii="uff", symbols=["C", "H"])
    assert field.radii[0].item() == pytest.approx(3.431 / 2)
    assert field.radii[1].item() == pytest.approx(2.571 / 2)


def test_invalid_inputs_are_rejected():
    cell = np.diag([10.0, 10.0, 10.0])
    with pytest.raises(ValueError, match="method must be"):
        _field(np.zeros((1, 3)), cell, method="magic")
    with pytest.raises(ValueError, match="positions must be"):
        _field(np.zeros((1, 4)), cell)
    with pytest.raises(ValueError, match="at least one atom"):
        _field(np.zeros((0, 3)), cell)
