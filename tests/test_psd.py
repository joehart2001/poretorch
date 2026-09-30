import numpy as np
import pytest
import torch

from poretorch.cell import mic_distance
from poretorch.field import distance_field
from poretorch.labels import label_pores
from poretorch.psd import (
    bin_distribution,
    covering_sizes,
    pore_size_distribution,
)


def _field(cell, positions, **kwargs):
    kwargs.setdefault("grid_spacing", 0.9)
    return distance_field(
        positions,
        cell,
        radii=1.7,
        device="cpu",
        torch_dtype=torch.float64,
        **kwargs,
    )


def _naive_covering(field):
    """Reference implementation: every void point against every candidate."""
    coords = field.void_coordinates()
    radii = field.void_clearance()
    distance = mic_distance(coords.unsqueeze(1) - coords.unsqueeze(0), field.mic)
    covered = distance <= radii.unsqueeze(0)
    return torch.where(
        covered, radii.unsqueeze(0).expand_as(distance), torch.full_like(distance, -np.inf)
    ).amax(dim=1)


def test_covering_matches_the_naive_definition(cell, random_structure):
    """The early-exit search must reproduce the plain all-pairs maximum."""
    field = _field(cell, random_structure(cell, n_atoms=30, seed=21))
    assert torch.allclose(covering_sizes(field), _naive_covering(field), atol=1e-12)


def test_covering_is_never_smaller_than_the_local_clearance(cell, random_structure):
    """A sphere centred elsewhere can be bigger, never smaller, than one here."""
    field = _field(cell, random_structure(cell, n_atoms=30, seed=22))
    assert bool((covering_sizes(field) >= field.void_clearance() - 1e-12).all())


def test_covering_is_insensitive_to_the_candidate_batch_size(cell, random_structure):
    field = _field(cell, random_structure(cell, n_atoms=30, seed=23))
    small = covering_sizes(field, center_batch_size=7)
    large = covering_sizes(field, center_batch_size=100_000)
    assert torch.allclose(small, large, atol=1e-12)


def test_local_maxima_centres_never_exceed_the_full_candidate_set(cell, random_structure):
    """Fewer candidates can only ever find a smaller covering sphere."""
    field = _field(cell, random_structure(cell, n_atoms=30, seed=24))
    subset = covering_sizes(field, center_mode="local_maxima")
    full = covering_sizes(field, center_mode="all_void")
    assert bool((subset <= full + 1e-12).all())


def test_all_void_conserves_the_void_volume(cell, random_structure):
    """Voxel weighting means the distribution integrates to the void volume."""
    field = _field(cell, random_structure(cell, n_atoms=30, seed=25))
    result = pore_size_distribution(field, method="all_void", bin_width=0.2)
    assert result.volume_A3.sum() == pytest.approx(field.void_volume, rel=1e-9)


def test_covering_conserves_the_void_volume(cell, random_structure):
    field = _field(cell, random_structure(cell, n_atoms=30, seed=26))
    result = pore_size_distribution(field, method="covering", bin_width=0.2)
    assert result.volume_A3.sum() == pytest.approx(field.void_volume, rel=1e-9)


def test_covering_sits_at_larger_diameters_than_all_void(cell, random_structure):
    field = _field(cell, random_structure(cell, n_atoms=30, seed=27))
    local = pore_size_distribution(field, method="all_void", bin_width=0.2)
    covering = pore_size_distribution(field, method="covering", bin_width=0.2)
    assert (
        covering.stats["median_diameter_A"] >= local.stats["median_diameter_A"]
    )
    assert covering.stats["mean_diameter_A"] >= local.stats["mean_diameter_A"]


def test_packed_spheres_do_not_overlap(cell, random_structure):
    """The whole point of packing: no void volume is counted twice."""
    from poretorch.psd import _packed_spheres

    field = _field(cell, random_structure(cell, n_atoms=30, seed=28))
    labels = label_pores(
        field.void_mask, field.clearance, field.grid.voxel_volume, field.mic.pbc
    )
    packing = _packed_spheres(field, labels, min_radius=0.3, overlap_tolerance=0.0, filter_size=3)
    if packing.n_spheres < 2:
        pytest.skip("too few spheres to test overlap")

    centers = torch.as_tensor(packing.centers)
    radii = torch.as_tensor(packing.radii)
    distance = mic_distance(centers.unsqueeze(1) - centers.unsqueeze(0), field.mic).numpy()
    limit = (radii[:, None] + radii[None, :]).numpy()
    np.fill_diagonal(distance, np.inf)
    assert (distance >= limit - 1e-9).all()


def test_packed_volume_stays_below_the_void_volume(cell, random_structure):
    field = _field(cell, random_structure(cell, n_atoms=30, seed=29))
    result = pore_size_distribution(field, method="packed", bin_width=0.2, min_radius=0.3)
    assert 0.0 < result.stats["filling_fraction"] < 1.0
    assert result.stats["n_spheres"] <= result.stats["n_candidates"]


def test_every_packed_sphere_belongs_to_a_pore(cell, random_structure):
    from poretorch.psd import _packed_spheres

    field = _field(cell, random_structure(cell, n_atoms=30, seed=30))
    labels = label_pores(
        field.void_mask, field.clearance, field.grid.voxel_volume, field.mic.pbc
    )
    packing = _packed_spheres(field, labels, 0.3, 0.0, 3)
    assert np.all(packing.pore_ids > 0)


def test_method_aliases_resolve(cell, random_structure):
    field = _field(cell, random_structure(cell, n_atoms=25, seed=31))
    assert pore_size_distribution(field, method="poreblazer").method == "covering"
    assert pore_size_distribution(field, method="sphere_packing").method == "packed"
    assert pore_size_distribution(field, method="local").method == "all_void"


def test_unknown_method_is_rejected(cell, random_structure):
    field = _field(cell, random_structure(cell, n_atoms=20, seed=32))
    with pytest.raises(ValueError, match="method must be one of"):
        pore_size_distribution(field, method="nonsense")


def test_min_radius_removes_the_smallest_pores(cell, random_structure):
    field = _field(cell, random_structure(cell, n_atoms=30, seed=33))
    loose = pore_size_distribution(field, method="all_void", bin_width=0.2)
    strict = pore_size_distribution(field, method="all_void", bin_width=0.2, min_radius=0.5)
    assert strict.volume_A3.sum() <= loose.volume_A3.sum()
    assert strict.stats["min_diameter_A"] >= 1.0


def test_reported_units_are_consistent(cell, random_structure):
    positions = random_structure(cell, n_atoms=30, seed=34)
    field = _field(cell, positions, masses=np.full(len(positions), 12.011))
    result = pore_size_distribution(field, method="all_void", bin_width=0.25)

    assert np.allclose(result.diameters_nm, result.diameters_A / 10.0)
    assert np.allclose(result.dV_dd_A2, result.volume_A3 / result.bin_width_A)
    expected = result.volume_A3 * 1e-24 / (result.bin_width_A / 10.0) / result.mass_g
    assert np.allclose(result.dV_dd_cc_per_nm_per_g, expected)
    assert np.trapezoid(result.normalised, result.diameters_A) == pytest.approx(1.0)
    assert result.cumulative_fraction[-1] == pytest.approx(1.0)


def test_per_gram_units_need_masses(cell, random_structure):
    field = _field(cell, random_structure(cell, n_atoms=20, seed=35))
    result = pore_size_distribution(field, method="all_void")
    with pytest.raises(ValueError, match="needs the structure mass"):
        _ = result.dV_dd_cc_per_nm_per_g


def test_binning_is_exact_for_known_input():
    centers, volume = bin_distribution([0.5, 1.0], [2.0, 3.0], bin_width=1.0, max_diameter=3.0)
    # Diameters are 1.0 and 2.0, landing in bins [1,2) and [2,3).
    assert np.allclose(centers, [0.5, 1.5, 2.5])
    assert np.allclose(volume, [0.0, 2.0, 3.0])


def test_binning_rejects_a_non_positive_width():
    with pytest.raises(ValueError, match="bin_width must be"):
        bin_distribution([1.0], [1.0], bin_width=0.0)


def test_median_is_interpolated_not_snapped_to_a_bin():
    """A coarse and a fine binning must agree on the median to within a bin."""
    cell = np.diag([12.0, 12.0, 12.0])
    rng = np.random.default_rng(36)
    field = _field(cell, rng.random((30, 3)) @ cell)
    coarse = pore_size_distribution(field, method="all_void", bin_width=0.5)
    fine = pore_size_distribution(field, method="all_void", bin_width=0.05)
    assert coarse.stats["median_diameter_A"] == pytest.approx(
        fine.stats["median_diameter_A"], abs=0.5
    )
