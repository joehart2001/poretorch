"""Weighting changes sample contributions, never geometry or selection."""
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from ase import Atoms

from poretorch import analyse, field_from_atoms, label_pores, psd
from poretorch.adapters import average_psds
from poretorch.io import maybe_save
from poretorch.psd import bin_distribution, pore_size_distribution


@pytest.fixture
def field():
    return field_from_atoms(
        Atoms('C', positions=[[0, 0, 0]], cell=[6, 6, 6], pbc=True),
        radii=1.7, grid_spacing=1, device='cpu', torch_dtype=torch.float64,
    )


def test_equal_counts_give_one_to_eight_sphere_weights(field):
    radii = torch.ones_like(field.clearance)
    radii.reshape(-1)[::2] = 2
    field = replace(field, clearance=radii, void_mask=radii > 0)
    voxel = pore_size_distribution(field, weighting='voxel', bin_width=.5)
    sphere = pore_size_distribution(field, weighting='sphere', bin_width=.5)
    assert voxel.volume_A3[4] == voxel.volume_A3[7]  # final edge includes diameter 4
    assert sphere.volume_A3[7] / sphere.volume_A3[4] == pytest.approx(8)
    for result in [voxel, sphere]:
        assert result.normalised.sum() * result.bin_width_A == pytest.approx(1)
        assert result.total_void_volume_A3 == field.void_volume
        assert result.stats['porosity'] == field.porosity
    assert sphere.volume_A3.sum() > sphere.total_void_volume_A3


@pytest.mark.parametrize('method,weighting', [('all_void', 'voxel'), ('covering', 'voxel'), ('packed', 'sphere')])
def test_explicit_defaults_identical(field, method, weighting):
    default = pore_size_distribution(field, method=method)
    explicit = pore_size_distribution(field, method=method, weighting=weighting)
    np.testing.assert_array_equal(default.volume_A3, explicit.volume_A3)
    assert default.stats == explicit.stats
    assert default.weighting == weighting


@pytest.mark.parametrize('method,weighting', [('covering', 'sphere'), ('packed', 'voxel'), ('all_void', 'bogus')])
def test_invalid_combinations(field, method, weighting):
    with pytest.raises(ValueError, match='weighting'):
        pore_size_distribution(field, method=method, weighting=weighting)


@pytest.mark.parametrize('pbc', [True, False, [True, False, True]])
def test_identical_samples_and_filters(field, pbc):
    # The end points form a periodic component, while the middle point is
    # isolated. A second radius filter removes one point from the kept pore.
    values = torch.full_like(field.clearance, -1)
    values[0, 0, 0], values[-1, 0, 0] = 1.5, 2.0
    values[2, 3, 3] = 3
    field = replace(field, clearance=values, void_mask=values >= 1.325, probe_radius=1.325)
    labels = label_pores(field.void_mask, voxel_volume=field.grid.voxel_volume, pbc=pbc, min_voxels=2)
    included = values.numpy()[(labels.labels > 0) & (values.numpy() >= 1.7)]
    for weighting in ['voxel', 'sphere']:
        result = pore_size_distribution(field, labels=labels, min_radius=1.7, weighting=weighting, max_diameter=8)
        weights = field.grid.voxel_volume if weighting == 'voxel' else 4 / 3 * np.pi * included**3
        centers, expected = bin_distribution(included, weights, max_diameter=8)
        np.testing.assert_array_equal(result.diameters_A, centers)
        np.testing.assert_allclose(result.volume_A3, expected)
        assert result.stats['n_samples'] == included.size
        assert result.normalised.sum() * result.bin_width_A == pytest.approx(1 if included.size else 0)


@pytest.mark.parametrize('suffix', ['npz', 'json', 'csv', 'tsv'])
@pytest.mark.parametrize('weighting', ['voxel', 'sphere'])
def test_exports(field, tmp_path, suffix, weighting):
    result = pore_size_distribution(field, weighting=weighting)
    target = tmp_path / f'psd.{suffix}'
    maybe_save(None, target, result)
    if suffix == 'npz':
        with np.load(target) as data:
            assert data['weighting'] == weighting
            assert data['volume_kind'] == result.volume_kind
            assert data['total_void_volume_A3'] == field.void_volume
    elif suffix == 'json':
        data = json.loads(target.read_text())
        assert data['weighting'] == weighting
        assert data['volume_kind'] == result.volume_kind
        assert data['total_void_volume_A3'] == field.void_volume
    else:
        text = target.read_text()
        assert f'# weighting: {weighting}' in text
        assert f'# volume_kind: {result.volume_kind}' in text
        data = np.loadtxt(target, delimiter=',' if suffix == 'csv' else '\t', skiprows=1)
        np.testing.assert_array_equal(data[:, 2], result.volume_A3)


def test_high_level_ensemble_and_summary():
    atoms = Atoms('C', positions=[[0, 0, 0]], cell=[6, 6, 6], pbc=True)
    options = dict(radii=1.7, grid_spacing=1, device='cpu', weighting='sphere')
    analysis = analyse(atoms, **options)
    result = psd([atoms, atoms], **options)
    assert result.weighting == 'sphere'
    assert result.n_frames == 2
    assert analysis.summary()['weighting'] == 'sphere'
    assert analysis.summary()['volume_kind'] == 'overlapping_sphere_volume'
    np.testing.assert_array_equal(result.volume_A3, analysis.psd.volume_A3)
    np.testing.assert_array_equal(result.se_volume_A3, 0)
    voxel = psd(atoms, **dict(options, weighting='voxel'))
    with pytest.raises(ValueError, match='weightings'):
        average_psds([voxel, result])
    alias = psd(atoms, method='local', **options)
    np.testing.assert_array_equal(alias.volume_A3, analysis.psd.volume_A3)


def test_plot_volume_does_not_claim_physical_pore_volume(field):
    from poretorch.plotting import plot_psd
    sphere = pore_size_distribution(field, weighting='sphere')
    voxel = pore_size_distribution(field)
    for results in [[sphere], [sphere, voxel], [voxel, sphere]]:
        fig = plot_psd(results, y='volume')
        assert 'overlapping spheres' in fig.gca().get_ylabel()
        assert 'Pore volume' not in fig.gca().get_ylabel()


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_paper_parameters_and_normalisation(device):
    """Independent histogram reference for the paper's sphere-weighted formula."""
    atoms = Atoms('C2', positions=[[0, 0, 0], [3, 3, 3]], cell=[7, 7, 7], pbc=True)
    options = dict(radii=1.7, grid_spacing=.3, probe_radius=1.325,
                   device=device, torch_dtype=torch.float64)
    field = field_from_atoms(atoms, **options)
    min_voxels = int(np.ceil(1.0 / field.grid.voxel_volume))
    labels = label_pores(field.void_mask, voxel_volume=field.grid.voxel_volume,
                        pbc=atoms.pbc, min_voxels=min_voxels)
    radii = field.clearance.cpu().numpy()
    radii = radii[(labels.labels > 0) & (radii >= .5)]
    edges = np.arange(0, 30.5, .5)
    expected, _ = np.histogram(2 * radii, bins=edges, weights=(4 / 3) * np.pi * radii**3)
    result = psd(atoms, weighting='sphere', min_voxels=min_voxels,
                 min_radius=.5, bin_width=.5, max_diameter=30, **options)
    np.testing.assert_array_equal(result.volume_A3, expected)
    np.testing.assert_allclose(result.normalised * .5, expected / expected.sum())
    np.testing.assert_allclose(result.dV_dd_cc_per_nm_per_g, expected * 1e-24 / .05 / field.mass_g)
