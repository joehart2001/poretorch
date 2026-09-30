import numpy as np
import pytest

from poretorch.labels import label_pores

SIZE = 8


def _empty():
    return np.zeros((SIZE, SIZE, SIZE), dtype=bool)


def _only(mask, **kwargs):
    """Label and return the single pore, asserting there is exactly one."""
    result = label_pores(mask, **kwargs)
    assert result.n_pores == 1
    return result.pores[0]


def test_isolated_blob_is_a_closed_pore():
    mask = _empty()
    mask[2:4, 2:4, 2:4] = True
    pore = _only(mask)
    assert pore.dimensionality == 0
    assert pore.kind == "closed"
    assert not pore.is_percolating
    assert pore.n_voxels == 8
    assert pore.spans_axes == (False, False, False)


def test_straight_channel_percolates_along_one_axis():
    mask = _empty()
    mask[3, 3, :] = True
    pore = _only(mask)
    assert pore.dimensionality == 1
    assert pore.kind == "channel"
    assert pore.percolation_vectors == ((0, 0, 1),)
    assert pore.spans_axes == (False, False, True)


def test_planar_void_percolates_in_two_directions():
    mask = _empty()
    mask[3, :, :] = True
    pore = _only(mask)
    assert pore.dimensionality == 2
    assert pore.kind == "planar"


def test_fully_open_void_percolates_in_three_directions():
    pore = _only(np.ones((SIZE, SIZE, SIZE), dtype=bool))
    assert pore.dimensionality == 3
    assert pore.kind == "network"
    assert pore.spans_axes == (True, True, True)


def test_blob_straddling_a_boundary_is_one_closed_pore():
    """The point of tracking images: wrapping is not the same as percolating."""
    mask = _empty()
    mask[[0, 1, SIZE - 1], 2:4, 2:4] = True
    pore = _only(mask)
    assert pore.n_voxels == 12  # merged into one pore
    assert pore.dimensionality == 0  # but it does not run through the cell


def test_separate_regions_stay_separate():
    mask = _empty()
    mask[1, 1, 1] = True
    mask[5, 5, 5] = True
    result = label_pores(mask)
    assert result.n_pores == 2
    assert {pore.n_voxels for pore in result.pores} == {1}


def test_channel_is_closed_when_its_axis_is_not_periodic():
    mask = _empty()
    mask[3, 3, :] = True
    assert _only(mask, pbc=(True, True, False)).dimensionality == 0
    assert _only(mask, pbc=(True, True, True)).dimensionality == 1


def test_diagonal_neighbours_need_higher_connectivity():
    mask = _empty()
    for i in range(SIZE):
        mask[i, 3, i] = True

    face_only = label_pores(mask, connectivity=1)
    assert face_only.n_pores == SIZE  # face-sharing does not link diagonals

    corners = label_pores(mask, connectivity=3)
    assert corners.n_pores == 1
    assert corners.pores[0].percolation_vectors == ((1, 0, 1),)


def test_min_voxels_discards_small_pores_and_their_voxels():
    mask = _empty()
    mask[1, 1, 1] = True
    mask[4:7, 4:7, 4:7] = True
    result = label_pores(mask, min_voxels=5)
    assert result.n_pores == 1
    assert result.pores[0].n_voxels == 27
    assert result.labels[1, 1, 1] == 0


def test_labels_are_consecutive_from_one():
    mask = _empty()
    mask[1, 1, 1] = True
    mask[3, 3, 3] = True
    mask[5, 5, 5] = True
    result = label_pores(mask)
    present = np.unique(result.labels[result.labels > 0])
    assert list(present) == [1, 2, 3]
    assert [pore.id for pore in result.pores] == [1, 2, 3]


def test_clearance_statistics_are_reported():
    mask = _empty()
    mask[2:4, 2:4, 2:4] = True
    clearance = np.zeros_like(mask, dtype=float)
    clearance[2:4, 2:4, 2:4] = np.arange(8).reshape(2, 2, 2)
    pore = _only(mask, clearance=clearance)
    assert pore.max_clearance == pytest.approx(7.0)
    assert pore.mean_clearance == pytest.approx(3.5)


def test_volume_scales_with_voxel_volume():
    mask = _empty()
    mask[2:4, 2:4, 2:4] = True
    pore = _only(mask, voxel_volume=0.125)
    assert pore.volume == pytest.approx(1.0)


def test_empty_void_gives_no_pores():
    result = label_pores(_empty())
    assert result.n_pores == 0
    assert result.total_volume == 0.0
    assert result.summary()["n_pores"] == 0


def test_summary_splits_percolating_from_closed():
    mask = _empty()
    mask[3, 3, :] = True  # channel
    mask[6, 6, 6] = True  # pocket
    summary = label_pores(mask, voxel_volume=1.0).summary()
    assert summary["n_pores"] == 2
    assert summary["n_percolating"] == 1
    assert summary["n_closed"] == 1
    assert summary["percolating_volume_A3"] == pytest.approx(8.0)
    assert summary["closed_volume_A3"] == pytest.approx(1.0)
    assert summary["counts_by_kind"]["channel"] == 1


def test_bad_arguments_are_rejected():
    with pytest.raises(ValueError, match="connectivity must be"):
        label_pores(_empty(), connectivity=4)
    with pytest.raises(ValueError, match="void_mask must be 3D"):
        label_pores(np.zeros((4, 4), dtype=bool))
