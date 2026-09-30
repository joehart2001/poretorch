import numpy as np
import pytest
import torch

from poretorch.cell import cell_tensor, cell_volume
from poretorch.grid import Grid


def _grid(cell, spacing=0.5):
    return Grid.from_cell(cell_tensor(cell, "cpu", torch.float64), spacing)


def test_voxels_tile_the_cell_exactly(cell):
    """Total voxel volume must equal the cell volume for any cell shape."""
    grid = _grid(cell)
    total = grid.voxel_volume * grid.n_points
    assert total == pytest.approx(cell_volume(cell_tensor(cell, "cpu", torch.float64)))


def test_realised_spacing_never_exceeds_the_request(cell):
    grid = _grid(cell, spacing=0.37)
    assert np.all(grid.spacing <= 0.37 + 1e-12)


def test_counts_follow_perpendicular_width_not_vector_length():
    """Shearing b shrinks the room across a, so a gets fewer points than |a|."""
    cell = np.array([[10.0, 0.0, 0.0], [8.0, 6.0, 0.0], [0.0, 0.0, 10.0]])
    grid = _grid(cell, spacing=1.0)
    # |a| = 10 but the a-perpendicular width is V / |b x c| = 600 / 100 = 6.
    assert grid.shape == (6, 6, 10)
    assert np.allclose(grid.spacing, [1.0, 1.0, 1.0])


def test_batches_reproduce_the_full_coordinate_list(cell):
    grid = _grid(cell, spacing=2.0)
    pieces = [coords for _, _, coords in grid.batches(7)]
    stitched = torch.cat(pieces, dim=0)
    assert torch.allclose(stitched, grid.coordinates())


def test_batches_cover_every_point_once(cell):
    grid = _grid(cell, spacing=2.0)
    spans = [(start, stop) for start, stop, _ in grid.batches(5)]
    assert spans[0][0] == 0
    assert spans[-1][1] == grid.n_points
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))


def test_fractional_coordinates_are_voxel_centres():
    grid = _grid(np.diag([4.0, 4.0, 4.0]), spacing=1.0)
    frac = grid.fractional(torch.tensor([0]))
    assert torch.allclose(frac, torch.full((1, 3), 0.125, dtype=torch.float64))


def test_zero_spacing_is_rejected():
    with pytest.raises(ValueError, match="spacing must be"):
        _grid(np.diag([4.0, 4.0, 4.0]), spacing=0.0)


def test_zero_batch_size_is_rejected():
    grid = _grid(np.diag([4.0, 4.0, 4.0]), spacing=1.0)
    with pytest.raises(ValueError, match="batch_size must be"):
        list(grid.batches(0))
