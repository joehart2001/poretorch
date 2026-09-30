import numpy as np
import pytest
import torch

from poretorch.cell import (
    cell_tensor,
    cell_volume,
    is_orthogonal,
    mic_distance,
    minimum_image_setup,
    pbc_tensor,
    perpendicular_widths,
)


def _setup(cell, pbc=(True, True, True)):
    cell_t = cell_tensor(cell, "cpu", torch.float64)
    return cell_t, minimum_image_setup(cell_t, pbc_tensor(pbc, "cpu"))


def test_cell_tensor_accepts_three_lengths():
    cell = cell_tensor([4.0, 5.0, 6.0], "cpu", torch.float64)
    assert torch.allclose(cell, torch.diag(torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)))


def test_cell_tensor_rejects_bad_shape():
    with pytest.raises(ValueError, match="cell must be"):
        cell_tensor(np.zeros((2, 2)), "cpu", torch.float64)


def test_cell_tensor_rejects_singular_cell():
    with pytest.raises(ValueError, match="singular"):
        cell_tensor(np.zeros((3, 3)), "cpu", torch.float64)


def test_pbc_tensor_accepts_scalar():
    assert torch.equal(pbc_tensor(True, "cpu"), torch.tensor([True, True, True]))


def test_cell_volume_matches_determinant():
    cell = cell_tensor(np.array([[3.0, 0, 0], [1.0, 4.0, 0], [0, 0, 5.0]]), "cpu", torch.float64)
    assert cell_volume(cell) == pytest.approx(60.0)


def test_perpendicular_widths_for_orthorhombic_are_the_lengths():
    cell = cell_tensor([3.0, 4.0, 5.0], "cpu", torch.float64)
    assert np.allclose(perpendicular_widths(cell).numpy(), [3.0, 4.0, 5.0])


def test_perpendicular_width_is_smaller_than_length_when_skewed():
    # A sheared cell has the same volume but less room across it.
    cell = cell_tensor(np.array([[10.0, 0, 0], [8.0, 6.0, 0], [0, 0, 10.0]]), "cpu", torch.float64)
    widths = perpendicular_widths(cell).numpy()
    lengths = np.linalg.norm(cell.numpy(), axis=1)
    assert widths[1] < lengths[1]
    assert widths[1] == pytest.approx(6.0)


def test_is_orthogonal_distinguishes_cells():
    assert is_orthogonal(cell_tensor([1.0, 2.0, 3.0], "cpu", torch.float64))
    skewed = cell_tensor(np.array([[1.0, 0, 0], [0.5, 1.0, 0], [0, 0, 1.0]]), "cpu", torch.float64)
    assert not is_orthogonal(skewed)


def test_mic_distance_matches_ase_find_mic(cell):
    """The exactness claim: our minimum image must agree with ASE's."""
    ase_geometry = pytest.importorskip("ase.geometry")
    _, mic = _setup(cell)
    rng = np.random.default_rng(11)
    vectors = (rng.random((500, 3)) - 0.5) * 60.0

    ours = mic_distance(torch.as_tensor(vectors), mic).numpy()
    _, reference = ase_geometry.find_mic(vectors, np.asarray(cell), pbc=True)
    assert np.allclose(ours, reference, atol=1e-10)


def test_mic_distance_never_exceeds_the_naive_wrap(cell):
    """The image search can only ever shorten the wrapped displacement."""
    cell_t, mic = _setup(cell)
    rng = np.random.default_rng(12)
    vectors = torch.as_tensor((rng.random((300, 3)) - 0.5) * 40.0)

    frac = vectors @ torch.linalg.inv(cell_t)
    naive = ((frac - torch.round(frac)) @ cell_t).norm(dim=-1)
    assert bool((mic_distance(vectors, mic) <= naive + 1e-12).all())


def test_open_axis_is_not_wrapped():
    cell_t, mic = _setup(np.diag([10.0, 10.0, 10.0]), pbc=(True, True, False))
    delta = torch.tensor([[0.0, 0.0, 9.0]], dtype=torch.float64)
    # A periodic z would report 1.0 via the image at -10.
    assert mic_distance(delta, mic).item() == pytest.approx(9.0)


def test_orthogonal_cell_skips_the_image_search():
    _, ortho = _setup(np.diag([5.0, 6.0, 7.0]))
    _, skewed = _setup(np.array([[5.0, 0, 0], [4.0, 5.0, 0], [3.0, 2.0, 5.0]]))
    assert ortho.orthogonal and ortho.shifts.shape[0] == 1
    assert not skewed.orthogonal and skewed.shifts.shape[0] == 27
