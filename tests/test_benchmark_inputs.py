"""Protect real benchmark geometry when matching PoreBlazer's grid."""

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import write

from benchmarks.compare_poreblazer import carbon_structure, matched_spacing, poreblazer_spacing
from poretorch.grid import Grid


@pytest.mark.parametrize('side,count', [(16.27341908881698, 32), (21.69789211842264, 43), (11.0, 22)])
def test_matched_grid_preserves_cell(side, count):
    cell = np.eye(3) * side
    spacing = matched_spacing(cell, [count, count, count, 1], side / count)
    grid = Grid.from_cell(torch.tensor(cell, dtype=torch.float64), spacing)
    assert grid.shape == (count, count, count)
    np.testing.assert_array_equal(grid.cell.numpy(), cell)
    np.testing.assert_allclose(grid.spacing, side / count, rtol=1e-14)


@pytest.mark.parametrize('shape,spacing', [([32, 32, 31, 1], .5), ([32, 32, 32, 1], 16 / 33)])
def test_rejects_grid_that_does_not_tile_cell(shape, spacing):
    with pytest.raises(ValueError, match='does not tile'):
        matched_spacing(np.eye(3)*16, shape, spacing)


def test_carbon_loader_ignores_image_flags_and_only_translates(tmp_path):
    path = tmp_path / 'carbon.data'
    path.write_text('''LAMMPS carbon benchmark fixture

2 atoms
1 atom types

0 16.27341908881698 xlo xhi
0 16.27341908881698 ylo yhi
0 16.27341908881698 zlo zhi

Masses

1 12.011

Atoms # atomic

1 1 1 2 3 0 3 1
2 1 4 5 6 -1 0 -2
''')
    positions, cell = carbon_structure(path)
    np.testing.assert_array_equal(positions, [[0, 0, 0], [3, 3, 3]])
    np.testing.assert_array_equal(cell, np.eye(3)*16.27341908881698)


@pytest.mark.parametrize('symbols,cell,message', [('O', [10, 10, 10], 'only carbon'), ('C', [10, 11, 10], 'cubic cell')])
def test_rejects_unmatched_chemistry_or_cell(tmp_path, symbols, cell, message):
    path = tmp_path / 'structure.data'
    write(path, Atoms(symbols, positions=[[1, 2, 3]], cell=cell, pbc=True),
          format='lammps-data', atom_style='atomic', masses=True)
    with pytest.raises(ValueError, match=message):
        carbon_structure(path)


def test_poreblazer_second_truncation_uses_a_finer_tiling_grid():
    side = 21.69789211842264
    cell = np.eye(3)*side
    original_count = int(side / .5)
    assert original_count == 43
    assert int(side / (side / original_count)) == 42
    requested = poreblazer_spacing(cell)
    count = int(side / requested)
    assert count == 44
    assert int(side / (side / count)) == count
    assert requested < .5
    np.testing.assert_array_equal(cell, np.eye(3)*side)


def test_poreblazer_stable_grid_keeps_requested_spacing():
    assert poreblazer_spacing(np.eye(3)*16.27341908881698) == .5
