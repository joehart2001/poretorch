import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the repo-local PoreTorch is importable when running tests in-tree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


CELLS = {
    "cubic": np.diag([12.0, 12.0, 12.0]),
    "orthorhombic": np.diag([13.0, 11.0, 9.0]),
    "mild_triclinic": np.array([[12.0, 0.0, 0.0], [3.0, 11.0, 0.0], [2.0, 1.5, 10.0]]),
    "skewed_triclinic": np.array([[12.0, 0.0, 0.0], [9.0, 8.0, 0.0], [8.0, 6.0, 7.0]]),
}


@pytest.fixture(params=sorted(CELLS), ids=sorted(CELLS))
def cell(request):
    """Each supported cell geometry in turn, so tests cover triclinic too."""
    return CELLS[request.param]


@pytest.fixture
def random_structure():
    """
    Build a reproducible random structure filling a given cell.

    Returns a callable ``(cell, n_atoms, seed) -> positions``.
    """

    def _make(cell, n_atoms=40, seed=0):
        rng = np.random.default_rng(seed)
        return rng.random((n_atoms, 3)) @ np.asarray(cell, dtype=float)

    return _make


@pytest.fixture
def carbon_atoms():
    """A small random carbon structure as an ASE Atoms object."""
    ase = pytest.importorskip("ase")
    from ase import Atoms

    rng = np.random.default_rng(7)
    cell = np.diag([14.0, 14.0, 14.0])
    positions = rng.random((60, 3)) @ cell
    return Atoms("C" * 60, positions=positions, cell=cell, pbc=True)
