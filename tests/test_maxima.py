import numpy as np

from poretorch.maxima import local_maxima


def test_local_maxima_respects_mixed_periodicity():
    """Periodic axes wrap even when another axis is open."""
    clearance = np.zeros((5, 3, 3), dtype=float)
    clearance[0, 1, 1] = 5.0
    clearance[-1, 1, 1] = 4.0
    mask = np.ones_like(clearance, dtype=bool)

    indices, _ = local_maxima(
        clearance,
        mask,
        pbc=(True, True, False),
        filter_size=3,
    )
    maxima = {tuple(index) for index in indices}

    assert (0, 1, 1) in maxima
    assert (4, 1, 1) not in maxima
