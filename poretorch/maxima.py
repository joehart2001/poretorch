"""Local maxima of the clearance field: the discrete pore centres.

A local maximum of the largest-empty-sphere field is a point where no nearby
point admits a bigger sphere, so it is the centre of a locally maximal inscribed
sphere. Collectively these approximate the medial axis of the void, and they are
the natural candidate set both for sphere packing and for the covering-sphere
pore-size definition.
"""

import numpy as np
from scipy.ndimage import maximum_filter1d


def local_maxima(
    clearance,
    void_mask,
    pbc=(True, True, True),
    min_radius: float = 0.0,
    filter_size: int = 3,
):
    """
    Locate local maxima of the clearance field within the void.

    Parameters
    ----------
    clearance
        Clearance grid shaped ``(nx, ny, nz)``, as a numpy array or torch
        tensor.
    void_mask
        Boolean grid of the same shape marking void points.
    pbc
        Per-axis periodicity. Fully periodic cells use a wrapping filter so
        maxima are found correctly across the boundary.
    min_radius
        Discard maxima whose clearance is below this, in Angstrom.
    filter_size
        Neighbourhood edge length for the maximum filter. 3 gives a 3x3x3
        window.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Grid indices of the maxima shaped ``(n_maxima, 3)``, and their clearance
        values shaped ``(n_maxima,)``.
    """
    clearance = _as_numpy(clearance)
    void_mask = _as_numpy(void_mask).astype(bool, copy=False)
    if clearance.shape != void_mask.shape:
        raise ValueError(
            f"clearance {clearance.shape} and void_mask {void_mask.shape} "
            "must have the same shape"
        )
    if filter_size < 3 or filter_size % 2 == 0:
        raise ValueError(f"filter_size must be an odd integer >= 3, got {filter_size}")

    # Solid points must not win the maximum filter, so push them to -inf.
    field = np.where(void_mask, clearance.astype(np.float64), -np.inf)

    periodic = _pbc_tuple(pbc)
    filtered = field
    # Apply the box maximum one axis at a time so mixed boundary conditions are
    # respected. A single mode cannot express periodic x/y and open z.
    for axis, wraps in enumerate(periodic):
        filtered = maximum_filter1d(
            filtered,
            size=filter_size,
            axis=axis,
            mode="wrap" if wraps else "nearest",
        )
    peaks = filtered == field
    peaks &= void_mask & (field >= min_radius)

    indices = np.argwhere(peaks)
    if indices.size == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float64)
    values = field[tuple(indices.T)]
    return indices, values


def _as_numpy(array):
    """
    Return a numpy view of a numpy array or torch tensor.

    Parameters
    ----------
    array
        Numpy array or torch tensor, on any device.

    Returns
    -------
    np.ndarray
        Equivalent numpy array.
    """
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _pbc_tuple(pbc) -> tuple[bool, bool, bool]:
    """
    Normalise periodicity flags to a three-tuple of booleans.

    Parameters
    ----------
    pbc
        A single boolean or an iterable of three.

    Returns
    -------
    tuple[bool, bool, bool]
        Per-axis periodicity.
    """
    if isinstance(pbc, (bool, np.bool_)):
        return (bool(pbc),) * 3
    values = tuple(bool(x) for x in _as_numpy(pbc).ravel())
    if len(values) != 3:
        raise ValueError(f"pbc must have three entries, got {len(values)}")
    return values
