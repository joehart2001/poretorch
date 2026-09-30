"""Cell geometry helpers: validation, widths, and exact minimum-image displacements.

Pore analysis needs distances between grid points and atoms under periodic
boundary conditions in cells of arbitrary shape. The naive orthorhombic rule
``d -= L * round(d / L)`` is only the true minimum image when the cell is close
to orthogonal. For a skewed triclinic cell it can return a vector that is not
the shortest one. This module therefore follows the same route as
:func:`ase.geometry.find_mic`: reduce the cell to a Minkowski-reduced basis,
wrap the displacement into that basis, and then search the 27 surrounding
lattice images. That search is exact for a reduced basis in three dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
from torch import Tensor

# A cell is treated as orthogonal when its off-diagonal Gram terms are this
# small relative to the cell scale. Below it the 27-image search is skipped.
ORTHO_TOL = 1e-8


def cell_tensor(cell, device, dtype) -> Tensor:
    """
    Convert an input cell into a torch tensor shaped ``(3, 3)``.

    Parameters
    ----------
    cell
        Cell matrix shaped ``(3, 3)`` with lattice vectors as rows (triclinic
        allowed), or three lengths for a diagonal cell.
    device
        Torch device for the returned tensor.
    dtype
        Torch dtype for the returned tensor.

    Returns
    -------
    Tensor
        Cell tensor shaped ``(3, 3)``.
    """
    cell_t = torch.as_tensor(cell, device=device, dtype=dtype)
    if cell_t.shape == (3,):
        cell_t = torch.diag(cell_t)
    if cell_t.shape != (3, 3):
        raise ValueError(f"cell must be (3,3) or (3,), got {tuple(cell_t.shape)}")
    if torch.det(cell_t).abs().item() <= 0.0:
        raise ValueError("cell is singular (zero volume). A periodic cell is required")
    return cell_t


def pbc_tensor(pbc, device) -> Tensor:
    """
    Convert periodic boundary flags to a boolean tensor shaped ``(3,)``.

    Parameters
    ----------
    pbc
        Iterable of three booleans, or a single boolean applied to all axes.
    device
        Torch device for the returned tensor.

    Returns
    -------
    Tensor
        Boolean PBC tensor shaped ``(3,)``.
    """
    if isinstance(pbc, (bool, np.bool_)):
        pbc = (bool(pbc),) * 3
    pbc_t = torch.as_tensor(pbc, device=device, dtype=torch.bool)
    if pbc_t.shape != (3,):
        raise ValueError(f"pbc must be (3,), got {tuple(pbc_t.shape)}")
    return pbc_t


def cell_volume(cell: Tensor) -> float:
    """
    Compute the volume of a cell tensor shaped ``(3, 3)``.

    Parameters
    ----------
    cell
        Cell tensor shaped ``(3, 3)``.

    Returns
    -------
    float
        Absolute value of the cell determinant (volume).
    """
    if cell.shape != (3, 3):
        raise ValueError(f"cell must be (3,3), got {tuple(cell.shape)}")
    return float(torch.det(cell).abs().item())


def perpendicular_widths(cell: Tensor) -> Tensor:
    """
    Compute the distance between opposite faces of the cell along each axis.

    For a triclinic cell the lattice vector lengths overestimate how much room
    there is across the cell. The perpendicular width ``V / |a_j x a_k|`` is the
    quantity that controls grid resolution and neighbour-bin sizing.

    Parameters
    ----------
    cell
        Cell tensor shaped ``(3, 3)`` with lattice vectors as rows.

    Returns
    -------
    Tensor
        Perpendicular widths shaped ``(3,)``, in the same units as ``cell``.
    """
    volume = torch.det(cell).abs()
    areas = torch.stack(
        [
            torch.linalg.cross(cell[1], cell[2]).norm(),
            torch.linalg.cross(cell[2], cell[0]).norm(),
            torch.linalg.cross(cell[0], cell[1]).norm(),
        ]
    )
    return volume / areas


def is_orthogonal(cell: Tensor, tol: float = ORTHO_TOL) -> bool:
    """
    Test whether the lattice vectors of a cell are mutually perpendicular.

    Parameters
    ----------
    cell
        Cell tensor shaped ``(3, 3)``.
    tol
        Relative tolerance on the off-diagonal Gram-matrix terms.

    Returns
    -------
    bool
        ``True`` when the cell is orthogonal (the cheap wrap is then exact).
    """
    gram = cell @ cell.T
    scale = torch.diagonal(gram).max()
    off = gram - torch.diag(torch.diagonal(gram))
    return bool((off.abs().max() <= tol * scale).item())


@dataclass(frozen=True)
class MinimumImage:
    """
    Precomputed quantities for exact minimum-image displacements.

    Attributes
    ----------
    rcell
        Minkowski-reduced cell shaped ``(3, 3)``. Spans the same lattice as the
        input cell, so it may be used for distances even when the grid itself is
        built on the original cell.
    rcell_inv
        Inverse of ``rcell``, mapping Cartesian vectors to reduced fractional
        coordinates.
    shifts
        Candidate Cartesian lattice shifts shaped ``(n_images, 3)``. Empty of all
        but the zero vector when the reduced cell is orthogonal, since the wrap
        alone is then exact.
    pbc
        Boolean periodicity flags shaped ``(3,)``.
    orthogonal
        Whether ``rcell`` is orthogonal.
    """

    rcell: Tensor
    rcell_inv: Tensor
    shifts: Tensor
    pbc: Tensor
    orthogonal: bool


def minimum_image_setup(cell: Tensor, pbc: Tensor) -> MinimumImage:
    """
    Build the reduced basis and image shifts needed for exact minimum imaging.

    Parameters
    ----------
    cell
        Cell tensor shaped ``(3, 3)``.
    pbc
        Boolean PBC tensor shaped ``(3,)``.

    Returns
    -------
    MinimumImage
        Reduced cell, its inverse, and the candidate lattice shifts.
    """
    from ase.geometry import minkowski_reduce

    device, dtype = cell.device, cell.dtype
    pbc_np = pbc.detach().cpu().numpy()

    # Minkowski reduction in float64 on the CPU. ASE only touches periodic axes.
    cell_np = cell.detach().cpu().numpy().astype(np.float64)
    rcell_np, _ = minkowski_reduce(cell_np, pbc=pbc_np)
    rcell = torch.as_tensor(np.asarray(rcell_np, dtype=np.float64), device=device, dtype=dtype)
    rcell_inv = torch.linalg.inv(rcell)

    orthogonal = is_orthogonal(rcell)
    if orthogonal:
        # The fractional wrap alone already yields the shortest vector.
        offsets = [(0, 0, 0)]
    else:
        ranges = [(-1, 0, 1) if periodic else (0,) for periodic in pbc_np]
        offsets = list(product(*ranges))
    shift_frac = torch.as_tensor(offsets, device=device, dtype=dtype)
    shifts = shift_frac @ rcell

    return MinimumImage(
        rcell=rcell,
        rcell_inv=rcell_inv,
        shifts=shifts,
        pbc=pbc,
        orthogonal=orthogonal,
    )


def mic_distance(delta: Tensor, mic: MinimumImage) -> Tensor:
    """
    Reduce Cartesian displacements to their minimum-image length.

    Parameters
    ----------
    delta
        Cartesian displacement vectors shaped ``(..., 3)``.
    mic
        Setup returned by :func:`minimum_image_setup`.

    Returns
    -------
    Tensor
        Minimum-image distances shaped ``delta.shape[:-1]``.
    """
    frac = delta @ mic.rcell_inv
    # Wrap only along periodic axes. Open axes keep their raw separation.
    wrapped = frac - torch.round(frac) * mic.pbc
    base = wrapped @ mic.rcell

    if mic.orthogonal:
        return base.norm(dim=-1)

    # Skewed cell: the wrap can leave a vector that a neighbouring image beats.
    candidates = base.unsqueeze(-2) + mic.shifts
    return candidates.norm(dim=-1).amin(dim=-1)


def to_fractional(positions: Tensor, cell: Tensor) -> Tensor:
    """
    Convert Cartesian coordinates to fractional coordinates of a cell.

    Parameters
    ----------
    positions
        Cartesian coordinates shaped ``(..., 3)``.
    cell
        Cell tensor shaped ``(3, 3)`` with lattice vectors as rows.

    Returns
    -------
    Tensor
        Fractional coordinates shaped like ``positions``.
    """
    return positions @ torch.linalg.inv(cell)
