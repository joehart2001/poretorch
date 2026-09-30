"""Greedy packing of non-overlapping inscribed spheres into the void.

The result is a discrete sphere-packing spectrum. It is not a partition or a
voxel-wise measure of the void: gaps remain between accepted spheres, and the
histogram is weighted by analytic sphere volume.

Candidates are the local maxima of the clearance field, taken largest first. A
candidate is accepted when it clears every sphere accepted so far. The pass is
global rather than per pore: spheres in different pores cannot overlap through a
solid wall anyway, and a single pass keeps the no-double-counting guarantee
unconditional.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .cell import MinimumImage, mic_distance


@dataclass
class SpherePacking:
    """
    A set of non-overlapping inscribed spheres.

    Attributes
    ----------
    centers
        Cartesian sphere centres shaped ``(n_spheres, 3)`` in Angstrom.
    radii
        Sphere radii shaped ``(n_spheres,)`` in Angstrom.
    pore_ids
        Pore label each sphere belongs to, shaped ``(n_spheres,)``. Zero when no
        labelling was supplied.
    n_candidates
        Number of candidate maxima considered, for reference against
        ``len(radii)``.
    """

    centers: np.ndarray
    radii: np.ndarray
    pore_ids: np.ndarray
    n_candidates: int = 0

    @property
    def n_spheres(self) -> int:
        """Number of accepted spheres."""
        return int(self.radii.size)

    @property
    def volume(self) -> float:
        """Combined volume of the packed spheres in cubic Angstrom."""
        return float((4.0 / 3.0) * np.pi * np.sum(self.radii**3))

    def fraction_of(self, reference_volume: float) -> float:
        """
        Packed volume as a fraction of a caller-supplied reference volume.

        Parameters
        ----------
        reference_volume
            Reference volume in cubic Angstrom, normally the cell volume.

        Returns
        -------
        float
            Packed volume divided by the reference volume.
        """
        if reference_volume <= 0.0:
            return 0.0
        return self.volume / reference_volume


def pack_spheres(
    centers: Tensor,
    radii: Tensor,
    mic: MinimumImage,
    pore_ids=None,
    overlap_tolerance: float = 0.0,
    max_spheres: int | None = None,
) -> SpherePacking:
    """
    Greedily accept the largest candidate spheres that do not overlap.

    Parameters
    ----------
    centers
        Candidate Cartesian centres shaped ``(n_candidates, 3)``.
    radii
        Candidate radii shaped ``(n_candidates,)``.
    mic
        Minimum-image setup, so overlap is tested across periodic boundaries.
    pore_ids
        Optional pore label per candidate, carried through to the result.
    overlap_tolerance
        Spheres are allowed to interpenetrate by this much, in Angstrom. Zero
        enforces strict non-overlap. A small positive value (around 0.1 A)
        loosens the test enough to tolerate grid discretisation.
    max_spheres
        Optional cap on the number of accepted spheres.

    Returns
    -------
    SpherePacking
        The accepted centres and radii.
    """
    centers_np = _as_numpy(centers).astype(np.float64, copy=False)
    radii_np = _as_numpy(radii).astype(np.float64, copy=False)
    if centers_np.shape[0] != radii_np.shape[0]:
        raise ValueError(
            f"centers ({centers_np.shape[0]}) and radii ({radii_np.shape[0]}) "
            "must have the same length"
        )
    n_candidates = int(radii_np.shape[0])
    if pore_ids is None:
        pore_ids_np = np.zeros(n_candidates, dtype=np.int64)
    else:
        pore_ids_np = _as_numpy(pore_ids).astype(np.int64, copy=False)

    if n_candidates == 0:
        return SpherePacking(
            centers=np.empty((0, 3)),
            radii=np.empty(0),
            pore_ids=np.empty(0, dtype=np.int64),
            n_candidates=0,
        )

    order = np.argsort(radii_np)[::-1]
    centers_np = centers_np[order]
    radii_np = radii_np[order]
    pore_ids_np = pore_ids_np[order]

    # The greedy pass is inherently sequential, so it runs on the CPU where the
    # per-candidate overlap test is a cheap vector operation with no launch cost.
    device = torch.device("cpu")
    dtype = torch.float64
    mic_cpu = _mic_to(mic, device, dtype)
    candidates = torch.as_tensor(centers_np, device=device, dtype=dtype)

    # Preallocated buffers, so accepting a sphere is a write rather than a
    # reallocation of everything accepted so far.
    accepted_centers = torch.empty((n_candidates, 3), device=device, dtype=dtype)
    accepted_radii = torch.empty(n_candidates, device=device, dtype=dtype)
    accepted_idx: list[int] = []
    count = 0

    for i in range(n_candidates):
        radius = float(radii_np[i])
        if count:
            delta = candidates[i].unsqueeze(0) - accepted_centers[:count]
            distance = mic_distance(delta, mic_cpu)
            limit = accepted_radii[:count] + radius - overlap_tolerance
            if bool((distance < limit).any().item()):
                continue
        accepted_centers[count] = candidates[i]
        accepted_radii[count] = radius
        accepted_idx.append(i)
        count += 1
        if max_spheres is not None and count >= max_spheres:
            break

    keep = np.asarray(accepted_idx, dtype=np.int64)
    return SpherePacking(
        centers=centers_np[keep],
        radii=radii_np[keep],
        pore_ids=pore_ids_np[keep],
        n_candidates=n_candidates,
    )


def _mic_to(mic: MinimumImage, device, dtype) -> MinimumImage:
    """
    Move a minimum-image setup to another device and dtype.

    Parameters
    ----------
    mic
        Setup to convert.
    device
        Target torch device.
    dtype
        Target torch dtype.

    Returns
    -------
    MinimumImage
        Equivalent setup on the requested device.
    """
    from .cell import MinimumImage as MI

    return MI(
        rcell=mic.rcell.to(device=device, dtype=dtype),
        rcell_inv=mic.rcell_inv.to(device=device, dtype=dtype),
        shifts=mic.shifts.to(device=device, dtype=dtype),
        pbc=mic.pbc.to(device=device),
        orthogonal=mic.orthogonal,
    )


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
