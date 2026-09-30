"""Backends for the grid-point to atomic-surface distance reduction.

Every pore-size definition in this package rests on one quantity: for each grid
point, the distance to the nearest atomic surface,

.. math:: d(g) = \\min_i \\left( |g - r_i|_\\mathrm{mic} - R_i \\right)

The brute-force kernel evaluates this against every atom, which costs
``O(n_grid * n_atoms)``. The cell-list kernel bins atoms once and then only
examines the bins around each grid point, which removes the atom-count factor
and is what makes large supercells tractable.

The cell list is exact, not approximate. Each query returns both a value and a
flag saying whether the search radius was provably large enough to have found
the true nearest surface. Unresolved points are retried on a wider ring and
finally handed to the brute-force kernel, so the two backends agree to
floating-point rounding by construction.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
from torch import Tensor

from .cell import MinimumImage, mic_distance, perpendicular_widths

# Cap on the size of the intermediate (points, candidates, images, 3) tensor,
# in number of elements. Candidate chunking is derived from this.
MAX_INTERMEDIATE = 32_000_000


@dataclass(frozen=True)
class AtomBins:
    """
    Atoms bucketed into a regular grid of bins for neighbour queries.

    Binning happens in the fractional coordinates of the Minkowski-reduced cell,
    the same basis used for minimum-image distances, so bin adjacency and
    spatial proximity agree even for a skewed cell.

    Attributes
    ----------
    table
        Atom indices per bin, shaped ``(n_bins, max_occupancy)`` and padded with
        ``-1``.
    counts
        Number of bins along each reduced-cell axis, shaped ``(3,)``.
    widths
        Perpendicular width of one bin along each axis, in Angstrom.
    positions
        Cartesian atom positions shaped ``(n_atoms, 3)``.
    radii
        Per-atom radii shaped ``(n_atoms,)``.
    max_radius
        Largest atomic radius, needed to bound how far a smaller-but-closer
        surface could hide outside the searched bins.
    mic
        Minimum-image setup shared with the rest of the calculation.
    """

    table: Tensor
    counts: Tensor
    widths: np.ndarray
    positions: Tensor
    radii: Tensor
    max_radius: float
    mic: MinimumImage

    @property
    def n_bins(self) -> int:
        """Total number of bins."""
        return int(self.counts.prod().item())

    def coverage_radius(self, ring: int) -> float:
        """
        Radius that a ring-``ring`` bin search is guaranteed to have covered.

        Any atom outside the searched bins lies at least this far from the query
        point, which is what makes the search verifiable rather than heuristic.

        Parameters
        ----------
        ring
            Number of bin layers searched around the query point's own bin.

        Returns
        -------
        float
            Guaranteed coverage radius in Angstrom, or ``inf`` when the ring
            already wraps around every axis and the search is exhaustive.
        """
        counts = self.counts.detach().cpu().numpy()
        # An axis with few enough bins is covered end to end by this ring and so
        # places no limit on the coverage radius.
        limited = [w for m, w in zip(counts, self.widths) if m > 2 * ring + 1]
        if not limited:
            return float("inf")
        return float(ring * min(limited))


def build_atom_bins(
    positions: Tensor,
    radii: Tensor,
    mic: MinimumImage,
    bin_width: float,
) -> AtomBins:
    """
    Bucket atoms into bins of at least ``bin_width`` along every axis.

    Parameters
    ----------
    positions
        Cartesian atom positions shaped ``(n_atoms, 3)``.
    radii
        Per-atom radii shaped ``(n_atoms,)``.
    mic
        Minimum-image setup from :func:`poretorch.cell.minimum_image_setup`.
    bin_width
        Target minimum bin width in Angstrom. Larger bins mean fewer, fatter
        candidate lists per query and a larger guaranteed coverage radius.

    Returns
    -------
    AtomBins
        Populated bin table plus the metadata needed to query it.
    """
    if bin_width <= 0.0:
        raise ValueError(f"bin_width must be > 0, got {bin_width}")

    device = positions.device
    rwidths = perpendicular_widths(mic.rcell).detach().cpu().numpy()
    counts_np = np.maximum(1, np.floor(rwidths / bin_width).astype(int))
    widths = rwidths / counts_np
    counts = torch.as_tensor(counts_np, device=device, dtype=torch.long)

    bin_idx = _bin_indices(positions, mic, counts)
    flat = _flatten_bin_indices(bin_idx, counts)

    n_bins = int(counts.prod().item())
    occupancy = torch.bincount(flat, minlength=n_bins)
    max_occupancy = int(occupancy.max().item()) if occupancy.numel() else 0
    max_occupancy = max(max_occupancy, 1)

    # Sort atoms by bin so each atom's slot within its bin is a simple offset.
    order = torch.argsort(flat, stable=True)
    flat_sorted = flat[order]
    starts = torch.cumsum(occupancy, dim=0) - occupancy
    slot = torch.arange(flat.numel(), device=device) - starts[flat_sorted]

    table = torch.full((n_bins, max_occupancy), -1, device=device, dtype=torch.long)
    table[flat_sorted, slot] = order

    return AtomBins(
        table=table,
        counts=counts,
        widths=widths,
        positions=positions,
        radii=radii,
        max_radius=float(radii.max().item()) if radii.numel() else 0.0,
        mic=mic,
    )


def _bin_indices(points: Tensor, mic: MinimumImage, counts: Tensor) -> Tensor:
    """
    Map Cartesian points to integer bin indices along each reduced-cell axis.

    Parameters
    ----------
    points
        Cartesian coordinates shaped ``(n, 3)``.
    mic
        Minimum-image setup providing the reduced basis.
    counts
        Bins per axis shaped ``(3,)``.

    Returns
    -------
    Tensor
        Integer bin indices shaped ``(n, 3)``.
    """
    frac = points @ mic.rcell_inv
    # Wrap into [0, 1) along periodic axes. Clamp the rest into the end bins.
    frac = torch.where(mic.pbc, frac - torch.floor(frac), frac)
    idx = torch.floor(frac * counts.to(frac.dtype)).to(torch.long)
    return torch.clamp(idx, torch.zeros_like(counts), counts - 1)


def _flatten_bin_indices(bin_idx: Tensor, counts: Tensor) -> Tensor:
    """
    Collapse per-axis bin indices to flat C-order bin ids.

    Parameters
    ----------
    bin_idx
        Integer bin indices shaped ``(..., 3)``.
    counts
        Bins per axis shaped ``(3,)``.

    Returns
    -------
    Tensor
        Flat bin ids shaped ``bin_idx.shape[:-1]``.
    """
    return (
        bin_idx[..., 0] * counts[1] * counts[2]
        + bin_idx[..., 1] * counts[2]
        + bin_idx[..., 2]
    )


def _ring_offsets(ring: int, device) -> Tensor:
    """
    Build the bin-index offsets making up a ring-``ring`` neighbourhood.

    Offsets are the same along every axis. Periodicity is applied later, when
    the offsets are added to a bin index, since an open axis must drop
    out-of-range neighbours rather than wrap them.

    Parameters
    ----------
    ring
        Number of bin layers around the central bin.
    device
        Torch device for the returned tensor.

    Returns
    -------
    Tensor
        Offsets shaped ``((2 * ring + 1) ** 3, 3)``.
    """
    span = range(-ring, ring + 1)
    offsets = list(product(span, span, span))
    return torch.as_tensor(offsets, device=device, dtype=torch.long)


def query_min_surface_distance(
    points: Tensor,
    bins: AtomBins,
    ring: int = 1,
) -> tuple[Tensor, Tensor]:
    """
    Find the nearest atomic-surface distance for each point via the cell list.

    Parameters
    ----------
    points
        Cartesian query coordinates shaped ``(n_points, 3)``.
    bins
        Bin table from :func:`build_atom_bins`.
    ring
        Number of bin layers to search around each point's own bin.

    Returns
    -------
    tuple[Tensor, Tensor]
        Nearest surface distance shaped ``(n_points,)``, and a boolean mask that
        is ``True`` where the value is provably the true minimum. Unresolved
        entries hold a valid upper bound, never a wrong answer presented as
        final.
    """
    mic = bins.mic
    device = points.device
    dtype = points.dtype
    n_points = points.shape[0]

    if n_points == 0:
        return (
            torch.empty(0, device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=torch.bool),
        )

    offsets = _ring_offsets(ring, device)
    bin_idx = _bin_indices(points, mic, bins.counts)

    neighbour = bin_idx.unsqueeze(1) + offsets.unsqueeze(0)  # (P, n_off, 3)
    # Periodic axes wrap. Open axes drop offsets that fall outside the cell.
    wrapped = neighbour % bins.counts
    in_range = (neighbour >= 0) & (neighbour < bins.counts)
    keep_axis = in_range | mic.pbc
    offset_ok = keep_axis.all(dim=-1)  # (P, n_off)

    flat = _flatten_bin_indices(wrapped, bins.counts)
    candidates = bins.table[flat]  # (P, n_off, max_occupancy)
    candidate_ok = (candidates >= 0) & offset_ok.unsqueeze(-1)

    candidates = candidates.reshape(n_points, -1)
    candidate_ok = candidate_ok.reshape(n_points, -1)

    best = _reduce_surface_distance(points, candidates, candidate_ok, bins)

    coverage = bins.coverage_radius(ring)
    if np.isinf(coverage):
        resolved = torch.ones(n_points, device=device, dtype=torch.bool)
    else:
        # A missed atom sits at distance >= coverage, so it can only beat the
        # current best if best + max_radius exceeds that.
        resolved = (best + bins.max_radius) <= coverage
    return best, resolved


def _reduce_surface_distance(
    points: Tensor,
    candidates: Tensor,
    candidate_ok: Tensor,
    bins: AtomBins,
) -> Tensor:
    """
    Minimise ``|g - r_i| - R_i`` over a padded candidate list per point.

    Parameters
    ----------
    points
        Cartesian query coordinates shaped ``(n_points, 3)``.
    candidates
        Candidate atom indices shaped ``(n_points, n_candidates)``.
    candidate_ok
        Boolean mask marking real (non-padding) candidates.
    bins
        Bin table supplying atom positions, radii, and the minimum-image setup.

    Returns
    -------
    Tensor
        Nearest surface distance shaped ``(n_points,)``.
    """
    device, dtype = points.device, points.dtype
    n_points, n_candidates = candidates.shape
    best = torch.full((n_points,), float("inf"), device=device, dtype=dtype)
    if n_candidates == 0:
        return best

    n_images = bins.mic.shifts.shape[0]
    chunk = max(1, MAX_INTERMEDIATE // max(1, n_points * n_images * 3))
    safe_idx = candidates.clamp(min=0)

    for start in range(0, n_candidates, chunk):
        stop = min(start + chunk, n_candidates)
        idx = safe_idx[:, start:stop]
        delta = points.unsqueeze(1) - bins.positions[idx]
        surface = mic_distance(delta, bins.mic) - bins.radii[idx]
        surface = torch.where(
            candidate_ok[:, start:stop], surface, torch.full_like(surface, float("inf"))
        )
        best = torch.minimum(best, surface.amin(dim=1))
    return best


def brute_min_surface_distance(
    points: Tensor,
    positions: Tensor,
    radii: Tensor,
    mic: MinimumImage,
    atom_chunk: int | None = None,
) -> Tensor:
    """
    Reference kernel: minimise over every atom, with no spatial pruning.

    Parameters
    ----------
    points
        Cartesian query coordinates shaped ``(n_points, 3)``.
    positions
        Cartesian atom positions shaped ``(n_atoms, 3)``.
    radii
        Per-atom radii shaped ``(n_atoms,)``.
    mic
        Minimum-image setup from :func:`poretorch.cell.minimum_image_setup`.
    atom_chunk
        Atoms per chunk. Defaults to a size that bounds the intermediate
        tensor.

    Returns
    -------
    Tensor
        Nearest surface distance shaped ``(n_points,)``.
    """
    device, dtype = points.device, points.dtype
    n_points = points.shape[0]
    n_atoms = positions.shape[0]
    best = torch.full((n_points,), float("inf"), device=device, dtype=dtype)
    if n_points == 0 or n_atoms == 0:
        return best

    n_images = mic.shifts.shape[0]
    if atom_chunk is None:
        atom_chunk = max(1, MAX_INTERMEDIATE // max(1, n_points * n_images * 3))

    for start in range(0, n_atoms, atom_chunk):
        stop = min(start + atom_chunk, n_atoms)
        delta = points.unsqueeze(1) - positions[start:stop]
        surface = mic_distance(delta, mic) - radii[start:stop]
        best = torch.minimum(best, surface.amin(dim=1))
    return best
