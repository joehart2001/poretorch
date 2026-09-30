"""The three geometric pore-size definitions, and the distributions they give.

All three read the same clearance field and differ only in what "the pore size
at this point" means. The first two work with **overlapping** spheres, the third
with **non-overlapping** ones, and that is the sharpest division between them:

``"all_void"`` (overlapping spheres)
    Every void point is sized by its own clearance: the largest empty sphere
    centred *on* the point. One sphere per void point, all free to overlap,
    since a point and its neighbour carry very nearly the same sphere. This is
    the local-pore-radius distribution of the void, in the spirit of the
    Gelb-Gubbins geometric PSD.

``"covering"`` (overlapping spheres)
    Every void point is sized by the largest empty sphere anywhere in the void
    that still *contains* the point. The candidate set is the same overlapping
    one-sphere-per-void-point family as ``"all_void"``, and a point takes the
    largest member that reaches it. A sphere centred elsewhere may be bigger and
    still cover the point, so this size is never smaller than the local
    clearance and the distribution sits at larger diameters. This is the
    definition PoreBlazer and Zeo++ use, and the one to compare against them.

``"packed"`` (non-overlapping spheres)
    The void is filled greedily with inscribed spheres, largest first, accepting
    one only if it clears every sphere already accepted. The accepted set is
    disjoint, so the distribution partitions the void instead of averaging it
    locally, at the cost of the interstitial gaps a packing must leave.

Overlap decides which size a point is labelled with, not how much volume it
contributes, so no method double-counts void volume. The two overlapping methods
weight each void *voxel* by its own voxel volume, counted exactly once, so they
integrate to the accessible void volume however much their spheres overlap.
``"packed"`` weights each *sphere* by its sphere volume, which is sound only
because the spheres are disjoint, and integrates to the packed volume, smaller
than the void volume by those gaps.
"""

from dataclasses import dataclass, field as dataclass_field

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from .cell import mic_distance
from .field import DistanceField
from .maxima import local_maxima
from .packing import SpherePacking, pack_spheres

METHODS = ("all_void", "covering", "packed")

# Aliases kept because these names are what the underlying definitions are
# usually called in the literature and in other codes.
METHOD_ALIASES = {
    "local": "all_void",
    "gelb_gubbins": "all_void",
    "poreblazer": "covering",
    "zeo": "covering",
    "sphere_packing": "packed",
}

DEFAULT_BIN_WIDTH = 0.25
MAX_INTERMEDIATE = 32_000_000

AMU_TO_GRAM = 1.66053906660e-24
A3_TO_CC = 1e-24


@dataclass
class PSD:
    """
    A pore-size distribution over diameter bins.

    The stored quantity is the void volume in each bin, in cubic Angstrom, which
    is convention-free. The properties convert that to the usual reporting
    forms.

    Attributes
    ----------
    diameters_A
        Bin centres in Angstrom, shaped ``(n_bins,)``.
    volume_A3
        Void volume falling in each bin, in cubic Angstrom.
    bin_width_A
        Bin width in Angstrom.
    method
        Which pore-size definition produced this distribution.
    total_void_volume_A3
        Accessible void volume of the cell, for reference against
        ``volume_A3.sum()``.
    mass_g
        Mass of the cell contents in grams, or ``None`` if unknown.
    stats
        Summary statistics. See :func:`pore_size_distribution`.
    se_volume_A3
        Standard error on ``volume_A3`` across frames, or ``None`` for a single
        structure.
    n_frames
        Number of frames averaged into this distribution.
    """

    diameters_A: np.ndarray
    volume_A3: np.ndarray
    bin_width_A: float
    method: str
    total_void_volume_A3: float
    mass_g: float | None = None
    stats: dict = dataclass_field(default_factory=dict)
    se_volume_A3: np.ndarray | None = None
    n_frames: int = 1

    @property
    def diameters_nm(self) -> np.ndarray:
        """Bin centres in nanometres."""
        return self.diameters_A / 10.0

    @property
    def dV_dd_cc_per_nm_per_g(self) -> np.ndarray:
        """
        Differential pore volume in cc/nm/g, the form adsorption data uses.

        Raises
        ------
        ValueError
            If the structure's mass is unknown.
        """
        if self.mass_g is None or self.mass_g <= 0.0:
            raise ValueError(
                "dV/dd per gram needs the structure mass. Pass masses= to "
                "distance_field(), or use an ASE Atoms object which carries them"
            )
        return self.volume_A3 * A3_TO_CC / (self.bin_width_A / 10.0) / self.mass_g

    @property
    def dV_dd_A2(self) -> np.ndarray:
        """Differential pore volume per cell in Angstrom squared, mass-free."""
        return self.volume_A3 / self.bin_width_A

    @property
    def normalised(self) -> np.ndarray:
        """
        Distribution scaled to unit area in Angstrom to the minus one.

        Use this to compare the *shape* of distributions from different
        structures, methods or codes without their absolute scales getting in
        the way.
        """
        area = float(np.trapezoid(self.volume_A3, self.diameters_A))
        if area <= 0.0:
            return np.zeros_like(self.volume_A3)
        return self.volume_A3 / area

    @property
    def se_dV_dd_cc_per_nm_per_g(self) -> np.ndarray | None:
        """Standard error on :attr:`dV_dd_cc_per_nm_per_g`, or ``None``."""
        if self.se_volume_A3 is None:
            return None
        if self.mass_g is None or self.mass_g <= 0.0:
            raise ValueError("standard error per gram needs the structure mass")
        return (
            self.se_volume_A3 * A3_TO_CC / (self.bin_width_A / 10.0) / self.mass_g
        )

    @property
    def se_normalised(self) -> np.ndarray | None:
        """
        Standard error on :attr:`normalised`, or ``None``.

        Scaled by the same factor as the mean curve, so the band stays
        consistent with it rather than being renormalised on its own.
        """
        if self.se_volume_A3 is None:
            return None
        area = float(np.trapezoid(self.volume_A3, self.diameters_A))
        if area <= 0.0:
            return np.zeros_like(self.se_volume_A3)
        return self.se_volume_A3 / area

    @property
    def cumulative_volume_A3(self) -> np.ndarray:
        """Cumulative void volume up to each bin, in cubic Angstrom."""
        return np.cumsum(self.volume_A3)

    @property
    def cumulative_fraction(self) -> np.ndarray:
        """Cumulative fraction of the binned volume up to each bin."""
        total = float(self.volume_A3.sum())
        if total <= 0.0:
            return np.zeros_like(self.volume_A3)
        return self.cumulative_volume_A3 / total

    def to_dict(self) -> dict:
        """
        Flatten to a plain dictionary of arrays and scalars.

        Returns
        -------
        dict
            Serialisable representation, suitable for ``np.savez`` or JSON after
            converting arrays to lists.
        """
        out = {
            "method": self.method,
            "diameter_A": self.diameters_A,
            "diameter_nm": self.diameters_nm,
            "volume_A3": self.volume_A3,
            "bin_width_A": self.bin_width_A,
            "total_void_volume_A3": self.total_void_volume_A3,
            "cumulative_volume_A3": self.cumulative_volume_A3,
            "normalised": self.normalised,
            "n_frames": self.n_frames,
            **{f"stat_{k}": v for k, v in self.stats.items()},
        }
        if self.mass_g:
            out["dV_dd_cc_per_nm_per_g"] = self.dV_dd_cc_per_nm_per_g
            out["mass_g"] = self.mass_g
        if self.se_volume_A3 is not None:
            out["se_volume_A3"] = self.se_volume_A3
            if self.mass_g:
                out["se_dV_dd_cc_per_nm_per_g"] = self.se_dV_dd_cc_per_nm_per_g
        return out


@torch.no_grad()
def covering_sizes(
    field: DistanceField,
    center_mode: str = "all_void",
    center_stride: int = 1,
    min_radius: float = 0.0,
    filter_size: int = 3,
    center_batch_size: int = 4096,
    progress: bool = False,
) -> Tensor:
    """
    Radius of the largest empty sphere containing each void point.

    Formally ``r(a) = max {r_c : |a - c| <= r_c}`` over candidate centres ``c``,
    each carrying its own clearance as its radius.

    The evaluation exploits the fact that candidates can be visited in
    decreasing radius order. The first sphere found to cover a point is
    therefore already the largest one that ever will, so points drop out of the
    search as soon as they are assigned. Since the biggest spheres cover most of
    the void, the working set collapses quickly and the cost is far below the
    naive all-points-against-all-centres product.

    Parameters
    ----------
    field
        Distance field from :func:`poretorch.field.distance_field`.
    center_mode
        ``"all_void"`` treats every void point as a candidate centre, which is
        the faithful definition. ``"local_maxima"`` uses only clearance maxima:
        much cheaper, and exact wherever the maxima capture the medial axis, but
        it can miss ridge points that the grid does not resolve.
    center_stride
        Subsampling stride for ``"all_void"`` candidates. Greater than 1 trades
        accuracy for speed.
    min_radius
        Ignore candidate centres with clearance below this, in Angstrom.
    filter_size
        Maximum-filter window for ``"local_maxima"``.
    center_batch_size
        Candidate centres per batch.
    progress
        Whether to show a progress bar over candidate batches.

    Returns
    -------
    Tensor
        Covering radius per void point, ordered like
        :meth:`poretorch.field.DistanceField.void_indices`.
    """
    void_coords = field.void_coordinates()
    void_clearance = field.void_clearance()
    n_void = void_coords.shape[0]
    device, dtype = void_coords.device, void_coords.dtype

    if n_void == 0:
        return torch.empty(0, device=device, dtype=dtype)

    centers, center_radii = _candidate_centers(
        field, center_mode, center_stride, min_radius, filter_size
    )
    if centers.shape[0] == 0:
        # Nothing to cover with: each point is its own largest sphere.
        return void_clearance.clone()

    # Descending radius order is what makes the early exit valid.
    order = torch.argsort(center_radii, descending=True)
    centers = centers[order]
    center_radii = center_radii[order]

    sizes = torch.full((n_void,), float("nan"), device=device, dtype=dtype)
    pending = torch.arange(n_void, device=device)

    n_centers = centers.shape[0]
    n_batches = (n_centers + center_batch_size - 1) // center_batch_size
    batch_starts = range(0, n_centers, center_batch_size)
    if progress:
        batch_starts = tqdm(
            batch_starts, total=n_batches, desc="Covering spheres", unit="batch"
        )

    for start in batch_starts:
        if pending.numel() == 0:
            break
        stop = min(start + center_batch_size, n_centers)
        batch_centers = centers[start:stop]
        batch_radii = center_radii[start:stop]

        best = _best_covering_radius(
            void_coords[pending], batch_centers, batch_radii, field
        )
        covered = torch.isfinite(best)
        if bool(covered.any().item()):
            sizes[pending[covered]] = best[covered]
            pending = pending[~covered]

    # Points no candidate covers fall back to their own clearance. With the
    # default settings this set is empty, since each point covers itself.
    uncovered = torch.isnan(sizes)
    if bool(uncovered.any().item()):
        sizes[uncovered] = void_clearance[uncovered]
    return sizes


def _candidate_centers(
    field: DistanceField,
    center_mode: str,
    center_stride: int,
    min_radius: float,
    filter_size: int,
) -> tuple[Tensor, Tensor]:
    """
    Build the candidate sphere centres for the covering definition.

    Parameters
    ----------
    field
        Distance field to draw candidates from.
    center_mode
        ``"all_void"`` or ``"local_maxima"``.
    center_stride
        Subsampling stride for ``"all_void"``.
    min_radius
        Minimum candidate clearance in Angstrom.
    filter_size
        Maximum-filter window for ``"local_maxima"``.

    Returns
    -------
    tuple[Tensor, Tensor]
        Cartesian centres shaped ``(n_centers, 3)`` and radii ``(n_centers,)``.
    """
    if center_stride < 1:
        raise ValueError(f"center_stride must be >= 1, got {center_stride}")

    if center_mode == "all_void":
        centers = field.void_coordinates()[::center_stride]
        radii = field.void_clearance()[::center_stride]
    elif center_mode == "local_maxima":
        indices, values = local_maxima(
            field.clearance,
            field.void_mask,
            pbc=field.mic.pbc,
            min_radius=min_radius,
            filter_size=filter_size,
        )
        nx, ny, nz = field.grid.shape
        flat = indices[:, 0] * ny * nz + indices[:, 1] * nz + indices[:, 2]
        flat_t = torch.as_tensor(flat, device=field.clearance.device)
        centers = field.grid.cartesian(flat_t)
        radii = torch.as_tensor(
            values, device=centers.device, dtype=centers.dtype
        )
    else:
        raise ValueError(
            f'center_mode must be "all_void" or "local_maxima", got {center_mode!r}'
        )

    keep = torch.isfinite(radii) & (radii >= min_radius)
    return centers[keep], radii[keep]


def _best_covering_radius(
    points: Tensor,
    centers: Tensor,
    radii: Tensor,
    field: DistanceField,
) -> Tensor:
    """
    Largest radius among the given centres whose sphere covers each point.

    Parameters
    ----------
    points
        Cartesian query coordinates shaped ``(n_points, 3)``.
    centers
        Candidate centres shaped ``(n_centers, 3)``.
    radii
        Candidate radii shaped ``(n_centers,)``.
    field
        Distance field supplying the minimum-image setup.

    Returns
    -------
    Tensor
        Best covering radius per point, ``-inf`` where nothing covers it.
    """
    n_points = points.shape[0]
    n_centers = centers.shape[0]
    device, dtype = points.device, points.dtype
    best = torch.full((n_points,), float("-inf"), device=device, dtype=dtype)
    if n_points == 0 or n_centers == 0:
        return best

    n_images = field.mic.shifts.shape[0]
    chunk = max(1, MAX_INTERMEDIATE // max(1, n_points * n_images * 3))
    for start in range(0, n_centers, chunk):
        stop = min(start + chunk, n_centers)
        delta = points.unsqueeze(1) - centers[start:stop]
        distance = mic_distance(delta, field.mic)
        chunk_radii = radii[start:stop]
        covered = distance <= chunk_radii
        candidate = torch.where(
            covered, chunk_radii.expand_as(distance), torch.full_like(distance, float("-inf"))
        )
        best = torch.maximum(best, candidate.amax(dim=1))
    return best


def bin_distribution(
    sizes,
    weights,
    bin_width: float = DEFAULT_BIN_WIDTH,
    max_diameter: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Histogram pore radii into diameter bins, weighted by volume.

    Parameters
    ----------
    sizes
        Pore radii in Angstrom.
    weights
        Volume in cubic Angstrom carried by each entry.
    bin_width
        Diameter bin width in Angstrom.
    max_diameter
        Upper edge of the last bin in Angstrom. Defaults to just above the
        largest diameter present.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Bin centres in Angstrom and the volume in each bin.
    """
    if bin_width <= 0.0:
        raise ValueError(f"bin_width must be > 0, got {bin_width}")

    sizes = _as_numpy(sizes).astype(np.float64, copy=False).ravel()
    weights = np.broadcast_to(
        _as_numpy(weights).astype(np.float64, copy=False), sizes.shape
    )

    finite = np.isfinite(sizes) & (sizes > 0.0)
    sizes, weights = sizes[finite], weights[finite]

    diameters = 2.0 * sizes
    if max_diameter is None:
        upper = float(diameters.max()) if diameters.size else bin_width
    else:
        upper = float(max_diameter)
    n_bins = max(1, int(np.ceil(upper / bin_width)))
    edges = np.arange(n_bins + 1, dtype=np.float64) * bin_width

    volume, _ = np.histogram(diameters, bins=edges, weights=weights)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, volume


@torch.no_grad()
def pore_size_distribution(
    field: DistanceField,
    method: str = "all_void",
    labels=None,
    bin_width: float = DEFAULT_BIN_WIDTH,
    max_diameter: float | None = None,
    min_radius: float = 0.0,
    overlap_tolerance: float = 0.0,
    center_mode: str = "all_void",
    center_stride: int = 1,
    filter_size: int = 3,
    center_batch_size: int = 4096,
    progress: bool = False,
) -> PSD:
    """
    Build a pore-size distribution from a precomputed distance field.

    Parameters
    ----------
    field
        Distance field from :func:`poretorch.field.distance_field`.
    method
        Which pore-size definition to use. ``"all_void"`` and ``"covering"``
        both work with overlapping spheres, one per void point, and weight each
        voxel by its own volume. ``"packed"`` uses non-overlapping spheres and
        weights each by its sphere volume. See the module docstring for the
        definitions. ``"poreblazer"`` and ``"sphere_packing"`` are accepted as
        aliases for ``"covering"`` and ``"packed"``.
    labels
        Optional :class:`poretorch.labels.PoreLabels`. Used by ``"packed"`` to
        record which pore each sphere came from, and to exclude points that
        labelling discarded.
    bin_width
        Diameter bin width in Angstrom.
    max_diameter
        Upper edge of the last bin in Angstrom. Defaults to the largest diameter
        present.
    min_radius
        Ignore pore radii below this, in Angstrom. For ``"packed"`` this is also
        the smallest sphere that may be accepted, and is the main control on
        cost.
    overlap_tolerance
        Permitted sphere interpenetration for ``"packed"``, in Angstrom.
    center_mode
        Candidate centre set for ``"covering"``: ``"all_void"`` or
        ``"local_maxima"``.
    center_stride
        Candidate subsampling stride for ``"covering"``.
    filter_size
        Maximum-filter window for local-maxima searches.
    center_batch_size
        Candidate centres per batch for ``"covering"``.
    progress
        Whether to show progress bars.

    Returns
    -------
    PSD
        The distribution, with ``stats`` holding the volume-weighted mean,
        median and modal diameters, the largest sphere found, and for
        ``"packed"`` the sphere count and filling fraction.
    """
    method = METHOD_ALIASES.get(method, method)
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    voxel_volume = field.grid.voxel_volume
    extra_stats: dict = {}

    if method == "packed":
        packing = _packed_spheres(
            field,
            labels=labels,
            min_radius=min_radius,
            overlap_tolerance=overlap_tolerance,
            filter_size=filter_size,
        )
        sizes = packing.radii
        weights = (4.0 / 3.0) * np.pi * packing.radii**3
        extra_stats = {
            "n_spheres": packing.n_spheres,
            "n_candidates": packing.n_candidates,
            "packed_volume_A3": packing.volume,
            "filling_fraction": packing.filling_fraction(field.void_volume),
        }
    else:
        if method == "covering":
            radii = covering_sizes(
                field,
                center_mode=center_mode,
                center_stride=center_stride,
                min_radius=min_radius,
                filter_size=filter_size,
                center_batch_size=center_batch_size,
                progress=progress,
            )
        else:
            radii = field.void_clearance()

        sizes = _as_numpy(radii)
        if labels is not None:
            # Respect voxels that labelling discarded (min_voxels filtering).
            kept = _as_numpy(labels.labels).reshape(-1)[
                _as_numpy(field.void_indices())
            ]
            sizes = sizes[kept > 0]
        sizes = sizes[sizes >= min_radius]
        weights = np.full(sizes.shape, voxel_volume, dtype=np.float64)

    centers, volume = bin_distribution(
        sizes, weights, bin_width=bin_width, max_diameter=max_diameter
    )
    stats = _distribution_stats(centers, volume, sizes)
    stats.update(extra_stats)
    stats["probe_radius_A"] = field.probe_radius
    stats["porosity"] = field.porosity

    return PSD(
        diameters_A=centers,
        volume_A3=volume,
        bin_width_A=float(bin_width),
        method=method,
        total_void_volume_A3=field.void_volume,
        mass_g=field.mass_g,
        stats=stats,
    )


def _packed_spheres(
    field: DistanceField,
    labels,
    min_radius: float,
    overlap_tolerance: float,
    filter_size: int,
) -> SpherePacking:
    """
    Find candidate maxima and greedily pack non-overlapping spheres.

    Parameters
    ----------
    field
        Distance field to pack.
    labels
        Optional pore labelling, used to tag spheres with their pore id.
    min_radius
        Smallest sphere that may be accepted, in Angstrom.
    overlap_tolerance
        Permitted interpenetration in Angstrom.
    filter_size
        Maximum-filter window for the candidate search.

    Returns
    -------
    SpherePacking
        The accepted spheres.
    """
    indices, values = local_maxima(
        field.clearance,
        field.void_mask,
        pbc=field.mic.pbc,
        min_radius=min_radius,
        filter_size=filter_size,
    )
    if indices.shape[0] == 0:
        return SpherePacking(
            centers=np.empty((0, 3)),
            radii=np.empty(0),
            pore_ids=np.empty(0, dtype=np.int64),
        )

    nx, ny, nz = field.grid.shape
    flat = indices[:, 0] * ny * nz + indices[:, 1] * nz + indices[:, 2]
    flat_t = torch.as_tensor(flat, device=field.clearance.device)
    centers = field.grid.cartesian(flat_t)
    radii = torch.as_tensor(values, device=centers.device, dtype=centers.dtype)

    pore_ids = None
    if labels is not None:
        label_grid = _as_numpy(labels.labels)
        pore_ids = label_grid[tuple(indices.T)]
        keep = pore_ids > 0
        centers, radii = centers[keep], radii[keep]
        pore_ids = pore_ids[keep]

    return pack_spheres(
        centers,
        radii,
        field.mic,
        pore_ids=pore_ids,
        overlap_tolerance=overlap_tolerance,
    )


def _distribution_stats(centers, volume, sizes) -> dict:
    """
    Summarise a binned distribution.

    Parameters
    ----------
    centers
        Bin centres in Angstrom.
    volume
        Volume per bin in cubic Angstrom.
    sizes
        The underlying radii, for exact extremes independent of binning.

    Returns
    -------
    dict
        Volume-weighted mean, median and modal diameters plus extremes.
    """
    total = float(volume.sum())
    sizes = np.asarray(sizes, dtype=np.float64)
    stats = {
        "n_samples": int(sizes.size),
        "binned_volume_A3": total,
        "max_diameter_A": float(2.0 * sizes.max()) if sizes.size else 0.0,
        "min_diameter_A": float(2.0 * sizes.min()) if sizes.size else 0.0,
    }
    if total <= 0.0:
        stats.update(
            mean_diameter_A=0.0, median_diameter_A=0.0, mode_diameter_A=0.0
        )
        return stats

    fraction = np.cumsum(volume) / total
    # Interpolate the median across the bin it falls in rather than snapping to
    # a bin centre, so it does not quantise with bin_width.
    index = int(np.searchsorted(fraction, 0.5))
    index = min(index, centers.size - 1)
    if index == 0:
        median = float(centers[0])
    else:
        f0, f1 = fraction[index - 1], fraction[index]
        d0, d1 = centers[index - 1], centers[index]
        weight = (0.5 - f0) / (f1 - f0) if f1 > f0 else 0.0
        median = float(d0 + weight * (d1 - d0))

    stats.update(
        mean_diameter_A=float(np.average(centers, weights=volume)),
        median_diameter_A=median,
        mode_diameter_A=float(centers[int(np.argmax(volume))]),
    )
    return stats


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
