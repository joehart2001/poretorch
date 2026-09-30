"""The largest-empty-sphere distance field, and the void it defines.

For every grid point this computes the distance to the nearest atomic surface,
which is the radius of the largest empty sphere that can be centred there. A
point belongs to the void when that clearance is at least the probe radius.

Note the probe convention: the probe decides *where* the void is, and does not
shrink the reported clearance. A point is accessible when a probe of radius
``probe_radius`` fits with its centre there, but the pore size assigned to the
point is still the full distance to the atomic surface. This is the convention
PoreBlazer and Zeo++ use for adsorptive-accessible pore sizes, and it is what
makes ``probe_radius="N2"`` results comparable with theirs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from .cell import (
    MinimumImage,
    cell_tensor,
    cell_volume,
    minimum_image_setup,
    pbc_tensor,
)
from .grid import Grid
from .neighbor import (
    brute_min_surface_distance,
    build_atom_bins,
    query_min_surface_distance,
)
from .radii import resolve_probe, resolve_radii

DEFAULT_GRID_SPACING = 0.3
DEFAULT_BATCH_SIZE = 65_536
DEFAULT_MAX_RING = 6


@dataclass
class DistanceField:
    """
    Clearance to the nearest atomic surface on a grid, and the resulting void.

    Attributes
    ----------
    clearance
        Distance to the nearest atomic surface at each grid point, shaped
        ``(nx, ny, nz)``. Negative inside atoms. This is the local
        largest-empty-sphere radius.
    void_mask
        Boolean grid, ``True`` where a probe of ``probe_radius`` fits.
    grid
        The grid the field was evaluated on.
    probe_radius
        Probe radius used for the accessibility threshold, in Angstrom.
    mic
        Minimum-image setup, reused by downstream pore-size definitions.
    positions
        Cartesian atom positions shaped ``(n_atoms, 3)``.
    radii
        Per-atom radii shaped ``(n_atoms,)``.
    masses
        Per-atom masses in atomic mass units, or ``None`` when unknown. Needed
        for pore volumes reported per gram.
    """

    clearance: Tensor
    void_mask: Tensor
    grid: Grid
    probe_radius: float
    mic: MinimumImage
    positions: Tensor
    radii: Tensor
    masses: np.ndarray | None = None

    @property
    def n_void(self) -> int:
        """Number of void grid points."""
        return int(self.void_mask.sum().item())

    @property
    def void_volume(self) -> float:
        """Accessible void volume in cubic Angstrom."""
        return self.n_void * self.grid.voxel_volume

    @property
    def cell_volume(self) -> float:
        """Cell volume in cubic Angstrom."""
        return cell_volume(self.grid.cell)

    @property
    def porosity(self) -> float:
        """Accessible void volume as a fraction of the cell volume."""
        return self.void_volume / self.cell_volume

    @property
    def mass_g(self) -> float | None:
        """Total mass of the cell contents in grams, or ``None`` if unknown."""
        if self.masses is None:
            return None
        # 1 amu = 1.66053906660e-24 g
        return float(self.masses.sum()) * 1.66053906660e-24

    @property
    def specific_void_volume(self) -> float | None:
        """Accessible void volume in cc/g, or ``None`` when masses are unknown."""
        mass = self.mass_g
        if mass is None or mass <= 0.0:
            return None
        return self.void_volume * 1e-24 / mass

    def void_indices(self) -> Tensor:
        """
        Flat C-order indices of the void grid points.

        Returns
        -------
        Tensor
            Integer indices shaped ``(n_void,)``.
        """
        return torch.nonzero(self.void_mask.reshape(-1), as_tuple=True)[0]

    def void_coordinates(self) -> Tensor:
        """
        Cartesian coordinates of the void grid points.

        Returns
        -------
        Tensor
            Coordinates shaped ``(n_void, 3)`` in Angstrom.
        """
        return self.grid.cartesian(self.void_indices())

    def void_clearance(self) -> Tensor:
        """
        Clearance at each void grid point, ordered like :meth:`void_indices`.

        Returns
        -------
        Tensor
            Clearance values shaped ``(n_void,)`` in Angstrom.
        """
        return self.clearance.reshape(-1)[self.void_indices()]


@torch.no_grad()
def distance_field(
    positions,
    cell,
    pbc=(True, True, True),
    radii=1.7,
    symbols=None,
    masses=None,
    grid_spacing: float = DEFAULT_GRID_SPACING,
    probe_radius=0.0,
    method: str = "cell_list",
    bin_width: float | None = None,
    max_ring: int = DEFAULT_MAX_RING,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | torch.device | None = "auto",
    torch_dtype: torch.dtype = torch.float32,
    progress: bool = False,
) -> DistanceField:
    """
    Compute the largest-empty-sphere distance field over a periodic cell.

    Parameters
    ----------
    positions
        Array-like Cartesian coordinates shaped ``(n_atoms, 3)``.
    cell
        Cell matrix shaped ``(3, 3)`` with lattice vectors as rows (triclinic
        allowed), or three lengths for a diagonal cell.
    pbc
        Iterable of three booleans for periodicity along each cell axis.
    radii
        Atomic radii: a scalar, a per-atom sequence, a ``{symbol: radius}``
        mapping, or ``"uff"`` / ``"vdw"`` to use a built-in table (which needs
        ``symbols``). Defaults to 1.7 A, the usual carbon value.
    symbols
        Optional chemical symbols, required only for table or mapping lookups.
    masses
        Optional per-atom masses in atomic mass units, used to report pore
        volumes per gram.
    grid_spacing
        Target grid spacing in Angstrom. The dominant convergence parameter:
        0.3 A is a good default, 0.5 A is noticeably cheaper and slightly
        coarser.
    probe_radius
        Accessibility threshold in Angstrom, or an adsorptive name such as
        ``"N2"``. Zero gives the bare geometric void.
    method
        ``"cell_list"`` for the binned search, or ``"brute"`` for the reference
        all-atom reduction. Both are exact. They differ only in cost.
    bin_width
        Cell-list bin width in Angstrom. Defaults to a value derived from the
        largest atomic radius.
    max_ring
        Largest bin ring to search before falling back to the brute-force
        kernel for whichever points are still unresolved.
    batch_size
        Grid points evaluated per batch. Lower this if GPU memory is tight.
    device
        Torch device used for computation. ``"auto"`` (the default) selects
        CUDA when available and otherwise uses the CPU.
    torch_dtype
        Torch dtype used for tensors. ``torch.float64`` if you need it.
    progress
        Whether to show a progress bar over grid batches.

    Returns
    -------
    DistanceField
        Clearance grid, void mask, and the metadata downstream analyses need.
    """
    if method not in {"cell_list", "brute"}:
        raise ValueError(f'method must be "cell_list" or "brute", got {method!r}')

    device = _resolve_device(device)
    pos_t = torch.as_tensor(positions, device=device, dtype=torch_dtype)
    if pos_t.ndim != 2 or pos_t.shape[1] != 3:
        raise ValueError(f"positions must be (n_atoms,3), got {tuple(pos_t.shape)}")
    if pos_t.shape[0] == 0:
        raise ValueError("at least one atom is required")

    cell_t = cell_tensor(cell, device=device, dtype=torch_dtype)
    pbc_t = pbc_tensor(pbc, device=device)
    mic = minimum_image_setup(cell_t, pbc_t)

    radii_np = resolve_radii(radii, pos_t.shape[0], symbols=symbols)
    radii_t = torch.as_tensor(radii_np, device=device, dtype=torch_dtype)
    probe = resolve_probe(probe_radius)

    grid = Grid.from_cell(cell_t, grid_spacing)

    bins = None
    if method == "cell_list":
        if bin_width is None:
            # Wide enough that ring 1 already resolves points near the surface.
            bin_width = max(3.0, 2.0 * float(radii_np.max()))
        bins = build_atom_bins(pos_t, radii_t, mic, bin_width)

    clearance = torch.empty(grid.n_points, device=device, dtype=torch_dtype)

    batches = grid.batches(batch_size)
    if progress:
        n_batches = (grid.n_points + batch_size - 1) // batch_size
        batches = tqdm(batches, total=n_batches, desc="Clearance", unit="batch")

    for start, stop, points in batches:
        if method == "brute":
            values = brute_min_surface_distance(points, pos_t, radii_t, mic)
        else:
            values = _cell_list_clearance(points, bins, max_ring, pos_t, radii_t, mic)
        clearance[start:stop] = values

    clearance = grid.reshape(clearance)
    void_mask = clearance >= probe

    masses_np = None if masses is None else np.asarray(masses, dtype=np.float64)
    if masses_np is not None and masses_np.shape != (pos_t.shape[0],):
        raise ValueError(
            f"masses must be ({pos_t.shape[0]},), got {masses_np.shape}"
        )

    return DistanceField(
        clearance=clearance,
        void_mask=void_mask,
        grid=grid,
        probe_radius=probe,
        mic=mic,
        positions=pos_t,
        radii=radii_t,
        masses=masses_np,
    )


def _resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve the public ``"auto"`` device spelling."""
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _cell_list_clearance(
    points: Tensor,
    bins,
    max_ring: int,
    positions: Tensor,
    radii: Tensor,
    mic: MinimumImage,
) -> Tensor:
    """
    Evaluate the clearance for one batch, widening the search until it is exact.

    Points in a large pore are far from every atom and so are not resolved by a
    narrow ring. Rather than guess a global search radius, the ring is widened
    only for the points that need it, and the brute-force kernel takes over once
    a wider ring would examine more candidates than there are atoms.

    Parameters
    ----------
    points
        Cartesian query coordinates shaped ``(n_points, 3)``.
    bins
        Bin table from :func:`poretorch.neighbor.build_atom_bins`.
    max_ring
        Largest ring to attempt before falling back to brute force.
    positions
        Cartesian atom positions shaped ``(n_atoms, 3)``.
    radii
        Per-atom radii shaped ``(n_atoms,)``.
    mic
        Minimum-image setup.

    Returns
    -------
    Tensor
        Exact clearance shaped ``(n_points,)``.
    """
    values, resolved = query_min_surface_distance(points, bins, ring=1)
    n_atoms = positions.shape[0]
    max_occupancy = bins.table.shape[1]

    ring = 1
    while not bool(resolved.all().item()) and ring < max_ring:
        ring += 1
        # Once a ring would sweep more candidate slots than there are atoms,
        # the unpruned reduction is the cheaper way to finish the job.
        if (2 * ring + 1) ** 3 * max_occupancy >= n_atoms:
            break
        pending = ~resolved
        sub_values, sub_resolved = query_min_surface_distance(
            points[pending], bins, ring=ring
        )
        values[pending] = sub_values
        resolved[pending] = sub_resolved

    if not bool(resolved.all().item()):
        pending = ~resolved
        values[pending] = brute_min_surface_distance(
            points[pending], positions, radii, mic
        )
    return values
