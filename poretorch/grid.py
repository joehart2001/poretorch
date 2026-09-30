"""The real-space grid that every pore-size definition is evaluated on.

Points sit at voxel centres in fractional coordinates, ``((i + 0.5) / nx, ...)``,
so the voxels tile the cell exactly and each one has volume
``V_cell / (nx * ny * nz)`` whatever the cell shape. Counts per axis come from
the perpendicular widths rather than the lattice-vector lengths, which is what
keeps the resolution uniform in a skewed triclinic cell.
"""

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from torch import Tensor

from .cell import cell_volume, perpendicular_widths


@dataclass(frozen=True)
class Grid:
    """
    A regular grid of voxel centres filling a periodic cell.

    Attributes
    ----------
    shape
        Number of voxels along each cell axis, ``(nx, ny, nz)``.
    cell
        Cell tensor shaped ``(3, 3)`` with lattice vectors as rows.
    requested_spacing
        The spacing asked for, in Angstrom. The realised spacing differs
        slightly because the counts must be integers.
    """

    shape: tuple[int, int, int]
    cell: Tensor
    requested_spacing: float

    @classmethod
    def from_cell(cls, cell: Tensor, spacing: float) -> "Grid":
        """
        Build a grid covering a cell at approximately the requested spacing.

        Parameters
        ----------
        cell
            Cell tensor shaped ``(3, 3)``.
        spacing
            Target spacing between neighbouring grid points, in Angstrom.

        Returns
        -------
        Grid
            Grid whose per-axis counts give a spacing no coarser than
            ``spacing`` along every perpendicular direction.
        """
        if spacing <= 0.0:
            raise ValueError(f"spacing must be > 0, got {spacing}")
        widths = perpendicular_widths(cell).detach().cpu().numpy()
        # Round up so the realised spacing never exceeds the request.
        counts = np.maximum(1, np.ceil(widths / spacing).astype(int))
        return cls(
            shape=(int(counts[0]), int(counts[1]), int(counts[2])),
            cell=cell,
            requested_spacing=float(spacing),
        )

    @property
    def n_points(self) -> int:
        """Total number of grid points."""
        nx, ny, nz = self.shape
        return nx * ny * nz

    @property
    def spacing(self) -> np.ndarray:
        """Realised spacing along each perpendicular direction, in Angstrom."""
        widths = perpendicular_widths(self.cell).detach().cpu().numpy()
        return widths / np.asarray(self.shape, dtype=float)

    @property
    def voxel_volume(self) -> float:
        """Volume of a single voxel in cubic Angstrom."""
        return cell_volume(self.cell) / self.n_points

    def fractional(self, flat_index: Tensor) -> Tensor:
        """
        Fractional coordinates of grid points given their flat C-order indices.

        Parameters
        ----------
        flat_index
            Integer tensor of flat indices into the ``(nx, ny, nz)`` grid.

        Returns
        -------
        Tensor
            Fractional coordinates shaped ``(len(flat_index), 3)``.
        """
        nx, ny, nz = self.shape
        i = torch.div(flat_index, ny * nz, rounding_mode="floor")
        j = torch.div(flat_index, nz, rounding_mode="floor") % ny
        k = flat_index % nz
        ijk = torch.stack([i, j, k], dim=-1).to(self.cell.dtype)
        counts = torch.tensor(self.shape, device=self.cell.device, dtype=self.cell.dtype)
        return (ijk + 0.5) / counts

    def cartesian(self, flat_index: Tensor) -> Tensor:
        """
        Cartesian coordinates of grid points given their flat C-order indices.

        Parameters
        ----------
        flat_index
            Integer tensor of flat indices into the ``(nx, ny, nz)`` grid.

        Returns
        -------
        Tensor
            Cartesian coordinates shaped ``(len(flat_index), 3)`` in Angstrom.
        """
        return self.fractional(flat_index) @ self.cell

    def coordinates(self) -> Tensor:
        """
        Cartesian coordinates of every grid point, in flat C-order.

        Returns
        -------
        Tensor
            Coordinates shaped ``(n_points, 3)``. Materialises the whole
            grid, so prefer :meth:`batches` for large cells.
        """
        idx = torch.arange(self.n_points, device=self.cell.device)
        return self.cartesian(idx)

    def batches(self, batch_size: int) -> Iterator[tuple[int, int, Tensor]]:
        """
        Iterate over the grid in batches of Cartesian coordinates.

        Parameters
        ----------
        batch_size
            Number of grid points per batch.

        Yields
        ------
        tuple[int, int, Tensor]
            Start index, stop index, and coordinates shaped ``(stop - start, 3)``.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        total = self.n_points
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            idx = torch.arange(start, stop, device=self.cell.device)
            yield start, stop, self.cartesian(idx)

    def reshape(self, flat: Tensor | np.ndarray):
        """
        Reshape a flat per-point array back onto the ``(nx, ny, nz)`` grid.

        Parameters
        ----------
        flat
            Array or tensor with ``n_points`` entries in flat C-order.

        Returns
        -------
        Tensor or np.ndarray
            The same data viewed as ``(nx, ny, nz)``.
        """
        return flat.reshape(self.shape)
