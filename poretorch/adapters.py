"""User-facing entry points: structures in, pore-size distributions out.

An ASE ``Atoms`` object already carries positions, cell, periodicity, chemical
symbols and masses, which is everything a pore analysis needs, so it is the
native input here. A path is read with :func:`ase.io.read`, which covers LAMMPS
data and dump files, extended XYZ, CIF and the rest.

Passing several frames averages the distribution over them and returns a
standard error alongside, since a single amorphous configuration is one sample of
an ensemble and its distribution is noisy on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .field import DEFAULT_GRID_SPACING, DistanceField, distance_field
from .labels import PoreLabels, label_pores
from .psd import DEFAULT_BIN_WIDTH, PSD, pore_size_distribution

try:
    from ase import Atoms
except ImportError:  # pragma: no cover (ASE is a hard dependency in practice)
    Atoms = None


@dataclass
class PoreAnalysis:
    """
    Everything one structure yields: the field, its pores, and the distribution.

    Attributes
    ----------
    field
        The clearance field and void mask.
    labels
        The void partitioned into pores.
    psd
        The pore-size distribution.
    """

    field: DistanceField
    labels: PoreLabels
    psd: PSD

    def summary(self) -> dict:
        """
        Condense the analysis to a flat dictionary of scalars.

        Returns
        -------
        dict
            Porosity, void volume, pore counts by kind, and the distribution
            statistics.
        """
        out = {
            "grid_shape": tuple(self.field.grid.shape),
            "grid_spacing_A": [float(s) for s in self.field.grid.spacing],
            "probe_radius_A": self.field.probe_radius,
            "porosity": self.field.porosity,
            "void_volume_A3": self.field.void_volume,
            "cell_volume_A3": self.field.cell_volume,
            "specific_void_volume_cc_per_g": self.field.specific_void_volume,
            "method": self.psd.method,
        }
        out.update(self.labels.summary())
        out.update({f"psd_{k}": v for k, v in self.psd.stats.items()})
        return out


def read_structures(obj, index=None, read_kwargs=None):
    """
    Normalise any accepted input into a list of ASE ``Atoms`` frames.

    Parameters
    ----------
    obj
        An ASE ``Atoms``, an iterable of ``Atoms``, or a path to any file ASE
        can read.
    index
        Optional index or slice string forwarded to :func:`ase.io.read` when
        ``obj`` is a path, or applied to an iterable of frames.
    read_kwargs
        Optional arguments forwarded to :func:`ase.io.read`, for example
        ``{"format": "lammps-data", "atom_style": "atomic"}``. Worth supplying
        for LAMMPS data files, whose ``.data`` extension ASE does not recognise
        on its own.

    Returns
    -------
    list
        One or more ``Atoms`` frames.
    """
    if Atoms is None:
        raise ImportError("ASE must be installed to read structures")

    if isinstance(obj, (str, Path)):
        from ase.io import read

        path = Path(obj)
        if not path.exists():
            raise FileNotFoundError(f"no such structure file: {path}")

        result = read(
            str(path), index=":" if index is None else index, **(read_kwargs or {})
        )
        frames = [result] if hasattr(result, "get_positions") else list(result)
        if not frames:
            # ASE guesses the format from the extension and quietly returns
            # nothing when it guesses wrong, which looks like an empty file.
            hint = ""
            if path.suffix == ".data":
                hint = (
                    " A LAMMPS data file needs an explicit format, for example "
                    'read_kwargs={"format": "lammps-data", "atom_style": "atomic"}.'
                )
            raise ValueError(
                f"ASE read no frames from {path}. Its guessed format may be "
                f"wrong, so pass the format explicitly via read_kwargs.{hint}"
            )
        return frames

    if hasattr(obj, "get_positions"):
        return [obj]

    frames = list(obj)
    if not frames:
        raise ValueError("no frames to analyse")
    if not all(hasattr(frame, "get_positions") for frame in frames):
        raise TypeError(
            "expected an ASE Atoms, an iterable of Atoms, or a readable path"
        )
    if index is not None:
        frames = frames[index] if isinstance(index, slice) else [frames[index]]
    return frames


def field_from_atoms(
    atoms,
    radii="uff",
    grid_spacing: float = DEFAULT_GRID_SPACING,
    probe_radius=0.0,
    method: str = "cell_list",
    device: str | torch.device | None = "auto",
    torch_dtype: torch.dtype = torch.float32,
    **kwargs,
) -> DistanceField:
    """
    Compute the clearance field for one ASE ``Atoms`` object.

    Parameters
    ----------
    atoms
        ASE ``Atoms`` with a defined cell.
    radii
        Atomic radii. ``"uff"`` (default) reads half the UFF sigma from the
        chemical symbols, ``"vdw"`` uses ASE van der Waals radii, or pass a
        scalar, per-atom sequence, or ``{symbol: radius}`` mapping.
    grid_spacing
        Target grid spacing in Angstrom.
    probe_radius
        Accessibility threshold in Angstrom, or an adsorptive name like
        ``"N2"``.
    method
        ``"cell_list"`` or ``"brute"``.
    device
        Torch device used for computation. ``"auto"`` selects CUDA when
        available and otherwise uses the CPU.
    torch_dtype
        Torch dtype used for tensors.
    **kwargs
        Further arguments forwarded to
        :func:`poretorch.field.distance_field`.

    Returns
    -------
    DistanceField
        Clearance grid and void mask for this frame.
    """
    cell = np.array(atoms.get_cell().array, dtype=np.float64)
    if not np.any(cell):
        raise ValueError(
            "the Atoms object has no cell. Pore analysis needs a periodic cell"
        )
    return distance_field(
        atoms.get_positions(),
        cell,
        pbc=tuple(bool(p) for p in atoms.get_pbc()),
        radii=radii,
        symbols=atoms.get_chemical_symbols(),
        masses=atoms.get_masses(),
        grid_spacing=grid_spacing,
        probe_radius=probe_radius,
        method=method,
        device=device,
        torch_dtype=torch_dtype,
        **kwargs,
    )


def analyse(
    obj,
    method: str = "all_void",
    radii="uff",
    grid_spacing: float = DEFAULT_GRID_SPACING,
    probe_radius=0.0,
    bin_width: float = DEFAULT_BIN_WIDTH,
    max_diameter: float | None = None,
    min_radius: float = 0.0,
    connectivity: int = 1,
    min_voxels: int = 1,
    field_method: str = "cell_list",
    device: str | torch.device | None = "auto",
    torch_dtype: torch.dtype = torch.float32,
    index=None,
    read_kwargs=None,
    progress: bool = False,
    **kwargs,
) -> PoreAnalysis:
    """
    Run the full analysis on a single structure: field, pores, distribution.

    Parameters
    ----------
    obj
        An ASE ``Atoms``, or a path to a structure file. An iterable of frames
        is accepted but only its first frame is analysed. Use :func:`psd` to
        average a distribution over frames.
    method
        Pore-size definition: ``"all_void"`` or ``"covering"`` for overlapping
        spheres, one per void point, or ``"packed"`` for non-overlapping ones.
    radii
        Atomic radii convention or explicit values.
    grid_spacing
        Target grid spacing in Angstrom.
    probe_radius
        Accessibility threshold in Angstrom, or an adsorptive name.
    bin_width
        Diameter bin width in Angstrom.
    max_diameter
        Upper edge of the last bin in Angstrom.
    min_radius
        Ignore pore radii below this, in Angstrom.
    connectivity
        Voxel connectivity for pore labelling: 1, 2 or 3.
    min_voxels
        Discard pores smaller than this many voxels.
    field_method
        Distance-field backend: ``"cell_list"`` or ``"brute"``.
    device
        Torch device used for computation. ``"auto"`` selects CUDA when
        available and otherwise uses the CPU.
    torch_dtype
        Torch dtype used for tensors.
    index
        Optional frame selector when ``obj`` is a path or an iterable.
    read_kwargs
        Optional arguments forwarded to :func:`ase.io.read` when ``obj`` is a
        path, for example ``{"format": "lammps-data", "atom_style": "atomic"}``.
    progress
        Whether to show progress bars.
    **kwargs
        Further arguments forwarded to
        :func:`poretorch.psd.pore_size_distribution`.

    Returns
    -------
    PoreAnalysis
        Field, pore labelling and distribution for the structure.
    """
    frames = read_structures(obj, index=index, read_kwargs=read_kwargs)
    atoms = frames[0]

    field = field_from_atoms(
        atoms,
        radii=radii,
        grid_spacing=grid_spacing,
        probe_radius=probe_radius,
        method=field_method,
        device=device,
        torch_dtype=torch_dtype,
        progress=progress,
    )
    labels = label_pores(
        field.void_mask,
        field.clearance,
        voxel_volume=field.grid.voxel_volume,
        pbc=field.mic.pbc,
        connectivity=connectivity,
        min_voxels=min_voxels,
    )
    distribution = pore_size_distribution(
        field,
        method=method,
        labels=labels,
        bin_width=bin_width,
        max_diameter=max_diameter,
        min_radius=min_radius,
        progress=progress,
        **kwargs,
    )
    return PoreAnalysis(field=field, labels=labels, psd=distribution)


def psd(
    obj,
    method: str = "all_void",
    radii="uff",
    grid_spacing: float = DEFAULT_GRID_SPACING,
    probe_radius=0.0,
    bin_width: float = DEFAULT_BIN_WIDTH,
    max_diameter: float | None = None,
    min_radius: float = 0.0,
    connectivity: int = 1,
    min_voxels: int = 1,
    field_method: str = "cell_list",
    device: str | torch.device | None = "auto",
    torch_dtype: torch.dtype = torch.float32,
    index=None,
    read_kwargs=None,
    outdir=None,
    output: str | None = None,
    progress: bool = False,
    **kwargs,
) -> PSD:
    """
    Compute a pore-size distribution, averaging over frames when given several.

    This is the one function most uses need.

    Parameters
    ----------
    obj
        An ASE ``Atoms``, an iterable of ``Atoms``, or a path to a structure or
        trajectory file.
    method
        Pore-size definition. ``"all_void"`` sizes each void point by the
        largest empty sphere centred on it, and ``"covering"`` by the largest
        empty sphere anywhere that contains it, as PoreBlazer defines it. Both
        use overlapping spheres, one per void point. ``"packed"`` instead fills
        the void with non-overlapping inscribed spheres and reports those.
    radii
        Atomic radii convention or explicit values.
    grid_spacing
        Target grid spacing in Angstrom. The main convergence parameter.
    probe_radius
        Accessibility threshold in Angstrom, or an adsorptive name such as
        ``"N2"``. Zero gives the bare geometric void.
    bin_width
        Diameter bin width in Angstrom.
    max_diameter
        Upper edge of the last bin in Angstrom.
    min_radius
        Ignore pore radii below this, in Angstrom.
    connectivity
        Voxel connectivity for pore labelling: 1, 2 or 3.
    min_voxels
        Discard pores smaller than this many voxels.
    field_method
        Distance-field backend: ``"cell_list"`` or ``"brute"``.
    device
        Torch device used for computation. ``"auto"`` selects CUDA when
        available and otherwise uses the CPU.
    torch_dtype
        Torch dtype used for tensors.
    index
        Optional frame selector.
    read_kwargs
        Optional arguments forwarded to :func:`ase.io.read` when ``obj`` is a
        path, for example ``{"format": "lammps-data", "atom_style": "atomic"}``.
    outdir
        Optional directory to write ``psd.npz`` into.
    output
        Optional explicit output path. ``.npz``, ``.csv`` or ``.json``.
    progress
        Whether to show progress bars.
    **kwargs
        Further arguments forwarded to
        :func:`poretorch.psd.pore_size_distribution`.

    Returns
    -------
    PSD
        The distribution. For several frames, bin volumes are the frame mean and
        ``se_volume_A3`` holds the standard error.
    """
    from .io import maybe_save

    frames = read_structures(obj, index=index, read_kwargs=read_kwargs)

    results = []
    for frame in frames:
        analysis = analyse(
            frame,
            method=method,
            radii=radii,
            grid_spacing=grid_spacing,
            probe_radius=probe_radius,
            bin_width=bin_width,
            max_diameter=max_diameter,
            min_radius=min_radius,
            connectivity=connectivity,
            min_voxels=min_voxels,
            field_method=field_method,
            device=device,
            torch_dtype=torch_dtype,
            progress=progress,
            **kwargs,
        )
        results.append(analysis.psd)

    combined = results[0] if len(results) == 1 else average_psds(results)
    maybe_save(outdir, output, combined)
    return combined


def average_psds(distributions) -> PSD:
    """
    Average several distributions bin by bin, with a standard error.

    Frames may produce histograms of different lengths, since each stops at the
    largest pore it actually contains. Shorter ones are zero-extended rather
    than truncated, so no volume is silently discarded.

    Parameters
    ----------
    distributions
        Iterable of :class:`poretorch.psd.PSD` sharing a bin width and method.

    Returns
    -------
    PSD
        Frame-mean distribution with ``se_volume_A3`` populated and
        ``n_frames`` set.
    """
    distributions = list(distributions)
    if not distributions:
        raise ValueError("no distributions to average")
    if len(distributions) == 1:
        return distributions[0]

    widths = {round(d.bin_width_A, 12) for d in distributions}
    if len(widths) != 1:
        raise ValueError(f"cannot average distributions with bin widths {widths}")
    methods = {d.method for d in distributions}
    if len(methods) != 1:
        raise ValueError(f"cannot average distributions from methods {methods}")

    bin_width = distributions[0].bin_width_A
    n_bins = max(d.volume_A3.size for d in distributions)
    stack = np.zeros((len(distributions), n_bins), dtype=np.float64)
    for row, dist in enumerate(distributions):
        stack[row, : dist.volume_A3.size] = dist.volume_A3

    mean = stack.mean(axis=0)
    se = stack.std(axis=0, ddof=1) / np.sqrt(len(distributions))
    centers = (np.arange(n_bins, dtype=np.float64) + 0.5) * bin_width

    masses = [d.mass_g for d in distributions if d.mass_g]
    mass_g = float(np.mean(masses)) if masses else None

    stats = _average_stats([d.stats for d in distributions])
    return PSD(
        diameters_A=centers,
        volume_A3=mean,
        bin_width_A=bin_width,
        method=distributions[0].method,
        total_void_volume_A3=float(
            np.mean([d.total_void_volume_A3 for d in distributions])
        ),
        mass_g=mass_g,
        stats=stats,
        se_volume_A3=se,
        n_frames=len(distributions),
    )


def _average_stats(stats_list) -> dict:
    """
    Average the numeric entries of several statistics dictionaries.

    Parameters
    ----------
    stats_list
        List of per-frame statistics dictionaries.

    Returns
    -------
    dict
        Mean of each numeric key, with ``<key>_se`` giving its standard error.
        Non-numeric entries are taken from the first frame.
    """
    keys = set().union(*(set(s) for s in stats_list))
    out: dict = {}
    for key in sorted(keys):
        values = [s[key] for s in stats_list if key in s]
        if all(isinstance(v, (int, float, np.floating, np.integer)) for v in values):
            array = np.asarray(values, dtype=np.float64)
            out[key] = float(array.mean())
            if array.size > 1:
                out[f"{key}_se"] = float(array.std(ddof=1) / np.sqrt(array.size))
        else:
            out[key] = values[0]
    return out
