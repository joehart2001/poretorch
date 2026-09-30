"""Atomic radii and probe sizes used to define the solid/void boundary.

A geometric pore size depends entirely on where the solid surface is taken to
be, so the radius convention is part of the result and not an implementation
detail. Two conventions are provided:

``"uff"``
    Half the UFF Lennard-Jones sigma. This is what PoreBlazer and Zeo++-style
    analyses use, so it is the right choice when comparing against them.
``"vdw"``
    Bondi van der Waals radii as shipped with ASE. Larger than UFF sigma/2 for
    carbon (1.70 A against 1.716 A they are near-identical, but they diverge for
    other elements), and the more common choice for pure void-space geometry.

Probe radii for the usual adsorptives are given by :data:`PROBE_RADII`, again as
sigma/2 so that they compose with the UFF table.
"""

import numpy as np

# UFF Lennard-Jones sigma in Angstrom, as distributed with PoreBlazer
# (UFF.atoms). Radii are sigma / 2.
UFF_SIGMA = {
    "H": 2.571, "He": 2.104, "Li": 2.184, "Be": 2.446, "B": 3.638,
    "C": 3.431, "N": 3.261, "O": 3.118, "F": 2.997, "Ne": 2.889,
    "Na": 2.658, "Mg": 2.691, "Al": 4.008, "Si": 3.826, "P": 3.695,
    "S": 3.595, "Cl": 3.516, "Ar": 3.446, "K": 3.396, "Ca": 3.028,
    "Sc": 2.936, "Ti": 2.829, "V": 2.801, "Cr": 2.693, "Mn": 2.638,
    "Fe": 2.594, "Co": 2.559, "Ni": 2.525, "Cu": 3.114, "Zn": 2.462,
    "Ga": 3.905, "Ge": 3.813, "As": 3.769, "Se": 3.746, "Br": 3.732,
    "Kr": 3.689, "Rb": 3.665, "Sr": 3.244, "Y": 2.980, "Zr": 2.783,
    "Nb": 2.820, "Mo": 2.719, "Ru": 2.640, "Rh": 2.609, "Pd": 2.583,
    "Ag": 2.805, "Cd": 2.537, "In": 3.976, "Sn": 3.913, "Sb": 3.938,
    "Te": 3.982, "I": 4.009, "Xe": 3.924, "Cs": 4.024, "Ba": 3.299,
    "W": 2.734, "Re": 2.632, "Os": 2.780, "Ir": 2.530, "Pt": 2.454,
    "Au": 2.934, "Hg": 2.410, "Pb": 3.828, "Bi": 3.893,
}

# Common adsorptive probe radii in Angstrom, taken as sigma / 2 so they are
# consistent with the UFF surface convention.
PROBE_RADII = {
    "geometric": 0.0,   # no probe: the bare geometric void
    "N2": 1.657,        # sigma = 3.314 A, the PoreBlazer default for N2
    "Ar": 1.717,        # sigma = 3.434 A
    "He": 1.29,         # sigma = 2.58 A, the PoreBlazer helium probe
    "CO2": 1.65,        # sigma = 3.30 A
}


def atomic_radii(symbols, table: str = "uff") -> np.ndarray:
    """
    Look up a radius for each atom from its chemical symbol.

    Parameters
    ----------
    symbols
        Iterable of chemical symbols, e.g. ``["C", "C", "O"]``.
    table
        ``"uff"`` for half the UFF Lennard-Jones sigma (PoreBlazer convention)
        or ``"vdw"`` for Bondi van der Waals radii from ASE.

    Returns
    -------
    np.ndarray
        Radii shaped ``(n_atoms,)`` in Angstrom.
    """
    symbols = list(symbols)
    if table == "uff":
        missing = sorted({s for s in symbols if s not in UFF_SIGMA})
        if missing:
            raise KeyError(
                f"No UFF sigma for element(s) {missing}. Pass radii explicitly "
                'or use table="vdw"'
            )
        return np.array([UFF_SIGMA[s] / 2.0 for s in symbols], dtype=np.float64)

    if table == "vdw":
        from ase.data import atomic_numbers, vdw_radii

        out = []
        for s in symbols:
            radius = vdw_radii[atomic_numbers[s]]
            if not np.isfinite(radius):
                raise KeyError(
                    f"ASE has no van der Waals radius for element {s!r}. Pass "
                    'radii explicitly or use table="uff"'
                )
            out.append(float(radius))
        return np.array(out, dtype=np.float64)

    raise ValueError(f'table must be "uff" or "vdw", got {table!r}')


def resolve_radii(radii, n_atoms: int, symbols=None) -> np.ndarray:
    """
    Normalise the many accepted spellings of ``radii`` to a per-atom array.

    Parameters
    ----------
    radii
        One of: a float applied to every atom, a sequence of ``n_atoms``
        radii, a ``{symbol: radius}`` mapping, or the strings ``"uff"`` /
        ``"vdw"`` to take the corresponding table (which requires
        ``symbols``).
    n_atoms
        Number of atoms the result must cover.
    symbols
        Optional chemical symbols, required for table lookups and mappings.

    Returns
    -------
    np.ndarray
        Radii shaped ``(n_atoms,)`` in Angstrom.
    """
    if isinstance(radii, str):
        if symbols is None:
            raise ValueError(f'radii="{radii}" needs chemical symbols to look up')
        result = atomic_radii(symbols, table=radii)
        return _validate_radii(result)

    if isinstance(radii, dict):
        if symbols is None:
            raise ValueError("a radii mapping needs chemical symbols to look up")
        missing = sorted({s for s in symbols if s not in radii})
        if missing:
            raise KeyError(f"radii mapping is missing element(s) {missing}")
        result = np.array([float(radii[s]) for s in symbols], dtype=np.float64)
        return _validate_radii(result)

    arr = np.asarray(radii, dtype=np.float64)
    if arr.ndim == 0:
        return _validate_radii(np.full(n_atoms, float(arr), dtype=np.float64))
    if arr.shape != (n_atoms,):
        raise ValueError(f"radii must be scalar or ({n_atoms},), got {arr.shape}")
    return _validate_radii(arr)


def _validate_radii(radii: np.ndarray) -> np.ndarray:
    """Reject radii that cannot define a physical atomic surface."""
    if not np.all(np.isfinite(radii)) or np.any(radii < 0.0):
        raise ValueError("atomic radii must be finite and non-negative")
    return radii


def resolve_probe(probe_radius) -> float:
    """
    Normalise a probe radius given either as a number or an adsorptive name.

    Parameters
    ----------
    probe_radius
        A radius in Angstrom, or a key of :data:`PROBE_RADII` such as ``"N2"``.

    Returns
    -------
    float
        Probe radius in Angstrom.
    """
    if isinstance(probe_radius, str):
        if probe_radius not in PROBE_RADII:
            raise KeyError(
                f"Unknown probe {probe_radius!r}. Known probes: "
                f"{sorted(PROBE_RADII)}"
            )
        return PROBE_RADII[probe_radius]
    value = float(probe_radius)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"probe_radius must be finite and >= 0, got {value}")
    return value
