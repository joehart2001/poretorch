"""
PoreTorch: GPU-accelerated geometric pore-size distributions with PyTorch.
"""

__version__ = "0.1.0"

from .adapters import (
    PoreAnalysis,
    analyse,
    average_psds,
    field_from_atoms,
    psd,
    read_structures,
)
from .field import DistanceField, distance_field
from .grid import Grid
from .labels import Pore, PoreLabels, label_pores
from .maxima import local_maxima
from .packing import SpherePacking, pack_spheres
from .plotting import plot_psd
from .psd import PSD, covering_sizes, pore_size_distribution
from .radii import PROBE_RADII, UFF_SIGMA, atomic_radii

__all__ = [
    "__version__",
    # Entry points
    "psd",
    "analyse",
    "plot_psd",
    # Results
    "PSD",
    "PoreAnalysis",
    "DistanceField",
    "PoreLabels",
    "Pore",
    "SpherePacking",
    "Grid",
    # Building blocks
    "distance_field",
    "field_from_atoms",
    "label_pores",
    "pore_size_distribution",
    "covering_sizes",
    "pack_spheres",
    "local_maxima",
    "average_psds",
    "read_structures",
    # Conventions
    "atomic_radii",
    "UFF_SIGMA",
    "PROBE_RADII",
]
