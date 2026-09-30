"""Overlay the three pore-size definitions for one structure.

Usage
-----
    python examples/compare_methods.py STRUCTURE [--probe N2] [--out psd.png]

STRUCTURE is anything ASE can read: a LAMMPS data file, an extended XYZ, a CIF.
Pass a trajectory and every frame is averaged, with an error band per bin.
"""

import argparse
from pathlib import Path

from poretorch import analyse, plot_psd, psd

METHODS = [
    ("all_void", "all-void, overlapping spheres"),
    ("covering", "covering, overlapping spheres"),
    ("packed", "packed, non-overlapping spheres"),
]


def main() -> None:
    """Read a structure, compute all three distributions, and plot them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure", type=Path)
    parser.add_argument("--probe", default="0.0", help='probe radius in A, or a name like "N2"')
    parser.add_argument("--spacing", type=float, default=0.3, help="grid spacing in A")
    parser.add_argument("--bin-width", type=float, default=0.25, help="diameter bin width in A")
    parser.add_argument("--out", type=Path, default=Path("psd.png"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atom-style", default="atomic",
                        help="LAMMPS atom_style, used for .data inputs")
    args = parser.parse_args()

    try:
        probe = float(args.probe)
    except ValueError:
        probe = args.probe

    # ASE does not recognise the .data extension on its own, so name the
    # format for LAMMPS files rather than letting the guess fail silently.
    read_kwargs = None
    if args.structure.suffix == ".data":
        read_kwargs = {"format": "lammps-data", "atom_style": args.atom_style}

    shared = dict(
        probe_radius=probe,
        grid_spacing=args.spacing,
        bin_width=args.bin_width,
        device=args.device,
        read_kwargs=read_kwargs,
    )

    # One pass to report the structure itself, then one distribution per method.
    overview = analyse(args.structure, method="all_void", **shared)
    print(f"{args.structure.name}")
    print(f"  grid            {overview.field.grid.shape}")
    print(f"  porosity        {overview.field.porosity:.4f}")
    specific = overview.field.specific_void_volume
    if specific is not None:
        print(f"  void volume     {specific:.4f} cc/g")
    summary = overview.labels.summary()
    print(f"  pores           {summary['n_pores']} "
          f"({summary['n_percolating']} percolating)")
    print(f"  by kind         {summary['counts_by_kind']}")

    curves, labels = [], []
    for method, label in METHODS:
        result = psd(args.structure, method=method, **shared)
        curves.append(result)
        labels.append(label)
        print(f"  {method:9s} median {result.stats['median_diameter_A']:6.2f} A"
              f"   mode {result.stats['mode_diameter_A']:6.2f} A"
              f"   max {result.stats['max_diameter_A']:6.2f} A")

    plot_psd(curves, labels=labels, path=args.out, xlim=(1.5, 25.0))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
