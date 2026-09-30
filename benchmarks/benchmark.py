"""Benchmark the distance field and the three pore-size definitions.

Structures are random points in cubic cells sized to hold a fixed number
density, so the atom count is the only thing that varies and the scaling is
clean. Random points are not a real amorphous carbon, but they load the same
kernels in the same proportions, and the timings track real structures closely
because the cost depends on the grid and atom counts rather than on the
structure itself.

Run with ``python -m benchmarks.benchmark`` from the repository root.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from poretorch import distance_field, pore_size_distribution

try:
    from plotting import plot_methods, plot_scaling
except ImportError:  # pragma: no cover (depends on how it is invoked)
    from .plotting import plot_methods, plot_scaling

SIZES = [500, 1000, 1728, 4000, 8000, 16000, 32000]
DENSITY = 0.05  # atoms per cubic Angstrom
GRID_SPACING = 0.4
ATOM_RADIUS = 1.7
BIN_WIDTH = 0.25
MIN_RADIUS = 0.3
METHOD_SIZES = [500, 1000, 1728, 4000, 8000]
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def make_structure(n_atoms: int, density: float = DENSITY, seed: int = 42):
    """
    Build a random structure at a fixed number density.

    Parameters
    ----------
    n_atoms
        Number of atoms.
    density
        Number density in atoms per cubic Angstrom.
    seed
        Random seed.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Positions shaped ``(n_atoms, 3)`` and the cubic cell.
    """
    box = float((n_atoms / density) ** (1.0 / 3.0))
    rng = np.random.default_rng(seed)
    return rng.random((n_atoms, 3)) * box, np.diag([box, box, box])


def _sync(device: str) -> None:
    """
    Wait for queued GPU work so wall-clock timings are meaningful.

    Parameters
    ----------
    device
        Device the work was submitted to.
    """
    if device == "cuda":
        torch.cuda.synchronize()


def _time(call, device: str, repeats: int) -> float:
    """
    Time a call, discarding one warm-up run and taking the best of the rest.

    Parameters
    ----------
    call
        Zero-argument callable to time.
    device
        Device in use, for synchronisation.
    repeats
        Number of timed runs.

    Returns
    -------
    float
        Best elapsed time in seconds.
    """
    call()
    _sync(device)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        _sync(device)
        timings.append(time.perf_counter() - start)
    return min(timings)


def benchmark_field(device: str, repeats: int, sizes=SIZES) -> list[dict]:
    """
    Time both distance-field backends across system sizes.

    Parameters
    ----------
    device
        ``"cuda"`` or ``"cpu"``.
    repeats
        Timed runs per point.
    sizes
        Atom counts to benchmark.

    Returns
    -------
    list[dict]
        One record per backend and size.
    """
    rows = []
    for n_atoms in sizes:
        positions, cell = make_structure(n_atoms)
        n_points = None
        for backend in ("cell_list", "brute"):
            def call(backend=backend):
                return distance_field(
                    positions,
                    cell,
                    radii=ATOM_RADIUS,
                    grid_spacing=GRID_SPACING,
                    method=backend,
                    device=device,
                )

            if backend == "brute" and n_atoms > 16000 and device == "cpu":
                print(f"  skipping brute force at {n_atoms} atoms on CPU")
                continue

            seconds = _time(call, device, repeats)
            if n_points is None:
                n_points = call().grid.n_points
            rows.append(
                {
                    "benchmark": "field",
                    "device": device,
                    "backend": backend,
                    "method": "",
                    "n_atoms": n_atoms,
                    "n_grid_points": n_points,
                    "seconds": seconds,
                }
            )
            print(f"  {n_atoms:6d} atoms  {backend:10s}  {seconds:8.3f} s")
    return rows


def benchmark_methods(device: str, repeats: int, sizes=METHOD_SIZES) -> list[dict]:
    """
    Time each pore-size definition on a precomputed field.

    Parameters
    ----------
    device
        ``"cuda"`` or ``"cpu"``.
    repeats
        Timed runs per point.
    sizes
        Atom counts to benchmark.

    Returns
    -------
    list[dict]
        One record per method and size.
    """
    rows = []
    for n_atoms in sizes:
        positions, cell = make_structure(n_atoms)
        field = distance_field(
            positions,
            cell,
            radii=ATOM_RADIUS,
            grid_spacing=GRID_SPACING,
            method="cell_list",
            device=device,
        )
        for method in ("all_void", "covering", "packed"):
            def call(method=method):
                return pore_size_distribution(
                    field,
                    method=method,
                    bin_width=BIN_WIDTH,
                    min_radius=MIN_RADIUS,
                )

            seconds = _time(call, device, repeats)
            rows.append(
                {
                    "benchmark": "method",
                    "device": device,
                    "backend": "cell_list",
                    "method": method,
                    "n_atoms": n_atoms,
                    "n_grid_points": field.grid.n_points,
                    "seconds": seconds,
                }
            )
            print(f"  {n_atoms:6d} atoms  {method:10s}  {seconds:8.3f} s")
    return rows


def read_csv(path: Path) -> list[dict]:
    """
    Load previously written benchmark records.

    Lets the figures be redrawn without paying for the timings again.

    Parameters
    ----------
    path
        CSV written by :func:`write_csv`.

    Returns
    -------
    list[dict]
        Benchmark records, with numeric fields converted back from text.
    """
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["n_atoms"] = int(row["n_atoms"])
        row["n_grid_points"] = int(row["n_grid_points"])
        row["seconds"] = float(row["seconds"])
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    """
    Write benchmark records to CSV.

    Parameters
    ----------
    rows
        Benchmark records.
    path
        Destination file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "benchmark",
        "device",
        "backend",
        "method",
        "n_atoms",
        "n_grid_points",
        "seconds",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the benchmarks and write results and figures under ``results/``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-atoms", type=int, default=max(SIZES))
    parser.add_argument(
        "--replot",
        action="store_true",
        help="redraw the figures from the existing results.csv without timing",
    )
    args = parser.parse_args()

    if args.replot:
        rows = read_csv(RESULTS_DIR / "results.csv")
        plot_scaling(
            [r for r in rows if r["benchmark"] == "field"], str(RESULTS_DIR / "results")
        )
        plot_methods(
            [r for r in rows if r["benchmark"] == "method"], str(RESULTS_DIR / "methods")
        )
        print(f"Redrew figures from {RESULTS_DIR / 'results.csv'}")
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    sizes = [n for n in SIZES if n <= args.max_atoms]
    method_sizes = [n for n in METHOD_SIZES if n <= args.max_atoms]

    print(f"Device: {args.device}")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"\nDistance field (grid spacing {GRID_SPACING} A, density {DENSITY} at/A^3)")
    rows = benchmark_field(args.device, args.repeats, sizes)
    print("\nPore-size definitions")
    rows += benchmark_methods(args.device, args.repeats, method_sizes)

    write_csv(rows, RESULTS_DIR / "results.csv")
    plot_scaling(
        [r for r in rows if r["benchmark"] == "field"], str(RESULTS_DIR / "results")
    )
    plot_methods(
        [r for r in rows if r["benchmark"] == "method"], str(RESULTS_DIR / "methods")
    )
    print(f"\nWrote {RESULTS_DIR / 'results.csv'} and figures alongside it")


if __name__ == "__main__":
    main()
