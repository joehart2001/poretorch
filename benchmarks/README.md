# PSD benchmarks against PoreBlazer

`compare_poreblazer.py` benchmarks all three PoreTorch definitions, including
both `all_void` weightings, against PoreBlazer 4.0. Real carbon results are in
`results/carbon/`; the original synthetic dataset is in `results/poreblazer/`.
The synthetic dataset is
64, 128 and 256 synthetic carbon atoms, not equilibrated carbon structures.
Positions are uniform random points (seed 42); overlaps are allowed. This is a
small workload comparison, not evidence of scaling on large or real materials.

## Reproduce

From the PoreTorch repository root, with its dependencies, GNU Fortran and make:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m benchmarks.compare_poreblazer \
    --poreblazer-src /path/to/PoreBlazer/src \
    --sizes 64 128 256 --repeats 3
```

Use `--devices cpu` on machines without CUDA. Start with `--sizes 64` for a
pilot. Each PoreBlazer subprocess has a 60-second timeout, configurable with
`--timeout`. The supplied PoreBlazer checkout is read only: source copies are
compiled in a temporary directory. The saved patches show every change.

The existing `benchmark.py` remains available for larger PoreTorch-only runs.
Its older minimum-of-repeats timings and float32 settings should not be mixed
with this comparison's float64 median timings.

## Amorphous-carbon structures

The real-carbon benchmark uses the supplied MACE-MP-0b3 `run_1` cooled structures:
6×6 (216 atoms) followed by 8×8 (512 atoms), liquid temperature 8000 K,
annealing temperature 3000 K, density 1 g/cm³, quench timestep 0.5 fs and
annealing timestep 1 fs. Original file paths and SHA-256 hashes are recorded in
`results/carbon/metadata.json`, and unchanged LAMMPS inputs are retained as
`216/source.data` and `512/source.data`.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m benchmarks.compare_poreblazer \
    --poreblazer-src /path/to/PoreBlazer/src \
    --structures benchmarks/results/carbon/216/source.data \
                 benchmarks/results/carbon/512/source.data \
    --repeats 3 --output /tmp/poretorch-carbon-repeat
```

`--structures` accepts fully periodic cubic carbon LAMMPS data files, in run
order. Image flags are ignored, preserving the wrapped coordinates in the file;
only the common origin translation described below is applied. Cell lengths and
relative atomic positions are never rescaled. All radius, probe, binning, dtype,
thread and timing settings remain the same as the synthetic comparison.

Unlike the synthetic cells, these cells are not rounded to multiples of the
requested spacing. PoreTorch instead matches PoreBlazer's actual grid. The 6×6
cell uses 32³ points at 0.5085443465 Å. For the 8×8 cell, requesting exactly
0.5 Å exposes PoreBlazer's second integer-truncation issue: it calculates spacing
from 43 points, then rounds the resulting count down to 42, leaving a grid plane
unrepresented. The benchmark selects the nearest stable finer grid, 44³ at
0.4931339118 Å, for both codes. The structure is unchanged and PoreBlazer's
source is not patched to change its grid algorithm. `grid.json` records requested
and actual spacings, and the harness rejects grids that do not tile the cell.

## Shared parameters and synthetic inputs

- Fully periodic cubic cells, approximately 0.05 atoms/Å³. Cell lengths are
  rounded up to a multiple of 0.5 Å so both grids coincide: PoreBlazer normally
  rounds the grid count down, while PoreTorch rounds it up.
- Grid spacing 0.5 Å, carbon radius 1.7 Å (PoreBlazer sigma 3.4 Å), point probe
  (radius zero), diameter bins 0.25 Å, upper diameter 30 Å.
- No minimum-radius or connected-pore filtering. Compare PoreBlazer's **Total**
  PSD, not its network-accessible PSD.
- Coordinates are shifted to have zero minima before either calculation, matching
  PoreBlazer's automatic input translation. Grid shapes and void counts are
  asserted equal for every structure and device.
- Double-precision geometry in both codes; PoreBlazer's histogram arrays remain
  single precision as in its source. PoreTorch uses the `cell_list` backend.

`covering` is the closest definition to PoreBlazer here. PoreTorch exhaustively
sizes every void point, whereas PoreBlazer samples 10,000 points with replacement
(seed 21908391). `all_void` and `packed` calculate different distributions;
their runtimes are reported for completeness, not as equivalent replacements.

PoreBlazer reports a survival distribution and a central-difference derivative.
Its cumulative x labels are half a bin below the corresponding lower edges.
The numerical check compares survival probabilities at matching bin edges,
not its smoothed derivative against PoreTorch's rectangular histogram density.
`comparison.json` records the maximum absolute survival difference.

## Isolating PSD

The [PoreBlazer source](https://github.com/SarkisovGitHub/PoreBlazer/blob/main/src/poreblazer.f90)
has separate `initialize`, `lattice_calculations` and `pore_distribution` routines.
The isolated driver calls only these. It omits limiting-diameter, surface-area,
volume-reporting and network-accessibility calculations, including the second PSD.
The PSD algorithm is unchanged.

The shared lattice routine still calculates helium masks and Lennard-Jones
quantities alongside distances. Those remain in the lattice timing; this is not
a fully stripped geometric distance-field implementation.

Timing columns in `results/poreblazer/timings.csv` mean:

| Column | PoreTorch | PoreBlazer |
|---|---|---|
| `field_seconds` | Distance-field construction | Shared lattice calculation |
| `psd_seconds` | Distribution from a precomputed field, including histogram and device-to-host transfer | Sorting, 10,000-point search and cumulative histogram, ending before output files and differentiation |
| `compute_seconds` | Measured field + distribution pipeline | Lattice + PSD timers summed within each run |
| `pipeline_wall_seconds` | Same in-memory pipeline | Whole isolated subprocess, including input, allocation, differentiation, output and process overhead |

PoreBlazer's PSD-only timer stops immediately before its output block. Cumulative normalization and the
central-difference derivative are interleaved with output there, so they are included
in whole-subprocess time, not PSD-core time. The README table uses compute time;
the raw file also exposes whole-subprocess time to make this boundary explicit.
No Python imports, CUDA context startup or compilation are included in compute
times. PoreTorch includes field setup and transfers; PoreBlazer initialization
is outside its compute timer. Both use one warm-up and the median of three timed
runs. CUDA work is synchronized, and PyTorch CPU operations use one thread.

At the smallest size, an additional full-program run validates isolation. Its
surface-area stage normally consumes `2 * atoms * 500` random draws before PSD.
For this check only, the isolated driver discards the same draws before PSD.
The resulting Total cumulative distribution must match the full program exactly.
The RNG-advancement validation run is excluded from timing statistics. The
full-program wall time is a single diagnostic observation, not a benchmark median.

## Recorded results

`results/poreblazer/` contains the timing CSV, individual measurements, hardware
and source hashes, instrumentation patches, compiler logs, inputs, per-method
PSD arrays, PoreBlazer curves and validation logs. `metadata.json` identifies the
PoreBlazer commit and compiler flags. The published run uses a Ryzen 9 9950X
(single CPU thread) and RTX 5090, with GNU Fortran 13.3 `-O2`.

## Synthetic reference timings

The earlier synthetic-structure results are retained for reference. These are
field + PSD compute seconds, with the same hardware and timing convention.

| Implementation / method | 64 atoms | 128 atoms | 256 atoms |
|---|---:|---:|---:|
| PoreBlazer CPU · sampled covering | 0.1882 | 0.4405 | 0.8904 |
| PoreTorch CPU · `all_void`, voxel | 0.1079 | 0.1820 | 0.5522 |
| PoreTorch CPU · `all_void`, sphere | 0.1068 | 0.1824 | 0.5516 |
| PoreTorch CPU · `covering` | 0.8110 | 1.9367 | 5.3069 |
| PoreTorch CPU · `packed` | 0.1076 | 0.1839 | 0.5553 |
| PoreTorch GPU · `all_void`, voxel | 0.0044 | 0.0064 | 0.0177 |
| PoreTorch GPU · `all_void`, sphere | 0.0043 | 0.0064 | 0.0177 |
| PoreTorch GPU · `covering` | 0.0314 | 0.0735 | 0.2005 |
| PoreTorch GPU · `packed` | 0.0053 | 0.0080 | 0.0209 |
