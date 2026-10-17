![PoreTorch logo](logo.svg)

# PoreTorch

PoreTorch computes geometric pore-size distributions from atomistic structures.
It supports periodic triclinic cells, runs on a GPU when one is available, and
falls back to the CPU automatically.

The package accepts ASE `Atoms`, trajectories, and structure files readable by
ASE.

## Installation

From PyPI:

```bash
python -m pip install PoreTorch
```

From the repository:

```bash
git clone https://github.com/joehart2001/poretorch.git
cd poretorch
python -m pip install -e .
```

## Quick start

```python
from ase.io import read
from poretorch import psd, plot_psd

atoms = read(
    "carbon.data",
    format="lammps-data",
    atom_style="atomic",
)

result = psd(
    atoms,
    method="covering",
    grid_spacing=0.3,
    probe_radius="N2",
    bin_width=0.25,
)

print(result.stats["median_diameter_A"])
plot_psd(result, path="psd.png")
```

`device="auto"` is the default. It selects CUDA when available and otherwise
uses the CPU. Pass `device="cpu"` or `device="cuda"` to choose explicitly.

## What is being measured?

At every grid point `g`, PoreTorch computes the clearance to the nearest atomic
surface:

```text
clearance(g) = min_i(distance(g, atom_i) - radius_i)
```

This is the radius of the largest empty sphere centred at `g`. A point is
included when its clearance is at least `probe_radius`.

The probe is an inclusion threshold only. It is not subtracted from the reported
pore radius. For example, an included point with 3.0 A clearance is reported as
a 6.0 A pore diameter, including when an N2 probe was used.

Probe-accessible here means that the probe fits locally. Closed cavities are
included; PoreTorch does not test whether a probe can enter from an external
reservoir.

## Pore-size definitions

| Method | Radius assigned | Histogram weight | Use it for |
|---|---|---|---|
| `all_void` | Clearance at the voxel itself | Voxel volume | Local-clearance distribution |
| `covering` | Largest empty sphere that contains the voxel | Voxel volume | PoreBlazer-style geometric PSD |
| `packed` | Radius of an accepted non-overlapping sphere | Sphere volume | Discrete sphere-packing spectrum |

`psd(atoms, method="all_void", weighting="sphere")` optionally weights each
local clearance by `4πr³/3`, using the same points and filters. This favours
larger clearances and counts overlapping space repeatedly: the integral is not
physical pore volume or porosity. Defaults remain unchanged (`voxel` for
`all_void`/`covering`, `sphere` for `packed`); other combinations are rejected.
Results and exports record `weighting` and `volume_kind`, and ensembles require
matching weightings. See [Methods](docs/source/methods.md#optional-sphere-weighting).


### `all_void`

Each included voxel is sized by the largest empty sphere centred exactly there.
By default every voxel contributes once, so the distribution integrates to the included
void volume unless an explicit filter is applied.

### `covering`

Each included voxel is sized by the largest empty sphere, centred anywhere in
the void, that contains it. The result is never smaller than `all_void` at the
same point.

The default `center_mode="all_void"` evaluates the full grid definition. It can
be expensive for large systems. `center_mode="local_maxima"` and
`center_stride>1` are faster lower-bound approximations.

### `packed`

Local clearance maxima are considered largest-first. A sphere is accepted only
when it does not overlap previously accepted spheres or its own periodic
replicas. Open cell boundaries also cap the sphere radius.

This method describes a set of discrete inscribed spheres. It is not a partition
of the void and its absolute scale should not be compared directly with the two
voxel-weighted methods. `stats["packed_cell_fraction"]` is the packed sphere
volume divided by cell volume.

## Atomic and probe radii

The ASE interface uses `radii="uff"` by default: half the UFF Lennard-Jones
sigma for each element. Other options are:

```python
psd(atoms, radii="vdw")
psd(atoms, radii=1.70)
psd(atoms, radii={"C": 1.70, "O": 1.52})
```

Named probe radii are `"N2"`, `"Ar"`, `"He"`, `"CO2"`, and `"geometric"`.
Atomic radii, probe radius, and grid spacing are part of the definition of a
result and should always be reported with it.

## Results

```python
result.diameters_A
result.volume_A3
result.dV_dd_cc_per_nm_per_g
result.normalised
result.cumulative_fraction
result.stats
```

`volume_A3` is the volume in each diameter bin. The per-gram conversion uses the
atomic masses stored by ASE.

For an ensemble of structures, pass several frames. PoreTorch returns the mean
bin volume and its standard error:

```python
frames = read("anneal.extxyz", index="-10:")
result = psd(frames, method="covering")

result.n_frames
result.se_volume_A3
```

Use `analyse()` when you also need connected pores and percolation information:

```python
from poretorch import analyse

analysis = analyse(atoms, method="covering", probe_radius="N2")

print(analysis.field.porosity)
print(analysis.labels.summary())
```

Pores are classified as `closed`, `channel`, `planar`, or `network` from their
periodic percolation dimensionality.

## Benchmarks

Small synthetic periodic carbon structures, with all three methods and both
`all_void` weightings. Times below are **field + PSD compute time in seconds**,
excluding imports, compilation and file output. Medians of three runs after
one warm-up; CUDA timings are synchronized.

Hardware: AMD Ryzen 9 9950X (one CPU thread), NVIDIA RTX 5090. PoreTorch uses
float64 and `cell_list`; PoreBlazer 4.0 uses GNU Fortran 13.3, `-O2`. Both use
identical grids at 0.5 Å spacing, atomic radius 1.7 Å, a point probe (radius 0),
0.25 Å diameter bins, and no connected-pore or minimum-radius filtering.

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

PoreBlazer is run through an isolated driver: surface area, limiting diameter,
volume reporting and network-accessibility stages are omitted. Its shared
lattice routine still includes helium-related work. For 256 atoms, its lattice
and PSD-core times are **0.0670 s** and **0.8233 s**; PoreTorch GPU field and
`covering` times are **0.0171 s** and **0.1831 s**. PoreBlazer's complete isolated
subprocess, including input and output, takes **0.9026 s**. Its PSD-core timer
excludes normalization and differentiation interleaved with file output.

`covering` is the closest comparison: PoreBlazer samples 10,000 points, while
PoreTorch evaluates every void point. GPU PoreTorch is 4.4–6.0× faster for
field + covering on these cases; CPU PoreTorch covering is slower. `all_void`
and `packed` have different definitions, so their timings are not equivalent
PSD speedups. The maximum cumulative-distribution differences are 0.61–0.73
percentage points, consistent with sampling differences. The isolated PoreBlazer
PSD matches its full-program result exactly when given the same random draws.

These are uniform random structures (overlaps allowed), not equilibrated carbon;
the results should not be extrapolated to large structures. Grid sizes are
10,648, 21,952 and 42,875 points. See the [benchmark protocol and reproduction
command](benchmarks/README.md), [raw timing breakdown](benchmarks/results/poreblazer/timings.csv)
and [hardware/source metadata](benchmarks/results/poreblazer/metadata.json).

## Numerical controls

- `grid_spacing` is the main convergence parameter. The default is 0.3 A.
- `bin_width` controls diameter histogram resolution.
- `min_radius` removes smaller radii or packed-sphere candidates.
- `max_diameter` fixes the histogram range. Values outside it raise an error
  unless `allow_truncation=True` is explicitly requested.
- `field_method="cell_list"` is the fast exact backend. `"brute"` is the
  reference implementation.

The grid uses voxel centres and tiles the cell exactly. Minimum-image distances
use a Minkowski-reduced basis, including for skewed triclinic cells.

PoreTorch is released under the MIT license.
