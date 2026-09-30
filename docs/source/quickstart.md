# Quickstart

Install for development:

```
pip install -e .
```

An NVIDIA GPU is strongly recommended but not required. Pass `device="cpu"` to
run anywhere.

## One structure, one distribution

```python
from ase.io import read
from poretorch import psd, plot_psd

atoms = read("carbon.data", format="lammps-data", atom_style="atomic")

result = psd(
    atoms,
    method="all_void",   # overlapping spheres. "covering" (PoreBlazer-style)
                         # is also overlapping, "packed" is non-overlapping
    grid_spacing=0.3,    # A, the main convergence parameter
    probe_radius="N2",   # accessibility threshold. 0.0 for the bare void
    bin_width=0.25,      # A, diameter bins
)

print(result.stats["median_diameter_A"], "A median pore diameter")
plot_psd(result, path="psd.png")
```

`psd()` accepts an ASE `Atoms`, an iterable of `Atoms`, or a path to anything
ASE can read: LAMMPS data and dump files, extended XYZ, CIF, ASE `.traj`.

## Reading the result

`PSD` stores the void volume per diameter bin, which carries no unit convention
of its own, and converts on demand:

```python
result.diameters_A               # bin centres, A
result.diameters_nm              # bin centres, nm
result.volume_A3                 # void volume per bin, A^3
result.dV_dd_cc_per_nm_per_g     # cc/nm/g, as adsorption data is reported
result.normalised                # unit area, for comparing shapes
result.cumulative_fraction       # cumulative pore volume fraction
```

`dV_dd_cc_per_nm_per_g` needs the structure mass. An ASE `Atoms` carries it, so
this works without extra input.

## Averaging over frames

A single amorphous configuration is one sample of an ensemble, so its
distribution is noisy. Pass several frames and the result is their mean with a
standard error:

```python
frames = read("anneal.extxyz", index="-10:")
result = psd(frames, method="covering")

print(result.n_frames)          # 10
result.se_volume_A3             # standard error per bin
result.se_dV_dd_cc_per_nm_per_g # the same, in reporting units
```

Frames that contain a bigger pore produce a longer histogram. Shorter ones are
zero-extended rather than truncated, so no volume is silently dropped.

## Comparing methods

```python
from poretorch import plot_psd, psd

# The first two use overlapping spheres, the third non-overlapping ones.
curves = [psd(atoms, method=m) for m in ("all_void", "covering", "packed")]
plot_psd(
    curves,
    labels=["all-void (overlapping)", "covering (overlapping)", "packed (disjoint)"],
    path="methods.png",
)
```

## Pores, not just sizes

`analyse()` returns the field and the pore labelling alongside the
distribution:

```python
result = analyse(atoms, method="covering", probe_radius="N2")

print(result.field.porosity)                     # accessible void fraction
print(result.field.specific_void_volume)         # cc/g
print(result.labels.summary())                   # counts and volumes by kind

for pore in result.labels.percolating():
    print(pore.id, pore.kind, pore.volume, pore.percolation_vectors)
```

Each pore is classified by how far it extends: `closed` (a finite pocket),
`channel`, `planar`, or `network`. This is a genuine periodic percolation test,
not a check for touching opposite faces.

## Saving

```python
psd(atoms, output="results/psd.csv")   # or .npz, or .json
psd(atoms, outdir="results")           # writes results/psd.npz
```

## Choosing the settings that matter

`grid_spacing` is the dominant convergence parameter: 0.3 A is a good default
and 0.5 A is noticeably cheaper. Halving it multiplies the grid by eight.

`radii` sets where the solid surface is, and so is part of the definition rather
than a detail. The default `"uff"` takes half the UFF Lennard-Jones sigma from
the chemical symbols, which is the PoreBlazer convention and the right choice
for comparing against it. `"vdw"` uses ASE van der Waals radii instead, and a
scalar, per-atom array, or `{symbol: radius}` mapping all work.

`probe_radius` decides which void a probe can reach. It does not shrink the
reported pore size: a point is accessible when the probe fits with its centre
there, but the size assigned to the point is still the full distance to the
atomic surface. That is the convention PoreBlazer and Zeo++ use.
