# Methods

## The shared foundation

Every pore-size definition here rests on one field: for each point `g` of a
regular grid over the cell, the distance to the nearest atomic surface,

```
d(g) = min_i ( |g - r_i|_mic - R_i )
```

where `R_i` is the radius of atom `i` and the displacement is taken under the
minimum-image convention. `d(g)` is the radius of the largest empty sphere that
can be centred at `g`. A point belongs to the void when `d(g) >= probe_radius`.
This is a local fit test, not a reservoir-reachability calculation, so closed
cavities are included.

Grid points sit at voxel centres in fractional coordinates, so the voxels tile
the cell exactly and each has volume `V_cell / (nx * ny * nz)` whatever the cell
shape. Counts per axis follow the perpendicular widths `V / |a_j x a_k|`, not the
lattice-vector lengths, which keeps the resolution uniform in a sheared cell.

## The three definitions

All three read the same field and differ only in what "the pore size at this
point" means.

| Method | Spheres | Size of a point | Weighting | Integrates to |
|---|---|---|---|---|
| `all_void` | overlapping | its own clearance | voxel volume | accessible void volume |
| `covering` | overlapping | largest empty sphere anywhere that contains it | voxel volume | accessible void volume |
| `packed` | non-overlapping | accepted local-maximum radius | sphere volume | packed sphere volume |

### Overlapping spheres against packed spheres

The overlap column is the sharpest division between the three.

`all_void` and `covering` place a sphere at **every** void point, each as large as
fits there. Neighbouring points differ by one voxel, so their spheres are nearly
the same sphere, and the set overlaps almost completely. No overlap test is
applied, and none is needed.

`packed` is the exception. Candidates are visited largest-first and one is
accepted only if it clears every sphere accepted so far, under the minimum-image
convention so the test holds across periodic boundaries. The accepted set is
therefore **disjoint**. Candidate radii are capped to avoid overlap with their
own periodic replicas or with open cell boundaries. The result is a discrete
sphere-packing spectrum, not a partition of the void.

For `all_void` and `covering`, overlap decides which *size* a point receives,
not how much volume it contributes. Each included voxel carries its own voxel
volume exactly once. `packed` instead weights each accepted sphere by its
analytic volume, so its absolute scale is not directly comparable with the two
voxel-weighted methods. `stats["packed_cell_fraction"]` reports packed sphere
volume divided by cell volume.

### all_void (overlapping spheres)

Every void point is sized by `d(g)`: the largest empty sphere centred *on* the
point. One sphere per void point, all of them free to overlap. This is the
local-pore-radius distribution of the void, in the spirit of the Gelb-Gubbins
geometric PSD. It answers: at a randomly chosen point in the void, how big is the
largest sphere centred there?

### covering (overlapping spheres)

Every void point is sized by the largest empty sphere anywhere in the void that
still *contains* the point. The candidate set is the same overlapping
one-sphere-per-void-point family as `all_void`, and a point simply takes the
largest member that reaches it,

```
r(a) = max { r_c : |a - c|_mic <= r_c }
```

over candidate centres `c`, each carrying its own clearance as its radius. A
sphere centred elsewhere may be bigger and still cover the point, so
`r(a) >= d(a)` always and this distribution sits at larger diameters than
`all_void`. This is the definition PoreBlazer and Zeo++ use, and the one to
compare against them. Equivalently it is the morphological opening size: the
largest sphere that fits entirely in the void and covers the point.

`center_mode="all_void"` treats every void point as a candidate, which is the
faithful definition. `center_mode="local_maxima"` restricts candidates to
clearance maxima: much cheaper, and a strict lower bound on the faithful result,
since fewer candidates can only ever find a smaller covering sphere.

### packed (non-overlapping spheres)

Inscribed spheres are considered greedily, largest first, and the distribution
is taken over the accepted spheres. The spheres are disjoint but leave
interstitial gaps, so this is a packing spectrum rather than a void-volume
partition.

Candidates are local maxima of the clearance field. The pass is global rather
than per pore. Periodic self-image and open-boundary caps ensure the packing is
valid within the simulation cell.

## Pores and percolation

Connected-component labelling of the void mask is only half the job: a pore that
leaves one face and re-enters through the opposite face is one pore, not two. The
merge tracks which periodic image each piece was reached in, which also settles
whether the pore is a closed pocket or runs through the material without end.

A pore percolates when it can be traced back to itself in a *different* periodic
image. The lattice vectors realised by such round trips generate a lattice whose
rank is the pore's dimensionality:

| Dimensionality | Kind | Meaning |
|---|---|---|
| 0 | `closed` | finite pocket in every direction |
| 1 | `channel` | infinite along one direction |
| 2 | `planar` | infinite within a plane |
| 3 | `network` | fully connected pore network |

This is stricter than asking whether a pore touches two opposite faces, which a
small pocket straddling a boundary also passes.

## Triclinic cells

The naive orthorhombic wrap `d -= L * round(d / L)` is not the true minimum image
in a skewed cell. PoreTorch instead reduces the cell to a Minkowski-reduced
basis, wraps into that basis, and searches the 27 surrounding lattice images,
which is exact in three dimensions. The result agrees with
`ase.geometry.find_mic` to floating-point rounding on cells from cubic to
strongly sheared, and the test suite checks that on every cell shape. Orthogonal
cells skip the image search, since the wrap alone is already exact for them.

## Exactness of the fast backend

The `cell_list` backend bins atoms once and examines only the bins around each
grid point, which removes the atom-count factor from the cost. It is exact, not
approximate. Each query returns both a value and a flag saying whether the
searched bins provably covered far enough to have found the true nearest surface:
any atom outside a ring-`k` neighbourhood lies at least `k * bin_width` away, so
a result `v` is final once `v + R_max <= k * bin_width`. Unresolved points are
retried on a wider ring, and once a wider ring would examine more candidate slots
than there are atoms, the brute-force kernel finishes the job. The two backends
therefore agree bit for bit, which the test suite asserts for uniform and
per-atom radii on every cell shape.

## Cost

Measured on an RTX 5090, random structures at 0.05 atoms per cubic Angstrom, 0.4
Angstrom grid spacing.

The clearance field with `cell_list` is linear in the grid size and effectively
independent of the atom count, so its advantage over brute force grows without
bound: 4x at 1728 atoms, 17x at 8000, 43x at 16000 and 87x at 32000, where
brute force takes 58.6 s against 0.672 s.

Among the distributions, `all_void` and `packed` are cheap, 0.012 s and 0.295 s
at 8000 atoms. `covering` is the expensive one at 29.7 s, quadratic in the
number of void points, because in the faithful mode every void point is also a
candidate centre. Descending-radius ordering with
early exit cuts the constant substantially, because the first sphere found to
cover a point is already the largest that ever will and points leave the search
as soon as they are assigned, but the scaling stands. For large cells use
`center_mode="local_maxima"`, or `center_stride` to subsample candidates, and be
aware that both give a lower bound on the faithful result.

## What these are not

These are purely geometric pore sizes computed from atomic coordinates. An
experimental N2-adsorption pore-size distribution is a thermodynamic,
model-dependent quantity: a measured isotherm inverted with a QSDFT or NLDFT
kernel. The two are different observables. Setting `probe_radius="N2"` mimics the
excluded-volume part of an adsorption measurement and makes the comparison
fairer, but a method-matched comparison would require simulating the isotherm and
inverting it with the same kernel the experiment used. Expect agreement in the
position and trend of the main pore mode, not in fine detail.
