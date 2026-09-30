# API

## Entry points

```{eval-rst}
.. autofunction:: poretorch.psd
.. autofunction:: poretorch.analyse
.. autofunction:: poretorch.plot_psd
```

## Results

```{eval-rst}
.. autoclass:: poretorch.PSD
   :members:
.. autoclass:: poretorch.PoreAnalysis
   :members:
.. autoclass:: poretorch.DistanceField
   :members:
.. autoclass:: poretorch.PoreLabels
   :members:
.. autoclass:: poretorch.Pore
   :members:
.. autoclass:: poretorch.SpherePacking
   :members:
.. autoclass:: poretorch.Grid
   :members:
```

## Building blocks

Use these when you want to reuse one field across several pore-size definitions,
or to drive the calculation from raw arrays rather than an ASE object.

```{eval-rst}
.. autofunction:: poretorch.distance_field
.. autofunction:: poretorch.field_from_atoms
.. autofunction:: poretorch.label_pores
.. autofunction:: poretorch.pore_size_distribution
.. autofunction:: poretorch.covering_sizes
.. autofunction:: poretorch.pack_spheres
.. autofunction:: poretorch.local_maxima
.. autofunction:: poretorch.average_psds
.. autofunction:: poretorch.read_structures
```

## Conventions

```{eval-rst}
.. autofunction:: poretorch.atomic_radii
```

The built-in lookup tables are available as `poretorch.UFF_SIGMA` and
`poretorch.PROBE_RADII`.

## Cell geometry

```{eval-rst}
.. automodule:: poretorch.cell
   :members:
```

## Neighbour search

```{eval-rst}
.. automodule:: poretorch.neighbor
   :members:
```
