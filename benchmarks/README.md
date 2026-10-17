# Carbon PSD benchmarks

PoreTorch GPU and PoreBlazer CPU timings for 216, 512, 1,728 and 4,096 atoms.
Each size uses carbon structures from simulation runs 1–3. Reported times are
the mean of the three per-structure medians, each based on three repeats.
Error bars show the standard deviation across structures.

## Reproduce

With PoreTorch, GNU Fortran and Make installed, run from the repository root:

```bash
python -m benchmarks.benchmark_end_to_end \
  --poreblazer-src /path/to/PoreBlazer/src \
  --runs 1 2 3 --devices cuda --repeats 3
python -m benchmarks.plot_end_to_end
```

Use `--output` to choose a results directory and `--resume` to continue an
interrupted run. Prepared inputs are in [inputs/carbon_runs](inputs/carbon_runs).

## Settings

PoreTorch uses warm starts after one initial call. PoreBlazer timings include
its full calculation. Input reading and saving are included.

The structures are MACE-MP-0b3 amorphous carbon at 1 g/cm³, cooled after
annealing at 3000 K. Settings are periodic cells, carbon radius 1.7 Å, probe
radius zero, approximately 0.5 Å grids and 0.25 Å diameter bins up to 30 Å.
No radius or connected-pore filter is applied. `all_void` uses default voxel
weighting. PoreTorch geometry uses float64.

PoreBlazer samples 10,000 points. PoreTorch `covering` evaluates every void
point. The three PoreTorch methods use different pore-size definitions.

Hardware is a Ryzen 9 9950X with one CPU thread and an RTX 5090.
Software is PyTorch 2.8.0, CUDA 12.8 and PoreBlazer 4.0 built with
GNU Fortran 13.3 and `-O2`.

[Mean times](results/carbon_three_runs/timings.csv),
[per-structure times](results/carbon_three_runs/per_structure_timings.csv),
[repeats](results/carbon_three_runs/raw_timings.json) and
[metadata](results/carbon_three_runs/metadata.json) accompany the PSDs.
