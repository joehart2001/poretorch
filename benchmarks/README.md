# Carbon PSD benchmarks

PoreTorch GPU and PoreBlazer CPU timings for carbon structures.
The 216–4,096 atom cases use simulation runs 1–3. Reported times are
the mean of the three per-structure medians, each based on three repeats.
Error bars show the standard deviation across structures.
The 32,768-atom case uses one uncooled structure and has no error bar.

## Reproduce

With PoreTorch, GNU Fortran and Make installed, run from the repository root:

```bash
python -m benchmarks.benchmark_end_to_end \
  --poreblazer-src /path/to/PoreBlazer/src \
  --runs 1 2 3 --devices cuda --repeats 3
python -m benchmarks.update_report
```

Use `--output` to choose a results directory and `--resume` to continue an
interrupted run. Prepared inputs are in [inputs/carbon_runs](inputs/carbon_runs).

The [32×32 jobs](results/carbon_32768/README.md) run separately. To update the
README table and plot as they finish:

```bash
python -m benchmarks.update_report --watch
```

Only methods with all three timing repeats are shown. Pending entries use `-`.
The watcher checks every 30 seconds and stops when both jobs end.

## Settings

PoreTorch uses warm starts after one initial call. PoreBlazer timings include
its full calculation. Input reading and saving are included.

The smaller structures are MACE-MP-0b3 amorphous carbon at 1 g/cm³, cooled
after annealing at 3000 K. The 32×32 input is the supplied liquid structure. Settings are periodic cells, carbon radius 1.7 Å, probe
radius zero, approximately 0.5 Å grids and 0.25 Å diameter bins up to 30 Å.
No radius or connected-pore filter is applied. `all_void` uses default voxel
weighting. PoreTorch geometry uses float64.

PoreBlazer samples 10,000 points. PoreTorch `covering` evaluates every void
point. The three PoreTorch methods use different pore-size definitions.

Hardware is a Ryzen 9 9950X with one CPU thread and an RTX 5090.
The 32×32 PoreBlazer job runs on fast9, which has the same CPU model.
Software is PyTorch 2.8.0, CUDA 12.8 and PoreBlazer 4.0 built with
GNU Fortran 13.3 and `-O2`.

[Reported times](results/carbon_report/timings.csv),
[per-structure times](results/carbon_three_runs/per_structure_timings.csv),
[repeats](results/carbon_three_runs/raw_timings.json) and
[metadata](results/carbon_three_runs/metadata.json) accompany the PSDs.
