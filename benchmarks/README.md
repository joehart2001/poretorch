# Carbon PSD benchmarks

Compare the time to save a PSD for 6×6, 8×8, 12×12 and 16×16 amorphous-carbon
structures (216, 512, 1,728 and 4,096 atoms). The [main README](../README.md#benchmarks)
reports median timings over three runs.

## Reproduce

With PoreTorch, ASE, NumPy, GNU Fortran and Make installed, run from the repository root:

```bash
python -m benchmarks.benchmark_end_to_end \
  --poreblazer-src /path/to/PoreBlazer/src \
  --sizes 216 512 1728 4096 --repeats 3
python -m benchmarks.plot_end_to_end
```

PoreTorch CPU runs stop at 512 atoms by default. Larger cases use GPU only.
Use `--cpu-max-atoms` to change this limit or `--devices cpu` for CPU only. Prepared structures and parameters are
in [inputs/carbon](inputs/carbon). Results are written to
`results/carbon_end_to_end`. Use `--output` to select another directory.

## Timing and settings

**New process** measures launch through successful exit with a saved PSD,
including imports and CUDA initialization. **Warm start** measures the
complete `psd()` call after one initial call. Both include input reading,
calculation and saving. PoreBlazer timings include its full calculation.
Installation, compilation, input preparation and plotting are excluded.
Startup times vary. Filesystem caches are not flushed.

The inputs are the MACE-MP-0b3 `run_1` carbon structures at 1 g/cm³, cooled
after annealing at 3000 K. Both programs use periodic cells, carbon radius
1.7 Å, probe radius zero, approximately 0.5 Å grids, 0.25 Å diameter bins
(up to 30 Å), and no radius or connected-pore exclusion. Exact grids and
source hashes are saved with the inputs.

PoreBlazer 4.0 samples 10,000 points. PoreTorch `covering` evaluates every void
point and is the closest PSD definition. `all_void` and `packed` measure
different distributions. PoreTorch uses float64 geometry.

Measurements use a Ryzen 9 9950X (one CPU thread) and RTX 5090, PyTorch 2.8.0
with CUDA 12.8, and GNU Fortran 13.3 with `-O2`.

[Timings](results/carbon_end_to_end/timings.csv),
[individual runs](results/carbon_end_to_end/raw_timings.json), and
[metadata](results/carbon_end_to_end/metadata.json) accompany the saved PSDs.
