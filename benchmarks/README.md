# Carbon PSD benchmarks

PoreTorch GPU and PoreBlazer CPU timings for 216–32,768 carbon atoms.
See the [main README](../README.md#benchmarks) for results.

- `inputs/<atoms>/run_<n>/`: structures and settings.
- `results/<atoms>/run_<n>/`: timings, PSDs and metadata.
- `report/`: combined timing table and plots.

Settings: carbon radius 1.7 Å, probe radius zero, approximately 0.5 Å grids
and 0.25 Å diameter bins. PoreTorch uses warm starts. PoreBlazer includes its
full calculation.

Run from the repository root with PoreTorch, GNU Fortran and Make installed:

```bash
python -m benchmarks.benchmark_end_to_end --poreblazer-src /path/to/PoreBlazer/src
python -m benchmarks.update_report
```

For 32,768 atoms, use `python -m benchmarks.benchmark_large --help`.
