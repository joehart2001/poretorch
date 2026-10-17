#!/usr/bin/env bash
set -u
cd /home/jh2536/poretorch
/home/jh2536/venvs/venv_mace/bin/python -u -m benchmarks.benchmark_large \
  --program poretorch \
  --input-dir benchmarks/inputs/carbon_32768 \
  --output benchmarks/results/carbon_32768/poretorch \
  --repeats 3 > benchmarks/results/carbon_32768/poretorch.log 2>&1
result=$?
printf '%s\n' "$result" > benchmarks/results/carbon_32768/poretorch.exitcode
exit "$result"
