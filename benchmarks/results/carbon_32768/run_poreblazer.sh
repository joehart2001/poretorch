#!/usr/bin/env bash
set -u
cd /home/jh2536/poretorch
python3 -u -m benchmarks.benchmark_large \
  --program poreblazer \
  --input-dir benchmarks/inputs/carbon_32768 \
  --output benchmarks/results/carbon_32768/poreblazer \
  --poreblazer-src /home/jh2536/PoreBlazer/src \
  --repeats 3 > benchmarks/results/carbon_32768/poreblazer.log 2>&1
result=$?
printf '%s\n' "$result" > benchmarks/results/carbon_32768/poreblazer.exitcode
exit "$result"
