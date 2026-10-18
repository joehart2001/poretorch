#!/usr/bin/env bash
set -eu
cd /home/jh2536/poretorch
python3 -u -m benchmarks.benchmark_large --program poreblazer --input-dir benchmarks/inputs/carbon_32768 --output benchmarks/results/carbon_32768_exact/poreblazer --repeats 1
/home/jh2536/venvs/venv_mace/bin/python -m benchmarks.update_report --large-results benchmarks/results/carbon_32768_exact
