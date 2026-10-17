#!/usr/bin/env bash
set -u
cd /home/jh2536/poretorch
/home/jh2536/venvs/venv_mace/bin/python -u -m benchmarks.update_report --watch \
  > benchmarks/results/carbon_32768/report.log 2>&1
result=$?
printf '%s\n' "$result" > benchmarks/results/carbon_32768/report.exitcode
exit "$result"
