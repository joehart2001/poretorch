"""Run a large single-structure benchmark in a persistent session."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

from benchmarks.benchmark_end_to_end import build_poreblazer


def main(args):
    if args.repeats < 1:
        raise ValueError('repeats must be positive')
    root = Path(__file__).resolve().parents[1]
    inputs = args.input_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output/'job.lock').open('x') as lock:
        lock.write(f'{platform.node()} {os.getpid()}\n')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    metadata = dict(host=platform.node(), program=args.program, pid=os.getpid(),
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    python=platform.python_version(), repeats=args.repeats, n_structures=1,
                    cpu=next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
                    cpu_threads=1, inputs=json.loads((inputs/'metadata.json').read_text()))
    records = []

    def status(state, **details):
        record = dict(state=state, updated_utc=datetime.now(timezone.utc).isoformat(), **details)
        temporary = output/'status.tmp'
        temporary.write_text(json.dumps(record, indent=2)+'\n')
        temporary.replace(output/'status.json')
        print(json.dumps(record), flush=True)

    def save():
        (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        (output/'raw_timings.json').write_text(json.dumps(records, indent=2)+'\n')
        if records:
            with (output/'timings.csv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=['code', 'device', 'method', 'weighting', 'n_atoms', 'n_structures', 'seconds'])
                writer.writeheader()
                for record in records:
                    writer.writerow({key: record[key] for key in writer.fieldnames if key != 'seconds'} |
                                    dict(seconds=statistics.median(record['samples'])))

    try:
        save()
        if args.program == 'poretorch':
            # Complete the shorter methods before the expensive covering calculation.
            for method, weighting in [('all_void', 'voxel'), ('packed', 'sphere'), ('covering', 'voxel')]:
                status('running', method=method)
                command = [sys.executable, '-m', 'benchmarks.benchmark_end_to_end', '--worker',
                           '--input', str(inputs/'atoms.extxyz'), '--psd-output', str(output/f'{method}.csv'),
                           '--spacing', str(metadata['inputs']['grid']['poretorch_requested_spacing_A']),
                           '--device', 'cuda', '--method', method, '--weighting', weighting,
                           '--warm-runs', str(args.repeats)]
                with (output/f'{method}.json').open('w') as log, (output/f'{method}.stderr').open('w') as err:
                    subprocess.run(command, cwd=root, env=env, stdout=log, stderr=err, check=True)
                worker = json.loads((output/f'{method}.json').read_text())
                samples = worker['seconds']
                if len(samples) != args.repeats or not all(math.isfinite(t) and t > 0 for t in samples):
                    raise ValueError('invalid timing samples')
                metadata.update(torch=worker['torch'], cuda=worker['cuda'], gpu=worker['gpu'])
                records.append(dict(code='PoreTorch', device='cuda', method=method, weighting=weighting,
                                    n_atoms=metadata['inputs']['n_atoms'], n_structures=1, samples=samples))
                save()
                status('running', completed_method=method, seconds=statistics.median(samples))
        else:
            status('building')
            metadata['compiler'] = subprocess.check_output(['gfortran', '--version'], text=True).splitlines()[0]
            metadata['compiler_flags'] = '-cpp -O2'
            metadata['poreblazer_commit'] = subprocess.check_output(
                ['git', '-C', str(args.poreblazer_src), 'rev-parse', 'HEAD'], text=True).strip()
            exe, hashes = build_poreblazer(args.poreblazer_src.resolve(), output/'build', output)
            metadata['source_sha256'] = hashes
            save()
            samples = []
            reference = None
            for repeat in range(args.repeats):
                run = output/f'repeat_{repeat+1}'
                run.mkdir()
                for name in ('atoms.extxyz', 'input.dat', 'atoms.dat', 'defaults.dat'):
                    shutil.copy2(inputs/name, run/name)
                status('running', repeat=repeat+1, repeats=args.repeats)
                with (run/'poreblazer.log').open('w') as log:
                    start = time.perf_counter()
                    subprocess.run([str(exe), 'input.dat'], cwd=run, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                    samples.append(time.perf_counter()-start)
                curve = (run/'Total_psd_cumulative.txt').read_text()
                if not all(math.isfinite(float(value)) for value in curve.split()):
                    raise ValueError('nonfinite PSD output')
                if reference is not None and curve != reference:
                    raise ValueError('PSD changed between identical repeats')
                reference = curve
                records[:] = [dict(code='PoreBlazer', device='cpu', method='full', weighting='voxel',
                                   n_atoms=metadata['inputs']['n_atoms'], n_structures=1, samples=list(samples))]
                save()
                status('running', completed_repeat=repeat+1, seconds=samples[-1])
        status('complete')
    except BaseException as error:
        status('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--program', choices=['poretorch', 'poreblazer'], required=True)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--poreblazer-src', type=Path, default=Path.home()/'PoreBlazer/src')
    parser.add_argument('--repeats', type=int, default=3)
    main(parser.parse_args())
