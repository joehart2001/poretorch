"""Benchmark PoreBlazer and PoreTorch on carbon structures."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

CONFIGS = [('all_void', 'voxel'),
           ('covering', 'voxel'), ('packed', 'sphere')]


def worker(args):
    """An ordinary PoreTorch script: public API, input reading and CSV output."""
    import torch
    from poretorch import psd

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    def calculate():
        begin = time.perf_counter()
        result = psd(
            args.input, method=args.method, weighting=args.weighting,
            radii=1.7, probe_radius=0., grid_spacing=args.spacing,
            min_radius=0., min_voxels=1, bin_width=.25, max_diameter=30.,
            device=args.device, torch_dtype=torch.float64,
            output=args.psd_output,
        )
        # CSV export has already copied the results to CPU and closed the file.
        if args.device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        return elapsed, result

    _, result = calculate()
    warmed = [calculate()[0] for _ in range(args.warm_runs)]
    print(json.dumps(dict(seconds=warmed, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
        total_void_volume_A3=result.total_void_volume_A3)), flush=True)


def build_poreblazer(source, directory, output):
    """Build PoreBlazer using its supplied Makefile."""
    directory.mkdir()
    hashes = {}
    for pattern in ('*.f90', '*.f', 'Makefile_gfort'):
        for path in source.glob(pattern):
            shutil.copy2(path, directory / path.name)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with (output / 'build_poreblazer.log').open('w') as log:
        subprocess.run(['make', '-f', 'Makefile_gfort'], cwd=directory,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    for name, digest in hashes.items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    return directory / 'poreblazer.exe', hashes


def prepare(case, target):
    """Use one extended XYZ for both codes, without changing the original cell."""
    target.mkdir(parents=True, exist_ok=True)
    lines = (case / 'atoms.xyz').read_text().splitlines()
    box = (case / 'input.dat').read_text().splitlines()
    lengths = [float(x) for x in box[1].split()]
    lattice = [lengths[0], 0, 0, 0, lengths[1], 0, 0, 0, lengths[2]]
    lines[1] = ('Lattice="' + ' '.join(str(x) for x in lattice)
                + '" Properties=species:S:1:pos:R:3 pbc="T T T"')
    (target / 'atoms.extxyz').write_text('\n'.join(lines) + '\n')
    box[0] = 'atoms.extxyz'
    (target / 'input.dat').write_text('\n'.join(box) + '\n')
    for name in ('atoms.dat', 'defaults.dat'):
        shutil.copy2(case / name, target / name)


TIMING_FIELDS = ('seconds',)
GROUP_FIELDS = ('code', 'device', 'method', 'weighting', 'n_atoms')
SUMMARY_FIELDS = (*GROUP_FIELDS, 'n_structures', *TIMING_FIELDS,
                  *(field + '_std' for field in TIMING_FIELDS))


def aggregate_timings(rows, run_ids):
    """Mean and sample SD of per-structure medians, for complete groups only."""
    expected = set(run_ids)
    if not expected or len(expected) != len(run_ids):
        raise ValueError('run IDs must be nonempty and unique')
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in GROUP_FIELDS)
        group = groups.setdefault(key, {})
        run_id = int(row['run_id'])
        if run_id not in expected or run_id in group:
            raise ValueError('unexpected or duplicate structure run')
        group[run_id] = row
    result = []
    for key, group in groups.items():
        if set(group) != expected:
            continue
        summary = dict(zip(GROUP_FIELDS, key))
        summary['n_structures'] = len(group)
        for field in TIMING_FIELDS:
            values = [group[run_id][field] for run_id in run_ids]
            if all(value == '' for value in values):
                summary[field] = summary[field + '_std'] = ''
            elif any(value == '' for value in values):
                raise ValueError('inconsistent timing availability across structures')
            else:
                values = [float(value) for value in values]
                summary[field] = statistics.mean(values)
                summary[field + '_std'] = statistics.stdev(values) if len(values) > 1 else 0.
        result.append(summary)
    return result


def main(args):
    import numpy as np

    aggregate_timings([], args.runs)
    if args.repeats < 1:
        raise ValueError('repeats must be positive')
    root = Path(__file__).resolve().parents[1]
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    metadata = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(), compiler=subprocess.check_output(
            ['gfortran', '--version'], text=True).splitlines()[0], compiler_flags='-cpp -O2',
        poreblazer_commit=subprocess.check_output(
            ['git', '-C', str(args.poreblazer_src), 'rev-parse', 'HEAD'], text=True).strip(),
        cpu=next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
        cpu_threads=1, repeats=args.repeats, sizes=args.sizes, runs=args.runs,
        devices=args.devices, cpu_max_atoms=args.cpu_max_atoms,
        timing=f'mean across structure runs of each structure\'s {args.repeats}-repeat median',
        variation='sample standard deviation across per-structure medians',
        excluded='Installation, compilation, common input preparation, loading the output into a plotter, and plotting.',
        warm_note='PoreTorch calls after one warm-up, including input reading, calculation and saving.',
        input_dataset=str(dataset), parameters=json.loads((dataset/'metadata.json').read_text()))
    rows, raw, comparisons = [], [], []
    previous = None
    if args.resume:
        previous = json.loads((output/'metadata.json').read_text())
        for key in ('parameters', 'repeats', 'poreblazer_commit', 'devices', 'sizes', 'runs'):
            if previous[key] != metadata[key]:
                raise ValueError(f'cannot resume with different {key}')
        with (output/'per_structure_timings.csv').open() as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            row['n_atoms'] = int(row['n_atoms'])
            row['run_id'] = int(row['run_id'])
        raw = json.loads((output/'raw_timings.json').read_text())
        comparisons = json.loads((output/'comparison.json').read_text())
        metadata['previous_batches'] = previous.get('previous_batches', []) + [
            dict(timestamp_utc=previous['timestamp_utc'], reused_records=len(rows))]
    completed = {(r['code'], r['device'], r['method'], r['weighting'],
                  int(r['n_atoms']), int(r['run_id'])) for r in rows}

    def save():
        with (output / 'per_structure_timings.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (output / 'timings.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(aggregate_timings(rows, args.runs))
        (output / 'raw_timings.json').write_text(json.dumps(raw, indent=2)+'\n')
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        (output / 'comparison.json').write_text(json.dumps(comparisons, indent=2)+'\n')

    with tempfile.TemporaryDirectory(prefix='poretorch-benchmark-') as temporary:
        temporary = Path(temporary)
        exe, hashes = build_poreblazer(args.poreblazer_src.resolve(), temporary/'poreblazer', output)
        metadata['source_sha256'] = hashes
        if previous is not None and previous['source_sha256'] != hashes:
            raise ValueError('cannot resume with different PoreBlazer sources')
        for n, run_id in itertools.product(args.sizes, args.runs):
            case = dataset/str(n)/f'run_{run_id}'
            case_output = output/str(n)/f'run_{run_id}'
            inputs = case_output/'inputs'
            prepare(case, inputs)
            grid = json.loads((case/'grid.json').read_text())
            if ('PoreBlazer', 'cpu', 'full', 'voxel', n, run_id) not in completed:
                print(f'{n} atoms, run {run_id}: PoreBlazer, {args.repeats} repeats', flush=True)
                pb_times = []
                pb_curves = []
                for repeat in range(args.repeats):
                    run = temporary/f'pb-{n}-{run_id}-{repeat}'
                    shutil.copytree(inputs, run)
                    with (case_output/f'poreblazer_{repeat}.log').open('w') as log:
                        start = time.perf_counter()
                        subprocess.run([str(exe), 'input.dat'], cwd=run, env=env,
                                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.timeout)
                        pb_times.append(time.perf_counter()-start)
                    # Check saved plots are ready only after the stopwatch stops.
                    curve = np.loadtxt(run/'Total_psd_cumulative.txt')
                    assert np.isfinite(curve).all()
                    pb_curves.append(curve)
                    for name in ('Total_psd.txt', 'Total_psd_cumulative.txt',
                                 'Network-accessible_psd.txt', 'summary.dat'):
                        shutil.copy2(run/name, case_output/name)
                for curve in pb_curves[1:]:
                    np.testing.assert_array_equal(curve, pb_curves[0])
                rows.append(dict(code='PoreBlazer', device='cpu', method='full', weighting='voxel',
                    n_atoms=n, run_id=run_id, seconds=statistics.median(pb_times)))
                raw.append(dict(code='PoreBlazer', device='cpu', method='full', weighting='voxel',
                    n_atoms=n, run_id=run_id, seconds=pb_times))
                save()
            pb_curve = np.loadtxt(case_output/'Total_psd_cumulative.txt')
            for device in args.devices:
                if device == 'cpu' and n > args.cpu_max_atoms:
                    continue
                for method, weighting in CONFIGS:
                    if ('PoreTorch', device, method, weighting, n, run_id) in completed:
                        continue
                    print(f'{n} atoms, run {run_id}: PoreTorch {device} {method}/{weighting}, warm start', flush=True)
                    csv_path = case_output/f'{device}_{method}_{weighting}.csv'
                    command = [sys.executable, '-m', 'benchmarks.benchmark_end_to_end', '--worker',
                        '--input', str(inputs/'atoms.extxyz'), '--psd-output', str(csv_path),
                        '--spacing', str(grid['poretorch_requested_spacing_A']),
                        '--device', device, '--method', method, '--weighting', weighting]
                    warm = subprocess.run(command+['--warm-runs', str(args.repeats)],
                        cwd=root, env=env, capture_output=True, text=True, check=True,
                        timeout=args.timeout*(args.repeats+1))
                    warm_record = json.loads(warm.stdout)
                    table = np.loadtxt(csv_path, delimiter=',', skiprows=1)
                    assert np.isfinite(table).all()
                    np.testing.assert_allclose(table[-1, 3], 1., atol=1e-10)
                    if device == 'cuda' and 'cpu' in args.devices and n <= args.cpu_max_atoms:
                        cpu = np.loadtxt(case_output/f'cpu_{method}_{weighting}.csv',
                                         delimiter=',', skiprows=1)
                        np.testing.assert_allclose(table, cpu, rtol=1e-8, atol=1e-8)
                    if method == 'covering':
                        survival = np.r_[1., 1.-table[:,3]][:-1]
                        comparisons.append(dict(n_atoms=n, run_id=run_id, device=device,
                            max_survival_difference=float(np.max(abs(survival-pb_curve[:,1])))))
                    metadata.update(torch=warm_record['torch'], cuda=warm_record['cuda'])
                    if warm_record['gpu']:
                        metadata['gpu'] = warm_record['gpu']
                    rows.append(dict(code='PoreTorch', device=device, method=method, weighting=weighting,
                        n_atoms=n, run_id=run_id, seconds=statistics.median(warm_record['seconds'])))
                    raw.append(dict(code='PoreTorch', device=device, method=method, weighting=weighting,
                        n_atoms=n, run_id=run_id, seconds=warm_record['seconds'], worker=warm_record))
                    save()
    print(f'Wrote benchmark results to {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--psd-output', type=Path)
    parser.add_argument('--spacing', type=float)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--method', choices=['all_void', 'covering', 'packed'])
    parser.add_argument('--weighting', choices=['voxel', 'sphere'])
    parser.add_argument('--warm-runs', type=int, default=0)
    parser.add_argument('--poreblazer-src', type=Path)
    parser.add_argument('--dataset', type=Path, default=Path('benchmarks/inputs'))
    parser.add_argument('--output', type=Path, default=Path('benchmarks/results'))
    parser.add_argument('--sizes', nargs='+', type=int, default=[216, 512, 1728, 4096])
    parser.add_argument('--runs', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--resume', action='store_true', help='Reuse completed per-structure measurements')
    parser.add_argument('--devices', nargs='+', choices=['cpu', 'cuda'], default=['cuda'])
    parser.add_argument('--cpu-max-atoms', type=int, default=512,
                        help='Run PoreTorch CPU methods only up to this size')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=600)
    arguments = parser.parse_args()
    if arguments.worker:
        worker(arguments)
    else:
        if arguments.poreblazer_src is None:
            parser.error('--poreblazer-src is required')
        main(arguments)
