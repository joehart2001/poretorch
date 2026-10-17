"""Measure time to a saved PSD for PoreBlazer and PoreTorch."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
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

CONFIGS = [('all_void', 'voxel'), ('all_void', 'sphere'),
           ('covering', 'voxel'), ('packed', 'sphere')]


def worker(args):
    """An ordinary PoreTorch script: public API, input reading and CSV output."""
    start = time.perf_counter()
    import torch
    from poretorch import psd

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    imports = time.perf_counter() - start

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

    first, result = calculate()
    warmed = [calculate()[0] for _ in range(args.warm_runs)]
    print(json.dumps(dict(import_seconds=imports, first_call_seconds=first,
        warm_call_seconds=warmed, torch=torch.__version__, cuda=torch.version.cuda,
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


def main(args):
    import numpy as np

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
        cpu_threads=1, repeats=args.repeats, sizes=args.sizes,
        devices=args.devices, cpu_max_atoms=args.cpu_max_atoms,
        timing='median process launch to successful exit with saved PSD',
        fresh_process_note='Fresh interpreter/CUDA context each run; filesystem caches are not flushed.',
        excluded='Installation, compilation, common input preparation, loading the output into a plotter, and plotting.',
        warm_note='Public psd() in an already-imported Python process: input read, field, labels, PSD, CSV write, synchronization.',
        input_dataset=str(dataset), parameters=json.loads((dataset/'metadata.json').read_text()))
    rows, raw, comparisons = [], [], []

    def save():
        with (output / 'timings.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (output / 'raw_timings.json').write_text(json.dumps(raw, indent=2)+'\n')
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        (output / 'comparison.json').write_text(json.dumps(comparisons, indent=2)+'\n')

    with tempfile.TemporaryDirectory(prefix='poretorch-benchmark-') as temporary:
        temporary = Path(temporary)
        exe, hashes = build_poreblazer(args.poreblazer_src.resolve(), temporary/'poreblazer', output)
        metadata['source_sha256'] = hashes
        for n in args.sizes:
            case = dataset/str(n)
            inputs = output/str(n)/'inputs'
            prepare(case, inputs)
            grid = json.loads((case/'grid.json').read_text())
            print(f'{n} atoms: PoreBlazer, {args.repeats} complete fresh runs', flush=True)
            pb_times = []
            pb_curves = []
            for repeat in range(args.repeats):
                run = temporary/f'pb-{n}-{repeat}'
                shutil.copytree(inputs, run)
                with (output/str(n)/f'poreblazer_{repeat}.log').open('w') as log:
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
                    shutil.copy2(run/name, output/str(n)/name)
            for curve in pb_curves[1:]:
                np.testing.assert_array_equal(curve, pb_curves[0])
            rows.append(dict(code='PoreBlazer', device='cpu', method='full', weighting='voxel',
                n_atoms=n, launch_to_saved_psd_seconds=statistics.median(pb_times),
                warm_session_to_saved_psd_seconds=''))
            raw.append(dict(code='PoreBlazer', n_atoms=n, launch_seconds=pb_times))
            save()
            for device in args.devices:
                if device == 'cpu' and n > args.cpu_max_atoms:
                    continue
                for method, weighting in CONFIGS:
                    print(f'{n} atoms: PoreTorch {device} {method}/{weighting}, fresh + warm', flush=True)
                    csv_path = output/str(n)/f'{device}_{method}_{weighting}.csv'
                    command = [sys.executable, '-m', 'benchmarks.benchmark_end_to_end', '--worker',
                        '--input', str(inputs/'atoms.extxyz'), '--psd-output', str(csv_path),
                        '--spacing', str(grid['poretorch_requested_spacing_A']),
                        '--device', device, '--method', method, '--weighting', weighting]
                    wall_times, worker_records = [], []
                    for repeat in range(args.repeats):
                        csv_path.unlink(missing_ok=True)
                        start = time.perf_counter()
                        proc = subprocess.run(command, cwd=root, env=env, capture_output=True,
                                              text=True, check=True, timeout=args.timeout)
                        wall_times.append(time.perf_counter()-start)
                        worker_records.append(json.loads(proc.stdout))
                        assert csv_path.is_file()
                    warm = subprocess.run(command+['--warm-runs', str(args.repeats)],
                        cwd=root, env=env, capture_output=True, text=True, check=True,
                        timeout=args.timeout*(args.repeats+1))
                    warm_record = json.loads(warm.stdout)
                    table = np.loadtxt(csv_path, delimiter=',', skiprows=1)
                    assert np.isfinite(table).all()
                    np.testing.assert_allclose(table[-1, 3], 1., atol=1e-10)
                    if device == 'cuda' and 'cpu' in args.devices and n <= args.cpu_max_atoms:
                        cpu = np.loadtxt(output/str(n)/f'cpu_{method}_{weighting}.csv',
                                         delimiter=',', skiprows=1)
                        np.testing.assert_allclose(table, cpu, rtol=1e-8, atol=1e-8)
                    if method == 'covering':
                        survival = np.r_[1., 1.-table[:,3]][:-1]
                        comparisons.append(dict(n_atoms=n, device=device,
                            max_survival_difference=float(np.max(abs(survival-pb_curves[0][:,1])))))
                    metadata.update(torch=warm_record['torch'], cuda=warm_record['cuda'])
                    if warm_record['gpu']:
                        metadata['gpu'] = warm_record['gpu']
                    rows.append(dict(code='PoreTorch', device=device, method=method, weighting=weighting,
                        n_atoms=n, launch_to_saved_psd_seconds=statistics.median(wall_times),
                        warm_session_to_saved_psd_seconds=statistics.median(warm_record['warm_call_seconds'])))
                    raw.append(dict(code='PoreTorch', device=device, method=method, weighting=weighting,
                        n_atoms=n, launch_seconds=wall_times, worker_records=worker_records,
                        warm_process=warm_record))
                    save()
    print(f'Wrote end-to-end results to {output}', flush=True)


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
    parser.add_argument('--dataset', type=Path, default=Path('benchmarks/inputs/carbon'))
    parser.add_argument('--output', type=Path, default=Path('benchmarks/results/carbon_end_to_end'))
    parser.add_argument('--sizes', nargs='+', type=int, default=[216, 512, 1728, 4096])
    parser.add_argument('--devices', nargs='+', choices=['cpu', 'cuda'], default=['cpu', 'cuda'])
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
