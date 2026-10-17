"""Small-system PSD timings against an isolated PoreBlazer 4.0 driver.

Run from the repository root; see benchmarks/README.md for timing boundaries.
The supplied PoreBlazer checkout is only read. Compilation and runs use a
TemporaryDirectory, and the driver/instrumentation diff is saved with results.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np
import torch

from poretorch import distance_field, pore_size_distribution

CONFIGS = [("all_void", "voxel"), ("all_void", "sphere"),
           ("covering", "voxel"), ("packed", "sphere")]
SPACING = 0.5
RADIUS = 1.7
BIN_WIDTH = 0.25
MAX_DIAMETER = 30.0
SEED = 21908391

DRIVER = '''program poreblazer
use parameters
use lattice, only: ncubesx, ncubesy, ncubesz, ng_cubes
use adsorbent, only: natoms
use random, only: rranf
implicit none
interface
    subroutine initialize(arg)
        character(len=*), optional :: arg
    end subroutine initialize
end interface
integer(kind=8) :: start, finish, rate
integer :: draw
real(kind=8) :: discarded
call initialize('input.dat')
call system_clock(start, rate)
call lattice_calculations
call system_clock(finish)
write(*,*) 'BENCH_LATTICE', real(finish-start,8)/real(rate,8)
write(*,*) 'BENCH_GRID', ncubesx, ncubesy, ncubesz, ng_cubes
property = 'Total'
! Validation only: consume the draws used by the full surface-area routine.
if (command_argument_count() > 1) then
    do draw = 1, 2*natoms*nsample
        discarded = rranf()
    end do
end if
call pore_distribution
end program
'''


def instrument(source: str, isolated: bool) -> str:
    """Keep calculation routines intact; add wall-clock timers."""
    if isolated:
        source, count = re.subn(r'^program poreblazer\b.*?^end program\s*$',
                               DRIVER, source, count=1, flags=re.M | re.S)
        if count != 1:
            raise ValueError('Cannot locate the PoreBlazer main program')
    start = source.index('subroutine pore_distribution\n')
    end = source.index('end subroutine pore_distribution', start)
    routine = source[start:end]
    anchor = '    integer,allocatable                   :: connolly_volume(:)'
    assert routine.count(anchor) == 1
    routine = routine.replace(anchor, anchor + '''
    integer(kind=8) :: bench_start, bench_finish, bench_rate

    call system_clock(bench_start, bench_rate)
''')
    anchor = '    open(13, file='
    assert routine.count(anchor) == 1
    routine = routine.replace(anchor, '''    call system_clock(bench_finish)
    write(*,*) 'BENCH_PSD', real(bench_finish-bench_start,8)/real(bench_rate,8)

''' + anchor)
    return source[:start] + routine + source[end:]


def build(source_dir: Path, work: Path, output: Path, isolated: bool) -> Path:
    work.mkdir()
    for pattern in ('*.f90', '*.f', 'Makefile_gfort'):
        for path in source_dir.glob(pattern):
            shutil.copy2(path, work / path.name)
    original = (source_dir / 'poreblazer.f90').read_text()
    modified = instrument(original, isolated)
    (work / 'poreblazer.f90').write_text(modified)
    name = 'isolated' if isolated else 'full'
    (output / f'poreblazer_{name}.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), modified.splitlines(True),
        fromfile='src/poreblazer.f90', tofile=f'{name}/poreblazer.f90')))
    with (output / f'build_{name}.log').open('w') as log:
        subprocess.run(['make', '-f', 'Makefile_gfort'], cwd=work,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    return work / 'poreblazer.exe'


def structure(n_atoms):
    # Exact half-Angstrom cell lengths align both codes' otherwise different
    # grid rounding (PoreBlazer floors; PoreTorch ceils).
    side = np.ceil((n_atoms / .05)**(1/3) / SPACING) * SPACING
    positions = np.random.default_rng(42).random((n_atoms, 3)) * side
    # PoreBlazer translates each coordinate minimum to zero during input.
    positions -= positions.min(axis=0)
    return positions, np.diag([side] * 3)


def inputs(directory, positions, cell):
    directory.mkdir()
    side = cell[0, 0]
    with (directory / 'atoms.xyz').open('w') as handle:
        handle.write(f'{len(positions)}\nsynthetic periodic carbon; seed 42\n')
        for xyz in positions:
            handle.write('C ' + ' '.join(f'{x:.17g}' for x in xyz) + '\n')
    (directory / 'input.dat').write_text(f'atoms.xyz\n{side} {side} {side}\n90 90 90\n')
    (directory / 'atoms.dat').write_text('1\nC 3.4 52.84 12.011\n')
    (directory / 'defaults.dat').write_text(
        f'atoms.dat\n2.58 10.22 298 12.8\n0.0\n500\n{SPACING}\n'
        f'{MAX_DIAMETER} {BIN_WIDTH}\n{SEED}\n0\n')


def pb_run(exe, directory, timeout, match_full_rng=False):
    start = time.perf_counter()
    completed = subprocess.run([str(exe), 'input.dat'] + (['match-full-rng'] if match_full_rng else []), cwd=directory,
                               capture_output=True, text=True, check=True, timeout=timeout)
    wall = time.perf_counter() - start
    (directory / 'run.log').write_text(completed.stdout + completed.stderr)
    times = {}
    for key in ('LATTICE', 'PSD'):
        match = re.search(rf'BENCH_{key}\s+(\S+)', completed.stdout)
        if match:
            times[key.lower()] = float(match[1])
    match = re.search(r'BENCH_GRID\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', completed.stdout)
    if match:
        times['grid'] = [int(x) for x in match.groups()]
    times['wall'] = wall
    return times


def timed(call, device):
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = call()
    if device == 'cuda':
        torch.cuda.synchronize()
    return time.perf_counter() - start, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poreblazer-src', type=Path, required=True)
    parser.add_argument('--sizes', nargs='+', type=int, default=[64, 128, 256])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--devices', nargs='+', choices=['cpu', 'cuda'], default=['cpu', 'cuda'])
    parser.add_argument('--output', type=Path, default=Path('benchmarks/results/poreblazer'))
    args = parser.parse_args()
    if args.repeats < 1 or min(args.sizes) < 64:
        parser.error('Use repeats >= 1 and sizes >= 64 (PoreBlazer needs room for 10,000 samples)')
    if 'cuda' in args.devices and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --devices cpu')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = args.poreblazer_src.resolve()
    metadata = dict(poreblazer_commit=subprocess.check_output(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob('*.f90'))},
        compiler=subprocess.check_output(['gfortran', '--version'], text=True).splitlines()[0],
        compiler_flags='-cpp -O2', python=platform.python_version(), torch=torch.__version__,
        numpy=np.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        cpu=next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
        cpu_threads=1, sizes=args.sizes, repeats=args.repeats, summary='median after one warm-up',
        spacing_A=SPACING, radius_A=RADIUS, probe_radius_A=0., bin_width_A=BIN_WIDTH,
        min_radius_A=0., connected_pore_filter=False, structure_seed=42, poreblazer_seed=SEED,
        poreblazer_samples=10000, timeout_seconds=args.timeout)
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    rows, comparisons, raw = [], [], []
    with tempfile.TemporaryDirectory(prefix='poretorch-pb-') as temp:
        temp = Path(temp)
        isolated = build(source, temp / 'isolated', out, True)
        full = build(source, temp / 'full', out, False)
        for n in args.sizes:
            positions, cell = structure(n)
            directory = temp / f'run-{n}'
            inputs(directory, positions, cell)
            print(f'{n} atoms: PoreBlazer isolated', flush=True)
            # Persist inputs, curves and the final run log, but no build files.
            saved = out / str(n)
            saved.mkdir(exist_ok=True)
            for path in directory.iterdir():
                shutil.copy2(path, saved / path.name)
            measurements = [pb_run(isolated, directory, args.timeout) for _ in range(args.repeats + 1)][1:]
            raw.append(dict(code='PoreBlazer', n_atoms=n, runs=measurements))
            grid = measurements[0]['grid']
            assert np.prod(grid[:3]) >= 10000
            pb_curve = np.loadtxt(directory / 'Total_psd_cumulative.txt')
            for filename in ('Total_psd.txt', 'Total_psd_cumulative.txt', 'run.log'):
                shutil.copy2(directory / filename, saved / filename)
            rows.append(dict(code='PoreBlazer', device='cpu', method='covering_MC', weighting='voxel',
                n_atoms=n, n_grid_points=int(np.prod(grid[:3])), n_void=grid[3],
                field_seconds=float(np.median([x['lattice'] for x in measurements])),
                psd_seconds=float(np.median([x['psd'] for x in measurements])),
                compute_seconds=float(np.median([x['lattice']+x['psd'] for x in measurements])),
                pipeline_wall_seconds=float(np.median([x['wall'] for x in measurements]))))
            # Full-program sanity check at the smallest size only. Advance the
            # isolated RNG by the two draws per surface-area trial, outside timings.
            if n == min(args.sizes):
                print(f'{n} atoms: full PoreBlazer validation', flush=True)
                full_times = pb_run(full, directory, args.timeout)
                full_curve = np.loadtxt(directory / 'Total_psd_cumulative.txt')
                shutil.copy2(directory / 'run.log', saved / 'full_run.log')
                pb_run(isolated, directory, args.timeout, match_full_rng=True)
                matched_curve = np.loadtxt(directory / 'Total_psd_cumulative.txt')
                np.testing.assert_array_equal(full_curve, matched_curve)
                (out / 'full_validation.json').write_text(json.dumps(dict(
                    n_atoms=n, times=full_times, total_cumulative_psd_exact_match=True), indent=2)+'\n')
            for device in args.devices:
                print(f'{n} atoms: PoreTorch {device}', flush=True)
                def make_field():
                    return distance_field(positions, cell, radii=RADIUS, probe_radius=0.,
                        grid_spacing=SPACING, device=device, torch_dtype=torch.float64, masses=np.full(n,12.011))
                _, field = timed(make_field, device)
                assert list(field.grid.shape) == grid[:3]
                assert field.n_void == grid[3]
                field_times = [timed(make_field, device)[0] for _ in range(args.repeats)]
                for method, weighting in CONFIGS:
                    def calculate(field=field):
                        return pore_size_distribution(field, method=method, weighting=weighting,
                            bin_width=BIN_WIDTH, max_diameter=MAX_DIAMETER, min_radius=0.)
                    timed(calculate, device)  # warm-up
                    samples = [timed(calculate, device) for _ in range(args.repeats)]
                    psd_times = [x[0] for x in samples]
                    result = samples[-1][1]
                    def pipeline():
                        return calculate(make_field())
                    timed(pipeline, device)
                    pipelines = [timed(pipeline, device)[0] for _ in range(args.repeats)]
                    raw.append(dict(code='PoreTorch', n_atoms=n, device=device, method=method,
                                    weighting=weighting, field=field_times, psd=psd_times, pipeline=pipelines))
                    rows.append(dict(code='PoreTorch', device=device, method=method, weighting=weighting,
                        n_atoms=n, n_grid_points=field.grid.n_points, n_void=field.n_void,
                        field_seconds=float(np.median(field_times)), psd_seconds=float(np.median(psd_times)),
                        compute_seconds=float(np.median(pipelines)), pipeline_wall_seconds=float(np.median(pipelines))))
                    np.savez(saved / f'{device}_{method}_{weighting}.npz', **result.to_dict())
                    if method == 'covering':
                        # PB stores a survival curve at bin lower edges but labels
                        # each x coordinate half a bin lower. Compare at edges.
                        survival = np.r_[1., 1.-result.cumulative_fraction][:-1]
                        assert len(survival) == len(pb_curve)
                        comparisons.append(dict(n_atoms=n, device=device,
                            max_survival_difference=float(np.max(abs(survival-pb_curve[:,1])))))
            with (out / 'timings.csv').open('w') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (out / 'raw_timings.json').write_text(json.dumps(raw, indent=2)+'\n')
            (out / 'comparison.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    print(f'Results written to {out}', flush=True)


if __name__ == '__main__':
    main()
