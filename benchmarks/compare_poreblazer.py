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
use lattice, only: ncubesx, ncubesy, ncubesz, ng_cubes, cube_size
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
write(*,*) 'BENCH_SPACING', cube_size
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


def carbon_structure(path):
    """Load a cubic carbon cell without applying LAMMPS image flags or scaling."""
    from ase.io import read

    atoms = read(path, format='lammps-data', atom_style='atomic', read_image_flags=False)
    cell = np.asarray(atoms.cell)
    if not np.all(atoms.numbers == 6):
        raise ValueError(f'{path}: only carbon structures are supported')
    if not np.all(atoms.pbc) or not np.allclose(cell, np.eye(3)*cell[0, 0], rtol=0, atol=1e-12):
        raise ValueError(f'{path}: a fully periodic cubic cell is required')
    positions = atoms.positions.copy()
    positions -= positions.min(axis=0)
    return positions, cell


def matched_spacing(cell, grid, pb_spacing):
    """Match PoreBlazer's actual grid without changing the structure or cell."""
    actual = cell[0, 0] / grid[0]
    if grid[:3] != [grid[0]] * 3 or not np.isclose(actual, pb_spacing, rtol=1e-12, atol=0):
        raise ValueError('PoreBlazer grid does not tile the cubic cell; cannot compare identical grids')
    # Grid.from_cell uses ceil; guard against roundoff in perpendicular widths.
    return actual * (1 + 1e-12)


def poreblazer_spacing(cell, target=SPACING):
    """Avoid PB's second integer truncation dropping a grid plane by roundoff."""
    side = float(cell[0, 0])
    count = int(side / target)
    if count < 1:
        raise ValueError('Grid spacing exceeds the cell length')
    if int(side / (side / count)) == count:
        return target
    # Prefer a finer grid over losing a cell plane; never rescale the cell.
    while int(side / (side / count)) != count:
        count += 1
    return side / count


def inputs(directory, positions, cell):
    directory.mkdir()
    side = cell[0, 0]
    spacing = poreblazer_spacing(cell)
    if int(side / spacing)**3 < 10000:
        raise ValueError('PoreBlazer needs at least 10,000 grid points for its sample buffer')
    with (directory / 'atoms.xyz').open('w') as handle:
        handle.write(f'{len(positions)}\nperiodic carbon benchmark\n')
        for xyz in positions:
            handle.write('C ' + ' '.join(f'{x:.17g}' for x in xyz) + '\n')
    (directory / 'input.dat').write_text(f'atoms.xyz\n{side} {side} {side}\n90 90 90\n')
    (directory / 'atoms.dat').write_text('1\nC 3.4 52.84 12.011\n')
    (directory / 'defaults.dat').write_text(
        f'atoms.dat\n2.58 10.22 298 12.8\n0.0\n500\n{spacing}\n'
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
    match = re.search(r'BENCH_SPACING\s+(\S+)', completed.stdout)
    if match:
        times['spacing_A'] = float(match[1])
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
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--sizes', nargs='+', type=int, default=[64, 128, 256])
    selection.add_argument('--structures', nargs='+', type=Path, help='Cubic carbon LAMMPS data files, run in the supplied order')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--devices', nargs='+', choices=['cpu', 'cuda'], default=['cpu', 'cuda'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or (not args.structures and min(args.sizes) < 64):
        parser.error('Use repeats >= 1 and sizes >= 64 (PoreBlazer needs room for 10,000 samples)')
    if 'cuda' in args.devices and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --devices cpu')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cases = []
    if args.structures:
        for path in args.structures:
            path = path.resolve()
            positions, cell = carbon_structure(path)
            cases.append((len(positions), positions, cell, path))
    else:
        for n in args.sizes:
            positions, cell = structure(n)
            cases.append((n, positions, cell, None))
    if len({case[0] for case in cases}) != len(cases):
        parser.error('Each input must have a distinct atom count for output directories')
    out = (args.output or Path('benchmarks/results/carbon' if args.structures else 'benchmarks/results/poreblazer')).resolve()
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
        cpu_threads=1, sizes=[case[0] for case in cases], repeats=args.repeats, summary='median after one warm-up',
        spacing_A=SPACING, radius_A=RADIUS, probe_radius_A=0., bin_width_A=BIN_WIDTH,
        min_radius_A=0., connected_pore_filter=False, structure_seed=None if args.structures else 42, poreblazer_seed=SEED,
        poreblazer_samples=10000, timeout_seconds=args.timeout)
    metadata['structures'] = [dict(n_atoms=n, cell_A=cell.tolist(),
        source=str(path) if path else 'synthetic',
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path else None)
        for n, _, cell, path in cases]
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    rows, comparisons, raw = [], [], []
    with tempfile.TemporaryDirectory(prefix='poretorch-pb-') as temp:
        temp = Path(temp)
        isolated = build(source, temp / 'isolated', out, True)
        full = build(source, temp / 'full', out, False)
        for n, positions, cell, source_path in cases:
            directory = temp / f'run-{n}'
            inputs(directory, positions, cell)
            print(f'{n} atoms: PoreBlazer isolated', flush=True)
            # Persist inputs, curves and the final run log, but no build files.
            saved = out / str(n)
            saved.mkdir(exist_ok=True)
            if source_path and source_path != saved / 'source.data':
                shutil.copy2(source_path, saved / 'source.data')
            for path in directory.iterdir():
                shutil.copy2(path, saved / path.name)
            measurements = [pb_run(isolated, directory, args.timeout) for _ in range(args.repeats + 1)][1:]
            raw.append(dict(code='PoreBlazer', n_atoms=n, runs=measurements))
            grid = measurements[0]['grid']
            assert np.prod(grid[:3]) >= 10000
            spacing = matched_spacing(cell, grid, measurements[0]['spacing_A'])
            (saved / 'grid.json').write_text(json.dumps(dict(shape=grid[:3], n_void=grid[3],
                requested_spacing_A=SPACING, poreblazer_requested_spacing_A=poreblazer_spacing(cell),
                actual_spacing_A=measurements[0]['spacing_A'],
                poretorch_requested_spacing_A=spacing), indent=2)+'\n')
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
            if n == min(case[0] for case in cases):
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
                        grid_spacing=spacing, device=device, torch_dtype=torch.float64, masses=np.full(n,12.011))
                _, field = timed(make_field, device)
                assert list(field.grid.shape) == grid[:3]
                assert field.n_void == grid[3]
                field_times = [timed(make_field, device)[0] for _ in range(args.repeats)]
                for method, weighting in CONFIGS:
                    print(f'  {method}/{weighting}', flush=True)
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
