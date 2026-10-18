"""Refresh the benchmark report as completed measurements become available."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import statistics
import time

CASES = [
    ('PoreBlazer', 'cpu', 'full', 'voxel', 'PoreBlazer CPU'),
    ('PoreTorch', 'cuda', 'all_void', 'voxel', 'PoreTorch GPU `all_void`'),
    ('PoreTorch', 'cuda', 'covering', 'voxel', 'PoreTorch GPU `covering`'),
    ('PoreTorch', 'cuda', 'packed', 'sphere', 'PoreTorch GPU `packed`'),
]
FIELDS = ['code', 'device', 'method', 'weighting', 'n_atoms', 'n_structures',
          'seconds', 'seconds_std', 'status']


def optional_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def collect_rows(small, large):
    with (small/'timings.csv').open() as f:
        rows = [dict(row, status='complete') for row in csv.DictReader(f)]
    for code, device, method, weighting, _ in CASES:
        row = dict(code=code, device=device, method=method, weighting=weighting,
                   n_atoms=32768, n_structures=1, seconds='', seconds_std='', status='pending')
        directory = large/('poretorch' if code == 'PoreTorch' else 'poreblazer')
        metadata = optional_json(directory/'metadata.json')
        records = optional_json(directory/'raw_timings.json') or []
        if metadata is not None:
            if metadata['inputs']['n_atoms'] != 32768 or metadata['n_structures'] != 1:
                raise ValueError('unexpected large-structure input')
            for record in records:
                if any(record[key] != row[key] for key in ('code', 'device', 'method', 'weighting', 'n_atoms')):
                    continue
                samples = record['samples']
                if len(samples) == metadata['repeats'] and all(math.isfinite(t) and t > 0 for t in samples):
                    row.update(seconds=statistics.median(samples), status='complete')
        rows.append(row)
    return rows


def update_table(text, rows):
    sizes = sorted({int(row['n_atoms']) for row in rows})
    table = ['| Time (s) | ' + ' | '.join(f'{n:,} atoms' for n in sizes) + ' |',
             '|---|' + '---:|'*len(sizes)]
    for code, device, method, weighting, label in CASES:
        values = []
        for size in sizes:
            match = next((r for r in rows if r['code'] == code and r['device'] == device
                          and r['method'] == method and r['weighting'] == weighting
                          and int(r['n_atoms']) == size), None)
            values.append(f"{float(match['seconds']):.3f}" if match and match['seconds'] != '' else '-')
        table.append('| ' + label + ' | ' + ' | '.join(values) + ' |')
    lines = text.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.startswith('| Time (s) |'))
    end = start
    while end < len(lines) and lines[end].startswith('|'):
        end += 1
    return ''.join(lines[:start]) + '\n'.join(table) + '\n' + ''.join(lines[end:])


def write_atomic(path, text):
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def refresh(args, rows):
    from benchmarks.plot_end_to_end import plot

    args.output.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    write_atomic(args.output/'timings.csv', stream.getvalue())
    plot(args.output)
    write_atomic(args.readme, update_table(args.readme.read_text(), rows))
    completed = sum(int(r['n_atoms']) == 32768 and r['status'] == 'complete' for r in rows)
    print(f'{datetime.now(timezone.utc).isoformat()} updated report: {completed}/4 large-case results complete', flush=True)


def main(args):
    if args.interval <= 0:
        raise ValueError('interval must be positive')
    previous = None
    while True:
        try:
            rows = collect_rows(args.small_results, args.large_results)
            signature = json.dumps(rows, sort_keys=True)
            if signature != previous:
                refresh(args, rows)
                previous = signature
            states = [optional_json(args.large_results/program/'status.json')
                      for program in ('poretorch', 'poreblazer')]
            if not args.watch:
                return
            if all(state and state['state'] in ('complete', 'failed') for state in states):
                final_rows = collect_rows(args.small_results, args.large_results)
                if json.dumps(final_rows, sort_keys=True) != previous:
                    refresh(args, final_rows)
                return
        except (json.JSONDecodeError, FileNotFoundError) as error:
            # Writers may be replacing a checkpoint while this process reads it.
            if not args.watch:
                raise
            print(f'Waiting for a readable checkpoint: {error}', flush=True)
        time.sleep(args.interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--small-results', type=Path, default=Path('benchmarks/results'))
    parser.add_argument('--large-results', type=Path, default=Path('benchmarks/results/32768/run_1'))
    parser.add_argument('--output', type=Path, default=Path('benchmarks/report'))
    parser.add_argument('--readme', type=Path, default=Path('README.md'))
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=30.)
    main(parser.parse_args())
