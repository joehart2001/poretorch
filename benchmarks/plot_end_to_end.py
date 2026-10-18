"""Plot carbon benchmark times, omitting pending measurements."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS = [
    ('all_void', 'voxel', '#0072B2', 'o'),
    ('covering', 'voxel', '#D55E00', 'D'),
    ('packed', 'sphere', '#009E73', '^'),
]


def plot(directory):
    with (directory/'timings.csv').open() as f:
        rows = list(csv.DictReader(f))
    if not rows or any(int(r['n_structures']) not in (1, 3) for r in rows):
        raise ValueError('expected one or three structures per result')
    sizes = sorted({int(r['n_atoms']) for r in rows})

    def series(code, method, weighting):
        selected = {int(r['n_atoms']): r for r in rows
                    if r['code'] == code and r['method'] == method
                    and r['weighting'] == weighting
                    and (code == 'PoreBlazer' or r['device'] == 'cuda')}
        return ([float(selected[n]['seconds']) if n in selected and selected[n]['seconds'] != ''
                 else float('nan') for n in sizes],
                [float(selected[n]['seconds_std']) if n in selected and selected[n]['seconds_std'] != ''
                 else float('nan') for n in sizes])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, weighting, color, marker in METHODS:
        times, deviations = series('PoreTorch', method, weighting)
        ax.errorbar(sizes, times, yerr=deviations, color=color, marker=marker,
                    capsize=3, linewidth=1.7, label=f'PoreTorch GPU {method}')
    pb, pb_std = series('PoreBlazer', 'full', 'voxel')
    ax.errorbar(sizes, pb, yerr=pb_std, color='#222222', marker='X', capsize=3,
                linewidth=1.7, label='PoreBlazer CPU')
    ax.set_ylabel('Time (s)')
    ax.set_title('PSD timings')
    ax.legend(frameon=False, fontsize=8)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(sizes, [f'{n:,}' for n in sizes])
    ax.minorticks_off()
    ax.set_xlabel('Carbon atoms')
    ax.grid(alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    for extension in ('svg', 'png'):
        path = directory/f'scaling.{extension}'
        temporary = directory/f'.scaling.tmp.{extension}'
        fig.savefig(temporary, dpi=180, bbox_inches='tight')
        if extension == 'svg':
            temporary.write_text('\n'.join(line.rstrip() for line in temporary.read_text().splitlines())+'\n')
        temporary.replace(path)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path,
                        default=Path('benchmarks/report'))
    plot(parser.parse_args().results)
