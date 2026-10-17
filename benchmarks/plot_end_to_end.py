"""Plot mean carbon benchmark times and covering speedups."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

METHODS = [
    ('all_void', 'voxel', '#0072B2', 'o'),
    ('covering', 'voxel', '#D55E00', 'D'),
    ('packed', 'sphere', '#009E73', '^'),
]


def plot(directory):
    with (directory/'timings.csv').open() as f:
        rows = list(csv.DictReader(f))
    if not rows or any(int(r['n_structures']) != 3 for r in rows):
        raise ValueError('plots require three structures per result')
    sizes = sorted({int(r['n_atoms']) for r in rows})

    def series(code, method, weighting):
        selected = {int(r['n_atoms']): r for r in rows
                    if r['code'] == code and r['method'] == method
                    and r['weighting'] == weighting
                    and (code == 'PoreBlazer' or r['device'] == 'cuda')}
        return ([float(selected[n]['seconds']) for n in sizes],
                [float(selected[n]['seconds_std']) for n in sizes])

    fig, (ax, ax_ratio) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for method, weighting, color, marker in METHODS:
        times, deviations = series('PoreTorch', method, weighting)
        ax.errorbar(sizes, times, yerr=deviations, color=color, marker=marker,
                    capsize=3, linewidth=1.7, label=f'PoreTorch GPU {method}')
    pb, pb_std = series('PoreBlazer', 'full', 'voxel')
    ax.errorbar(sizes, pb, yerr=pb_std, color='#222222', marker='X', capsize=3,
                linewidth=1.7, label='PoreBlazer CPU')
    ax.set_ylabel('Time (s)')
    ax.set_title('Mean time across three structures')
    ax.legend(frameon=False, fontsize=8)

    covering, _ = series('PoreTorch', 'covering', 'voxel')
    ratios = [baseline / time for baseline, time in zip(pb, covering)]
    ax_ratio.plot(sizes, ratios, color='#D55E00', marker='D', linewidth=1.7)
    ax_ratio.axhline(1, color='#777777', linewidth=1, linestyle=':')
    ax_ratio.set_ylabel('Speedup (×)')
    ax_ratio.set_title('Covering relative to PoreBlazer')
    for axis in (ax, ax_ratio):
        axis.set_xscale('log')
        axis.set_yscale('log')
        axis.set_xticks(sizes, [f'{n:,}' for n in sizes])
        axis.minorticks_off()
        axis.set_xlabel('Carbon atoms')
        axis.grid(alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
    ax_ratio.set_yscale('linear')
    ax_ratio.set_ylim(.8, max(ratios) * 1.1)
    ax_ratio.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax_ratio.yaxis.set_major_formatter(ScalarFormatter())
    fig.tight_layout()
    for extension in ('svg', 'png'):
        path = directory/f'scaling.{extension}'
        fig.savefig(path, dpi=180, bbox_inches='tight')
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)
    with (directory/'speedups.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['n_atoms', 'poreblazer_over_poretorch'])
        writer.writeheader()
        writer.writerows(dict(n_atoms=n, poreblazer_over_poretorch=ratio)
                         for n, ratio in zip(sizes, ratios))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path,
                        default=Path('benchmarks/results/carbon_three_runs'))
    plot(parser.parse_args().results)
