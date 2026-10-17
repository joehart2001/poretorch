"""Plot saved carbon benchmark timings and covering speedups."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter

METHODS = [
    ('all_void', 'voxel', 'all_void', '#0072B2', 'o'),
    ('covering', 'voxel', 'covering', '#D55E00', 'D'),
    ('packed', 'sphere', 'packed', '#009E73', '^'),
]
TIMINGS = [
    ('launch_to_saved_psd_seconds', 'New process'),
    ('warm_session_to_saved_psd_seconds', 'Warm start'),
]


def plot(directory):
    with (directory/'timings.csv').open() as f:
        rows = list(csv.DictReader(f))
    sizes = sorted({int(r['n_atoms']) for r in rows})
    pb = {int(r['n_atoms']): float(r[TIMINGS[0][0]])
          for r in rows if r['code'] == 'PoreBlazer'}

    def values(device, method, weighting, field):
        found = {int(r['n_atoms']): float(r[field]) for r in rows
                 if r['code'] == 'PoreTorch' and r['device'] == device
                 and r['method'] == method and r['weighting'] == weighting}
        return [found.get(n, float('nan')) for n in sizes]

    def axes_style(ax, ylabel):
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(sizes, [f'{n:,}' for n in sizes])
        ax.minorticks_off()
        ax.set_xlabel('Carbon atoms')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)

    def save(fig, name):
        for extension in ('svg', 'png'):
            fig.savefig(directory/f'{name}.{extension}', dpi=180, bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, (field, title) in zip(axes, TIMINGS):
        for method, weighting, label, color, marker in METHODS:
            ax.plot(sizes, values('cuda', method, weighting, field),
                    color=color, marker=marker, linewidth=1.7,
                    markersize=5, label=f'GPU {label}')
        ax.plot(sizes, [pb[n] for n in sizes], color='#222222', marker='X',
                linewidth=2, label='PoreBlazer CPU (new process)')
        axes_style(ax, 'Time to saved PSD (s)')
        ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
               fontsize=8.5, bbox_to_anchor=(.5, -.09))
    fig.suptitle('PSD runtime · amorphous carbon · median of three runs')
    fig.tight_layout(rect=(0, .12, 1, .96))
    save(fig, 'scaling')

    fig, ax = plt.subplots(figsize=(7.5, 4))
    speedups = []
    for (field, title), dash, marker in zip(TIMINGS, ['--', '-'], ['s', 'o']):
        times = values('cuda', 'covering', 'voxel', field)
        ratios = [pb[n]/t for n, t in zip(sizes, times)]
        ax.plot(sizes, ratios, color='#D55E00', linestyle=dash, marker=marker,
                label=f'GPU · {title.lower()}')
        speedups.extend(dict(n_atoms=n, device='cuda', start=title,
                             poreblazer_over_poretorch=ratio)
                        for n, ratio in zip(sizes, ratios) if math.isfinite(ratio))
    ax.axhline(1, color='#777777', linewidth=1, linestyle=':')
    axes_style(ax, 'PoreBlazer time / PoreTorch time (×)')
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_title('Covering PSD · speed relative to PoreBlazer')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, 'covering_speedup')
    with (directory/'speedups.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(speedups[0]))
        writer.writeheader()
        writer.writerows(speedups)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path,
                        default=Path('benchmarks/results/carbon_end_to_end'))
    plot(parser.parse_args().results)
