"""Plots for the PoreTorch benchmarks."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "cell_list": {"color": "#0072B2", "marker": "o", "label": "PoreTorch (cell list)"},
    "brute": {"color": "#D55E00", "marker": "s", "label": "PoreTorch (brute force)"},
}


def plot_scaling(rows, path_stem):
    """
    Plot field time against atom count for each backend, plus the speed-up.

    Parameters
    ----------
    rows
        Benchmark records with keys ``backend``, ``n_atoms``, ``seconds``.
    path_stem
        Output path without an extension. PNG and SVG are both written.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    fig, (ax, ax_ratio) = plt.subplots(1, 2, figsize=(11, 4.4))

    per_backend = {}
    for row in rows:
        per_backend.setdefault(row["backend"], []).append(row)

    for backend, records in per_backend.items():
        records.sort(key=lambda r: r["n_atoms"])
        style = STYLE.get(backend, {"label": backend})
        ax.plot(
            [r["n_atoms"] for r in records],
            [r["seconds"] for r in records],
            linewidth=2,
            **style,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Atoms")
    ax.set_ylabel("Clearance field time (s)")
    ax.set_title("Distance-field cost at fixed density")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    fast = {r["n_atoms"]: r["seconds"] for r in per_backend.get("cell_list", [])}
    slow = {r["n_atoms"]: r["seconds"] for r in per_backend.get("brute", [])}
    shared = sorted(set(fast) & set(slow))
    if shared:
        ax_ratio.plot(
            shared,
            [slow[n] / fast[n] for n in shared],
            color="#009E73",
            marker="D",
            linewidth=2,
        )
        ax_ratio.axhline(1.0, color="black", linestyle=":", linewidth=1)
    ax_ratio.set_xscale("log")
    ax_ratio.set_xlabel("Atoms")
    ax_ratio.set_ylabel("Speed-up over brute force")
    ax_ratio.set_title("Benefit of pruning the atom loop")
    ax_ratio.grid(alpha=0.3, which="both")

    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(f"{path_stem}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_methods(rows, path_stem):
    """
    Plot the cost of each pore-size definition against atom count.

    Parameters
    ----------
    rows
        Benchmark records with keys ``method``, ``n_atoms``, ``seconds``.
    path_stem
        Output path without an extension.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    colors = {"all_void": "#984EA3", "covering": "#0072B2", "packed": "#009E73"}
    # Name the overlap character in the legend, since it is the sharpest
    # difference between the definitions.
    labels = {
        "all_void": "all_void (overlapping)",
        "covering": "covering (overlapping)",
        "packed": "packed (non-overlapping)",
    }

    per_method = {}
    for row in rows:
        per_method.setdefault(row["method"], []).append(row)

    for method, records in per_method.items():
        records.sort(key=lambda r: r["n_atoms"])
        ax.plot(
            [r["n_atoms"] for r in records],
            [r["seconds"] for r in records],
            marker="o",
            linewidth=2,
            color=colors.get(method),
            label=labels.get(method, method),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Atoms")
    ax.set_ylabel("Distribution time (s)")
    ax.set_title("Cost of each pore-size definition")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(f"{path_stem}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig
