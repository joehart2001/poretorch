"""A quick look at a pore-size distribution.

Pore sizes span roughly a decade, from sub-nanometre micropores to mesopores, so
the diameter axis defaults to logarithmic. A linear axis crowds everything
interesting into its left edge.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_psd(
    distributions,
    labels=None,
    y: str = "auto",
    path=None,
    show: bool = False,
    log_x: bool = True,
    xlim=None,
    title=None,
    ax=None,
):
    """
    Plot one or several pore-size distributions on shared axes.

    Parameters
    ----------
    distributions
        A :class:`poretorch.psd.PSD`, or an iterable of them to overlay.
    labels
        Optional legend labels, one per distribution. Defaults to each
        distribution's method name.
    y
        Which curve to draw: ``"dV_dd"`` for cc/nm/g, ``"normalised"`` for unit
        area, ``"volume"`` for the raw per-bin volume, or ``"auto"`` to use
        ``dV_dd`` when masses are known and ``normalised`` otherwise.
    path
        Optional output path for the figure.
    show
        Whether to display the plot interactively.
    log_x
        Whether the diameter axis is logarithmic.
    xlim
        Optional ``(low, high)`` diameter limits in Angstrom.
    title
        Optional plot title.
    ax
        Optional existing axes to draw into.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.
    """
    if hasattr(distributions, "diameters_A"):
        distributions = [distributions]
    distributions = list(distributions)
    if not distributions:
        raise ValueError("nothing to plot")

    if labels is None:
        labels = [dist.method for dist in distributions]
    if len(labels) != len(distributions):
        raise ValueError(
            f"got {len(labels)} labels for {len(distributions)} distributions"
        )

    mode = _resolve_mode(y, distributions)
    fig = plt.figure() if ax is None else ax.figure
    ax = fig.gca() if ax is None else ax

    ylabel = ""
    for dist, label in zip(distributions, labels):
        values, errors, ylabel = _curve(dist, mode)
        # A log diameter axis cannot show the zero-diameter edge of the first
        # bin, and empty leading bins would stretch the axis for nothing.
        keep = dist.diameters_A > 0
        x = dist.diameters_A[keep]
        line, = ax.plot(x, values[keep], label=label)
        if errors is not None and np.any(errors):
            ax.fill_between(
                x,
                (values - errors)[keep],
                (values + errors)[keep],
                color=line.get_color(),
                alpha=0.2,
                linewidth=0,
            )

    if log_x:
        ax.set_xscale("log")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Pore diameter (Å)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if len(distributions) > 1:
        ax.legend()

    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    if show:  # pragma: no cover (interactive only)
        plt.show()
    else:
        plt.close(fig)
    return fig


def _resolve_mode(y: str, distributions) -> str:
    """
    Decide which y quantity to plot.

    Parameters
    ----------
    y
        Requested mode, possibly ``"auto"``.
    distributions
        The distributions to be plotted.

    Returns
    -------
    str
        A concrete mode: ``"dV_dd"``, ``"normalised"`` or ``"volume"``.
    """
    if y == "auto":
        return (
            "dV_dd"
            if all(dist.mass_g for dist in distributions)
            else "normalised"
        )
    if y not in {"dV_dd", "normalised", "volume"}:
        raise ValueError(
            f'y must be "auto", "dV_dd", "normalised" or "volume", got {y!r}'
        )
    return y


def _curve(distribution, mode: str):
    """
    Extract the requested curve, its error band, and an axis label.

    Parameters
    ----------
    distribution
        The distribution to read.
    mode
        ``"dV_dd"``, ``"normalised"`` or ``"volume"``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None, str]
        Values, standard error (or ``None``), and the y-axis label.
    """
    if mode == "dV_dd":
        return (
            distribution.dV_dd_cc_per_nm_per_g,
            distribution.se_dV_dd_cc_per_nm_per_g,
            "dV/dd (cc nm$^{-1}$ g$^{-1}$)",
        )
    if mode == "normalised":
        return (
            distribution.normalised,
            distribution.se_normalised,
            "Normalised PSD (Å$^{-1}$)",
        )
    return distribution.volume_A3, distribution.se_volume_A3, "Pore volume (Å$^3$)"
