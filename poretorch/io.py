"""Writing distributions to disk in the usual plain formats."""

import json
from pathlib import Path

import numpy as np


def maybe_save(outdir, output, distribution) -> Path | None:
    """
    Persist a distribution when a destination was requested.

    Parameters
    ----------
    outdir
        Optional directory. A ``psd.npz`` is written inside it.
    output
        Optional explicit path. The suffix chooses the format: ``.npz``,
        ``.csv`` / ``.tsv``, or ``.json``. Anything else falls back to ``.npz``.
    distribution
        The :class:`poretorch.psd.PSD` to write.

    Returns
    -------
    Path or None
        Where the file went, or ``None`` if nothing was requested.
    """
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
    elif outdir:
        directory = Path(outdir)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "psd.npz"
    else:
        return None

    suffix = target.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        _save_table(target, distribution, delimiter="," if suffix == ".csv" else "\t")
    elif suffix == ".json":
        _save_json(target, distribution)
    else:
        np.savez(target, **_arrays(distribution))
    return target


def _arrays(distribution) -> dict:
    """
    Collect a distribution's array and scalar payload for ``np.savez``.

    Parameters
    ----------
    distribution
        The distribution to flatten.

    Returns
    -------
    dict
        Name to value mapping of arrays and scalars.
    """
    return {
        key: value
        for key, value in distribution.to_dict().items()
        if value is not None
    }


def _columns(distribution):
    """
    Build the tabular view of a distribution: one row per diameter bin.

    Parameters
    ----------
    distribution
        The distribution to tabulate.

    Returns
    -------
    tuple[list[str], np.ndarray]
        Column names and the stacked column data.
    """
    names = ["diameter_A", "diameter_nm", "volume_A3", "cumulative_fraction"]
    columns = [
        distribution.diameters_A,
        distribution.diameters_nm,
        distribution.volume_A3,
        distribution.cumulative_fraction,
    ]
    if distribution.mass_g:
        names.append("dV_dd_cc_per_nm_per_g")
        columns.append(distribution.dV_dd_cc_per_nm_per_g)
    names.append("normalised_A-1")
    columns.append(distribution.normalised)
    if distribution.se_volume_A3 is not None:
        names.append("se_volume_A3")
        columns.append(distribution.se_volume_A3)
    return names, np.column_stack(columns)


def _save_table(target: Path, distribution, delimiter: str) -> None:
    """
    Write the distribution as a delimited text table with a header row.

    Parameters
    ----------
    target
        Destination path.
    distribution
        The distribution to write.
    delimiter
        Field separator.
    """
    names, data = _columns(distribution)
    np.savetxt(
        target,
        data,
        delimiter=delimiter,
        header=delimiter.join(names),
        comments="",
    )


def _save_json(target: Path, distribution) -> None:
    """
    Write the distribution as JSON, converting arrays to lists.

    Parameters
    ----------
    target
        Destination path.
    distribution
        The distribution to write.
    """
    payload = {}
    for key, value in distribution.to_dict().items():
        if isinstance(value, np.ndarray):
            payload[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            payload[key] = value.item()
        else:
            payload[key] = value
    with open(target, "w") as handle:
        json.dump(payload, handle, indent=2)
