import json

import numpy as np
import pytest
import torch

from poretorch.adapters import analyse, average_psds, psd, read_structures

CPU = {"device": "cpu", "torch_dtype": torch.float64, "grid_spacing": 1.0}


def test_analyse_returns_a_consistent_bundle(carbon_atoms):
    result = analyse(carbon_atoms, method="all_void", **CPU)
    assert result.field.n_void > 0
    assert result.labels.n_pores > 0
    assert result.psd.volume_A3.sum() == pytest.approx(result.field.void_volume, rel=1e-9)

    summary = result.summary()
    assert summary["method"] == "all_void"
    assert 0.0 < summary["porosity"] < 1.0
    assert summary["specific_void_volume_cc_per_g"] > 0.0
    assert summary["n_pores"] == result.labels.n_pores


def test_ase_atoms_supply_symbols_and_masses(carbon_atoms):
    """Radii come from the symbols and dV/dd per gram works without extra input."""
    result = analyse(carbon_atoms, **CPU)
    assert result.field.radii[0].item() == pytest.approx(3.431 / 2)
    assert result.psd.mass_g == pytest.approx(carbon_atoms.get_masses().sum() * 1.66053906660e-24)
    assert np.all(np.isfinite(result.psd.dV_dd_cc_per_nm_per_g))


def test_psd_of_one_frame_has_no_error_band(carbon_atoms):
    result = psd(carbon_atoms, **CPU)
    assert result.n_frames == 1
    assert result.se_volume_A3 is None
    assert result.se_normalised is None


def test_psd_averages_over_frames(carbon_atoms):
    rng = np.random.default_rng(41)
    frames = []
    for _ in range(3):
        frame = carbon_atoms.copy()
        frame.positions += rng.normal(scale=0.3, size=frame.positions.shape)
        frames.append(frame)

    result = psd(frames, **CPU)
    assert result.n_frames == 3
    assert result.se_volume_A3 is not None
    assert result.se_volume_A3.shape == result.volume_A3.shape
    assert np.all(result.se_volume_A3 >= 0.0)
    assert result.se_normalised is not None


def test_averaging_zero_extends_rather_than_truncating(carbon_atoms):
    """A frame with a bigger pore must not be clipped to a shorter histogram."""
    short = psd(carbon_atoms, bin_width=0.5, max_diameter=3.0, **CPU)
    long = psd(carbon_atoms, bin_width=0.5, max_diameter=9.0, **CPU)
    combined = average_psds([short, long])

    assert combined.volume_A3.size == long.volume_A3.size
    assert combined.volume_A3.sum() == pytest.approx(
        0.5 * (short.volume_A3.sum() + long.volume_A3.sum())
    )


def test_averaging_rejects_mismatched_distributions(carbon_atoms):
    coarse = psd(carbon_atoms, bin_width=0.5, **CPU)
    fine = psd(carbon_atoms, bin_width=0.25, **CPU)
    with pytest.raises(ValueError, match="bin widths"):
        average_psds([coarse, fine])

    local = psd(carbon_atoms, method="all_void", **CPU)
    packed = psd(carbon_atoms, method="packed", **CPU)
    with pytest.raises(ValueError, match="methods"):
        average_psds([local, packed])


def test_averaged_stats_carry_standard_errors(carbon_atoms):
    rng = np.random.default_rng(42)
    frames = []
    for _ in range(4):
        frame = carbon_atoms.copy()
        frame.positions += rng.normal(scale=0.2, size=frame.positions.shape)
        frames.append(frame)
    result = psd(frames, **CPU)
    assert "median_diameter_A" in result.stats
    assert "median_diameter_A_se" in result.stats
    assert result.stats["median_diameter_A_se"] >= 0.0


def test_read_structures_normalises_every_input(carbon_atoms, tmp_path):
    from ase.io import write

    assert len(read_structures(carbon_atoms)) == 1
    assert len(read_structures([carbon_atoms, carbon_atoms])) == 2

    path = tmp_path / "structure.extxyz"
    write(str(path), [carbon_atoms, carbon_atoms])
    assert len(read_structures(path)) == 2
    assert len(read_structures(str(path), index=0)) == 1


def test_read_structures_rejects_nonsense():
    with pytest.raises(TypeError, match="expected an ASE Atoms"):
        read_structures([1, 2, 3])
    with pytest.raises(ValueError, match="no frames"):
        read_structures([])


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such structure file"):
        read_structures(tmp_path / "absent.xyz")


def test_read_kwargs_reach_ase(carbon_atoms, tmp_path):
    """A LAMMPS .data file needs an explicit format, since ASE guesses wrong."""
    from ase.io import write

    path = tmp_path / "structure.data"
    write(str(path), carbon_atoms, format="lammps-data", atom_style="atomic")

    frames = read_structures(
        path, read_kwargs={"format": "lammps-data", "atom_style": "atomic"}
    )
    assert len(frames) == 1
    assert len(frames[0]) == len(carbon_atoms)


def test_a_misguessed_format_explains_itself(carbon_atoms, tmp_path):
    """ASE returns no frames rather than raising, so we must not hide that."""
    from ase.io import write

    path = tmp_path / "structure.data"
    write(str(path), carbon_atoms, format="lammps-data", atom_style="atomic")

    with pytest.raises(ValueError, match="read no frames"):
        read_structures(path)
    with pytest.raises(ValueError, match="lammps-data"):
        read_structures(path)


def test_read_kwargs_flow_through_the_entry_points(carbon_atoms, tmp_path):
    from ase.io import write

    path = tmp_path / "structure.data"
    write(str(path), carbon_atoms, format="lammps-data", atom_style="atomic")
    read_kwargs = {"format": "lammps-data", "atom_style": "atomic"}

    from_path = psd(path, read_kwargs=read_kwargs, **CPU)
    assert from_path.volume_A3.sum() > 0.0
    assert analyse(path, read_kwargs=read_kwargs, **CPU).labels.n_pores > 0


def test_a_cell_is_required():
    from ase import Atoms

    atoms = Atoms("C2", positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="no cell"):
        analyse(atoms, **CPU)


def test_probe_by_name_reduces_the_accessible_void(carbon_atoms):
    bare = analyse(carbon_atoms, probe_radius=0.0, **CPU)
    probed = analyse(carbon_atoms, probe_radius="N2", **CPU)
    assert probed.field.probe_radius == pytest.approx(1.657)
    assert probed.field.n_void <= bare.field.n_void


@pytest.mark.parametrize("suffix", [".npz", ".csv", ".json"])
def test_output_formats_round_trip(carbon_atoms, tmp_path, suffix):
    target = tmp_path / f"psd{suffix}"
    result = psd(carbon_atoms, output=str(target), **CPU)
    assert target.exists()

    if suffix == ".npz":
        loaded = np.load(target, allow_pickle=True)
        assert np.allclose(loaded["diameter_A"], result.diameters_A)
        assert np.allclose(loaded["volume_A3"], result.volume_A3)
    elif suffix == ".csv":
        table = np.loadtxt(target, delimiter=",", skiprows=1)
        assert table.shape[0] == result.diameters_A.size
        assert np.allclose(table[:, 0], result.diameters_A)
    else:
        payload = json.loads(target.read_text())
        assert np.allclose(payload["volume_A3"], result.volume_A3)
        assert payload["method"] == result.method


def test_outdir_writes_a_default_filename(carbon_atoms, tmp_path):
    psd(carbon_atoms, outdir=str(tmp_path / "out"), **CPU)
    assert (tmp_path / "out" / "psd.npz").exists()


def test_plotting_produces_a_figure(carbon_atoms, tmp_path):
    from poretorch.plotting import plot_psd

    local = psd(carbon_atoms, method="all_void", **CPU)
    covering = psd(carbon_atoms, method="covering", **CPU)

    target = tmp_path / "psd.png"
    figure = plot_psd([local, covering], labels=["all-void", "covering"], path=target)
    assert target.exists()
    assert len(figure.gca().lines) == 2

    with pytest.raises(ValueError, match="labels for"):
        plot_psd([local, covering], labels=["only one"])
    with pytest.raises(ValueError, match='y must be'):
        plot_psd(local, y="bogus")
