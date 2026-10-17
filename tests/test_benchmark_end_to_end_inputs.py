"""Both programs receive the same prepared geometry."""

import numpy as np
import pytest
from ase.io import read

from benchmarks.benchmark_end_to_end import prepare


@pytest.mark.parametrize('side', [16.27341908881698, 21.69789211842264])
def test_common_native_input_preserves_geometry_and_configuration(tmp_path, side):
    case = tmp_path/'case'
    case.mkdir()
    original = {
        'atoms.xyz': '2\ncarbon\nC 0 0 0\nC 1.2345678901234567 2.5 3.75\n',
        'input.dat': f'atoms.xyz\n{side} {side} {side}\n90 90 90\n',
        'atoms.dat': '1\nC 3.4 52.84 12.011\n',
        'defaults.dat': 'atoms.dat\n2.58 10.22 298 12.8\n0\n500\n0.5\n30 0.25\n21908391\n0\n',
    }
    for name, text in original.items():
        (case/name).write_text(text)
    target = tmp_path/'prepared'
    prepare(case, target)
    atoms = read(target/'atoms.extxyz')
    np.testing.assert_array_equal(atoms.cell.array, np.eye(3)*side)
    np.testing.assert_array_equal(atoms.positions, [[0, 0, 0], [1.2345678901234567, 2.5, 3.75]])
    assert atoms.pbc.all()
    assert atoms.get_chemical_symbols() == ['C', 'C']
    pb_input = (target/'input.dat').read_text().splitlines()
    assert pb_input[0] == 'atoms.extxyz'
    assert pb_input[1:] == original['input.dat'].splitlines()[1:]
    for name in ('atoms.dat', 'defaults.dat'):
        assert (target/name).read_text() == original[name]
    for name, text in original.items():
        assert (case/name).read_text() == text


def _timing(run_id, seconds):
    return dict(code='PoreTorch', device='cuda', method='covering', weighting='voxel',
                n_atoms=216, run_id=run_id, seconds=seconds)


def test_structure_mean_and_sample_variation():
    from benchmarks.benchmark_end_to_end import aggregate_timings

    result, = aggregate_timings([_timing(1, 1), _timing(2, 2), _timing(3, 9)], [1, 2, 3])
    assert result['n_structures'] == 3
    assert result['seconds'] == 4
    assert result['seconds_std'] == pytest.approx(np.sqrt(19))


def test_incomplete_structure_group_is_not_reported():
    from benchmarks.benchmark_end_to_end import aggregate_timings

    assert aggregate_timings([_timing(1, 1), _timing(2, 2)], [1, 2, 3]) == []


def test_duplicate_structure_cannot_replace_missing_run():
    from benchmarks.benchmark_end_to_end import aggregate_timings

    with pytest.raises(ValueError, match='duplicate'):
        aggregate_timings([_timing(1, 1), _timing(2, 2), _timing(2, 3)], [1, 2, 3])
