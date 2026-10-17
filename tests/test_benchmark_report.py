"""Only complete timing samples should enter the published report."""
import json

from benchmarks.update_report import collect_rows, update_table


def test_partial_repeat_is_pending_but_completed_method_is_published(tmp_path):
    small = tmp_path/'small'
    small.mkdir()
    (small/'timings.csv').write_text(
        'code,device,method,weighting,n_atoms,n_structures,seconds,seconds_std\n'
        'PoreBlazer,cpu,full,voxel,216,3,1.0,0.1\n')
    large = tmp_path/'large'
    for program, code, device, method, weighting, samples in (
        ('poretorch', 'PoreTorch', 'cuda', 'all_void', 'voxel', [4., 5., 6.]),
        ('poreblazer', 'PoreBlazer', 'cpu', 'full', 'voxel', [100., 110.]),
    ):
        directory = large/program
        directory.mkdir(parents=True)
        (directory/'metadata.json').write_text(json.dumps(
            dict(inputs=dict(n_atoms=32768), n_structures=1, repeats=3)))
        (directory/'raw_timings.json').write_text(json.dumps([
            dict(code=code, device=device, method=method, weighting=weighting,
                 n_atoms=32768, samples=samples)]))
    rows = collect_rows(small, large)
    pb = next(r for r in rows if r['code']=='PoreBlazer' and int(r['n_atoms'])==32768)
    pt = next(r for r in rows if r['method']=='all_void')
    assert pb['status']=='pending' and pb['seconds']==''
    assert pt['status']=='complete' and pt['seconds']==5.
    assert pt['n_structures']==1 and pt['seconds_std']==''
    path = large/'poreblazer/raw_timings.json'
    records = json.loads(path.read_text())
    records[0]['samples'].append(120.)
    path.write_text(json.dumps(records))
    updated = collect_rows(small, large)
    pb = next(r for r in updated if r['code']=='PoreBlazer' and int(r['n_atoms'])==32768)
    assert pb['status']=='complete' and pb['seconds']==110.


def test_report_uses_hyphens_and_preserves_surrounding_text():
    text = 'Intro\n\n| Time (s) | old |\n|---|---|\n| old | 2 |\n\n## Quick start\nKeep this.\n'
    rows = [dict(code='PoreBlazer', device='cpu', method='full', weighting='voxel',
                 n_atoms=32768, seconds='')]
    result = update_table(text, rows)
    assert '| PoreBlazer CPU | - |' in result
    assert '| PoreTorch GPU `all_void` | - |' in result
    assert result.startswith('Intro\n\n')
    assert result.endswith('\n## Quick start\nKeep this.\n')
