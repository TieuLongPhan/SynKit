import json
from pathlib import Path

import pytest

from Experiment.Synister import storage_report


def test_partial_missing_and_empty_exports(tmp_path, monkeypatch):
    cases = tmp_path / 'cases'
    cases.mkdir()
    rows = [{'stage': 'exact', 'status': 'unresolved', 'labels': [],
             'joint_labels': [], 'emitted_representatives': 0},
            {'stage': 'exact', 'status': 'hard_timeout'}]
    for index, row in enumerate(rows):
        (cases / f'{index}.json').write_text(json.dumps(row))
    monkeypatch.setattr(storage_report, 'resource_report', lambda _: {
        'primary_audit_sha256': 'fixture',
        'record_hashes': {p.name: storage_report.sha(p) for p in cases.iterdir()}})
    result = storage_report.report(tmp_path)['stages']['exact']
    assert result['files'] == 2
    assert result['bytes'] == sum(p.stat().st_size for p in cases.iterdir())
    assert result['labels'] == {'observed': 1, 'missing': 1, 'total': 0, 'maximum': 0}


@pytest.mark.parametrize('cohort', ['c1', 'c2'])
def test_archive_storage_counts(cohort):
    root = Path(__file__).resolve().parents[3]
    directory = root / f'paper/synister/evidence/identifiability_{cohort}_primary_v1'
    result = storage_report.report(directory)
    saved = json.loads((root / f'paper/synister/evidence/{cohort}_storage_v1.json').read_text())
    assert result == saved
    tex = (root / 'paper/synister/supplementary.tex').read_text()
    exact = result['stages']['exact']
    assert f'{exact["bytes"]:,} bytes for {cohort.upper()}' in tex
    for field in ('labels', 'joint_labels', 'emitted_representatives'):
        assert f'{exact[field]["total"]:,}' in tex
    for stage, values in result['stages'].items():
        paths = list((directory / 'cases').glob(f'*.{stage}.json'))
        assert values['files'] == len(paths)
        assert values['bytes'] == sum(path.stat().st_size for path in paths)
        if stage == 'exact':
            rows = [json.loads(path.read_text()) for path in paths]
            for field in ('labels', 'joint_labels', 'emitted_representatives'):
                counts = [len(row[field]) if field != 'emitted_representatives'
                          else row[field] for row in rows if row.get(field) is not None]
                assert values[field] == {'observed': len(counts),
                    'missing': len(rows)-len(counts), 'total': sum(counts),
                    'maximum': max(counts)}
