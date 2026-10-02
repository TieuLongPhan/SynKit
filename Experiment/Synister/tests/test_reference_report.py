import pytest
import json
from pathlib import Path

from Experiment.Synister.reference_report import summarize
from Experiment.Synister.reference_report import report, sha


def test_unresolved_weight_and_distinct_admission_statuses():
    rows = [
        {'status': 'evaluated', 'reference': {'mapping_in_minimum': True},
         'metrics': {'bond_f1': {'status': 'complete', 'reference_status': 'complete',
                    'reference_orbit_in_minimum': True,
                    'lower': {'difference': '-1/2'}, 'upper': {'difference': '1/4'}}}},
        {'status': 'evaluated', 'reference': {'mapping_in_minimum': True}, 'metrics': {}},
        {'status': 'hard_timeout'},
    ]
    result = summarize(rows)
    assert result['admission']['mapping'] == {'admitted': 2, 'unassessed': 1}
    assert result['admission']['bond'] == {'admitted': 1, 'unassessed': 2}
    assert result['admission']['atom'] == {'unassessed': 3}
    group = result['strata']['admitted']
    assert group['conditional_envelope'] == ['-1/2', '1/4']
    assert group['outer_envelope'] == ['-3/4', '5/8']
    assert group['local_reversal'] == 1
    assert result['strata']['unassessed']['conditional_envelope'] is None


def test_invalid_membership_is_not_treated_as_boolean():
    with pytest.raises(ValueError, match='Invalid reference membership'):
        summarize([{'status': 'evaluated', 'reference': {'mapping_in_minimum': 1}}])


def test_outside_reference_stratum_is_retained():
    result = summarize([{'status': 'evaluated',
                         'reference': {'mapping_in_minimum': False}}])
    assert result['strata']['outside']['selected'] == 1
    assert result['strata']['outside']['outer_envelope'] == ['-1', '1']


def test_changed_record_rejected_against_replay_binding(tmp_path):
    (tmp_path / 'cases').mkdir()
    (tmp_path / 'inputs.json').write_text(json.dumps([{'case_id': 'case_0'}]))
    (tmp_path / 'manifest.json').write_text(json.dumps(
        {'inputs_sha256': sha(tmp_path / 'inputs.json')}))
    record = tmp_path / 'cases/case_0.annotations.json'
    record.write_text(json.dumps({'case_id': 'case_0', 'status': 'hard_timeout'}))
    replay = tmp_path / 'replay.json'
    replay.write_text(json.dumps({'parent_manifest_sha256': sha(tmp_path / 'manifest.json'),
                                 'artifact_sha256': {str(record): sha(record)}}))
    assert report(tmp_path, replay)['strata']['unassessed']['selected'] == 1
    record.write_text('{}')
    with pytest.raises(ValueError, match='Record hash mismatch'):
        report(tmp_path, replay)


def test_all_saved_records_match_independent_stratum_arithmetic():
    from fractions import Fraction
    root = Path(__file__).resolve().parents[3] / 'paper/synister/evidence'
    directory = root / 'identifiability_c1_annotations_v1'
    result = report(directory, root / 'release_annotation_replay_v1/manifest.json')
    records = [json.loads(p.read_text()) for p in sorted((directory / 'cases').glob('*.annotations.json'))]
    assert len(records) == 1000
    for category, flag in [('admitted', True), ('outside', False), ('unassessed', None)]:
        group = [r for r in records if r.get('reference', {}).get('mapping_in_minimum') is flag]
        completed = [r['metrics']['bond_f1'] for r in group
                     if r.get('metrics', {}).get('bond_f1', {}).get('status') == 'complete']
        actual = result['strata'][category]
        assert actual['selected'] == len(group)
        assert actual['resolved'] == len(completed)
        for index, endpoint in enumerate(['lower', 'upper']):
            total = sum((Fraction(r[endpoint]['difference']) for r in completed), Fraction())
            missing = len(group)-len(completed)
            bound = (total + (-missing if index == 0 else missing))/len(group)
            assert Fraction(actual['outer_envelope'][index]) == bound
        assert actual['local_reversal'] == sum(Fraction(r['lower']['difference']) < 0 <
                                               Fraction(r['upper']['difference']) for r in completed)
