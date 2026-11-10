"""Protect retained evidence and refusal behavior after the paper cleanup."""
from pathlib import Path

import pytest

from Experiment.Stereo.paths import CORE, recorded_path
from Experiment.Stereo.run import rerun, validate_synthetic, verify


def test_relocated_paper_evidence_and_all_registration_partitions_verify():
    result = verify(replay=True)
    assert result['native_result_files'] == 15
    assert result['independent']['assignments_and_renumberings'] == 4878
    assert result['independent']['positive_mirror_witnesses_replayed'] == 370


def test_original_absolute_paths_resolve_without_changing_report_content():
    old = '/home/lolo/Documents/TACsy/SynEco/SynKit/Experiment/Stereo/Data/Revision/NativeScientificCpp_PhysicalV2_20261102/execution/independent.json'
    assert recorded_path(old) == CORE/'execution/independent.json'


def test_incorrect_relation_result_is_rejected(tmp_path,monkeypatch):
    import Experiment.Stereo.run as runner
    original=runner.load
    def changed(path):
        report=original(path)
        if Path(path).name=='relations_cpp.json':
            report['records'][0]['correct']=False
        return report
    monkeypatch.setattr(runner,'load',changed)
    with pytest.raises(ValueError,match='relation classifications'):
        verify()


def test_synthetic_negatives_require_completed_invariant_native_outcomes():
    inputs={'rows':[{'id':'negative','proof_kind':'no_supported_axis'}]}
    report={'records':[{'id':'negative','complete':True,'erased_renumbering_invariant':True,
                        'configuration_erased':{'elements':[]}}]}
    assert validate_synthetic(inputs, report)['passed']==1
    report['records'][0]['erased_renumbering_invariant']=False
    with pytest.raises(ValueError, match='failed'):
        validate_synthetic(inputs, report)


def test_rerun_refuses_to_overwrite_retained_or_existing_output(tmp_path):
    with pytest.raises(ValueError, match='new output directory'):
        rerun(Path('/no-build-needed'), tmp_path)
