from copy import deepcopy
import json
from pathlib import Path

import pytest

from Experiment.Synister.replay_annotations import compare
from synkit.Chem.Mapper.annotation_evaluation import analyze_annotations
from synkit.Chem.Mapper.identifiability import Endpoint


def test_replay_detects_policy_reference_and_extremum_tampering():
    r = Endpoint((6, 6), (0, 1), (4, 3), ())
    result = analyze_annotations(r, r, [(0, 1), (1, 0)], (0, 1), (1, 0),
                                 joint_labels_complete=True, minimum=0, reference_mapping=(0, 1))
    compare(result, deepcopy(result))
    bad = deepcopy(result)
    bad["policies"]["canonical_its"]["difference"] = "1/7"
    with pytest.raises(AssertionError):
        compare(result, bad)
    bad = deepcopy(result)
    bad["metrics"]["atom_f1"]["reference_scores"]["a"] = "1/7"
    with pytest.raises(AssertionError):
        compare(result, bad)
    bad = deepcopy(result)
    bad["metrics"]["atom_f1"]["lower"]["difference"] = "1/7"
    with pytest.raises(AssertionError):
        compare(result, bad)


def test_c1_replay_preserves_unreplayable_timeout(monkeypatch, tmp_path):
    from Experiment.Synister import replay_annotations as runner
    root = Path(__file__).resolve().parents[3] / "paper/synister/evidence"
    seen = []
    def fake_execute(task, deadline, cases):
        assert task["stage"] == "annotation_replay"
        assert task["old"]["status"] == "evaluated"
        assert deadline == 180
        assert (cases.parent / "manifest.json").exists()
        seen.append(task["case_id"])
        return {"status": "verified", "reference_mapping_in_minimum": True}
    monkeypatch.setattr(runner, "execute", fake_execute)
    monkeypatch.setattr(runner, "source_contents", lambda: {"test": "fixed"})
    runner.run(root / "identifiability_c1_annotations_v1",
               root / "identifiability_c1_selection_v1/references.json",
               tmp_path / "replay", root / "identifiability_c1_primary_v1")
    summary = json.loads((tmp_path / "replay/summary.json").read_text())
    assert len(seen) == len(set(seen)) == 999
    assert "case_0002" not in seen
    assert summary["not_replayable"] == 1
    assert summary["all_attempted_verified"]
    assert not summary["all_verified"]
