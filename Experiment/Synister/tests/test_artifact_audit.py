"""Ensure artifact replay detects changed witnesses and frozen predictions."""

from dataclasses import asdict
import json

import pytest

from Experiment.Synister.audit_development import audit, sha
from Experiment.Synister.development import save
from Experiment.Synister.worker import perform
from synkit.Chem.Mapper.prediction_adapter import align_mapped_prediction


@pytest.fixture
def fixture_archive(tmp_path):
    (tmp_path / "cases").mkdir()
    case = {"case_id": "c0", "reaction_id": "synthetic", "reaction": "CO>>CO", "status": "eligible"}
    save(tmp_path / "inputs.json", [case])
    save(tmp_path / "source_snapshot.json", {"fixture": "unit test, not a real campaign"})
    save(tmp_path / "manifest.json", {"inputs_sha256": sha(tmp_path / "inputs.json"),
                                     "source_snapshot_sha256": sha(tmp_path / "source_snapshot.json")})
    a = perform(dict(case, stage="slap"))
    raw = "[CH3:1][OH:2]>>[CH3:1][OH:2]"
    b = {"status": "valid", "raw_prediction": {"mapped_rxn": raw},
         "prediction": asdict(align_mapped_prediction(case["reaction"], raw)), "label": a["label"]}
    save(tmp_path / "cases/c0.slap.json", a)
    save(tmp_path / "cases/c0.rxnmapper.json", b)
    save(tmp_path / "prediction_freeze.json", {
        f"c0.{method}": sha(tmp_path / f"cases/c0.{method}.json") for method in ("slap", "rxnmapper")})
    exact = perform(dict(case, stage="exact", search_seconds=5))
    score = perform(dict(case, stage="score", score_seconds=5, labels=exact["labels"],
                         prediction_a=a["prediction"]["mapping"], prediction_b=b["prediction"]["mapping"]))
    save(tmp_path / "cases/c0.exact.json", exact)
    save(tmp_path / "cases/c0.score.json", score)
    save(tmp_path / "summary.json", {
        "selected": 1, "eligible": 1, "common_valid_predictions": 1,
        "resolved_comparisons": 1, "exact_searches_closed": 1, "positive_paired_width_cases": 0,
        "resolved_conditional_envelope": ["0", "0"], "common_valid_outer_envelope": ["0", "0"],
        "unresolved_weight": "0"})
    return tmp_path


def test_independent_artifact_replay(fixture_archive):
    assert audit(fixture_archive)["summary"]["resolved_comparisons"] == 1


def test_changed_extremum_rejected(fixture_archive):
    path = fixture_archive / "cases/c0.score.json"
    value = json.loads(path.read_text())
    value["lower"]["difference"] = "1/2"
    path.write_text(json.dumps(value))
    with pytest.raises(AssertionError):
        audit(fixture_archive)


def test_modified_frozen_prediction_rejected(fixture_archive):
    path = fixture_archive / "cases/c0.rxnmapper.json"
    value = json.loads(path.read_text())
    value["prediction"]["mapping"] = [1, 0]
    path.write_text(json.dumps(value))
    with pytest.raises(AssertionError):
        audit(fixture_archive)
