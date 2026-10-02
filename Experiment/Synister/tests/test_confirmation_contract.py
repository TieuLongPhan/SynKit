from copy import deepcopy
import pytest
from Experiment.Synister.confirmation_contract import SETTINGS, check_settings, validate_sources


def test_locked_settings_reject_resource_and_method_changes():
    check_settings(SETTINGS)
    for name in SETTINGS:
        altered = deepcopy(SETTINGS)
        altered[name] = None
        with pytest.raises(ValueError, match=name):
            check_settings(altered)


def test_source_drift_is_rejected():
    with pytest.raises(ValueError, match="implementation changed"):
        validate_sources({"all_sources_sha256": "0"*64})


@pytest.mark.parametrize("count,protocol,scope", [
    (1000,"identifiability-c1-confirmation-v1","prospective_C1_confirmation"),
    (500,"identifiability-c2-rhea-v1","prospective_C2_replication")])
def test_confirmation_pipeline_barriers_joint_exports_and_failures(tmp_path, monkeypatch, count, protocol, scope):
    import csv
    import gzip
    import json
    import sys
    from Experiment.Synister import development, confirmation_contract

    dataset, output = tmp_path / "inputs.csv.gz", tmp_path / "campaign"
    with gzip.open(dataset, "wt") as stream:
        writer = csv.DictWriter(stream, fieldnames=("source_line", "reaction_id", "mapped_reaction"))
        writer.writeheader()
        writer.writerows(dict(source_line=str(i), reaction_id=f"{i}:1", mapped_reaction="C>>C") for i in range(count))
    lockfile = tmp_path / "lock.json"
    lockfile.write_text("{}")
    lock = {"protocol": protocol, "all_sources_sha256": "fixture", "settings":dict(SETTINGS,limit=count)}
    monkeypatch.setattr(confirmation_contract, "validate_launch", lambda args: lock)
    monkeypatch.setattr(confirmation_contract, "validate_sources", lambda value: None)
    monkeypatch.setattr(confirmation_contract, "source_contents", lambda: {})
    monkeypatch.setattr(development, "snapshot", lambda path: "fixture")
    monkeypatch.setattr(development, "environment", lambda: {})
    stages = []

    def execute(task, timeout, directory):
        stage, key = task["stage"], task["case_id"]
        if stage in ("slap", "rxnmapper"):
            status = "invalid_prediction" if (key, stage) == ("case_0000", "rxnmapper") else "valid"
            value = dict(status=status, prediction={"mapping": [0]})
        elif stage == "exact":
            assert len([x for x in stages if x in ("slap", "rxnmapper")]) == 2*count
            assert (output / "prediction_freeze.json").exists()
            assert task["export_joint_labels"] is True and timeout == 65
            value = dict(status="unresolved" if key == "case_0001" else "complete", labels=[{"mapping": [0]}])
        else:
            assert stage == "score" and stages.count("exact") == count and timeout == 35
            value = dict(status="complete", lower={"difference": "0"}, upper={"difference": "0"}, width="0")
        stages.append(stage)
        return dict(value, case_id=key, stage=stage)

    monkeypatch.setattr(development, "execute", execute)
    args = ["development", "--dataset", str(dataset), "--output", str(output), "--confirmation-lock", str(lockfile)]
    for name, value in dict(SETTINGS,limit=count).items():
        args.extend(["--"+name.replace("_", "-"), str(value)])
    monkeypatch.setattr(sys, "argv", args)
    development.main()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["scope"] == scope
    assert summary["selected"] == count and summary["common_valid_predictions"] == count-1
    assert summary["resolved_comparisons"] == count-2
    assert summary["unresolved_weight"] == f"1/{count-1}"
