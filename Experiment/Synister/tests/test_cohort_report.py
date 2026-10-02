import json
from fractions import Fraction

import pytest

from Experiment.Synister.audit_development import sha
from Experiment.Synister.cohort_report import report


def fixture_archive(path, confirmation, replication=False):
    def save(name, value):
        (path / name).write_text(json.dumps(value))
    save("execution_lock.json", {"margin": "1/50"})
    save("manifest.json", {"execution_lock_sha256": sha(path / "execution_lock.json")})
    summary = {"common_valid_predictions": 2, "scope":
               "prospective_C2_replication" if replication else
               "prospective_C1_confirmation" if confirmation else "development_only"}
    save("summary.json", summary)
    save("audit.json", {"summary": summary,
         "summary_sha256": sha(path / "summary.json"),
         "manifest_sha256": sha(path / "manifest.json"),
         "resolved_rows": [{"lower": "-1/2", "upper": "1/2", "bond_label_orbits": 2}]})


def test_confirmation_scope_and_fixed_margin(tmp_path):
    fixture_archive(tmp_path, True)
    result = report(tmp_path, Fraction(1, 50))
    assert result["margin_status"] == "fixed pre-outcome C1 margin"
    assert result["bounds"]["unresolved_weight"] == "1/2"
    with pytest.raises(ValueError, match="margin"):
        report(tmp_path, Fraction(0))
    (tmp_path / "execution_lock.json").write_text('{"margin": "0"}')
    with pytest.raises(ValueError, match="hash"):
        report(tmp_path, Fraction(0))


def test_development_keeps_exploratory_margin(tmp_path):
    fixture_archive(tmp_path, False)
    result = report(tmp_path, Fraction(0))
    assert result["margin_status"] == "proposed development interpretation, not a confirmation decision"


def test_replication_scope_and_fixed_margin(tmp_path):
    fixture_archive(tmp_path, True, replication=True)
    result = report(tmp_path, Fraction(1, 50))
    assert result["margin_status"] == "fixed pre-outcome C2 margin inherited from C1"
    assert "C2" in result["scope"]
    assert result["bounds"]["unresolved_weight"] == "1/2"
    with pytest.raises(ValueError, match="margin"):
        report(tmp_path, Fraction(0))
    (tmp_path / "execution_lock.json").write_text('{"margin": "0"}')
    with pytest.raises(ValueError, match="hash"):
        report(tmp_path, Fraction(0))
