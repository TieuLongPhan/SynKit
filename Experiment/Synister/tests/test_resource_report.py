from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

from Experiment.Synister.resource_report import summarize


def test_resource_summary_keeps_failed_attempts_and_missing_measurements():
    result = summarize([
        {"status": "complete", "parent_seconds": 2, "worker_seconds": 1,
         "peak_rss_kib": 100},
        {"status": "hard_timeout", "parent_seconds": 65},
        {"status": "error", "parent_seconds": 3, "worker_seconds": 2,
         "peak_rss_kib": 300},
    ])
    assert result["attempts"] == 3
    assert result["parent_seconds"]["median"] == 3
    assert result["parent_seconds"]["maximum"] == 65
    assert result["peak_rss_kib"] == {
        "observed": 2, "missing": 1, "median": 200, "maximum": 300}
    assert result["status_counts"]["hard_timeout"] == 1
    assert summarize([])["worker_seconds"]["median"] is None


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_invalid_resource_measurements_are_refused(value):
    with pytest.raises(ValueError):
        summarize([{"status": "complete", "parent_seconds": value}])


@pytest.mark.parametrize("cohort,selected,attempts", [
    ("c1", 1000, 4000), ("c2", 500, 1991)])
def test_archived_resources_and_manuscript_table(cohort, selected, attempts):
    root = Path(__file__).resolve().parents[3]
    evidence = root / "paper/synister/evidence"
    saved = json.loads((evidence / f"{cohort}_resources_v1.json").read_text())
    parent = evidence / f"identifiability_{cohort}_primary_v1"
    paths = sorted((parent / "cases").glob("*.json"))
    assert len(paths) == attempts == len(saved["record_hashes"])
    assert saved["selected_inputs"] == selected
    assert saved["primary_audit_sha256"] == hashlib.sha256(
        (parent / "audit.json").read_bytes()).hexdigest()
    records = []
    for path in paths:
        assert saved["record_hashes"][path.name] == hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(json.loads(path.read_text()))
    tex = (root / "paper/synister/supplementary.tex").read_text()
    names = {"slap": "SLAP", "rxnmapper": "RXNMapper",
             "exact": "Search", "score": "Scoring"}
    for stage, name in names.items():
        rows = [r for r in records if r["stage"] == stage]
        summary = saved["stages"][stage]
        assert summary["attempts"] == len(rows)
        assert summary["status_counts"] == dict(Counter(r["status"] for r in rows))
        for field in ("parent_seconds", "worker_seconds", "peak_rss_kib"):
            values = sorted(r[field] for r in rows if field in r)
            n = len(values)
            middle = (values[(n-1)//2] + values[n//2])/2 if n else None
            assert summary[field] == {
                "observed": n, "missing": len(rows)-n, "median": middle,
                "maximum": values[-1] if n else None}
        time = summary["parent_seconds"]
        line = (f'{cohort.upper()} & {name} & {len(rows)} & '
                f'{time["median"]:.3f} & {time["maximum"]:.3f} & '
                f'{summary["peak_rss_kib"]["maximum"]}')
        assert line in tex
