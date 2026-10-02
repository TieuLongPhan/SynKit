"""Historical paper evidence checks; run explicitly with the local records."""

import json
from pathlib import Path

import pytest

from scripts.run_synister_alternative_its import _payload_sha256
from scripts.summarize_synister_evidence import summarize

ROOT = Path(__file__).resolve().parents[3]


def local_evidence(relative):
    """Locate optional local evidence outside the library regression suite."""
    path = ROOT / relative
    if not path.exists():
        pytest.skip(f"local historical evidence unavailable: {relative}")
    return path


def test_frozen_pilot_has_reproducible_alternative_its_application_yield():
    campaign = local_evidence("paper/synister/evidence/pilot100_v4")
    modes = summarize(campaign)["modes"]

    assert modes["minimal"]["alternative_its_application_complete_cases"] == 46
    assert modes["minimal"]["cases_with_alternative_its"] == 16
    assert modes["minimal"]["alternative_its_classes_relative_to_reference"] == 39
    assert modes["reference_cd"]["alternative_its_application_complete_cases"] == 52
    assert modes["reference_cd"]["cases_with_alternative_its"] == 18
    assert modes["reference_cd"]["alternative_its_classes_relative_to_reference"] == 54


def test_historical_30_second_campaign_is_recomputed_from_verified_records():
    campaign = local_evidence("Experiment/Synister/benchmark_results/synister_global_shells_flower10k_v4_30s")
    modes = summarize(campaign)["modes"]

    minimum = modes["minimal"]
    reference = modes["reference_cd"]
    assert minimum["cases"] == minimum["structure_complete"] == 244
    assert minimum["multiple_exact_its_classes"] == 81
    assert minimum["reference_its_class_observed"] == 231
    assert reference["cases"] == 280
    assert reference["structure_complete"] == 279
    assert reference["multiple_exact_its_classes"] == 103
    assert reference["reference_its_class_observed"] == 279


def test_frozen_alternative_its_case_payload_and_semantics_replay():
    record_path = local_evidence("paper/synister/evidence/alternative_its_case_v1/record.json")
    record = json.loads(record_path.read_text(encoding="ascii"))
    claimed = record.pop("record_sha256")

    assert claimed == _payload_sha256(record)
    assert record["implementation_sha256"] == (
        "d48db5c30262d1c1a327002e7f89c32ac76b54ede52f0bba19c11b6ab3389856"
    )
    grouped = {}
    identifiers = {}
    for query in record["queries"]:
        shell = query["result"]["shell"]
        target = str(query["requested_target"])
        grouped.setdefault(target, set()).add(
            (
                shell["status"],
                shell["shell_labeled_mapping_count"],
                shell["shell_its_class_count"],
                shell["reference_its_class_observed"],
                shell["alternative_its_class_count"],
            )
        )
        identifiers.setdefault(target, set()).add(
            tuple(sorted(item["its_class_id"] for item in shell["alternatives"]))
        )
    assert all(len(values) == 1 for values in identifiers.values())
    assert grouped["minimal"] == {("complete", 8, 2, False, 2)}
    assert grouped["reference"] == {("complete", 16, 4, True, 3)}
    assert grouped["4.0"] == {("no_solutions", 0, 0, False, 0)}
    assert grouped["10.0"] == {("complete", 52, 13, False, 13)}
    assert grouped["12.0"] == {("complete", 180, 45, False, 45)}
