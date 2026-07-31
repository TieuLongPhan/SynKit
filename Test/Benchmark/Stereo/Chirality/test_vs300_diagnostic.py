"""Permanent gates for the VS300 exact/topology mismatch explanation."""

from Experiment.Stereo.Chirality.vs300_diagnostic import (
    build_minimal_fixtures,
    build_vs300_diagnostic,
)


def test_vs300_two_input_contracts_are_explicit_and_reproducible() -> None:
    report = build_vs300_diagnostic()

    assert report["baseline_mismatch"] == {
        "exact_configured_chemical": "achiral",
        "acs_reference": "chiral",
        "classification": "global_topology_capability_boundary",
    }
    contracts = report["input_contracts"]
    assert contracts["rdkit_sanitized"]["configured_atom_tags"] == []
    assert [
        item["atom"]
        for item in contracts["source_tags_restored"]["configured_atom_tags"]
    ] == [2, 4, 18]

    paths = report["classification_paths"]
    assert paths["exact_source_declared"]["status"] == "achiral"
    assert paths["sanitized_without_topology_completion"]["status"] == "Achiral"
    assert paths["sanitized_with_topology_completion"]["status"] == "Chiral"
    assert paths["sanitized_with_topology_completion"]["completed_centers"] == [
        2,
        4,
        18,
    ]
    assert paths["restored_with_topology_completion"]["status"] == "Achiral"
    assert paths["restored_with_topology_completion"]["completed_centers"] == []


def test_vs300_mirror_witness_exposes_coupled_cage_frame_difference() -> None:
    report = build_vs300_diagnostic()

    comparison = report["descriptor_comparison"]
    assert comparison["shared_center_relations"] == [
        {"center": 2, "relation": "opposite"},
        {"center": 4, "relation": "equivalent"},
        {"center": 18, "relation": "opposite"},
    ]

    audit = report["representative_mirror_witness_audit"]
    assert audit["source_restored"]["blocking_source_centers"] == []
    assert audit["topology_completion"]["blocking_source_centers"] == [18, 4]


def test_minimal_global_fixture_pair_and_achiral_near_miss() -> None:
    report = build_minimal_fixtures()
    fixtures = {fixture["fixture_id"]: fixture for fixture in report["fixtures"]}

    positive = fixtures["global-minimal-positive"]
    assert positive["atom_count"] == 9
    assert positive["observed"] == {
        "exact_supplied": "achiral",
        "without_topology_completion": "Achiral",
        "with_topology_completion": "Chiral",
        "completed_centers": [2],
    }

    near_miss = fixtures["global-minimal-achiral-near-miss"]
    assert near_miss["atom_count"] == 9
    assert near_miss["observed"]["with_topology_completion"] == "Achiral"
    assert report["configured_pair"]["relation"] == "enantiomers"
