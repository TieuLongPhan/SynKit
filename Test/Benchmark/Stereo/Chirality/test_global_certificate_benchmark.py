"""Protocol-boundary gates for the exact-plus-global ACS result."""

from Experiment.Stereo.Chirality.global_certificate_benchmark import (
    SCHEMA,
    classify_record,
)
from Experiment.Stereo.Chirality.published import load_dataset
from Experiment.Stereo.Chirality.global_stereo_performance import measure_case


def test_global_certificate_protocol_is_separate_and_label_blind() -> None:
    assert SCHEMA == "synkit.exact-plus-global-certificate-acs/1"
    records = {row["ID"]: row for row in load_dataset()}

    vs170 = classify_record(records["VS170"])
    vs215 = classify_record(records["VS215"])
    vs216 = classify_record(records["VS216"])
    vs300 = classify_record(records["VS300"])

    assert vs170["source_declared_status"] == "chiral"
    assert vs170["status"] == "chiral"
    assert vs170["correct"] is False
    assert vs215["status"] == "achiral"
    assert vs216["status"] == "achiral"
    assert vs300["source_declared_status"] == "achiral"
    assert vs300["global_necessarily_chiral"] is True
    assert vs300["status"] == "chiral"
    assert vs300["correct"] is True


def test_global_auxiliary_graph_size_and_near_miss_are_stable() -> None:
    positive = measure_case("C1C2(OCC1)OCCC2", repeats=1)
    near_miss = measure_case("C1C2(CCC1)CCCC2", repeats=1)

    assert positive["status"] == "chiral"
    assert near_miss["status"] == "achiral"
    assert positive["auxiliary_nodes"] == near_miss["auxiliary_nodes"] == 90
    assert positive["auxiliary_edges"] == near_miss["auxiliary_edges"] == 148
    assert positive["peak_traced_bytes"] > 0
    assert positive["source_declared_baseline"]["peak_traced_bytes"] > 0
    assert positive["auxiliary_node_growth"] == 10.0
