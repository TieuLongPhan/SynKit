"""One-sided observations retain typed ports without asserting material loss."""

import json
from pathlib import Path

import pytest

from synkit.Rule import WildcardRuleExtractor, WildcardRuleExtractionError
from synkit.Synthesis.Reactor import WildcardEngine
from synkit.Synthesis.Reactor.core.completion import validate_lewis_endpoints
from Test.Synthesis.Reactor.helpers import canonical_fragments

CASES = json.loads((Path(__file__).parent / "fixtures/wildcard_observed_examples.json").read_text())["cases"]


@pytest.fixture(scope="module")
def native_library(reactor_native_library):
    return reactor_native_library


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("invert", [False, True], ids=["fw", "bw"])
@pytest.mark.parametrize("partial", [False, True], ids=["full", "partial"])
def test_observation_rule_replays_both_endpoints_with_typed_ports(native_library, case, invert, partial):
    extracted = WildcardRuleExtractor(endpoint_policy="observed").extract(case["reaction"])
    assert extracted.to_dict()["endpoint_policy"] == "observed"
    source, target = map(canonical_fragments, case["reaction"].split(">>"))
    if invert:
        source, target = target, source
    results = []
    for backend in ("python", "cpp"):
        engine = WildcardEngine(
            ".".join(source),
            extracted.rule,
            invert=invert,
            partial=partial,
            backend=backend,
            library_path=native_library,
            electron_diagnostics=True,
        )
        outcomes = {canonical_fragments(value.split(">>")[0 if invert else 1]) for value in engine.smarts_list}
        assert target in outcomes
        results.append(outcomes)
        assert engine.rule.rc.raw.graph["endpoint_policy"] == "observed"
        assert all(set(m) <= set(engine.rule.left.raw) for m in engine.mappings)
        for graph in engine.its_list:
            validate_lewis_endpoints(graph, engine.rule.rc.raw, observation=True)
            assert any(not all(a.get("present", (True, True))) for _, a in graph.nodes(data=True))
        if case.get("local_model", True):
            assert all(report["mismatch_count"] == 0 for report in engine.diagnostics)
    assert results[0] == results[1]


@pytest.mark.parametrize("invert", [False, True])
def test_observation_native_kernel_has_no_python_execution_fallback(native_library, monkeypatch, invert):
    from synkit.Synthesis.Reactor.core import completion, rewrite, product
    from synkit.Graph.Matcher.partial_matcher import PartialMatcher
    from synkit.Graph.Matcher.subgraph_matcher import SubgraphSearchEngine

    case = CASES[0]
    rule = WildcardRuleExtractor(endpoint_policy="observed").extract(case["reaction"]).rule
    source = canonical_fragments(case["reaction"].split(">>")[1 if invert else 0])

    def forbidden(*args, **kwargs):
        raise AssertionError("Native observation execution called a Python kernel.")

    monkeypatch.setattr(completion, "complete_rule_host", forbidden)
    monkeypatch.setattr(rewrite, "_glue_graph", forbidden)
    monkeypatch.setattr(product, "_reperceive_product_kekule_phase", forbidden)
    monkeypatch.setattr(PartialMatcher, "get_mappings", forbidden)
    monkeypatch.setattr(SubgraphSearchEngine, "find_subgraph_mappings", forbidden)
    engine = WildcardEngine(".".join(source), rule, invert=invert, backend="cpp", library_path=native_library)
    assert engine.its_list
    assert engine.smarts_list


def test_observation_policy_is_explicit_and_survives_double_reversal():
    with pytest.raises(WildcardRuleExtractionError, match="same observed atoms"):
        WildcardRuleExtractor().extract(CASES[0]["reaction"])
    rule = WildcardRuleExtractor(endpoint_policy="observed").extract(CASES[0]["reaction"]).rule
    assert rule.reversed().reversed().rc.raw.graph["endpoint_policy"] == "observed"
    assert rule.rc.raw.graph["wildcard_extraction"]["ports"]


def test_observed_port_domains_still_reject_a_wrong_element(native_library):
    rule = (
        WildcardRuleExtractor(endpoint_policy="observed")
        .extract("[CH3:1][NH:2][C:3](=[O:4])[O:5][CH3:6]>>[CH3:1][NH2:2]")
        .rule
    )
    # The persistent substituent root on the nitrogen was observed to be C.
    for backend in ("python", "cpp"):
        engine = WildcardEngine("ON", rule, invert=True, partial=False, backend=backend, library_path=native_library)
        assert not engine.smarts_list


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_empty_observed_center_is_the_identity(native_library, backend):
    rule = WildcardRuleExtractor(endpoint_policy="observed").extract("[CH4:1]>>[CH4:1]").rule
    engine = WildcardEngine("C", rule, backend=backend, library_path=native_library)
    assert len(engine.mappings) == 1
    assert {canonical_fragments(s.split(">>")[1]) for s in engine.smarts_list} == {("C",)}


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("invert", [False, True])
def test_first_native_witness_is_valid_and_marks_search_incomplete(native_library, case, invert):
    rule = WildcardRuleExtractor(endpoint_policy="observed").extract(case["reaction"]).rule
    sides = [canonical_fragments(side) for side in case["reaction"].split(">>")]
    engine = WildcardEngine(
        ".".join(sides[int(invert)]),
        rule,
        backend="cpp",
        library_path=native_library,
        invert=invert,
        partial=False,
        first_application=True,
    )
    assert len(engine.smarts_list) == 1
    assert canonical_fragments(engine.smarts_list[0].split(">>")[0 if invert else 1]) == sides[1 - int(invert)]
    assert engine.native_statistics["search_complete"] is False
    for graph in engine.its_list:
        validate_lewis_endpoints(graph, engine.rule.rc.raw, observation=True)
