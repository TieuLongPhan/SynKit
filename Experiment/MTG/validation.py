#!/usr/bin/env python
"""Run bounded, stage-specific native composition and MTG validation."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter_ns
import tracemalloc
from typing import Any, Callable

import networkx as nx

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from synkit.Graph.Morphism import LabelSchema, LewisLabelledGraph  # noqa: E402
from synkit.Graph.MTG import (  # noqa: E402
    ChoiceWitness,
    IndependenceWitness,
    MaterialBinding,
    MaterialOccurrence,
    OccurrenceMTG,
    OccurrenceMTGFamily,
    OccurrenceProcess,
    OccurrenceProcessFamily,
    ProcessAlternative,
    RuleOccurrence,
)
from synkit.Rule import SynRule  # noqa: E402
from synkit.Rule.Apply import RuleSpan  # noqa: E402
from synkit.Rule.Compose import (  # noqa: E402
    CompositionError,
    CompositionWitness,
    OverlapSearchError,
    OverlapSearchLimits,
    canonical_overlap_digest,
    compose_rules,
    commute_independent,
    enumerate_overlaps,
    extended_component_match_matrix,
    quotient_composition_witnesses,
    search_compositions,
)

SCHEMA = LabelSchema(
    node_identity=("kind",),
    node_state=("state",),
    edge_state=("weight",),
    name="mtg-validation/1",
)
LIMITS = OverlapSearchLimits(
    max_states=10_000,
    max_overlaps=100,
    max_component_embeddings=100,
    max_canonical_permutations=10_000,
)
BUDGETS = {
    "median_stage_ms": 1_000.0,
    "peak_python_mib": 128.0,
    "raw_overlaps": 50,
    "exact_classes": 10,
}
FIXTURE_DESCRIPTOR = {
    "composition": "two indistinguishable preserved components, both updated",
    "process": "two-step causal state chain",
    "ambiguity": "two isomorphic GA3P occurrences, one TKT consumer",
    "native_synrule": "C-C to C=C to C#C, tuple electron labels",
}


def _llg(
    nodes: list[tuple[Any, int]],
    edges: tuple[tuple[Any, Any, int], ...] = (),
) -> LewisLabelledGraph:
    graph = nx.Graph()
    for node, state in nodes:
        graph.add_node(node, kind="X", state=state)
    for left, right, weight in edges:
        graph.add_edge(left, right, weight=weight)
    return LewisLabelledGraph.from_networkx(graph, SCHEMA)


def _rule_pair(prefix: str = "base") -> tuple[RuleSpan, RuleSpan]:
    first = RuleSpan.from_mapping(
        _llg([(f"{prefix}:l1", 0), (f"{prefix}:l2", 0)]),
        _llg([(f"{prefix}:r1", 0), (f"{prefix}:r2", 0)]),
        {
            f"{prefix}:l1": f"{prefix}:r1",
            f"{prefix}:l2": f"{prefix}:r2",
        },
        name="identity-two",
    )
    second = RuleSpan.from_mapping(
        _llg([(f"{prefix}:a", 0), (f"{prefix}:b", 0)]),
        _llg([(f"{prefix}:x", 1), (f"{prefix}:y", 1)]),
        {f"{prefix}:a": f"{prefix}:x", f"{prefix}:b": f"{prefix}:y"},
        name="update-two",
    )
    return first, second


def _state_rule(before: int, after: int, name: str) -> RuleSpan:
    return RuleSpan.from_mapping(
        _llg([(f"{name}:left", before)]),
        _llg([(f"{name}:right", after)]),
        {f"{name}:left": f"{name}:right"},
        name=name,
    )


def _material(identifier: str, state: int) -> MaterialOccurrence:
    return MaterialOccurrence(identifier, _llg([(f"{identifier}:node", state)]))


def _binding(
    material: MaterialOccurrence, endpoint: LewisLabelledGraph
) -> MaterialBinding:
    endpoint_node = next(iter(endpoint.node_ids))
    material_node = next(iter(material.value.node_ids))
    return MaterialBinding(
        material.occurrence_id,
        frozenset((endpoint_node,)),
        ((endpoint_node, material_node),),
    )


def _event(
    identifier: str,
    rule: RuleSpan,
    input_material: MaterialOccurrence,
    output_material: MaterialOccurrence,
) -> RuleOccurrence:
    return RuleOccurrence(
        identifier,
        rule,
        (_binding(input_material, rule.left),),
        (_binding(output_material, rule.right),),
    )


def _chain_process() -> OccurrenceProcess:
    first = _state_rule(0, 1, "first")
    second = _state_rule(1, 2, "second")
    initial = _material("chain-initial", 0)
    intermediate = _material("chain-intermediate", 1)
    final = _material("chain-final", 2)
    return OccurrenceProcess(
        "validation-chain",
        (initial, intermediate, final),
        (
            _event("E1", first, initial, intermediate),
            _event("E2", second, intermediate, final),
        ),
    )


def _ga3p_family() -> OccurrenceProcessFamily:
    fba = _state_rule(0, 1, "fba")
    tpi = _state_rule(0, 1, "tpi")
    tkt = _state_rule(1, 2, "tkt")
    fbp_fragment = _material("fbp-fragment", 0)
    dhap = _material("dhap", 0)
    direct = _material("ga3p-direct", 1)
    converted = _material("ga3p-from-tpi", 1)
    direct_product = _material("tkt-product-direct", 2)
    converted_product = _material("tkt-product-converted", 2)
    fba_tpi = commute_independent(
        fba,
        {"fba:left": "a"},
        tpi,
        {"tpi:left": "b"},
        _llg([("a", 0), ("b", 0)]),
    )
    tpi_tkt = commute_independent(
        tpi,
        {"tpi:left": "b"},
        tkt,
        {"tkt:left": "a"},
        _llg([("a", 1), ("b", 0)]),
    )
    fba_tkt = commute_independent(
        fba,
        {"fba:left": "a"},
        tkt,
        {"tkt:left": "b"},
        _llg([("a", 0), ("b", 1)]),
    )
    shared = (fbp_fragment, dhap, direct, converted)
    direct_process = OccurrenceProcess(
        "consume-direct",
        shared + (direct_product,),
        (
            _event("FBA", fba, fbp_fragment, direct),
            _event("TPI", tpi, dhap, converted),
            _event("TKT", tkt, direct, direct_product),
        ),
        (
            IndependenceWitness("FBA", "TPI", fba_tpi),
            IndependenceWitness("TPI", "TKT", tpi_tkt),
        ),
    )
    converted_process = OccurrenceProcess(
        "consume-converted",
        shared + (converted_product,),
        (
            _event("FBA", fba, fbp_fragment, direct),
            _event("TPI", tpi, dhap, converted),
            _event("TKT", tkt, converted, converted_product),
        ),
        (
            IndependenceWitness("FBA", "TPI", fba_tpi),
            IndependenceWitness("FBA", "TKT", fba_tkt),
        ),
    )
    return OccurrenceProcessFamily(
        (
            ProcessAlternative("direct", direct_process),
            ProcessAlternative("converted", converted_process),
        ),
        (
            ChoiceWitness(
                "direct",
                "converted",
                "material_assignment",
                "TKT consumes one of two isomorphic GA3P occurrences.",
            ),
        ),
    )


def _measure(
    operation: Callable[[], Any], iterations: int, *, work_units: int = 1
) -> dict[str, float | int]:
    samples = []
    for _ in range(iterations):
        start = perf_counter_ns()
        operation()
        samples.append((perf_counter_ns() - start) / 1_000_000)
    median = statistics.median(samples)
    return {
        "iterations": iterations,
        "work_units_per_iteration": work_units,
        "median_ms": round(median, 6),
        "median_ms_per_unit": round(median / work_units, 6),
        "minimum_ms": round(min(samples), 6),
        "maximum_ms": round(max(samples), 6),
    }


def _accepted_witnesses(
    first: RuleSpan,
    second: RuleSpan,
    overlaps: tuple[Any, ...],
) -> tuple[CompositionWitness, ...]:
    accepted = []
    for overlap in overlaps:
        digest = canonical_overlap_digest(
            overlap,
            permutation_limit=LIMITS.max_canonical_permutations,
        )
        try:
            composition = compose_rules(first, second, overlap)
        except CompositionError:
            continue
        accepted.append(CompositionWitness(overlap, digest, composition))
    return tuple(accepted)


def _dataset_summary() -> tuple[dict[str, Any], str]:
    paths = {
        "aldol": REPOSITORY_ROOT / "synkit/Data/aldol.json.gz",
        "multistep_synthesis": REPOSITORY_ROOT / "synkit/Data/paracetamol.json.gz",
    }
    payloads = {name: path.read_bytes() for name, path in paths.items()}
    aldol = json.loads(payloads["aldol"])
    synthesis = json.loads(payloads["multistep_synthesis"])
    mechanisms = aldol[0]["mechanisms"]
    digest = sha256(
        json.dumps(FIXTURE_DESCRIPTOR, sort_keys=True).encode()
        + b"".join(payloads[name] for name in sorted(payloads))
    ).hexdigest()
    return (
        {
            "aldol": {
                "evidence_kind": "curated input audit",
                "formal_proof": False,
                "mechanism_count": len(mechanisms),
                "step_counts": [len(item["steps"]) for item in mechanisms],
                "all_steps_have_mapped_reaction": all(
                    ">>" in step.get("smart_string", "")
                    for mechanism in mechanisms
                    for step in mechanism["steps"]
                ),
            },
            "multistep_synthesis": {
                "evidence_kind": "curated mapped-sequence input audit",
                "formal_proof": False,
                "step_count": len(synthesis),
                "all_steps_have_aam": all(
                    ">>" in item.get("aam", "") for item in synthesis
                ),
            },
        },
        digest,
    )


def _native_synrule_case() -> dict[str, Any]:
    first = SynRule.from_smart(
        "[C:1][C:2]>>[C:1]=[C:2]",
        name="single-to-double",
        canon=False,
        implicit_h=False,
        format="tuple",
    )
    second = SynRule.from_smart(
        "[C:1]=[C:2]>>[C:1]#[C:2]",
        name="double-to-triple",
        canon=False,
        implicit_h=False,
        format="tuple",
    )
    explicit = first.compose(second, {1: 1, 2: 2})
    family = first.composition_candidates(second, limits=LIMITS)
    return {
        "evidence_kind": "native tuple-rule operational witness",
        "formal_proof": False,
        "explicit_replay": explicit.certificate.replay().valid,
        "raw_overlaps": family.raw_overlap_count,
        "accepted": family.accepted_count,
        "exact_classes": family.exact_class_count,
        "extended_match_matrix": family.match_matrix.counts,
    }


def validation_report(*, iterations: int = 7) -> dict[str, Any]:
    """Return bounded evidence with per-stage timings and explicit claim types."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    first, second = _rule_pair()
    matrix = extended_component_match_matrix(
        first.right,
        second.left,
        max_embeddings=LIMITS.max_component_embeddings,
    )
    overlaps, explored = enumerate_overlaps(
        first.right, second.left, limits=LIMITS, _match_matrix=matrix
    )
    accepted = _accepted_witnesses(first, second, overlaps)
    classes = quotient_composition_witnesses(accepted, limits=LIMITS)
    process = _chain_process()
    history = OccurrenceMTG.from_process(process)

    timings = {
        "rule_construction": _measure(_rule_pair, iterations),
        "extended_match_matrix": _measure(
            lambda: extended_component_match_matrix(
                first.right,
                second.left,
                max_embeddings=LIMITS.max_component_embeddings,
            ),
            iterations,
        ),
        "overlap_enumeration": _measure(
            lambda: enumerate_overlaps(
                first.right,
                second.left,
                limits=LIMITS,
                _match_matrix=matrix,
            ),
            iterations,
        ),
        "composite_construction_family": _measure(
            lambda: _accepted_witnesses(first, second, overlaps),
            iterations,
            work_units=len(overlaps),
        ),
        "exact_quotient": _measure(
            lambda: quotient_composition_witnesses(accepted, limits=LIMITS),
            iterations,
            work_units=len(accepted),
        ),
        "certificate_replay_family": _measure(
            lambda: tuple(
                witness.composition.certificate.replay() for witness in accepted
            ),
            iterations,
            work_units=len(accepted),
        ),
        "process_construction": _measure(_chain_process, iterations),
        "mtg_derivation": _measure(
            lambda: OccurrenceMTG.from_process(process), iterations
        ),
    }
    tracemalloc.start()
    memory_first, memory_second = _rule_pair("memory")
    memory_matrix = extended_component_match_matrix(
        memory_first.right,
        memory_second.left,
        max_embeddings=LIMITS.max_component_embeddings,
    )
    memory_overlaps, _ = enumerate_overlaps(
        memory_first.right,
        memory_second.left,
        limits=LIMITS,
        _match_matrix=memory_matrix,
    )
    memory_witnesses = _accepted_witnesses(
        memory_first, memory_second, memory_overlaps
    )
    quotient_composition_witnesses(memory_witnesses, limits=LIMITS)
    OccurrenceMTG.from_process(_chain_process())
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    relabeled_first, relabeled_second = _rule_pair("relabeled")
    relabeled = search_compositions(relabeled_first, relabeled_second, limits=LIMITS)
    canonical_ids = tuple(group.canonical_id for group in classes)
    corruption_code = None
    try:
        enumerate_overlaps(
            first.right,
            second.left,
            limits=OverlapSearchLimits(max_overlap_nodes=0),
        )
    except OverlapSearchError as error:
        corruption_code = error.issue.code.value

    ga3p = _ga3p_family()
    ga3p_history = OccurrenceMTGFamily.from_process_family(ga3p)
    datasets, input_digest = _dataset_summary()
    native_case = _native_synrule_case()
    observed = {
        "raw_overlaps": len(overlaps),
        "accepted_witnesses": len(accepted),
        "exact_classes": len(classes),
        "explored_states": explored,
        "class_witness_counts": sorted(len(group.witnesses) for group in classes),
        "matrix": matrix.counts,
        "peak_python_mib": round(peak / (1024 * 1024), 3),
        "retained_python_mib": round(current / (1024 * 1024), 3),
    }
    checks = {
        "composition_counts": (
            observed["raw_overlaps"],
            observed["accepted_witnesses"],
            observed["exact_classes"],
        )
        == (7, 7, 3),
        "all_composition_certificates_replay": all(
            witness.composition.certificate.replay().valid for witness in accepted
        ),
        "exact_quotient_keeps_witnesses": sum(
            len(group.witnesses) for group in classes
        )
        == len(accepted),
        "carrier_relabeling_invariant": canonical_ids
        == tuple(group.canonical_id for group in relabeled.classes),
        "typed_resource_refusal": corruption_code == "OVERLAP_SEARCH_NODE_LIMIT",
        "process_roundtrip": history.replay().valid
        and history.to_process().causal_pairs == (("E1", "E2"),),
        "ga3p_alternatives_retained": len(ga3p_history.alternatives) == 2
        and {item.alternative_id for item in ga3p_history.alternatives}
        == {"direct", "converted"},
        "native_tuple_synrule_replays": native_case["explicit_replay"],
        "stage_time_budget": all(
            item["median_ms"] <= BUDGETS["median_stage_ms"]
            for item in timings.values()
        ),
        "memory_budget": observed["peak_python_mib"] <= BUDGETS["peak_python_mib"],
        "search_budget": observed["raw_overlaps"] <= BUDGETS["raw_overlaps"]
        and observed["exact_classes"] <= BUDGETS["exact_classes"],
        "curated_inputs_readable": datasets["aldol"][
            "all_steps_have_mapped_reaction"
        ]
        and datasets["multistep_synthesis"]["all_steps_have_aam"],
    }
    cases = {
        **datasets,
        "glycolysis_ga3p": {
            "evidence_kind": "occurrence-identity ambiguity witness",
            "formal_proof": False,
            "alternative_count": len(ga3p.alternatives),
            "consumed_occurrences": {
                item.alternative_id: item.process.event_by_id["TKT"]
                .inputs[0]
                .material_id
                for item in ga3p.alternatives
            },
        },
        "native_multistep_rule": native_case,
    }
    return {
        "schema": "synkit.mtg-validation/1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "claim_boundary": (
            "Formal checks replay stated finite-graph premises; timings and curated "
            "chemical inputs are operational evidence, not mathematical proofs or "
            "claims about kinetics, energetics, yield, or mechanism preference."
        ),
        "environment": {
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "input_sha256": input_digest,
        "limits": {
            "max_states": LIMITS.max_states,
            "max_overlaps": LIMITS.max_overlaps,
            "max_component_embeddings": LIMITS.max_component_embeddings,
            "max_canonical_permutations": LIMITS.max_canonical_permutations,
        },
        "budgets": BUDGETS,
        "observed": observed,
        "stage_timings": timings,
        "checks": checks,
        "case_studies": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = validation_report(iterations=arguments.iterations)
    payload = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
