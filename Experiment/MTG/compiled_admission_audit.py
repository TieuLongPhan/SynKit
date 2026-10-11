"""Complete planned-admission costs against replay and a strong finite baseline."""

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
from math import ceil
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Experiment.MTG.compositional_condition_compiler import (  # noqa: E402
    compile_program as read_compile,
)
from Experiment.MTG.frame_condition_compiler import (  # noqa: E402
    Origin,
    compile_program as full_compile,
)
from Experiment.MTG.frame_condition_audit import (  # noqa: E402
    _raw_iso,
    coherent_graph,
    frame_labels,
    lifted_word,
    raw_projection,
    raw_rule,
)
from Experiment.MTG.verified_cases import _unpack_certificate  # noqa: E402
from synkit.Rule.Apply import DPOError, apply_dpo  # noqa: E402

METHODS = ("direct_replay", "full_world_finite", "read_supported_finite")


@dataclass(frozen=True)
class ReplayPlan:
    applications: tuple
    matches: tuple


def prepare_replay(applications):
    previous = applications[0].certificate.host
    origins = {n: Origin("initial", -1, n) for n in previous.node_ids}
    matches = []
    for step, app in enumerate(applications):
        c = app.certificate
        if c.host != previous or c.result != app.result or not c.replay().valid:
            raise ValueError(
                "Direct replay requires a consecutive valid source diagram."
            )
        matches.append(tuple((n, origins[h]) for n, h in c.match))
        before, after = dict(c.context_to_host), dict(c.context_to_result)
        origins = {after[n]: origins[before[n]] for n in c.context.node_ids}
        right = dict(c.right_to_result)
        for n in c.rule.right.node_ids - set(c.rule.right_arm.mapping.values()):
            origins[right[n]] = Origin("fresh", step, n)
        if set(origins) != app.result.node_ids:
            raise ValueError(
                "Direct replay origin plan does not cover its source result."
            )
        previous = app.result
    return ReplayPlan(tuple(applications), tuple(matches))


def evaluate(method, prepared, host, anchors):
    if method == "direct_replay":
        return lifted_word(prepared, host, dict(anchors))
    match = {n: anchors[n] for n in prepared.rule.left.node_ids}
    app = apply_dpo(prepared.rule, host, match)
    assert app.certificate.replay().valid
    return app


def outcome(method, prepared, host, anchors):
    try:
        return evaluate(method, prepared, host, anchors)
    except DPOError:
        return None


def compare(prepared, compiled, reference, host, anchors):
    assert (compiled is None) == (reference is None)
    if reference is None:
        return
    final, transport, _ = reference
    correspondence = {
        n: n for n in host.node_ids - {anchors[n] for n in prepared.rule.left.node_ids}
    }
    right = dict(compiled.certificate.right_to_result)
    for n, origin in prepared.final_origins:
        correspondence[right[n]] = transport[origin]
    _raw_iso(compiled.result, final, correspondence)


def probes(applications, program):
    initial = applications[0].certificate.host
    anchors = {n: n for n in initial.node_ids}
    yield "original_complete_world", initial, anchors
    graph = initial.to_networkx()
    prototype = initial.node_labels(next(iter(initial.node_ids)), semantic=True)
    for index in range(16):
        graph.add_node(("outside-admission-frame", index), **prototype)
    yield "sixteen_isolated_frame_atoms", coherent_graph(graph, initial.schema), anchors
    for index, pair in enumerate(program.rule.condition.pairs):
        graph = initial.to_networkx()
        assert pair not in initial.edge_keys
        graph.add_edge(*tuple(pair), **frame_labels(initial.schema))
        yield f"essential_absence_{index}", coherent_graph(
            graph, initial.schema
        ), anchors
    preserved = {
        program.rule.left_arm.mapping[k]: program.rule.right_arm.mapping[k]
        for k in program.rule.interface.node_ids
    }
    net_pairs = [
        frozenset((a, b))
        for a, b in combinations(sorted(preserved, key=repr), 2)
        if frozenset((a, b)) not in initial.edge_keys
        and frozenset((preserved[a], preserved[b])) in program.rule.right.edge_keys
    ]
    for index, pair in enumerate(net_pairs):
        graph = initial.to_networkx()
        graph.add_edge(*tuple(pair), **frame_labels(initial.schema))
        yield f"net_created_pair_{index}", coherent_graph(
            graph, initial.schema
        ), anchors
    graph = initial.to_networkx()
    node = min(program.read_nodes, key=repr)
    graph.nodes[node]["element"] = "Xe" if graph.nodes[node]["element"] != "Xe" else "C"
    yield "invalid_read_identity", coherent_graph(graph, initial.schema), anchors


def build_cases():
    for name in ("isotope_case.json", "mcsa155_case.json"):
        path = ROOT / "Experiment/MTG/evidence" / name
        report = json.loads(path.read_text())
        candidates = list(report.get("candidates", []))
        for case in report.get("cases", []):
            candidates.extend(case.get("interpretations", [case]))
        for index, candidate in enumerate(candidates):
            started = perf_counter()
            certificates = [_unpack_certificate(c) for c in candidate["certificates"]]
            original = [apply_dpo(c.rule, c.host, dict(c.match)) for c in certificates]
            assert all(app.certificate.replay().valid for app in original)
            projected = name == "isotope_case.json"
            if projected:
                current, applications = raw_projection(certificates[0].host), []
                for c in certificates:
                    assert current == raw_projection(c.host)
                    app = apply_dpo(raw_rule(c.rule), current, dict(c.match))
                    assert app.certificate.replay().valid
                    applications.append(app)
                    current = app.result
            else:
                applications = original
            common_seconds = perf_counter() - started
            yield name, index, tuple(applications), projected, common_seconds


def measure_case(name, index, applications, projected, common_seconds):
    builders = dict(zip(METHODS, (prepare_replay, full_compile, read_compile)))
    preparation, prepared = {m: [] for m in METHODS}, {}
    for repetition in range(3):
        order = METHODS[repetition:] + METHODS[:repetition]
        for method in order:
            started = perf_counter()
            prepared[method] = builders[method](applications)
            preparation[method].append(perf_counter() - started)
    panel = tuple(probes(applications, prepared["read_supported_finite"]))
    references = {
        key: outcome("direct_replay", prepared["direct_replay"], host, anchors)
        for key, host, anchors in panel
    }
    assert references["original_complete_world"] is not None
    assert references["sixteen_isolated_frame_atoms"] is not None
    assert references["invalid_read_identity"] is None
    for key in references:
        if key.startswith(("essential_absence_", "net_created_pair_")):
            assert references[key] is None
    rounds = []
    validations = 0
    for repetition in range(3):
        order = METHODS[repetition:] + METHODS[:repetition]
        probes_order = panel[repetition:] + panel[:repetition]
        durations = {m: [] for m in METHODS}
        for key, host, anchors in probes_order:
            for method in order:
                started = perf_counter()
                actual = outcome(method, prepared[method], host, anchors)
                seconds = perf_counter() - started
                durations[method].append({"probe": key, "seconds": seconds})
                if method == "direct_replay":
                    assert (actual is None) == (references[key] is None)
                    if actual is not None:
                        assert actual[0] == references[key][0]
                else:
                    compare(prepared[method], actual, references[key], host, anchors)
                validations += 1
        rounds.append(
            {
                "repetition": repetition,
                "method_order": list(order),
                "durations": durations,
            }
        )
        print("Admission timing", name, index, "round", repetition, "PASS", flush=True)
    prep_medians = {m: statistics.median(preparation[m]) for m in METHODS}
    workload_medians = {
        m: statistics.median(
            sum(r["seconds"] for r in row["durations"][m]) for row in rounds
        )
        for m in METHODS
    }
    break_even = {}
    for reference in ("direct_replay", "full_world_finite"):
        saving = workload_medians[reference] - workload_medians["read_supported_finite"]
        extra = prep_medians["read_supported_finite"] - prep_medians[reference]
        break_even[reference] = {
            "query_workload_saving_seconds": saving,
            "extra_preparation_seconds": extra,
            "complete_probe_workloads_required": (
                max(1, ceil(extra / saving)) if saving > 0 else None
            ),
            "role": "Descriptive median cost model on this exact probe mixture; no external speedup inference.",
        }
    totals = {
        str(repeats): {
            m: common_seconds + prep_medians[m] + repeats * workload_medians[m]
            for m in METHODS
        }
        for repeats in (1, 10, 100)
    }
    return {
        "input": name,
        "candidate": index,
        "events": len(applications),
        "raw_projection_only": projected,
        "original_schema_admission": not projected,
        "initial_world_nodes": len(applications[0].certificate.host.node_ids),
        "read_pattern_nodes": len(prepared["read_supported_finite"].rule.left.node_ids),
        "common_input_reconstruction_validation_seconds": common_seconds,
        "preparation_seconds": preparation,
        "preparation_median_seconds": prep_medians,
        "probes": [
            {
                "name": key,
                "world_nodes": len(host.node_ids),
                "admitted": references[key] is not None,
            }
            for key, host, _ in panel
        ],
        "timing_rounds": rounds,
        "query_workload_median_seconds": workload_medians,
        "break_even": break_even,
        "modelled_pipeline_seconds": totals,
        "validated_timed_outcomes": validations,
        "all_anchored_answers_agree": True,
        "status": "PASS",
    }


def main():
    rows = [measure_case(*case) for case in build_cases()]
    base = ROOT / "Experiment/MTG"
    paths = [
        Path(__file__),
        base / "compiled_admission_protocol.json",
        base / "compositional_condition_compiler.py",
        base / "compositional_condition_audit.py",
        base / "frame_condition_compiler.py",
        base / "frame_condition_audit.py",
        base / "verified_cases.py",
        base / "evidence/compositional_condition_audit.json",
        base / "evidence/isotope_case.json",
        base / "evidence/mcsa155_case.json",
        ROOT / "synkit/Rule/Apply/dpo.py",
        ROOT / "synkit/Rule/Apply/validation.py",
        ROOT / "synkit/Graph/Morphism/llg.py",
    ]
    report = {
        "schema": "synkit.compiled-admission-audit/1",
        "status": "PASS",
        "cases": rows,
        "summary": {
            "cases": len(rows),
            "original_schema_cases": sum(not r["raw_projection_only"] for r in rows),
            "raw_projection_cases": sum(r["raw_projection_only"] for r in rows),
            "probe_instances": sum(len(r["probes"]) for r in rows),
            "validated_timed_outcomes": sum(
                r["validated_timed_outcomes"] for r in rows
            ),
            "preparation_measurements": len(rows) * 9,
            "all_anchored_answers_agree": True,
        },
        "environment": {
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "platform": platform.platform(),
            "hardware_controlled": False,
        },
        "external_tool_timing_comparison": False,
        "physical_validation": False,
        "general_network_observation_speedup": False,
        "consequential_external_advantage_established": False,
        "ncs_readiness_established": False,
        "sha256": {
            str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in paths
        },
    }
    output = base / "evidence/compiled_admission_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS", output, flush=True)


if __name__ == "__main__":
    main()
