#!/usr/bin/env python
"""Execute MØD on the same fully labelled, anchored supplied histories.

This is an equal-information adapter experiment, not the historical atom/bond
projection. It tests selected witnesses, not arbitrary-host guarded equivalence.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.MTG.verified_cases import _unpack_certificate  # noqa: E402
from synkit.Rule.Apply import apply_dpo  # noqa: E402


from Experiment.MTG.mod_encoding import (  # noqa: E402
    graph_gml as _graph_gml,
    rule_gml as _rule_gml,
    expected_signature as _expected_signature,
    decode_graphs as _decode_graphs,
)


def _mod_signature(graphs):
    nodes, edges = [], []
    for graph in graphs:
        labels = {node.id: node.stringLabel for node in graph.vertices}
        nodes.extend(labels.values())
        edges.extend(
            (tuple(sorted((labels[e.source.id], labels[e.target.id]))), e.stringLabel)
            for e in graph.edges
        )
    return sorted(nodes), sorted(edges)


def _apply(mod, hosts, rule):
    dg = mod.DG(graphDatabase=hosts)
    with dg.build() as builder:
        edges = builder.apply(hosts, rule, onlyProper=True, verbosity=0)
    return dg, edges


def _history(mod, name, certificates, observation=None):
    parsed = [_unpack_certificate(c) for c in certificates]
    rows = []
    mod_rules = []
    mod_rule_maps = []
    initial_hosts = None
    actual_oxygen_roles = []
    for index, c in enumerate(parsed):
        started = perf_counter()
        graph_gml = _graph_gml(c.host)
        rule_gml, left_ids, right_ids = _rule_gml(c.rule)
        encoding_seconds = perf_counter() - started
        started = perf_counter()
        hosts = mod.Graph.fromGMLStringMulti(graph_gml, add=False)
        rule = mod.ruleGMLString(rule_gml, add=False)
        parsing_seconds = perf_counter() - started
        assert _mod_signature(hosts) == _expected_signature(c.host)
        if initial_hosts is None:
            initial_hosts = hosts
        mod_rules.append(rule)
        mod_rule_maps.append((left_ids, right_ids))
        started = perf_counter()
        dg, derivations = _apply(mod, hosts, rule)
        mod_seconds = perf_counter() - started
        expected = _expected_signature(c.result)
        actual = [
            _mod_signature([v.graph for v in edge.targets]) for edge in derivations
        ]
        if observation is not None:
            from Experiment.MTG.isotope_case import oxygen_role

            actual_oxygen_roles.append(
                [
                    oxygen_role(_decode_graphs([v.graph for v in edge.targets]))
                    for edge in derivations
                ]
            )
        started = perf_counter()
        native = apply_dpo(c.rule, c.host, dict(c.match))
        native_seconds = perf_counter() - started
        rows.append(
            {
                "transition": index + 1,
                "mod_derivations": len(actual),
                "all_targets_equal": bool(actual)
                and all(g == expected for g in actual),
                "native_certificate_valid": native.certificate.replay().valid,
                "encoding_seconds": encoding_seconds,
                "mod_parsing_seconds": parsing_seconds,
                "mod_application_seconds": mod_seconds,
                "native_application_seconds": native_seconds,
            }
        )
    composed = None
    if len(mod_rules) == 2:
        match = mod.RCMatch(*mod_rules)
        first_right = mod_rule_maps[0][1]
        second_left = mod_rule_maps[1][0]
        assert set(first_right) == set(second_left)
        for key in sorted(first_right, key=repr):
            match.push(
                mod_rules[0].getVertexFromExternalId(first_right[key]).right,
                mod_rules[1].getVertexFromExternalId(second_left[key]).left,
            )
        outer = match.compose()
        if outer is None:
            raise ValueError("The supplied full overlap did not compose in MØD.")
        dg, derivations = _apply(mod, initial_hosts, outer)
        expected = _expected_signature(parsed[-1].result)
        actual = [
            _mod_signature([v.graph for v in edge.targets]) for edge in derivations
        ]
        composed = {
            "derivations": len(actual),
            "all_targets_equal": bool(actual) and all(g == expected for g in actual),
        }
    assert all(
        row["all_targets_equal"] and row["native_certificate_valid"] for row in rows
    )
    assert composed is None or composed["all_targets_equal"]
    return {
        "name": name,
        "transitions": rows,
        "fixed_overlap_composition": composed,
        "mod_oxygen_roles_after_each_transition": actual_oxygen_roles,
        "observation_compatible": (
            all(
                role == observation["labelled_product_role"]
                for role in actual_oxygen_roles[-1]
            )
            if observation is not None
            else None
        ),
        "scope": "Explicit selected hosts and matches; no general guarded-rule equivalence or speed advantage claimed.",
    }


def run(case_path, isotope_path=None):
    import mod

    source = json.loads(case_path.read_text())
    histories = []
    for case in source["cases"]:
        for index, item in enumerate(case["interpretations"]):
            histories.append(
                _history(
                    mod,
                    f"SynEPD {case['id']} interpretation {index + 1}",
                    item["certificates"],
                )
            )
    inputs = {str(case_path): sha256(case_path.read_bytes()).hexdigest()}
    if isotope_path is not None:
        isotope = json.loads(isotope_path.read_text())
        for case in isotope["cases"]:
            histories.append(
                _history(
                    mod, case["name"], case["certificates"], isotope["observation"]
                )
            )
        inputs[str(isotope_path)] = sha256(isotope_path.read_bytes()).hexdigest()
    return {
        "schema": "synkit.mtg-mod-history-comparison/1",
        "environment": {"python": platform.python_version(), "mod": str(mod.version())},
        "inputs": inputs,
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "adapter": (
            "Coordinate-incidence graphs encode every semantic field separately; undefined "
            "coordinates delete/create labelled leaves. Supplied atom anchors and bond "
            "incidence are retained."
        ),
        "shared_information": (
            "The same states, coordinate-preserving rule interfaces, isotope identity when "
            "declared, anchors and supplied full overlaps."
        ),
        "histories": histories,
        "status": "PASS",
        "claim_limits": (
            "Historical installed MØD release; selected-host correctness only. This engine "
            "comparison alone does not evaluate catalytic overlay or semigroup abstractions."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--isotope-cases", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.cases, args.isotope_cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"histories": len(report["histories"]), "status": report["status"]})
    )


if __name__ == "__main__":
    main()
