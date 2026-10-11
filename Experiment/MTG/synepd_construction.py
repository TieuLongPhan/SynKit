#!/usr/bin/env python
"""Audit supplied SynEPD arrows and bounded grouping interpretations.

This research experiment preserves the input population. Passing electron
replay does not certify physical intermediates or chemical event grouping.
Natural DPO square checks certify only the declared exact finite schema.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import networkx as nx
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synkit.Graph.Morphism import (  # noqa: E402
    ELECTRON_LLG_SCHEMA,
    LewisLabelledGraph,
    derive_electron_labeled_graph,
)
from synkit.Graph.MTG import (  # noqa: E402
    MaterialBinding,
    MaterialOccurrence,
    OccurrenceMTG,
    OccurrenceProcess,
    RuleOccurrence,
)
from synkit.Mechanism.adapters import (  # noqa: E402
    group_from_legacy_epd,
    mechanism_from_legacy_epd,
)
from synkit.Mechanism.model import MechanisticStep  # noqa: E402
from synkit.Mechanism.replay import MechanismReplayer  # noqa: E402
from synkit.Rule.Apply import RuleSpan, apply_dpo  # noqa: E402


def _record(source, cuts, *, honor_source_policy=True):
    record = mechanism_from_legacy_epd(source["rsmi"], source["epd"])
    starts = (0, *cuts)
    ends = (*cuts, len(source["epd"]))
    steps = tuple(
        MechanisticStep(
            step_id=f"s{index}",
            groups=(
                group_from_legacy_epd(source["epd"][start:end], group_id=f"g{index}"),
            ),
        )
        for index, (start, end) in enumerate(zip(starts, ends), 1)
    )
    metadata = {}
    representation = source.get("epd_representation", {})
    if honor_source_policy and representation:
        if representation["mode"] != "closed_shell_pair":
            raise ValueError(
                f"Unsupported declared representation: {representation['mode']}"
            )
        metadata = {
            "endpoint_resource_policy": "closed_shell_pair",
            "closed_shell_atom_maps": representation["closed_shell_atom_maps"],
        }
    return replace(record, steps=steps, metadata=metadata)


def _states(replay):
    return [replay.mtg.nodes[0]["graph"], *replay.intermediates]


def _hydrogen_duet_excess(graph):
    excess = []
    for node, attrs in graph.nodes(data=True):
        if attrs["element"] != "H":
            continue
        shell = (
            2 * attrs["lone_pairs"]
            + attrs["radical"]
            + 2 * attrs["hcount"]
            + 2
            * sum(
                edge["sigma_order"] + edge["pi_order"] for edge in graph[node].values()
            )
        )
        if shell > 2:
            excess.append({"atom_map": attrs["atom_map"], "shell_electrons": shell})
    return excess


def _replay(source, cuts, *, honor_source_policy=True):
    try:
        result = MechanismReplayer(validation="strict", repair=False).replay(
            _record(source, cuts, honor_source_policy=honor_source_policy)
        )
        certificate = result.certificate
        graphs = _states(result)
        return {
            "status": certificate.status,
            "endpoint_match": certificate.endpoint_match,
            "verification_level": certificate.verification_level,
            "endpoint_resource_policy": certificate.endpoint_resource_policy,
            "normalization_evidence": list(certificate.normalization_evidence),
            "committed_groups": len(result.intermediates),
            "issue_codes": sorted({str(issue.code) for issue in certificate.issues}),
            "internal_hydrogen_duet_excess": (
                [
                    {"state_index": index, "atoms": excess}
                    for index, graph in enumerate(graphs[1:-1], 1)
                    if (excess := _hydrogen_duet_excess(graph))
                ]
                if certificate.status == "VALID"
                else []
            ),
            "boundary_hydrogen_duet_excess": _hydrogen_duet_excess(graphs[0]),
        }, result
    except Exception as error:
        return {
            "status": "ERROR",
            "error_type": type(error).__name__,
            "message": str(error)[:300],
        }, None


def _llg(graph):
    mapped = nx.relabel_nodes(
        graph, {n: a["atom_map"] for n, a in graph.nodes(data=True)}
    )
    # Audit the declared electron schema; source provenance remains in the report.
    # Unrecognized metadata are not certified by this schema (including isotope).
    for _node, attrs in mapped.nodes(data=True):
        for key in set(attrs) - set(ELECTRON_LLG_SCHEMA.declared_node_keys):
            del attrs[key]
    for _left, _right, attrs in mapped.edges(data=True):
        for key in set(attrs) - set(ELECTRON_LLG_SCHEMA.declared_edge_keys):
            del attrs[key]
    return LewisLabelledGraph.from_networkx(
        derive_electron_labeled_graph(mapped), ELECTRON_LLG_SCHEMA
    )


def _whole_state_audit(replay, *, construct_process=False):
    """Check actual API constructions while retaining their formal limitations."""
    graphs = [_llg(graph) for graph in _states(replay)]
    rules = []
    conflicts = 0
    certificates = []
    for before, after in zip(graphs, graphs[1:]):
        if before.node_ids != after.node_ids:
            raise ValueError("Replay changed the atom carrier set.")
        rule = RuleSpan.from_mapping(
            before, after, {n: n for n in before.node_ids}, boundary="closed"
        )
        application = apply_dpo(rule, before, {n: n for n in before.node_ids})
        if (
            application.result.nodes != after.nodes
            or application.result.edges != after.edges
        ):
            raise ValueError(
                "Whole-state application does not reproduce the replay state."
            )
        certificate = application.certificate
        for node, image in certificate.context_to_result:
            left = certificate.context.node_labels(node, semantic=True)
            right = certificate.result.node_labels(image, semantic=True)
            conflicts += sum(value != right.get(key) for key, value in left.items())
        for edge in certificate.context.edge_keys:
            mapping = dict(certificate.context_to_result)
            image = frozenset(mapping[n] for n in edge)
            left = certificate.context.edge_labels(edge, semantic=True)
            right = certificate.result.edge_labels(image, semantic=True)
            conflicts += sum(value != right.get(key) for key, value in left.items())
        rules.append(rule)
        certificates.append(certificate.replay().valid)
    output = {
        "status": "CERTIFIED_NATURAL_DPO",
        "transitions": len(rules),
        "natural_square_replay_accepts_all": all(certificates),
        "defined_context_coordinates_not_preserved": conflicts,
        "formal_dpo_certified": all(certificates),
    }
    if not construct_process:
        return output
    materials = []
    layers = []
    for state_index, graph in enumerate(graphs):
        layer = []
        for component_index, nodes in enumerate(
            nx.connected_components(graph.to_networkx())
        ):
            nodes = frozenset(nodes)
            value = LewisLabelledGraph.from_networkx(
                graph.to_networkx().subgraph(nodes).copy(), graph.schema
            )
            material = MaterialOccurrence(f"m{state_index}:{component_index}", value)
            materials.append(material)
            layer.append(
                MaterialBinding(
                    material.occurrence_id, nodes, tuple((n, n) for n in nodes)
                )
            )
        layers.append(tuple(layer))
    events = tuple(
        RuleOccurrence(f"e{index}", rule, layers[index - 1], layers[index])
        for index, rule in enumerate(rules, 1)
    )
    process = OccurrenceProcess(
        "supplied-ordered-electron-history", tuple(materials), events
    )
    history = OccurrenceMTG.from_process(process)
    extension = tuple(event.occurrence_id for event in events)
    output["occurrence"] = {
        "status": "CONSTRUCTED_OPERATIONALLY",
        "materials": len(materials),
        "events": len(events),
        "node_lineages": len(history.node_lineages),
        "edge_lineages": len(history.edge_lineages),
        "causal_pairs": [list(pair) for pair in process.causal_pairs],
        "replay": history.replay().valid,
        "full_history_nodes": history.full_history(extension).number_of_nodes(),
        "changed_core_nodes": history.minimal_changed_core(extension).number_of_nodes(),
        "scope": "Whole-state rules retain supplied order and can impose artificial spectator dependencies.",
    }
    output["states"] = [
        {
            "state_index": index,
            "atoms": [
                {
                    "atom_map": attrs["atom_map"],
                    **{
                        k: attrs[k]
                        for k in (
                            "element",
                            "hcount",
                            "charge",
                            "lone_pairs",
                            "radical",
                        )
                    },
                }
                for _, attrs in sorted(
                    graph.nodes(data=True), key=lambda pair: pair[1]["atom_map"]
                )
            ],
            "bonds": sorted(
                [
                    min(graph.nodes[a]["atom_map"], graph.nodes[b]["atom_map"]),
                    max(graph.nodes[a]["atom_map"], graph.nodes[b]["atom_map"]),
                    data["order"],
                ]
                for a, b, data in graph.edges(data=True)
            ),
        }
        for index, graph in enumerate(_states(replay))
    ]
    return output


def run(path, *, partition_max_arrows=4, limit=None, example_ids=(1, 67, 475)):
    raw = path.read_bytes()
    source = json.loads(raw)
    population = source["records"]
    if source["count"] != len(population) or len({r["id"] for r in population}) != len(
        population
    ):
        raise ValueError("Source count or record identifiers are inconsistent.")
    selected = population[:limit] if limit is not None else population
    started = perf_counter()
    rows, examples = [], []
    summary = Counter(records=len(selected))
    for index, record in enumerate(selected, 1):
        n = len(record["epd"])
        if not n:
            raise ValueError(
                "Empty supplied arrow lists require a separately defined interpretation."
            )
        sequential_cuts = tuple(range(1, n))
        cache = {}
        for cuts in (sequential_cuts, ()):
            if cuts not in cache:
                cache[cuts] = _replay(record, cuts)
        sequential, sequential_result = cache[sequential_cuts]
        grouped, grouped_result = cache[()]
        row = {
            "id": record["id"],
            "arrows": n,
            "ordered": sequential,
            "one_group": grouped,
        }
        if record.get("epd_representation"):
            row["declared_representation"] = record["epd_representation"]
            strict, _result = _replay(
                record, sequential_cuts, honor_source_policy=False
            )
            row["strict_endpoint_comparison"] = strict
            summary["records_with_declared_endpoint_policy"] += 1
            summary[f"declared_policy_strict_endpoint_{strict['status']}"] += 1
        for name, profile in (("ordered", sequential), ("one_group", grouped)):
            summary[f"{name}_{profile['status']}"] += 1
        summary[f"paired_{sequential['status']}_{grouped['status']}"] += 1
        if sequential.get("internal_hydrogen_duet_excess"):
            summary["ordered_valid_with_internal_hydrogen_duet_excess"] += 1
        if sequential["status"] == "VALID":
            try:
                row["whole_state"] = _whole_state_audit(sequential_result)
                summary["whole_state_constructed"] += 1
                if row["whole_state"]["defined_context_coordinates_not_preserved"]:
                    summary["whole_state_with_context_morphism_conflict"] += 1
            except Exception as error:
                row["whole_state"] = {
                    "status": "ERROR",
                    "error_type": type(error).__name__,
                    "message": str(error)[:300],
                }
                summary["whole_state_ERROR"] += 1
        if n <= partition_max_arrows:
            valid = []
            statuses = Counter()
            for mask in range(1 << (n - 1)):
                cuts = tuple(i for i in range(1, n) if mask & (1 << (i - 1)))
                if cuts not in cache:
                    cache[cuts] = _replay(record, cuts)
                profile, _result = cache[cuts]
                statuses[profile["status"]] += 1
                if profile["status"] == "VALID":
                    valid.append(cuts)
            row["contiguous_partitions"] = {
                "complete_in_declared_family": True,
                "tested": sum(statuses.values()),
                "statuses": dict(statuses),
                "valid_cuts": [list(cuts) for cuts in valid],
                "group_counts": sorted({len(cuts) + 1 for cuts in valid}),
            }
            summary["partition_complete_records"] += 1
            summary["partition_trials"] += sum(statuses.values())
            summary["partition_records_with_multiple_valid_groupings"] += len(valid) > 1
            summary["partition_records_with_no_valid_grouping"] += not valid
        else:
            row["contiguous_partitions"] = {
                "complete_in_declared_family": False,
                "reason": "arrow_count_limit",
            }
            summary["partition_limited_records"] += 1
        rows.append(row)
        if record["id"] in example_ids:
            example = {
                "id": record["id"],
                "reaction_name": record["reaction_name"],
                "arrows": n,
            }
            for name, profile, result in (
                ("ordered", sequential, sequential_result),
                ("one_group", grouped, grouped_result),
            ):
                if profile["status"] == "VALID":
                    try:
                        example[name] = _whole_state_audit(
                            result, construct_process=True
                        )
                    except Exception as error:
                        example[name] = {
                            "status": "ERROR",
                            "error_type": type(error).__name__,
                            "message": str(error)[:300],
                        }
                else:
                    example[name] = profile
            examples.append(example)
        if index % 100 == 0:
            print(
                f"Audited {index}/{len(selected)} records", file=sys.stderr, flush=True
            )
    return {
        "schema": "synkit.mtg-synepd-construction-research/2",
        "source": {
            "schema": source["schema"],
            "sha256": sha256(raw).hexdigest(),
            "declared_records": len(population),
            "audited_records": len(selected),
        },
        "environment": {
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "rdkit": rdBase.rdkitVersion,
            "experiment_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
        "policy": {
            "stereo_verification": "off",
            "endpoint_resources": "strict except explicitly declared source closed_shell_pair comparisons",
            "repair": False,
            "grouping": "Two extreme interpretations; exhaustive contiguous partitions only within limit.",
            "partition_max_arrows": partition_max_arrows,
            "chemical_event_grouping_inferred": False,
            "formal_dpo_certified": "per-transition exact natural-square checks; no general concurrency theorem",
        },
        "summary": dict(sorted(summary.items())),
        "examples": examples,
        "elapsed_seconds": perf_counter() - started,
        "records": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--partition-max-arrows", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not 1 <= args.partition_max_arrows <= 8 or (
        args.limit is not None and args.limit < 1
    ):
        parser.error("Partition limit must be 1..8 and record limit must be positive.")
    report = run(
        args.source, partition_max_arrows=args.partition_max_arrows, limit=args.limit
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.summary_output:
        compact = {key: value for key, value in report.items() if key != "records"}
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(compact, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
