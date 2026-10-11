"""Curated registration with independent labels and matched comparison scope.

The six conventional frame families form the timed common subset. Expected
groups come from direct reference assignment orbits before predictions. New
proper-frame/renumbering copies preserve that truth. Profile-sensitive and
strict incomplete-input examples are reported separately from the timed set.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from itertools import product
import json
import os
from pathlib import Path
import time

import networkx as nx
from rdkit import Chem, RDLogger

from Experiment.Stereo.Canonicalization.exact_code_store import ExactCodeStore
from Experiment.Stereo.Canonicalization.global_local import _case_time_limit
from Experiment.Stereo.Canonicalization.independent_fixtures import fixtures
from Experiment.Stereo.Canonicalization.independent_model import (
    formal_states, proper_group, state_key, to_descriptor, transport,
)
from Experiment.Stereo.Canonicalization.independent_orbits import color_signature, reference_partition
from Experiment.Stereo.Canonicalization.performance import distribution, hardware
from Experiment.Stereo.Canonicalization.performance_cases import state_from_json
from Experiment.Stereo.Canonicalization.runtime import peak_rss_mib
from Experiment.Stereo.provenance import capture_source, finish_source, sha256
from synkit.Graph.Stereo import (
    StereographMirrorStatus, canonicalize_configured_registry,
    classify_rdkit_stereograph_mirror, descriptor_id, stereo_isomorphic,
)
from synkit.Graph.Stereo.rdkit_identity import canonicalize_rdkit_identity

SIZES = (8, 16, 32, 64, 128)
BASES = ("tetrahedral:distinct", "planar_bond:distinct", "square_planar:distinct",
         "trigonal_bipyramidal:distinct", "octahedral:distinct", "tetrahedral:two:meso",
         "mixed:tetrahedral+square_planar:distinct", "tetrahedral:distinct:H")


def _prototypes():
    catalog = {value.name: value for value in fixtures()}
    result, evidence = [], []
    for name in BASES:
        fixture = catalog[name]
        choices = tuple(formal_states(value) for value in fixture.seeds)
        truth = reference_partition(fixture.graph, choices, atom_color="color", bond_color="color")
        if truth["status"] != "complete":
            raise ValueError(f"Registration ground truth is incomplete:{name}")
        class_of = {member: index for index, group in enumerate(truth["partition"]) for member in group}
        assignments = tuple(product(*(range(len(local)) for local in choices)))
        if name.startswith(("trigonal_bipyramidal:", "octahedral:", "mixed:")):
            assignments = assignments[:2]
        for assignment in assignments:
            states = tuple(local[index] for local, index in zip(choices, assignment, strict=True))
            result.append((fixture.graph, states, (name, class_of[assignment])))
        evidence.append({"basis": name, "reference": truth, "selected_assignments": assignments})
    return tuple(result), evidence


def collection(size):
    """Freeze expected classes independently, then generate admissible copies."""
    prototypes, evidence = _prototypes()
    rows = []
    for index in range(size):
        graph, states, expected = prototypes[index % len(prototypes)]
        variants = []
        for state in states:
            group = proper_group(state.family)
            image = group[(index // len(prototypes)) % len(group)]
            variant = replace(state, atoms=tuple(state.atoms[position] for position in image))
            if state_key(variant) != state_key(state):
                raise ValueError("Declared proper-frame copy changes its independent configuration.")
            variants.append(variant)
        mapping = {node: 1000 * (index + 1) + 3 * position for position, node in enumerate(reversed(tuple(graph)))}
        renamed = nx.relabel_nodes(graph, mapping)
        rows.append({
            "id": f"registration:{index:04d}", "expected_group": expected,
            "nodes": [(node, dict(data)) for node, data in reversed(tuple(renamed.nodes(data=True)))],
            "edges": [(a, b, dict(data)) for a, b, data in reversed(tuple(renamed.edges(data=True)))],
            "states": [asdict(transport(value, mapping)) for value in variants],
            "transformations": "proper local frame, atom renumbering, reverse insertion order",
        })
    return rows, evidence


def partition(labels):
    groups = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    return tuple(sorted(tuple(group) for group in groups.values()))


def _prepare(rows):
    result = []
    for row in rows:
        graph = nx.Graph()
        graph.add_nodes_from(row["nodes"])
        graph.add_edges_from(row["edges"])
        values = tuple(to_descriptor(state_from_json(value)) for value in row["states"])
        graph.graph["stereo_descriptors"] = {descriptor_id(value): value for value in values}
        result.append(graph)
    return result


def pairwise_groups(graphs):
    """Register against existing group representatives; avoid a contrived all-pairs cost."""
    representatives, labels, comparisons = [], [], 0
    node_match = lambda a, b: color_signature(a.get("color")) == color_signature(b.get("color"))  # noqa: E731
    edge_match = lambda a, b: color_signature(a.get("color")) == color_signature(b.get("color"))  # noqa: E731
    for graph in graphs:
        for index, representative in enumerate(representatives):
            comparisons += 1
            if stereo_isomorphic(graph, representative, node_match=node_match, edge_match=edge_match):
                labels.append(index)
                break
        else:
            labels.append(len(representatives))
            representatives.append(graph)
    return labels, comparisons


def certificate_groups(graphs):
    """Compute one full code per row and index candidates with exact byte equality."""
    codes, calculation_seconds, bytes_total = [], 0.0, 0
    for graph in graphs:
        started = time.perf_counter()
        result = canonicalize_configured_registry(
            graph, graph.graph["stereo_descriptors"], atom_color="color", bond_color="color",
            enumerate_automorphism_group=False,
        )
        if not result.complete or not result.exact:
            raise ValueError("A registration certificate is incomplete.")
        calculation_seconds += time.perf_counter() - started
        codes.append(result.canonical_code)
        bytes_total += len(result.canonical_code.encode("utf-8"))
    started = time.perf_counter()
    with ExactCodeStore() as store:
        labels = [store.intern(code) for code in codes]
        disk_bytes = store.bytes_on_disk
    return labels, {"certificate_seconds": calculation_seconds,
                    "index_and_storage_seconds": time.perf_counter() - started,
                    "input_certificate_bytes_total": bytes_total, "temporary_disk_bytes": disk_bytes}


def workload(rows, method, timeout_seconds):
    """Include input preparation in both from-scratch registration pipelines."""
    started = time.perf_counter()
    graphs = _prepare(rows)
    preparation_seconds = time.perf_counter() - started
    expected = partition(tuple(row["expected_group"]) for row in rows)
    started = time.perf_counter()
    try:
        with _case_time_limit(timeout_seconds):
            if method == "pairwise":
                labels, comparisons = pairwise_groups(graphs)
                details = {"comparison_seconds": time.perf_counter() - started, "pairwise_calls": comparisons}
            elif method == "certificates":
                labels, details = certificate_groups(graphs)
            else:
                raise ValueError(f"Unknown registration method:{method}")
    except TimeoutError:
        return {"status": "timeout", "preparation_seconds": preparation_seconds,
                "budget_seconds": timeout_seconds, "total_seconds": preparation_seconds + time.perf_counter() - started,
                "partition_passed": False}
    observed = partition(labels)
    result = {"status": "complete", "preparation_seconds": preparation_seconds, **details,
              "total_seconds": preparation_seconds + time.perf_counter() - started,
              "observed_partition": observed, "expected_partition": expected,
              "partition_passed": observed == expected, "group_count": len(observed)}
    started = time.perf_counter()
    json.dumps(result)
    result["report_serialization_seconds"] = time.perf_counter() - started
    return result


def boundary_examples():
    """Separate chemical profile/refusal demonstrations from the timed shared scope."""
    left = Chem.MolFromSmiles("COP(=O)(N)[18O-]")
    right = Chem.MolFromSmiles("COP(=[18O])(N)[O-]")
    rows = []
    for profile, expected_equal in (("chemical", True), ("lewis_state", False), ("acs_topology", True)):
        original = canonicalize_rdkit_identity(left, identity_profile=profile)
        alternate = canonicalize_rdkit_identity(right, identity_profile=profile)
        equal = original.canonical_code == alternate.canonical_code
        rows.append({"profile": profile, "expected_equal": expected_equal, "observed_equal": equal,
                     "passed": equal == expected_equal, "schema": original.schema})
    partial = Chem.MolFromSmiles("C[C@H](Cl)C(Cl)C")
    verdict = classify_rdkit_stereograph_mirror(partial, identity_profile="chemical", require_complete=True)
    return {"profile_pair": rows, "strict_partial_input_status": verdict.status.value,
            "strict_partial_input_refuses": verdict.status is StereographMirrorStatus.INCOMPLETE,
            "scope": ("source-declared profile identity pair and strict partial-input refusal; "
                      "outside timed pairwise scope"),
            "passed": all(row["passed"] for row in rows) and verdict.status is StereographMirrorStatus.INCOMPLETE}


def run(args):
    source = capture_source()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    maximum_rows, evidence = collection(max(args.size))
    manifest_path = args.output.with_name(args.output.stem + "_inputs.json")
    manifest_path.write_text(json.dumps({"schema": "synkit.registration-inputs/1", "rows": maximum_rows,
                                        "independent_reference": evidence, "sizes": args.size}, indent=2) + "\n")
    records = []
    for size in args.size:
        for repetition in range(args.repetitions):
            # Alternate method order to expose rather than silently favor warmed imports/caches.
            methods = ("pairwise", "certificates") if repetition % 2 == 0 else ("certificates", "pairwise")
            for method in methods:
                result = workload(maximum_rows[:size], method, args.timeout)
                result.update({"size": size, "repetition": repetition, "method": method})
                records.append(result)
        print(json.dumps({"size": size, "all_partitions_passed": all(
            row["partition_passed"] for row in records if row["size"] == size)}), flush=True)
    summary = []
    for size in args.size:
        row = {"size": size}
        for method in ("pairwise", "certificates"):
            selected = [value for value in records if value["size"] == size and value["method"] == method]
            row[method] = {"outcomes": dict(Counter(value["status"] for value in selected)),
                           "total_seconds_completed_only": distribution(
                               [value["total_seconds"] for value in selected if value["status"] == "complete"])}
        left = row["pairwise"]["total_seconds_completed_only"]["median"]
        right = row["certificates"]["total_seconds_completed_only"]["median"]
        row["pairwise_over_certificates_median_ratio"] = left / right if left is not None and right else None
        summary.append(row)
    boundaries = boundary_examples()
    report = {"schema": "synkit.curated-stereo-registration/1", "provenance": finish_source(source),
              "hardware": hardware(), "inputs": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
              "protocol": {"sizes": args.size, "repetitions": args.repetitions,
                           "timeout_seconds_per_pipeline": args.timeout,
                           "pairwise_strategy": "compare with existing group representatives",
                           "certificate_strategy": ("one full certificate per record; "
                                                    "digest index plus exact byte equality"),
                           "timing_scope": ("input preparation + computation + grouping/storage; "
                                            "report serialization separate"),
                           "worker_processes": 1, "method_order": "alternated by repetition",
                           "native_thread_environment": {key: os.environ.get(key, "unset") for key in (
                               "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                           "memory_scope": "whole study process peak RSS; not isolated per pipeline",
                           "chemical_holdout": False},
              "records": records, "summary": summary, "boundary_examples": boundaries,
              "process_peak_rss_mib": peak_rss_mib(),
              "passed": boundaries["passed"] and all(row["partition_passed"] for row in records),
              "claim_boundary": ("An empirical crossover is measured; it is not assumed. "
                                 "Unresolved inputs get no identity group.")}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, action="append")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    args.size = args.size or list(SIZES)
    if args.repetitions < 1 or args.timeout <= 0 or min(args.size) < 1:
        parser.error("Positive sizes, repetition count and timeout are required.")
    RDLogger.DisableLog("rdApp.*")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
