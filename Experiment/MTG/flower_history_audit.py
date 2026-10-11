#!/usr/bin/env python
"""Audit a pinned published FlowER prediction graph without running inference.

The released example is a pilot, not a representative accuracy benchmark.
Endpoint label projections are compared with supplied atom-origin queries.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import importlib.metadata as metadata
from pathlib import Path
import pickle
import platform
import sys

import networkx as nx
from networkx.classes import coreviews, reportviews
import numpy as np
from numpy.core.multiarray import scalar
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.MTG.isotope_case import isotope_llg  # noqa: E402
from synkit.Graph.Morphism import LLGError  # noqa: E402
from synkit.IO.mol_to_graph import MolToGraph  # noqa: E402
from synkit.Rule.Apply import DPOError, RuleSpan, apply_dpo  # noqa: E402

PUBLISHED_COMMIT = "00b93aa7f46c266a94d249066edfb1ee833cb1bc"
PUBLISHED_FILE = "examples/samples/unrecognized_pistachio.pkl"


class _ExampleReader(pickle.Unpickler):
    def find_class(self, module, name):
        known = {
            ("networkx.classes.digraph", "DiGraph"): nx.DiGraph,
            ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "scalar"): scalar,
        }
        if (module, name) in known:
            return known[(module, name)]
        if module == "networkx.classes.reportviews" and name in (
            "NodeView",
            "InDegreeView",
            "OutDegreeView",
            "OutEdgeView",
            "InEdgeView",
        ):
            return getattr(reportviews, name)
        if module == "networkx.classes.coreviews" and name in (
            "AdjacencyView",
            "AtlasView",
        ):
            return getattr(coreviews, name)
        raise ValueError(f"Unsupported released-example object: {module}.{name}")


def _molecule(smiles):
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    molecule = Chem.MolFromSmiles(smiles, parser)
    if molecule is None:
        raise ValueError(
            "Mapped prediction cannot be parsed under the stated RDKit policy."
        )
    maps = [atom.GetAtomMapNum() for atom in molecule.GetAtoms()]
    if any(value <= 0 for value in maps) or len(set(maps)) != len(maps):
        raise ValueError(
            "Every explicit predicted atom needs one distinct supplied map."
        )
    return molecule


def _state(smiles):
    molecule = _molecule(smiles)
    flags = {a.GetAtomMapNum(): a.GetIsAromatic() for a in molecule.GetAtoms()}
    Chem.Kekulize(molecule, clearAromaticFlags=True)
    graph = MolToGraph(include_stereo_descriptors=False).transform(molecule)
    for _, attrs in graph.nodes(data=True):
        attrs["aromatic"] = flags[attrs["atom_map"]]
    return isotope_llg(graph)


def _published_projection(smiles):
    # This reproduces the projection's documented fields, not the complete
    # published ranking pipeline. Stereo/isotope are lost by isomeric=False.
    molecule = _molecule(smiles)
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, isomericSmiles=False)


def _oxygen_hydrogen_origins(state):
    graph = state.to_networkx()
    return tuple(
        sorted(
            (
                node,
                tuple(
                    sorted(n for n in graph[node] if graph.nodes[n]["element"] == "H")
                ),
            )
            for node, attrs in graph.nodes(data=True)
            if attrs["element"] == "O"
        )
    )


def _duet_excess(state):
    graph = state.to_networkx()
    return [
        n
        for n, a in graph.nodes(data=True)
        if a["element"] == "H"
        and (
            2 * a["lone_pairs"]
            + a["radical"]
            + 2 * a["hcount"]
            + 2 * sum(e["sigma_order"] + e["pi_order"] for e in graph[n].values())
        )
        > 2
    ]


def _audit_graph(graph, root, goals, attained):
    states, projections, node_rows = {}, {}, []
    classes = defaultdict(list)
    for index, smiles in enumerate(graph):
        row = {"node": index, "depth": graph.nodes[smiles].get("depth")}
        try:
            states[smiles] = _state(smiles)
            projections[smiles] = _published_projection(smiles)
            classes[projections[smiles]].append(smiles)
            row.update(
                {
                    "status": "SUPPORTED",
                    "hydrogen_duet_excess": _duet_excess(states[smiles]),
                }
            )
        except (ValueError, LLGError) as error:
            row.update(
                {"status": "UNSUPPORTED_REPRESENTATION", "message": str(error)[:240]}
            )
        node_rows.append(row)
    index_of = {s: i for i, s in enumerate(graph)}
    edges = []
    terminal_loops = []
    for source, target, edge_metadata in graph.edges(data=True):
        row = {
            "source": index_of[source],
            "target": index_of[target],
            "rank": int(edge_metadata["rank"]),
            "sample_count": int(edge_metadata["count"]),
        }
        if source == target:
            row["status"] = "TERMINAL_SELF_LOOP"
            terminal_loops.append(row)
            continue
        if source not in states or target not in states:
            row["status"] = "UNSUPPORTED_ENDPOINT"
        else:
            row["source_derived_net_charge"] = sum(
                a["charge"] for _, a in states[source].to_networkx().nodes(data=True)
            )
            row["target_derived_net_charge"] = sum(
                a["charge"] for _, a in states[target].to_networkx().nodes(data=True)
            )
            try:
                left, right = states[source], states[target]
                if set(left.node_ids) != set(right.node_ids):
                    raise ValueError(
                        "A prediction edge changes the supplied atom inventory."
                    )
                rule = RuleSpan.from_mapping(
                    left, right, {n: n for n in left.node_ids}, boundary="closed"
                )
                app = apply_dpo(rule, left, {n: n for n in left.node_ids})
                assert app.certificate.replay().valid
                assert app.result.is_isomorphic(right)
                row["status"] = "CERTIFIED_NATURAL_DPO"
            except (DPOError, LLGError, ValueError) as error:
                row["status"] = "REFUSED_UNDER_EXACT_REPRESENTATION"
                row["message"] = str(error)[:240]
        edges.append(row)
    collisions = []
    for projection, members in classes.items():
        origins = {_oxygen_hydrogen_origins(states[s]) for s in members}
        if len(origins) > 1:
            collisions.append(
                {
                    "members": [index_of[s] for s in members],
                    "origin_query_values": [list(v) for v in sorted(origins)],
                    "projected_state": projection,
                }
            )
    terminals = tuple(nx.nodes_with_selfloops(graph))
    nonloops = nx.DiGraph((a, b) for a, b in graph.edges if a != b)
    nonloops.add_nodes_from(graph)
    paths = []
    # Freeze a maximum of 15 edges, matching the released search-depth setting.
    for target in terminals:
        for path in nx.all_simple_paths(nonloops, root, target, cutoff=15):
            paths.append([index_of[s] for s in path])
    return {
        "nodes": node_rows,
        "transitions": edges,
        "terminal_self_loops": terminal_loops,
        "summary": {
            "nodes": len(graph),
            "nonterminal_edges": len(edges),
            "terminal_self_loops": len(terminal_loops),
            "node_status": dict(Counter(r["status"] for r in node_rows)),
            "edge_status": dict(Counter(r["status"] for r in edges)),
            "nodes_with_hydrogen_duet_excess": sum(
                bool(r.get("hydrogen_duet_excess")) for r in node_rows
            ),
            "projected_state_classes": len(classes),
            "classes_with_different_oxygen_hydrogen_origins": len(collisions),
            "simple_paths_to_terminal_with_at_most_15_edges": len(paths),
            "acyclic_without_self_loops": nx.is_directed_acyclic_graph(nonloops),
        },
        "projection_collisions": collisions,
        "bounded_terminal_paths": paths,
        "declared_product_targets": list(goals[1]),
        "published_attained_targets": sorted(attained),
        "scope": (
            "One author-released example. A supplied map is an anchor, not experimentally "
            "established atom identity."
        ),
    }


def run(path):
    RDLogger.DisableLog("rdApp.warning")
    with path.open("rb") as stream:
        examples = _ExampleReader(stream).load()
    reports = [_audit_graph(*item) for item in examples]
    return {
        "schema": "synkit.mtg-flower-published-example-audit/1",
        "source": {
            "repository": "https://github.com/FongMunHong/FlowER",
            "tag": "2.0.0",
            "commit": PUBLISHED_COMMIT,
            "file": PUBLISHED_FILE,
            "sha256": sha256(path.read_bytes()).hexdigest(),
        },
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "packages": {
                key: metadata.version(key) for key in ("networkx", "rdkit", "numpy")
            },
        },
        "policy": {
            "local_rdkit_sanitization": True,
            "explicit_hydrogens": True,
            "kekule_conversion": "Explicit bonds, retaining original atom aromatic flags.",
            "chemical_event_grouping_inferred": False,
            "inference_rerun": False,
            "origin_query": "Supplied hydrogen carriers bonded to each anchored oxygen.",
            "duet_test": (
                "Additional boundary predicate in the represented pair-bond model; not "
                "independent physical mechanism truth."
            ),
        },
        "examples": reports,
        "claim_limits": (
            "No population error rate, quantitative isotope prediction, kinetic equivalence "
            "or comparison with complete isotope-aware history baselines."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("released_example", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.released_example)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps([item["summary"] for item in report["examples"]]))


if __name__ == "__main__":
    main()
