"""Replay the chemical and graph examples used by the two explanatory figures."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository.resolve()
    sys.path.insert(0, str(root))
    import networkx as nx
    from rdkit import Chem
    from synkit.Graph.Canon.exact import ExactColoredGraphCanonicalizer
    from synkit.Graph.Stereo import classify_rdkit_configured_stereograph_mirror

    source = (root / "paper/stereo_graph/fig/incomplete_completions.tex").read_text()
    if (r"\node[atom] at ({\x-1.05},-2.0) {H};" not in source or
            r"\node[atom] at ({\x+1.05},-2.0) {Cl};" not in source):
        raise RuntimeError("The upper Fischer arms changed.")
    pattern = r"\\foreach \\x/\\armleft/\\armright in \{([^}]+)\}"
    arms = [row.split("/") for row in re.search(pattern, source).group(1).split(",")]
    lower_labels = [fischer_cip(1 if right == "Cl" else -1, 1) for _x, _left, right in arms]
    upper_label = fischer_cip(1, -1)
    cases = [("partial", "C[C@H](Cl)C(Cl)C", "incomplete", [upper_label, "?"]),
             ("meso", "C[C@H](Cl)[C@H](Cl)C", "achiral", [upper_label, lower_labels[0]]),
             ("chiral", "C[C@H](Cl)[C@@H](Cl)C", "chiral", [upper_label, lower_labels[1]])]
    chemical = []
    for name, smiles, expected, cip in cases:
        molecule = Chem.MolFromSmiles(smiles)
        labels = [label for _index, label in
                  Chem.FindMolChiralCenters(molecule, includeUnassigned=True, useLegacyImplementation=False)]
        if labels != cip:
            raise RuntimeError("The figure's Fischer convention disagrees with source configuration.")
        rows = []
        for profile in ("chemical", "lewis_state", "acs_topology"):
            for reverse in (False, True):
                order = list(range(molecule.GetNumAtoms()))
                if reverse:
                    order.reverse()
                result = classify_rdkit_configured_stereograph_mirror(
                    Chem.RenumberAtoms(molecule, order), identity_profile=profile)
                if result.status.value != expected:
                    raise RuntimeError("A depicted strict mirror outcome differs.")
                rows.append({"profile": profile, "reverse_atom_order": reverse,
                             "strict_status": result.status.value})
        chemical.append({"case": name, "smiles": smiles, "cip_at_carbon_2_and_3": labels, "checks": rows})
    graphs = [nx.cycle_graph(range(1, 7))]
    graphs.append(nx.relabel_nodes(graphs[0], dict(zip(range(1, 7), (4, 1, 6, 2, 5, 3)))))
    graphs.append(nx.compose(nx.cycle_graph((1, 2, 3)), nx.cycle_graph((4, 5, 6))))
    codes, references, signatures = [], [], []
    for graph in graphs:
        nx.set_node_attributes(graph, "blue", "color")
        nx.set_edge_attributes(graph, "uniform", "color")
        result = ExactColoredGraphCanonicalizer(graph).canonicalize()
        codes.append(result.canonical_code)
        signatures.append((len(graph), tuple(sorted(dict(graph.degree()).values()))))
        # Separate exhaustive serialization on six vertices, without the
        # production canonizer or its refinement/search implementation.
        encodings = (tuple(int(graph.has_edge(order[i], order[j]))
                           for i in range(6) for j in range(i + 1, 6))
                     for order in itertools.permutations(graph))
        references.append(min(encodings))
    if not (signatures[0] == signatures[1] == signatures[2] and
            codes[0] == codes[1] != codes[2] and references[0] == references[1] != references[2]):
        raise RuntimeError("The depicted invariant collision or exact partition is false.")
    figures = ["incomplete_completions.tex", "invariant_vs_certificate.tex"]
    report = {"schema": "synkit.stereo-explanatory-figure-examples/1", "passed": True,
              "driver_sha256": sha256(Path(__file__)),
              "figure_source_sha256": {name: sha256(root / "paper/stereo_graph/fig" / name) for name in figures},
              "chemical_cases": chemical, "strict_chemical_checks": 18,
              "fischer_projection_convention_checked": True,
              "schematic_graphs": 3, "reference_serializations_enumerated": 2160,
              "count_degree_invariants_equal": True, "native_and_independent_exact_partition": [[0, 1], [2]],
              "boundary": (
                  "Three retained molecules replay under three profiles and two atom "
                  "orders. The drawn Fischer arms determine S/R after reversing for "
                  "horizontal, viewer-facing H; those labels match the source strings. "
                  "Independent exhaustive adjacency serialization verifies the schematic "
                  "graph partition. These are explanatory examples, not new benchmark "
                  "populations or experimental stability evidence.")}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "chemical_checks": 18, "independent_serializations": 2160}))
    return 0


def fischer_cip(chlorine_side, adjacent_carbon_direction):
    """Cl > adjacent configured carbon > Me > H; horizontal H faces viewer."""
    first = (chlorine_side, 0)
    second = (0, adjacent_carbon_direction)
    third = (0, -adjacent_carbon_direction)
    cross = ((second[0] - first[0]) * (third[1] - first[1]) -
             (second[1] - first[1]) * (third[0] - first[0]))
    return "R" if cross > 0 else "S"


if __name__ == "__main__":
    raise SystemExit(main())
