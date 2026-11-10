#!/usr/bin/env python
"""Compare explicit and Lewis-state hydrogen models in native rule expansion.

Explicit hydrogens are graph vertices that rule templates match one by one, and
RDKit restores the hydrogen counts of product atoms from default valences. The
Lewis-state model keeps hydrogens as per-atom count labels, compiles each mapped
H move of a rule into count deltas, and sets product counts exactly.

Two controls are reported. The formose network must be identical under both
models at every depth, and the Lewis model must be faster where matching
dominates. Three small rule sets probe hydrogen bookkeeping: hydrogen-atom
transfer, proton transfer between bracket atoms with fixed hydrogen counts,
and an incomplete insertion rule that leaves an open valence. Every reaction is
checked for atom (including H) and charge balance.

Emits schema ``synkit.crn-hydrogen/1``.

.. rubric:: Example

.. code-block:: bash

    python Experiment/CRN/studies/hydrogen_accounting.py --output crn-hydrogen.json
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Experiment.CRN.studies.formose_case_study import RULES, SEEDS  # noqa: E402
from Experiment.CRN.tools.common import environment, write_report  # noqa: E402
from synkit.CRN import CRNExpand  # noqa: E402

SCHEMA = "synkit.crn-hydrogen/1"
MODELS = ("explicit", "lewis")
FORMOSE_DEPTHS = (4, 5, 6)

#: Small rule sets whose hydrogen bookkeeping differs between the models.
BALANCE_CASES: List[Dict[str, Any]] = [
    {
        "name": "hydrogen_atom_transfer",
        "label": "H-atom transfer",
        "rules": ["[C:1][H:2].[O:3]>>[C:1].[O:3][H:2]"],
        "seeds": ["CC", "[OH]"],
        "repeats": 1,
    },
    {
        "name": "proton_transfer",
        "label": "Proton transfer",
        "rules": [
            "[N+:1][H:2].[O-:3]>>[N:1].[O:3][H:2]",
            "[N:1].[O:3][H:2]>>[N+:1][H:2].[O-:3]",
        ],
        "seeds": ["C[NH3+]", "CC(=O)[O-]", "[NH4+]"],
        "repeats": 2,
    },
    {
        "name": "incomplete_insertion",
        "label": "Incomplete insertion rule",
        "rules": [
            "[C:1]=[C:2].[Pd:3]>>[C:1]([Pd:3])[C:2]",
            "[C:1]([Pd:3])[C:2]>>[C:1]=[C:2].[Pd:3]",
        ],
        "seeds": ["C=CC=O", "[Pd]"],
        "repeats": 3,
    },
]


def _inventory(smiles: str) -> Dict[str, int]:
    """Count atoms, hydrogens included, and the formal charge of one species."""
    from rdkit import Chem

    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    counts: Dict[str, int] = defaultdict(int)
    for atom in molecule.GetAtoms():
        counts[atom.GetSymbol()] += 1
    counts["charge"] = Chem.GetFormalCharge(molecule)
    return counts


def _reactions(graph) -> List[Dict[str, Any]]:
    """Return each reaction with its sides and net atom/charge change."""
    smiles = {n: d["smiles"] for n, d in graph.nodes(data=True) if d["kind"] == "species"}
    out = []
    for node, data in graph.nodes(data=True):
        if data["kind"] != "rule":
            continue
        net: Dict[str, int] = defaultdict(int)
        left, right = [], []
        for u, _, edge in graph.in_edges(node, data=True):
            left += [smiles[u]] * edge["stoich"]
            for key, value in _inventory(smiles[u]).items():
                net[key] -= value * edge["stoich"]
        for _, v, edge in graph.out_edges(node, data=True):
            right += [smiles[v]] * edge["stoich"]
            for key, value in _inventory(smiles[v]).items():
                net[key] += value * edge["stoich"]
        imbalance = {key: value for key, value in net.items() if value}
        out.append(
            {
                "rule_index": int(data["rule_index"]),
                "reactants": sorted(left),
                "products": sorted(right),
                "imbalance": imbalance,
            }
        )
    return out


def _signature(graph):
    """Species and reaction sets independent of node numbering."""
    from rdkit import Chem

    species = {
        n: Chem.MolToSmiles(Chem.MolFromSmiles(d["smiles"]))
        for n, d in graph.nodes(data=True)
        if d["kind"] == "species"
    }
    reactions = sorted(
        (
            d["rule_index"],
            tuple(sorted((species[u], e["stoich"]) for u, _, e in graph.in_edges(n, data=True))),
            tuple(sorted((species[v], e["stoich"]) for _, v, e in graph.out_edges(n, data=True))),
        )
        for n, d in graph.nodes(data=True)
        if d["kind"] == "rule"
    )
    return sorted(species.values()), reactions


def _expand(rules, seeds, repeats, model):
    builder = CRNExpand(
        rules=rules,
        repeats=repeats,
        keep_aam=False,
        rule_backend="native",
        hydrogen_model=model,
    )
    started = time.perf_counter()
    graph = builder.build(list(seeds))
    seconds = time.perf_counter() - started
    return graph, seconds, builder.native_report


def hydrogen_report(depths=FORMOSE_DEPTHS) -> Dict[str, Any]:
    """Run both hydrogen models on formose and on the balance cases."""
    from synkit.CRN.Construct.native_expand import resolve_expander

    if resolve_expander() is None:
        return {
            "schema": SCHEMA,
            "status": "FAIL",
            "error": "a native rule expander is required",
            "checks": {"native_expander_available": False},
            "environment": environment(),
        }

    # Start the expander process before timing so no row pays start-up.
    for model in MODELS:
        _expand(RULES, SEEDS, 1, model)
    formose = []
    for depth in depths:
        row: Dict[str, Any] = {"depth": depth}
        signatures = {}
        for model in MODELS:
            graph, seconds, native = _expand(RULES, SEEDS, depth, model)
            signatures[model] = _signature(graph)
            steps = native["steps"]
            row[model] = {
                "n_species": len(signatures[model][0]),
                "n_reactions": len(signatures[model][1]),
                "seconds": seconds,
                "rule_applications": sum(s["applications"] for s in steps),
                "match_outcomes": sum(s["outcomes"] for s in steps),
                "lewis_rejected": sum(s["lewis_rejected"] for s in steps),
                "rule_models": [r["hydrogen_model"] for r in native["rules"]],
            }
        row["identical"] = signatures["explicit"] == signatures["lewis"]
        row["speedup"] = row["explicit"]["seconds"] / row["lewis"]["seconds"]
        formose.append(row)

    cases = []
    for case in BALANCE_CASES:
        entry: Dict[str, Any] = {key: case[key] for key in ("name", "label", "rules", "seeds", "repeats")}
        for model in MODELS:
            graph, _, native = _expand(case["rules"], case["seeds"], case["repeats"], model)
            reactions = _reactions(graph)
            entry[model] = {
                "n_reactions": len(reactions),
                "n_balanced": sum(1 for r in reactions if not r["imbalance"]),
                "reactions": reactions,
                "rule_models": [r["hydrogen_model"] for r in native["rules"]],
            }
        cases.append(entry)

    deepest = formose[-1]
    checks = {
        "formose_identical_at_every_depth": all(row["identical"] for row in formose),
        "formose_rules_compiled_to_lewis": all(m == "lewis" for row in formose for m in row["lewis"]["rule_models"]),
        "lewis_faster_at_deepest_formose": deepest["speedup"] > 1.0,
        "lewis_reactions_all_balanced": all(c["lewis"]["n_balanced"] == c["lewis"]["n_reactions"] > 0 for c in cases),
    }
    return {
        "schema": SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "claim_boundary": (
            "Balance checks concern atom and charge conservation of the "
            "generated reactions. They do not assess whether a rule or its "
            "products are chemically plausible; the incomplete insertion rule "
            "is deliberately underspecified."
        ),
        "environment": environment(),
        "checks": checks,
        "formose": formose,
        "balance_cases": cases,
    }


def main() -> int:
    """Parse arguments, run the study, and write the evidence file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--depths", type=int, nargs="+", default=list(FORMOSE_DEPTHS))
    parser.add_argument("--output", type=Path, help="write the evidence file here")
    arguments = parser.parse_args()
    logging.disable(logging.INFO)
    return write_report(hydrogen_report(arguments.depths), arguments.output)


from Experiment.CRN.tools.common import native_experiment  # noqa: E402

main = native_experiment(main, expansion=True)

if __name__ == "__main__":
    raise SystemExit(main())
