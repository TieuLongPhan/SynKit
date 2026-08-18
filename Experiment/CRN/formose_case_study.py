#!/usr/bin/env python
"""Expand the formose network from rules and analyse it structurally.

Generates a chemical space from two seed molecules (formaldehyde and
glycolaldehyde) and four rules --- aldol addition, retro-aldol cleavage, and
reversible keto--enol tautomerization --- then analyses the resulting network
with the same stack applied to curated pathways.

The headline check is that the single conservation law of the iteration-4
network assigns each species a coefficient equal to its carbon count. Carbon
conservation is therefore *recovered* from network structure: no atom count,
molecular formula, or chemical annotation enters the structural computation.

Emits schema ``synkit.crn-formose/1``.

.. rubric:: Example

.. code-block:: bash

    python Experiment/CRN/formose_case_study.py --output crn-formose.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Experiment.CRN.common import environment, timed, write_report  # noqa: E402
from synkit.CRN import (  # noqa: E402
    CRNExpand,
    SynCRN,
    conserved_moieties,
    crnt_summary,
    find_siphons,
    integer_conservation_laws,
    siphon_persistence_details,
)
from synkit.CRN.Props.helper import _species_and_rule_order  # noqa: E402

SCHEMA = "synkit.crn-formose/1"

#: Formaldehyde and glycolaldehyde.
SEEDS = ["C=O", "OCC=O"]

#: Keto--enol tautomerization (forward and reverse), aldol addition, retro-aldol.
RULES = [
    "[C:1]([C:2]=[O:3])[H:4]>>[C:1]=[C:2][O:3][H:4]",
    "[C:1]=[C:2][O:3][H:4].[O:5]=[C:6]>>[C:1]([C:2]=[O:3])[C:6][O:5][H:4]",
    "[C:1]=[C:2][O:3][H:4]>>[C:1]([C:2]=[O:3])[H:4]",
    "[C:1]([C:2]=[O:3])[C:6][O:5][H:4]>>[C:1]=[C:2][O:3][H:4].[O:5]=[C:6]",
]


def _carbon_count(smiles: str) -> int:
    """Count carbon atoms in a SMILES string.

    :param smiles: SMILES string.
    :type smiles: str
    :return: Number of carbon atoms, or ``-1`` when the string does not parse.
    :rtype: int
    """
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return -1
    return sum(1 for atom in molecule.GetAtoms() if atom.GetSymbol() == "C")


def _matrix_species_smiles(crn: SynCRN) -> List[str]:
    """Return species SMILES in stoichiometric-matrix row order.

    :param crn: Network to inspect.
    :type crn: SynCRN
    :return: SMILES aligned with the rows of ``S``.
    :rtype: List[str]
    """
    order, _, _, _ = _species_and_rule_order(crn)
    lookup: Dict[str, str] = {}
    for sid, record in crn.species.items():
        value = record.smiles or record.label or str(sid)
        lookup[str(sid)] = value
        lookup[str(record.source_node_id)] = value
    return [lookup.get(str(node), str(node)) for node in order]


def expand(repeats: int) -> SynCRN:
    """Expand the formose network for a given number of iterations.

    :param repeats: Number of rule-application iterations.
    :type repeats: int
    :return: Resulting network.
    :rtype: SynCRN
    """
    graph = CRNExpand(rules=RULES, repeats=repeats, keep_aam=False).build(SEEDS)
    return SynCRN.from_digraph(graph)


def formose_report(iterations: Sequence[int], *, analyse_at: int) -> Dict[str, Any]:
    """Expand across iterations and analyse the chosen one structurally.

    :param iterations: Iteration counts to expand and record sizes for.
    :type iterations: Sequence[int]
    :param analyse_at: Iteration to analyse in full.
    :type analyse_at: int
    :return: Report payload.
    :rtype: Dict[str, Any]
    """
    growth = []
    networks: Dict[int, SynCRN] = {}
    for repeats in iterations:
        seconds, crn, error = timed(lambda r=repeats: expand(r))
        if crn is None:
            growth.append({"iteration": repeats, "error": error})
            continue
        networks[repeats] = crn
        growth.append(
            {
                "iteration": repeats,
                "n_species": crn.n_species,
                "n_reactions": crn.n_reactions,
                "seconds": seconds,
            }
        )

    crn = networks.get(analyse_at)
    if crn is None:
        return {
            "schema": SCHEMA,
            "status": "FAIL",
            "error": f"iteration {analyse_at} was not expanded",
            "environment": environment(),
            "growth": growth,
        }

    report = crnt_summary(crn)
    laws = integer_conservation_laws(crn)
    moieties = conserved_moieties(crn)
    smiles = _matrix_species_smiles(crn)

    carbon_law = None
    if len(moieties) == 1:
        pairs = [
            {"smiles": smiles[i], "coefficient": int(c), "carbons": _carbon_count(smiles[i])}
            for i, c in enumerate(moieties[0])
            if c
        ]
        carbon_law = {
            "n_species": len(pairs),
            "coefficient_equals_carbon_count": all(
                p["coefficient"] == p["carbons"] for p in pairs
            ),
            "entries": pairs,
        }

    siphon_seconds, siphons, _ = timed(lambda: find_siphons(crn))
    persistence_seconds, persistence, _ = timed(
        lambda: siphon_persistence_details(crn).persistence_ok
    )

    checks = {
        "iteration_4_size_is_stable": any(
            g.get("iteration") == 4
            and g.get("n_species") == 37
            and g.get("n_reactions") == 44
            for g in growth
        ),
        "deficiency_identity_holds": report.deficiency
        == report.n_complexes - report.n_linkage_classes - report.rank,
        "single_conservation_law": len(laws) == 1,
        "conservation_law_is_carbon_count": bool(
            carbon_law and carbon_law["coefficient_equals_carbon_count"]
        ),
    }

    return {
        "schema": SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "claim_boundary": (
            "The expansion enumerates rule applications; it does not model "
            "kinetics, selectivity, thermodynamic favourability or yield. "
            "Recovering carbon conservation is a statement about the generated "
            "stoichiometry, not evidence about prebiotic plausibility."
        ),
        "environment": environment(),
        "inputs": {"seeds": SEEDS, "rules": RULES, "analysed_iteration": analyse_at},
        "checks": checks,
        "growth": growth,
        "structure": {
            "n_species": report.n_species,
            "n_reactions": report.n_reactions,
            "n_complexes": report.n_complexes,
            "n_linkage_classes": report.n_linkage_classes,
            "n_terminal_strong_linkage_classes": (
                report.n_terminal_strong_linkage_classes
            ),
            "rank": report.rank,
            "deficiency": report.deficiency,
            "weakly_reversible": report.is_weakly_reversible,
            "reversible": report.is_reversible,
            "n_conservation_laws": len(laws),
            "n_conserved_moieties": len(moieties),
            "n_minimal_siphons": None if siphons is None else len(siphons),
            "siphon_seconds": siphon_seconds,
            "persistent": persistence,
            "persistence_seconds": persistence_seconds,
        },
        "carbon_conservation": carbon_law,
    }


def main() -> int:
    """Parse arguments, run the study, and write the evidence file.

    :return: Process exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--analyse-at", type=int, default=4)
    parser.add_argument("--output", type=Path, help="write the evidence file here")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="keep the expansion's INFO logging",
    )
    arguments = parser.parse_args()

    if not arguments.verbose:
        logging.disable(logging.INFO)

    report = formose_report(arguments.iterations, analyse_at=arguments.analyse_at)
    return write_report(report, arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
