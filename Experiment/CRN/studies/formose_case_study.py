#!/usr/bin/env python
"""Expand the formose network from rules and analyse it structurally.

Starts from formaldehyde and glycolaldehyde and applies four rules for aldol
addition, retro-aldol cleavage, and reversible keto--enol tautomerization.
The resulting network is analysed with the same methods as the curated pathways.

The primary check is that the single conservation law of the iteration-4
network assigns each species a coefficient equal to its carbon count. Carbon
conservation is recovered from network structure. No atom count,
molecular formula, or chemical annotation enters the structural computation.

Emits schema ``synkit.crn-formose/1``.

.. rubric:: Example

.. code-block:: bash

    python Experiment/CRN/studies/formose_case_study.py --output crn-formose.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Experiment.CRN.tools.common import environment, timed, write_report, native_experiment  # noqa: E402
from synkit.CRN import (  # noqa: E402
    CRNExpand,
    SynCRN,
    conserved_moieties,
    crnt_summary,
    find_siphons,
    find_t_semiflows,
    find_traps,
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


def expand(
    repeats: int,
    record: Optional[Dict[str, Any]] = None,
    *,
    rule_backend: Optional[str] = None,
) -> SynCRN:
    """Expand the formose network for a given number of iterations.

    The SynReactor C++ DPO controller is selected when its own native context
    is active. Otherwise the genuine Python SynReactor is used. RDKit reaction
    execution remains available only as an explicitly requested control.

    :param repeats: Number of rule-application iterations.
    :type repeats: int
    :param record: Optional mapping that receives the expansion backend and
        the native controller's per-step statistics.
    :type record: Optional[Dict[str, Any]]
    :param rule_backend: Explicit generator selection, or ``None`` to prefer
        the native DPO context and otherwise use Python SynReactor.
    :type rule_backend: Optional[str]
    :return: Resulting network.
    :rtype: SynCRN
    """
    from synkit.CRN.Construct.native_dpo import resolve_native_dpo_expander

    if rule_backend is None:
        rule_backend = "synreactor-native" if resolve_native_dpo_expander() is not None else "synreactor"
    if rule_backend == "rdkit-native":
        rule_backend = "native"
    dpo = rule_backend.startswith("synreactor")
    full_native_dpo = rule_backend == "synreactor-native"
    builder = CRNExpand(
        rules=RULES,
        repeats=repeats,
        keep_aam=False,
        rule_backend=rule_backend,
        explicit_h=False,
        implicit_temp=dpo,
        hydrogen_model="lewis" if full_native_dpo or rule_backend == "native" else "explicit",
        # Generic mapped formose templates do not constrain radical counts.
        dpo_radical_policy="ignore" if dpo else "strict",
        dpo_component_policy="union",
    )
    crn = SynCRN.from_digraph(builder.build(SEEDS))
    if record is not None:
        native = builder.dpo_report if dpo else builder.native_report
        record.update(
            rule_backend=rule_backend,
            engine=(native or {}).get("engine", "synreactor_python" if dpo else "rdkit_reaction_python"),
            reaction_engine="SynReactor DPO" if dpo else "RDKit reaction SMARTS",
            controller=(native or {}).get(
                "controller", "cpp" if full_native_dpo or rule_backend == "native" else "python"
            ),
            kernel=(native or {}).get(
                "kernel", "synkit_dpo_cpp" if full_native_dpo else "python" if dpo else "rdkit_cpp"
            ),
            hydrogen_model=(native or {}).get("hydrogen_model", "implicit" if dpo else builder.hydrogen_model),
            implicit_h=dpo or builder.hydrogen_model == "lewis",
            radical_policy="ignore" if dpo else None,
            component_policy=(native or {}).get("component_policy", "union" if dpo else None),
            native=native,
        )
    return crn


def formose_report(
    iterations: Sequence[int],
    *,
    analyse_at: int,
    rule_backend: Optional[str] = None,
) -> Dict[str, Any]:
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
        expansion: Dict[str, Any] = {}
        seconds, crn, error = timed(lambda r=repeats: expand(r, expansion, rule_backend=rule_backend))
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
                "rule_backend": expansion.get("rule_backend"),
                "engine": expansion.get("engine"),
                "reaction_engine": expansion.get("reaction_engine"),
                "controller": expansion.get("controller"),
                "kernel": expansion.get("kernel"),
                "hydrogen_model": expansion.get("hydrogen_model"),
                "implicit_h": expansion.get("implicit_h"),
                "radical_policy": expansion.get("radical_policy"),
                "component_policy": expansion.get("component_policy"),
                "native_steps": (expansion.get("native") or {}).get("steps"),
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
            {
                "smiles": smiles[i],
                "coefficient": int(c),
                "carbons": _carbon_count(smiles[i]),
            }
            for i, c in enumerate(moieties[0])
            if c
        ]
        carbon_law = {
            "n_species": len(pairs),
            "coefficient_equals_carbon_count": all(p["coefficient"] == p["carbons"] for p in pairs),
            "entries": pairs,
        }

    siphon_seconds, siphons, _ = timed(lambda: find_siphons(crn, strict_limit=True))
    trap_seconds, traps, _ = timed(lambda: find_traps(crn, strict_limit=True))
    t_semiflow_seconds, t_semiflows, _ = timed(lambda: find_t_semiflows(crn))
    persistence_seconds, persistence, _ = timed(lambda: siphon_persistence_details(crn).persistence_ok)

    checks = {
        "all_structural_analyses_complete": all(
            value is not None for value in (siphons, traps, t_semiflows, persistence)
        ),
        "iteration_4_size_is_stable": any(
            g.get("iteration") == 4 and g.get("n_species") == 37 and g.get("n_reactions") == 44 for g in growth
        ),
        "deficiency_identity_holds": report.deficiency == report.n_complexes - report.n_linkage_classes - report.rank,
        "single_conservation_law": len(laws) == 1,
        "single_minimal_p_semiflow": len(moieties) == 1,
        "two_minimal_siphons": siphons is not None and len(siphons) == 2,
        "conservation_law_is_carbon_count": bool(carbon_law and carbon_law["coefficient_equals_carbon_count"]),
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
        "generation": {
            key: next(row for row in growth if row["iteration"] == analyse_at).get(key)
            for key in (
                "rule_backend",
                "engine",
                "reaction_engine",
                "controller",
                "kernel",
                "hydrogen_model",
                "implicit_h",
                "radical_policy",
                "component_policy",
            )
        },
        "checks": checks,
        "growth": growth,
        "structure": {
            "n_species": report.n_species,
            "n_reactions": report.n_reactions,
            "n_complexes": report.n_complexes,
            "n_linkage_classes": report.n_linkage_classes,
            "n_terminal_strong_linkage_classes": (report.n_terminal_strong_linkage_classes),
            "rank": report.rank,
            "deficiency": report.deficiency,
            "weakly_reversible": report.is_weakly_reversible,
            "reversible": report.is_reversible,
            "n_conservation_laws": len(laws),
            "n_conserved_moieties": len(moieties),
            "n_minimal_p_semiflows": len(moieties),
            "n_minimal_t_semiflows": (None if t_semiflows is None else int(t_semiflows.shape[1])),
            "right_nullity": report.n_reactions - report.rank,
            "n_minimal_siphons": None if siphons is None else len(siphons),
            "minimal_siphons": (None if siphons is None else [sorted(group) for group in siphons]),
            "siphon_seconds": siphon_seconds,
            "n_minimal_traps": None if traps is None else len(traps),
            "trap_seconds": trap_seconds,
            "t_semiflow_seconds": t_semiflow_seconds,
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
        "--generation-backend",
        choices=["auto", "synreactor-native", "synreactor", "rdkit", "rdkit-native"],
        default="auto",
        help="auto prefers full SynReactor C++ DPO; RDKit engines are explicit controls",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="keep the expansion's INFO logging",
    )
    arguments = parser.parse_args()

    if not arguments.verbose:
        logging.disable(logging.INFO)

    report = formose_report(
        arguments.iterations,
        analyse_at=arguments.analyse_at,
        rule_backend=None if arguments.generation_backend == "auto" else arguments.generation_backend,
    )
    return write_report(report, arguments.output)


main = native_experiment(main, expansion=True)

if __name__ == "__main__":
    raise SystemExit(main())
