#!/usr/bin/env python
"""Audit the printed SI reaction list and numerical conservation reconstruction.

This independent audit reads the reaction list exactly as supplied to reviewers,
records the equations in JSON, and replays the two cofactor-sensitive routes.
The numerical comparison deliberately distinguishes a valid floating subspace
from naive rounding into alleged integer conservation laws.
"""

from pathlib import Path
import argparse
import hashlib
import logging
import re
import sys

import numpy as np
from scipy.linalg import null_space

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Experiment.CRN.common import environment, write_report
from Experiment.CRN.review_experiments import execute
from Experiment.CRN.formose_case_study import expand
from synkit.CRN import SynCRN, crnt_summary, integer_conservation_laws
from synkit.CRN.Props.stoich import build_S_minus_plus
from synkit.CRN.Benchmark.networks import BENCHMARK_NETWORKS
from synkit.CRN.Symmetry.automorphism import CRNAutomorphism
from synkit.CRN.Symmetry._common import SymmetryConfig


def numerical_comparison():
    rows = []
    for fixture in BENCHMARK_NETWORKS:
        crn = fixture.build()
        _, _, minus, plus = build_S_minus_plus(crn)
        matrix = plus - minus
        numerical = null_space(matrix.T)
        rounded = np.rint(1000 * numerical).astype(np.int64)
        exact = np.asarray(integer_conservation_laws(crn), dtype=object)
        if exact.size:
            residual = exact @ matrix.astype(object).astype(int)
            exact_ok = all(x == 0 for x in residual.flat)
        else:
            exact_ok = True
        raw = matrix.T @ numerical
        rounded_residual = matrix.astype(np.int64).T @ rounded
        rows.append(
            {
                "network": fixture.name,
                "nullity": numerical.shape[1],
                "svd_max_abs_residual": float(np.max(np.abs(raw), initial=0)),
                "rounded_max_abs_residual": int(
                    np.max(np.abs(rounded_residual), initial=0)
                ),
                "rounded_invalid_vectors": int(
                    np.count_nonzero(np.any(rounded_residual != 0, axis=0))
                ),
                "exact_identity_holds": exact_ok,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--si", type=Path, default=ROOT / "paper/synkit_crn/SI/si.tex")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.disable(logging.INFO)
    source = args.si.read_text()
    rows = re.findall(r"^r(\d+) & (.*?) \\\\", source, re.M)
    equations = {
        f"r{n}": equation.replace(r"\(\rightarrow\)", ">>") for n, equation in rows
    }
    crn = SynCRN.from_reaction_strings(list(equations.values()))
    for reaction, label in zip(crn.reactions.values(), equations):
        reaction.label = label
    report = crnt_summary(crn)
    routes = []
    for atp in (1, 2):
        for name, flow in (
            ("ATP-dependent", {"r10": 1, "r15": 1, "r3": 1, "r7": 1}),
            ("ADP-dependent", {"r10": 1, "r15": 1, "r12": 1, "r7": 1}),
        ):
            routes.append(
                {
                    "name": name,
                    "atp_tokens": atp,
                    "outcome": execute(crn, flow, {"C": atp, "S": 1}),
                }
            )
    formose = expand(4)
    symmetry = {}
    for name, network in [("glycolysis", crn), ("formose", formose)]:
        summary = CRNAutomorphism(network, config=SymmetryConfig.topological()).summary(
            max_count=1024, timeout_sec=30
        )
        symmetry[name] = {
            "automorphism_count": summary.automorphism_count,
            "complete": not summary.stopped_early,
        }
    # Resolve the selected routes by their equations, never by generated ids.
    for species in formose.species.values():
        species.label = species.smiles or species.label
    target4 = "O=CC(O)C(O)CO"
    target6 = "O=CC(O)C(O)C(O)C(O)CO"
    steps = [
        ({"O=CCO": 1}, {"OC=CO": 1}),
        ({"O=CCO": 1, "OC=CO": 1}, {target4: 1}),
        ({target4: 1, "OC=CO": 1}, {target6: 1}),
    ]
    labels = []
    for lhs, rhs in steps:
        matches = [
            r
            for r in formose.reactions.values()
            if {formose.species[s].label: n for s, n in r.lhs.items()} == lhs
            and {formose.species[s].label: n for s, n in r.rhs.items()} == rhs
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected one generated reaction for each selected formose step"
            )
        labels.append(matches[0].id)
    for reaction in formose.reactions.values():
        reaction.label = reaction.id
    formose_routes = []
    for name, formaldehyde, glycolaldehyde in [
        ("M1", 1, 1),
        ("M2", 2, 1),
        ("M3", 4, 1),
        ("M4", 2, 2),
        ("C2-only", 0, 3),
    ]:
        for target, flow in [
            ("T4", {labels[0]: 1, labels[1]: 1}),
            ("T6", {labels[0]: 2, labels[1]: 1, labels[2]: 1}),
        ]:
            formose_routes.append(
                {
                    "pool": name,
                    "target": target,
                    "outcome": execute(
                        formose, flow, {"C=O": formaldehyde, "O=CCO": glycolaldehyde}
                    ),
                }
            )
    numerical = numerical_comparison()
    checks = {
        "printed_network_dimensions": (crn.n_species, crn.n_reactions, report.rank)
        == (40, 34, 23),
        "cofactor_controls": [r["outcome"]["status"] for r in routes]
        == ["unrealizable", "realizable", "realizable", "realizable"],
        "replay": all(
            r["outcome"]["independent_replay_ok"] is not False for r in routes
        ),
        "exact_conservation": all(r["exact_identity_holds"] for r in numerical),
        "numerical_subspace_residual": all(
            r["svd_max_abs_residual"] < 1e-10 for r in numerical
        ),
        "symmetry_complete": all(r["complete"] for r in symmetry.values()),
        "symmetry_counts": [r["automorphism_count"] for r in symmetry.values()]
        == [2, 512],
        "formose_routes": [r["outcome"]["status"] for r in formose_routes]
        == ["unrealizable"] * 6
        + ["realizable", "unrealizable", "realizable", "realizable"],
        "formose_replay": all(
            r["outcome"]["independent_replay_ok"] is not False for r in formose_routes
        ),
    }
    return write_report(
        {
            "schema": "synkit.crn-supplement-audit/1",
            "status": "PASS" if all(checks.values()) else "FAIL",
            "environment": environment(),
            "checks": checks,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "equations": equations,
            "rank": report.rank,
            "routes": routes,
            "symmetry": symmetry,
            "formose_routes": formose_routes,
            "numerical_method": "scipy.linalg.null_space(S.T); naive integer reconstruction = rint(1000*basis); exact residual checked on integer matrix",
            "numerical_comparison": numerical,
        },
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
