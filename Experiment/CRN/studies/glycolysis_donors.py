#!/usr/bin/env python
"""Check upper-glycolytic routes after ATP or F6P supplementation.

Loads the cached directed glycolysis equations and records every prescribed-flux outcome.
It does not search for alternative routes in the full network.
"""

from pathlib import Path
import argparse
import logging
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from Experiment.CRN.tools.common import environment, write_report, source_hashes
from Experiment.CRN.studies.representation import execute
from Experiment.CRN.benchmarks.glycolysis import load_glycolysis


def donor_report(fixture=None):
    """Analyse the cached directed glycolysis fixture and replay its routes."""
    logging.disable(logging.INFO)
    crn, equations = load_glycolysis(fixture)
    rows = []
    for pool, marking in (
        ("one_atp", {"S": 1, "C": 1}),
        ("two_atp", {"S": 1, "C": 2}),
        ("added_f6p", {"S": 1, "C": 1, "G": 1}),
    ):
        for name, step in (("ATP-dependent", "r3"), ("ADP-dependent", "r12")):
            flux = {"r10": 1, "r15": 1, step: 1, "r7": 1}
            rows.append(
                {"pool": pool, "name": name, "outcome": execute(crn, flux, marking)}
            )
    from synkit.CRN.backend import resolve_backend
    from synkit.CRN.native_analysis import qualitative_path

    _, library = resolve_backend()
    qualitative = []
    if library is not None:
        from copy import deepcopy

        species_ids = {s.label: sid for sid, s in crn.species.items()}
        for name, step in [("ATP-dependent", "r3"), ("ADP-dependent", "r12")]:
            selected = deepcopy(crn)
            selected.reactions = {
                rid: r
                for rid, r in selected.reactions.items()
                if r.label in {"r10", "r15", step, "r7"}
            }
            outcome = qualitative_path(
                selected,
                [species_ids["S"], species_ids["C"]],
                [species_ids["J"]],
                library_path=library,
            )
            qualitative.append(
                {
                    "name": name,
                    "semantics": "qualitative set reachability",
                    "outcome": outcome,
                }
            )
    checks = {
        "all_decisions_complete": all(
            row["outcome"]["decision_complete"] for row in rows
        ),
        "qualitative_routes_native": library is None
        or (
            len(qualitative) == 2
            and all(
                r["outcome"]["status"] == "realizable"
                and len(r["outcome"]["certificate"]) == 4
                for r in qualitative
            )
        ),
        "network_size": (crn.n_species, crn.n_reactions) == (40, 34),
        "expected_outcomes": [r["outcome"]["status"] for r in rows]
        == [
            "unrealizable",
            "realizable",
            "realizable",
            "realizable",
            "unrealizable",
            "realizable",
        ],
        "positive_replays": all(
            r["outcome"]["independent_replay_ok"]
            for r in rows
            if r["outcome"]["status"] == "realizable"
        ),
        "negative_searches_exhaustive": all(
            r["outcome"]["exhaustive"]
            for r in rows
            if r["outcome"]["status"] == "unrealizable"
        ),
    }
    report = {
        "schema": "synkit.crn-glycolysis-donors/1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "environment": environment(),
        "equations": equations,
        "routes": rows,
        "qualitative_routes": qualitative,
        "checks": checks,
        "source_sha256": source_hashes(),
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fixture", type=Path, help="glycolysis input JSON (default: cached fixture)"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    return write_report(donor_report(args.fixture), args.output)


from Experiment.CRN.tools.common import native_experiment

main = native_experiment(main)

if __name__ == "__main__":
    raise SystemExit(main())
