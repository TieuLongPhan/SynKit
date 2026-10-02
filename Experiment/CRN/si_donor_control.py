#!/usr/bin/env python
"""Check upper-glycolytic routes after ATP or F6P supplementation.

Reads the printed SI equations and records every prescribed-flux outcome.
It does not search for alternative routes in the full network.
"""

from pathlib import Path
import argparse
import hashlib
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Experiment.CRN.common import environment, write_report
from Experiment.CRN.review_experiments import execute
from synkit.CRN import SynCRN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--si", type=Path, default=ROOT / "paper/synkit_crn/SI/si.tex")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    equations = {
        f"r{n}": equation.replace(r"\(\rightarrow\)", ">>")
        for n, equation in re.findall(
            r"^r(\d+) & (.*?) \\\\", args.si.read_text(), re.M
        )
    }
    crn = SynCRN.from_reaction_strings(list(equations.values()))
    for reaction, label in zip(crn.reactions.values(), equations):
        reaction.label = label
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
    checks = {
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
    sources = [Path(__file__), ROOT / "Experiment/CRN/review_experiments.py"]
    sources.extend(sorted((ROOT / "synkit/CRN").rglob("*.py")))
    report = {
        "schema": "synkit.crn-si-donor-control/1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "environment": environment(),
        "equations": equations,
        "routes": rows,
        "checks": checks,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
    }
    write_report(report, args.output)
    print(f"Supplementary donor controls: {report['status']} ({len(rows)} cases)")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
