#!/usr/bin/env python
"""Validate SynKit CRN against reference networks with independent cross-checks.

Runs the shipped validation set. Each network's structural verdicts are compared
against the values the benchmark asserts, and every quantity is additionally
recomputed by a second algorithm sharing no code path with the production one:
deficiency through the rank identity ``rank(Ia) - rank(Y Ia)`` which never counts
linkage classes, siphons by exhaustive subset enumeration, semiflows by checking
their defining equations. Agreement is evidence; disagreement is a defect.

Emits schema ``synkit.crn-validation/1``. The command exits non-zero if any
network fails, so it is usable as a gate.

.. rubric:: Example

.. code-block:: bash

    python Experiment/CRN/validation.py --output crn-validation.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Experiment.CRN.common import environment, timed, write_report  # noqa: E402
from synkit.CRN.Benchmark import BENCHMARK_NETWORKS, run_validation  # noqa: E402

SCHEMA = "synkit.crn-validation/1"


def validation_report() -> Dict[str, Any]:
    """Run the validation set and assemble the evidence report.

    :return: Report payload.
    :rtype: Dict[str, Any]
    """
    seconds, report, error = timed(lambda: run_validation(crosscheck=True))
    if report is None:
        return {
            "schema": SCHEMA,
            "status": "FAIL",
            "error": error,
            "environment": environment(),
        }

    networks = []
    for result in report.results:
        entry = next(e for e in BENCHMARK_NETWORKS if e.name == result.name)
        networks.append(
            {
                "name": result.name,
                "source": result.source,
                "description": entry.description,
                "tags": list(entry.tags),
                "reactions": list(entry.reactions),
                "passed": result.passed,
                "computed": result.computed,
                "expected": {k: v for k, v in result.expected.items() if v is not None},
                "mismatches": result.mismatches,
                "crosschecks": result.crosschecks,
                "crosscheck_failures": result.crosscheck_failures,
            }
        )

    checks = {
        "all_networks_reproduce_expected_verdicts": not any(
            n["mismatches"] for n in networks
        ),
        "all_networks_agree_with_independent_recomputation": not any(
            n["crosscheck_failures"] for n in networks
        ),
        "deficiency_identity_holds_everywhere": all(
            n["computed"]["deficiency"]
            == n["computed"]["n_complexes"]
            - n["computed"]["n_linkage_classes"]
            - n["computed"]["rank"]
            for n in networks
        ),
    }

    return {
        "schema": SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "claim_boundary": (
            "Structural verdicts hold for every choice of positive rate "
            "constants and assert nothing about behaviour at any particular "
            "parametrization. The Angeli-De Leenheer-Sontag persistence test is "
            "sufficient, not necessary, so a negative verdict does not prove "
            "that a species can be driven to extinction."
        ),
        "environment": environment(),
        "seconds": seconds,
        "n_networks": len(networks),
        "checks": checks,
        "networks": networks,
    }


def main() -> int:
    """Parse arguments, run the validation, and write the evidence file.

    :return: Process exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="write the evidence file here")
    arguments = parser.parse_args()
    return write_report(validation_report(), arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
