#!/usr/bin/env python
"""Measure how the SynKit CRN analysis stack scales with network size.

Sweeps synthetic network families of increasing size and times each analysis.
Networks are generated from a fixed seed, so a run is reproducible; timings are
of course machine-dependent, which is why the environment block is recorded.

The point of the sweep is to locate each analysis's practical ceiling rather
than to claim a complexity bound. Minimal-siphon enumeration is the analysis
whose ceiling moved most relative to exhaustive subset search; exact minimal
semiflow computation is the one that dominates at large sizes.

Emits schema ``synkit.crn-scaling/1``.

.. rubric:: Example

.. code-block:: bash

    python Experiment/CRN/scaling.py --sizes 50 100 200 400 --output crn-scaling.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Experiment.CRN.common import environment, write_report  # noqa: E402
from synkit.CRN.Benchmark import run_scaling_benchmark  # noqa: E402

SCHEMA = "synkit.crn-scaling/1"

DEFAULT_SIZES = (50, 100, 200, 400)
DEFAULT_FAMILIES = ("chain", "reversible_chain", "random_sparse")
DEFAULT_TASKS = (
    "rank",
    "crnt_summary",
    "conserved_moieties",
    "minimal_siphons",
    "canonical_form",
)


def scaling_report(
    *,
    sizes: Sequence[int],
    families: Sequence[str],
    tasks: Sequence[str],
    repeats: int,
    seed: int,
    time_budget: float,
) -> Dict[str, Any]:
    """Run the sweep and assemble the evidence report.

    :param sizes: Size parameters to sweep, increasing.
    :type sizes: Sequence[int]
    :param families: Network families to include.
    :type families: Sequence[str]
    :param tasks: Analyses to time.
    :type tasks: Sequence[str]
    :param repeats: Timing repeats per measurement; the best is reported.
    :type repeats: int
    :param seed: Seed for stochastic families.
    :type seed: int
    :param time_budget: Per-measurement budget in seconds.
    :type time_budget: float
    :return: Report payload.
    :rtype: Dict[str, Any]
    """
    records = run_scaling_benchmark(
        sizes=tuple(sizes),
        families=tuple(families),
        tasks=tuple(tasks),
        repeats=repeats,
        seed=seed,
        time_budget=time_budget,
    )
    rows = [record.to_dict() for record in records]

    errored = [row for row in rows if row["error"]]
    checks = {
        "every_measurement_completed": not errored,
        "every_task_ran_at_the_smallest_size": all(
            any(
                row["task"] == task and row["size"] == min(sizes)
                for row in rows
            )
            for task in tasks
        ),
    }

    return {
        "schema": SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "claim_boundary": (
            "Wall-clock timings are operational measurements on the recorded "
            "machine. They locate practical ceilings and do not establish "
            "asymptotic complexity."
        ),
        "environment": environment(),
        "parameters": {
            "sizes": list(sizes),
            "families": list(families),
            "tasks": list(tasks),
            "repeats": repeats,
            "seed": seed,
            "time_budget": time_budget,
        },
        "checks": checks,
        "errors": errored,
        "measurements": rows,
    }


def main() -> int:
    """Parse arguments, run the sweep, and write the evidence file.

    :return: Process exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--time-budget",
        type=float,
        default=300.0,
        help="skip a task for larger sizes once it exceeds this many seconds",
    )
    parser.add_argument("--output", type=Path, help="write the evidence file here")
    arguments = parser.parse_args()

    report = scaling_report(
        sizes=arguments.sizes,
        families=arguments.families,
        tasks=arguments.tasks,
        repeats=arguments.repeats,
        seed=arguments.seed,
        time_budget=arguments.time_budget,
    )
    return write_report(report, arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
