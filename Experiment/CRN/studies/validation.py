#!/usr/bin/env python
"""Validate SynKit CRN against reference networks with independent cross-checks."""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import argparse

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from synkit.CRN.Petrinet.persistence import siphon_persistence_details  # noqa: E402
from synkit.CRN.Props.deficiency import crnt_summary  # noqa: E402
from synkit.CRN.Props.stoich import integer_conservation_laws  # noqa: E402
from Experiment.CRN.studies.crosschecks import (  # noqa: E402
    check_conservation_laws,
    check_semiflows,
    deficiency_by_rank_identity,
    persistence_by_brute_force,
    rank_by_svd,
)
from Experiment.CRN.studies.networks import BENCHMARK_NETWORKS, BenchmarkNetwork  # noqa: E402
from Experiment.CRN.tools.common import environment, timed, write_report, native_experiment  # noqa: E402

__all__ = [
    "NetworkValidation",
    "ValidationReport",
    "run_validation",
    "validation_table",
]

#: Species count above which the exponential cross-checks are skipped.
CROSSCHECK_SPECIES_LIMIT = 14


@dataclass
class NetworkValidation:
    """Validation outcome for one benchmark network.

    :param name:
        Benchmark network name.
    :type name: str

    :param source:
        Where the network comes from.
    :type source: str

    :param computed:
        Values computed by :mod:`synkit.CRN`.
    :type computed: Dict[str, Any]

    :param expected:
        Values the benchmark asserts; ``None`` entries are not checked.
    :type expected: Dict[str, Any]

    :param mismatches:
        Quantities where computed and expected disagree, as
        ``name -> (expected, computed)``.
    :type mismatches: Dict[str, Any]

    :param crosschecks:
        Results of the independent recomputations.
    :type crosschecks: Dict[str, Any]

    :param crosscheck_failures:
        Names of independent checks that disagreed or failed.
    :type crosscheck_failures: List[str]
    """

    name: str
    source: str
    computed: Dict[str, Any] = field(default_factory=dict)
    expected: Dict[str, Any] = field(default_factory=dict)
    mismatches: Dict[str, Any] = field(default_factory=dict)
    crosschecks: Dict[str, Any] = field(default_factory=dict)
    crosscheck_failures: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Return whether this network passed every check.

        :return:
            ``True`` when there is no mismatch and no cross-check failure.
        :rtype: bool
        """
        return not self.mismatches and not self.crosscheck_failures


@dataclass
class ValidationReport:
    """Validation outcome for the whole benchmark set.

    :param results:
        Per-network outcomes, in benchmark order.
    :type results: List[NetworkValidation]
    """

    results: List[NetworkValidation] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        """Return whether every network passed.

        :return:
            ``True`` when no network reported a failure.
        :rtype: bool
        """
        return all(result.passed for result in self.results)

    @property
    def n_networks(self) -> int:
        """Return the number of networks validated.

        :return:
            Network count.
        :rtype: int
        """
        return len(self.results)

    @property
    def failures(self) -> List[NetworkValidation]:
        """Return the networks that failed.

        :return:
            Failing outcomes.
        :rtype: List[NetworkValidation]
        """
        return [result for result in self.results if not result.passed]

    def to_dict(self) -> Dict[str, Any]:
        """Return the report as a serializable mapping.

        :return:
            Report contents.
        :rtype: Dict[str, Any]
        """
        return {
            "all_passed": self.all_passed,
            "n_networks": self.n_networks,
            "results": [
                {
                    "name": result.name,
                    "source": result.source,
                    "passed": result.passed,
                    "computed": result.computed,
                    "expected": result.expected,
                    "mismatches": result.mismatches,
                    "crosschecks": result.crosschecks,
                    "crosscheck_failures": result.crosscheck_failures,
                }
                for result in self.results
            ],
        }


def _persistence(crn: Any) -> Optional[bool]:
    """Compute the structural persistence verdict, tolerating failure.

    :param crn:
        CRN-like input.
    :type crn: Any

    :return:
        Verdict, or ``None`` when persistence could not be decided.
    :rtype: Optional[bool]
    """
    try:
        return siphon_persistence_details(crn).persistence_ok
    except Exception:  # pragma: no cover - defensive
        return None


def validate_network(
    entry: BenchmarkNetwork,
    *,
    crosscheck: bool = True,
) -> NetworkValidation:
    """Validate one benchmark network.

    :param entry:
        Benchmark entry to validate.
    :type entry: BenchmarkNetwork

    :param crosscheck:
        Whether the independent recomputations run. Disable for speed on large
        networks.
    :type crosscheck: bool

    :return:
        Validation outcome.
    :rtype: NetworkValidation
    """
    crn = entry.build()
    report = crnt_summary(crn)
    laws = integer_conservation_laws(crn)

    computed: Dict[str, Any] = {
        "n_species": report.n_species,
        "n_reactions": report.n_reactions,
        "n_complexes": report.n_complexes,
        "n_linkage_classes": report.n_linkage_classes,
        "rank": report.rank,
        "deficiency": report.deficiency,
        "weakly_reversible": report.is_weakly_reversible,
        "n_conservation_laws": len(laws),
        "persistent": _persistence(crn),
    }

    expected = entry.expected()
    mismatches = {
        key: (value, computed.get(key))
        for key, value in expected.items()
        if value is not None and computed.get(key) != value
    }

    result = NetworkValidation(
        name=entry.name,
        source=entry.source,
        computed=computed,
        expected=expected,
        mismatches=mismatches,
    )

    if not crosscheck:
        return result

    checks: Dict[str, Any] = {}
    failures: List[str] = []

    identity_ok = report.deficiency == report.n_complexes - report.n_linkage_classes - report.rank
    checks["deficiency_identity"] = identity_ok
    if not identity_ok:
        failures.append("deficiency_identity")

    independent_delta = deficiency_by_rank_identity(crn)
    checks["deficiency_by_rank_identity"] = independent_delta
    if independent_delta != report.deficiency:
        failures.append("deficiency_by_rank_identity")

    svd_rank = rank_by_svd(crn)
    checks["rank_by_svd"] = svd_rank
    if svd_rank != report.rank:
        failures.append("rank_by_svd")

    law_checks = check_conservation_laws(crn, laws, expected_count=entry.n_conservation_laws)
    checks["conservation_laws"] = law_checks
    if not law_checks["ok"]:
        failures.append("conservation_laws")

    for kind in ("p", "t"):
        semiflow_checks = check_semiflows(crn, kind=kind)
        checks[f"{kind}_semiflows"] = semiflow_checks
        if not semiflow_checks["ok"]:
            failures.append(f"{kind}_semiflows")

    if report.n_species <= CROSSCHECK_SPECIES_LIMIT:
        brute = persistence_by_brute_force(crn)
        checks["persistence_by_brute_force"] = brute
        if computed["persistent"] is not None and brute != computed["persistent"]:
            failures.append("persistence_by_brute_force")
    else:  # pragma: no cover - benchmark networks are small
        checks["persistence_by_brute_force"] = "skipped (too many species)"

    result.crosschecks = checks
    result.crosscheck_failures = failures
    return result


def run_validation(
    entries: Optional[Sequence[BenchmarkNetwork]] = None,
    *,
    crosscheck: bool = True,
) -> ValidationReport:
    """Validate the whole benchmark set.

    :param entries:
        Networks to validate. Defaults to
        :data:`~Experiment.CRN.studies.networks.BENCHMARK_NETWORKS`.
    :type entries: Optional[Sequence[BenchmarkNetwork]]

    :param crosscheck:
        Whether the independent recomputations run.
    :type crosscheck: bool

    :return:
        Report over every network.
    :rtype: ValidationReport

    .. rubric:: Example

    .. code-block:: python

        report = run_validation()
        print(report.all_passed)
    """
    selected = BENCHMARK_NETWORKS if entries is None else entries
    return ValidationReport(results=[validate_network(entry, crosscheck=crosscheck) for entry in selected])


def validation_table(
    report: Optional[ValidationReport] = None,
    *,
    fmt: str = "markdown",
) -> str:
    """Render a validation report as a table.

    :param report:
        Report to render. A fresh :func:`run_validation` is used when omitted.
    :type report: Optional[ValidationReport]

    :param fmt:
        ``"markdown"`` or ``"rst"``.
    :type fmt: str

    :return:
        Rendered table.
    :rtype: str

    :raises ValueError:
        If ``fmt`` is not a supported format.

    .. rubric:: Example

    .. code-block:: python

        print(validation_table())
    """
    if fmt not in {"markdown", "rst"}:
        raise ValueError(f"fmt must be 'markdown' or 'rst', got {fmt!r}")

    report = report or run_validation()

    headers = [
        "network",
        "species",
        "rxns",
        "n",
        "l",
        "s",
        "delta",
        "WR",
        "laws",
        "persistent",
        "status",
    ]

    rows: List[List[str]] = []
    for result in report.results:
        c = result.computed
        rows.append(
            [
                result.name,
                str(c["n_species"]),
                str(c["n_reactions"]),
                str(c["n_complexes"]),
                str(c["n_linkage_classes"]),
                str(c["rank"]),
                str(c["deficiency"]),
                "yes" if c["weakly_reversible"] else "no",
                str(c["n_conservation_laws"]),
                ("-" if c["persistent"] is None else ("yes" if c["persistent"] else "no")),
                "pass" if result.passed else "FAIL",
            ]
        )

    widths = [
        (max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])) for i in range(len(headers))
    ]

    def line(cells: Sequence[str], pad: str = " ") -> str:
        return "| " + " | ".join(cell.ljust(widths[i], pad) for i, cell in enumerate(cells)) + " |"

    if fmt == "markdown":
        out = [line(headers), line(["-" * w for w in widths])]
        out.extend(line(row) for row in rows)
    else:
        rule = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        heavy = "+" + "+".join("=" * (w + 2) for w in widths) + "+"
        out = [rule, line(headers), heavy]
        for row in rows:
            out.append(line(row))
            out.append(rule)

    verdict = (
        "all networks passed"
        if report.all_passed
        else (f"{len(report.failures)} of {report.n_networks} networks FAILED")
    )
    out.append("")
    out.append(f"{report.n_networks} networks; {verdict}.")
    return "\n".join(out)


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
        "all_networks_reproduce_expected_verdicts": not any(n["mismatches"] for n in networks),
        "all_networks_agree_with_independent_recomputation": not any(n["crosscheck_failures"] for n in networks),
        "deficiency_identity_holds_everywhere": all(
            n["computed"]["deficiency"]
            == n["computed"]["n_complexes"] - n["computed"]["n_linkage_classes"] - n["computed"]["rank"]
            for n in networks
        ),
    }

    return {
        "schema": SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "claim_boundary": (
            "These are structural, operation-specific verdicts rather than "
            "predictions at a fitted parametrization. Deficiency-theorem "
            "conclusions apply to mass-action systems under their reported "
            "hypotheses. The Angeli-De Leenheer-Sontag persistence test is "
            "sufficient, not necessary, so a failing condition does not prove "
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


main = native_experiment(main)

if __name__ == "__main__":
    raise SystemExit(main())
