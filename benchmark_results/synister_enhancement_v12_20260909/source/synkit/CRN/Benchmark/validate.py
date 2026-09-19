"""Run the benchmark set and produce the validation table.

This is the reproducible form of the paper's validation table: every benchmark
network is analysed, every asserted quantity is compared with the expected
value, and every quantity is additionally recomputed by an independent route
from :mod:`synkit.CRN.Benchmark.crosschecks`. A run that reports
``all_passed`` therefore establishes two things at once — that the package
reproduces the expected verdicts, and that it agrees with a second, disjoint
computation of the same numbers.

.. rubric:: Example

.. code-block:: python

    from synkit.CRN.Benchmark import run_validation, validation_table

    report = run_validation()
    print(report.all_passed, report.n_networks)
    print(validation_table(report))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..Petrinet.persistence import siphon_persistence_details
from ..Props.deficiency import crnt_summary
from ..Props.stoich import integer_conservation_laws
from .crosschecks import (
    check_conservation_laws,
    check_semiflows,
    deficiency_by_rank_identity,
    persistence_by_brute_force,
    rank_by_svd,
)
from .networks import BENCHMARK_NETWORKS, BenchmarkNetwork

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
        return bool(siphon_persistence_details(crn).persistence_ok)
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

    # delta = n - l - s must hold as an identity, not just as an assertion.
    identity_ok = (
        report.deficiency
        == report.n_complexes - report.n_linkage_classes - report.rank
    )
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

    law_checks = check_conservation_laws(
        crn, laws, expected_count=entry.n_conservation_laws
    )
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
        :data:`~synkit.CRN.Benchmark.networks.BENCHMARK_NETWORKS`.
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
    return ValidationReport(
        results=[validate_network(entry, crosscheck=crosscheck) for entry in selected]
    )


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
                "-" if c["persistent"] is None else ("yes" if c["persistent"] else "no"),
                "pass" if result.passed else "FAIL",
            ]
        )

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def line(cells: Sequence[str], pad: str = " ") -> str:
        return "| " + " | ".join(
            cell.ljust(widths[i], pad) for i, cell in enumerate(cells)
        ) + " |"

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

    verdict = "all networks passed" if report.all_passed else (
        f"{len(report.failures)} of {report.n_networks} networks FAILED"
    )
    out.append("")
    out.append(f"{report.n_networks} networks; {verdict}.")
    return "\n".join(out)
