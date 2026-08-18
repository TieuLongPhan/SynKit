"""Scaling benchmark for the :mod:`synkit.CRN` analysis stack.

Measures wall-clock time of each analysis against network size, over families
of synthetic networks whose size is a single parameter. The point is to show
where the practical ceiling of each analysis lies, since the analyses have very
different complexity: stoichiometric rank and conservation laws are polynomial,
minimal semiflows and canonicalization are worst-case exponential in theory but
tractable on the sparse networks chemistry produces, and minimal-siphon
enumeration is the one that used to be the wall.

Network families
----------------

``chain``
    ``S0 -> S1 -> ... -> Sn``. The sparsest possible network; the shape of a
    linear metabolic pathway.
``reversible_chain``
    The same chain with every reaction reversed as well. Weakly reversible, so
    every analysis has to do real work rather than bail out early.
``cycle``
    A closed chain ``S0 -> ... -> Sn -> S0``, which is weakly reversible and has
    a single conservation law.
``random_sparse``
    Random bimolecular reactions at a fixed reaction-to-species ratio. The
    hardest of the four, and the closest to a rule-expanded network.

.. rubric:: Example

.. code-block:: python

    from synkit.CRN.Benchmark import run_scaling_benchmark, scaling_table

    records = run_scaling_benchmark(sizes=(10, 20), families=("chain",))
    print(scaling_table(records))
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..Petrinet.structure import find_siphons
from ..Props.deficiency import crnt_summary
from ..Props.stoich import conserved_moieties, stoichiometric_rank
from ..Structure.syncrn import SynCRN

__all__ = [
    "NETWORK_FAMILIES",
    "ScalingRecord",
    "TASKS",
    "generate_network",
    "run_scaling_benchmark",
    "scaling_table",
]


# ---------------------------------------------------------------------------
# Network families
# ---------------------------------------------------------------------------


def _chain(size: int, *, seed: int = 0) -> List[str]:
    """Build a linear chain of ``size`` irreversible reactions.

    :param size: Number of reactions.
    :type size: int
    :param seed: Unused; present for a uniform family signature.
    :type seed: int
    :return: Reaction strings.
    :rtype: List[str]
    """
    return [f"S{i}>>S{i + 1}" for i in range(size)]


def _reversible_chain(size: int, *, seed: int = 0) -> List[str]:
    """Build a linear chain with every reaction reversible.

    :param size: Number of forward reactions; the network has ``2 * size``.
    :type size: int
    :param seed: Unused; present for a uniform family signature.
    :type seed: int
    :return: Reaction strings.
    :rtype: List[str]
    """
    out: List[str] = []
    for i in range(size):
        out.append(f"S{i}>>S{i + 1}")
        out.append(f"S{i + 1}>>S{i}")
    return out


def _cycle(size: int, *, seed: int = 0) -> List[str]:
    """Build a closed cycle of ``size`` reactions.

    :param size: Number of reactions and of species.
    :type size: int
    :param seed: Unused; present for a uniform family signature.
    :type seed: int
    :return: Reaction strings.
    :rtype: List[str]
    """
    return [f"S{i}>>S{(i + 1) % size}" for i in range(size)]


def _random_sparse(size: int, *, seed: int = 0) -> List[str]:
    """Build random bimolecular reactions over ``size`` species.

    Each reaction consumes two distinct species and produces one, at a fixed
    ratio of 1.5 reactions per species. The generator is seeded, so a benchmark
    run is reproducible.

    :param size: Number of species.
    :type size: int
    :param seed: Seed for the random generator.
    :type seed: int
    :return: Reaction strings.
    :rtype: List[str]
    """
    rng = random.Random(seed)
    species = [f"S{i}" for i in range(size)]
    n_reactions = max(1, int(1.5 * size))

    out: List[str] = []
    for _ in range(n_reactions):
        a, b, c = rng.sample(species, 3) if size >= 3 else (species * 3)[:3]
        out.append(f"{a}+{b}>>{c}")
    return out


#: Registry of network families, keyed by name.
NETWORK_FAMILIES: Dict[str, Callable[..., List[str]]] = {
    "chain": _chain,
    "reversible_chain": _reversible_chain,
    "cycle": _cycle,
    "random_sparse": _random_sparse,
}


def generate_network(family: str, size: int, *, seed: int = 0) -> SynCRN:
    """Build one benchmark network from a family and a size.

    :param family:
        Family name; one of the keys of :data:`NETWORK_FAMILIES`.
    :type family: str

    :param size:
        Size parameter, interpreted per family.
    :type size: int

    :param seed:
        Seed for stochastic families.
    :type seed: int

    :return:
        Generated network.
    :rtype: SynCRN

    :raises ValueError:
        If ``family`` is not a known family.

    .. rubric:: Example

    .. code-block:: python

        crn = generate_network("chain", 10)
        print(crn.n_species, crn.n_reactions)
        # 11 10
    """
    try:
        builder = NETWORK_FAMILIES[family]
    except KeyError as exc:
        raise ValueError(
            f"Unknown family {family!r}. Available: "
            f"{', '.join(sorted(NETWORK_FAMILIES))}"
        ) from exc
    return SynCRN.from_reaction_strings(builder(size, seed=seed))


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def _task_rank(crn: SynCRN) -> Any:
    """Time the stoichiometric rank.

    :param crn: Network to analyse.
    :type crn: SynCRN
    :return: Rank of ``S``.
    :rtype: Any
    """
    return stoichiometric_rank(crn)


def _task_crnt(crn: SynCRN) -> Any:
    """Time the full CRNT summary, including deficiency.

    :param crn: Network to analyse.
    :type crn: SynCRN
    :return: Deficiency.
    :rtype: Any
    """
    return crnt_summary(crn).deficiency


def _task_moieties(crn: SynCRN) -> Any:
    """Time exact minimal conservation-law computation.

    :param crn: Network to analyse.
    :type crn: SynCRN
    :return: Number of conserved moieties.
    :rtype: Any
    """
    return len(conserved_moieties(crn))


def _task_siphons(crn: SynCRN) -> Any:
    """Time minimal-siphon enumeration.

    :param crn: Network to analyse.
    :type crn: SynCRN
    :return: Number of minimal siphons.
    :rtype: Any
    """
    return len(find_siphons(crn))


def _task_canonical(crn: SynCRN) -> Any:
    """Time whole-network canonicalization.

    :param crn: Network to analyse.
    :type crn: SynCRN
    :return: Node count of the canonical graph.
    :rtype: Any
    """
    from ..Symmetry import canonical

    return canonical(crn).number_of_nodes()


def _task_sbml(crn: SynCRN) -> Any:
    """Time SBML export.

    :param crn: Network to analyse.
    :type crn: SynCRN
    :return: Length of the SBML document.
    :rtype: Any
    """
    from ..IO import crn_to_sbml

    return len(crn_to_sbml(crn, pretty=False))


#: Registry of timed analyses, keyed by name.
TASKS: Dict[str, Callable[[SynCRN], Any]] = {
    "rank": _task_rank,
    "crnt_summary": _task_crnt,
    "conserved_moieties": _task_moieties,
    "minimal_siphons": _task_siphons,
    "canonical_form": _task_canonical,
    "sbml_export": _task_sbml,
}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@dataclass
class ScalingRecord:
    """One timing measurement.

    :param family:
        Network family name.
    :type family: str

    :param size:
        Size parameter passed to the generator.
    :type size: int

    :param n_species:
        Species count of the generated network.
    :type n_species: int

    :param n_reactions:
        Reaction count of the generated network.
    :type n_reactions: int

    :param task:
        Timed analysis name.
    :type task: str

    :param seconds:
        Best wall-clock time over the repeats, or ``None`` when the task was
        cut off by ``time_budget``.
    :type seconds: Optional[float]

    :param result:
        Value the task returned, kept so a timing run doubles as a smoke test.
    :type result: Any

    :param error:
        Exception text when the task raised.
    :type error: Optional[str]
    """

    family: str
    size: int
    n_species: int
    n_reactions: int
    task: str
    seconds: Optional[float] = None
    result: Any = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the record as a serializable mapping.

        :return: Record contents.
        :rtype: Dict[str, Any]
        """
        return {
            "family": self.family,
            "size": self.size,
            "n_species": self.n_species,
            "n_reactions": self.n_reactions,
            "task": self.task,
            "seconds": self.seconds,
            "result": self.result,
            "error": self.error,
        }


def _time_task(
    task: Callable[[SynCRN], Any],
    crn: SynCRN,
    *,
    repeats: int,
) -> Tuple[Optional[float], Any, Optional[str]]:
    """Time one task, reporting the best of ``repeats`` runs.

    The best rather than the mean, because the slow runs measure the machine's
    other work rather than the algorithm.

    :param task: Callable to time.
    :type task: Callable[[SynCRN], Any]
    :param crn: Network to pass to the task.
    :type crn: SynCRN
    :param repeats: Number of repeats.
    :type repeats: int
    :return: Tuple ``(seconds, result, error)``.
    :rtype: Tuple[Optional[float], Any, Optional[str]]
    """
    best: Optional[float] = None
    result: Any = None

    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        try:
            result = task(crn)
        except Exception as exc:  # pragma: no cover - reported, not raised
            return None, None, f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        best = elapsed if best is None else min(best, elapsed)

    return best, result, None


def run_scaling_benchmark(
    sizes: Sequence[int] = (10, 20, 40, 80),
    *,
    families: Sequence[str] = ("chain", "reversible_chain", "random_sparse"),
    tasks: Sequence[str] = ("rank", "crnt_summary", "conserved_moieties",
                            "minimal_siphons"),
    repeats: int = 1,
    seed: int = 0,
    time_budget: Optional[float] = 30.0,
) -> List[ScalingRecord]:
    """Time each analysis across network families and sizes.

    Once a task exceeds ``time_budget`` on some size of a family, it is skipped
    for every larger size of that family, so one slow analysis cannot make the
    whole run open-ended.

    :param sizes:
        Size parameters to sweep, in increasing order.
    :type sizes: Sequence[int]

    :param families:
        Network families to include.
    :type families: Sequence[str]

    :param tasks:
        Analyses to time.
    :type tasks: Sequence[str]

    :param repeats:
        Timing repeats per measurement; the best is reported.
    :type repeats: int

    :param seed:
        Seed for stochastic families.
    :type seed: int

    :param time_budget:
        Per-measurement budget in seconds, or ``None`` for no budget.
    :type time_budget: Optional[float]

    :return:
        One record per (family, size, task) actually measured.
    :rtype: List[ScalingRecord]

    :raises ValueError:
        If a family or task name is unknown.

    .. rubric:: Example

    .. code-block:: python

        records = run_scaling_benchmark(sizes=(10, 20), families=("chain",))
        print(len(records))
    """
    for family in families:
        if family not in NETWORK_FAMILIES:
            raise ValueError(
                f"Unknown family {family!r}. Available: "
                f"{', '.join(sorted(NETWORK_FAMILIES))}"
            )
    for task in tasks:
        if task not in TASKS:
            raise ValueError(
                f"Unknown task {task!r}. Available: {', '.join(sorted(TASKS))}"
            )

    records: List[ScalingRecord] = []
    exhausted: set = set()

    for family in families:
        for size in sizes:
            crn = generate_network(family, size, seed=seed)
            for task in tasks:
                if (family, task) in exhausted:
                    continue

                seconds, result, error = _time_task(
                    TASKS[task], crn, repeats=repeats
                )
                records.append(
                    ScalingRecord(
                        family=family,
                        size=size,
                        n_species=crn.n_species,
                        n_reactions=crn.n_reactions,
                        task=task,
                        seconds=seconds,
                        result=result,
                        error=error,
                    )
                )

                over_budget = (
                    time_budget is not None
                    and seconds is not None
                    and seconds > time_budget
                )
                if error is not None or over_budget:
                    exhausted.add((family, task))

    return records


def scaling_table(records: Sequence[ScalingRecord], *, fmt: str = "markdown") -> str:
    """Render scaling records as a table.

    :param records:
        Records from :func:`run_scaling_benchmark`.
    :type records: Sequence[ScalingRecord]

    :param fmt:
        ``"markdown"`` or ``"csv"``.
    :type fmt: str

    :return:
        Rendered table.
    :rtype: str

    :raises ValueError:
        If ``fmt`` is not supported.

    .. rubric:: Example

    .. code-block:: python

        print(scaling_table(run_scaling_benchmark(sizes=(10,))))
    """
    if fmt not in {"markdown", "csv"}:
        raise ValueError(f"fmt must be 'markdown' or 'csv', got {fmt!r}")

    headers = ["family", "size", "species", "reactions", "task", "seconds", "result"]

    def cells(record: ScalingRecord) -> List[str]:
        seconds = (
            record.error
            if record.error
            else ("-" if record.seconds is None else f"{record.seconds:.4f}")
        )
        return [
            record.family,
            str(record.size),
            str(record.n_species),
            str(record.n_reactions),
            record.task,
            seconds,
            "" if record.result is None else str(record.result),
        ]

    rows = [cells(record) for record in records]

    if fmt == "csv":
        return "\n".join(
            [",".join(headers)] + [",".join(row) for row in rows]
        )

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        if rows
        else len(headers[i])
        for i in range(len(headers))
    ]

    def line(values: Sequence[str]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[i]) for i, value in enumerate(values)
        ) + " |"

    return "\n".join(
        [line(headers), line(["-" * w for w in widths])] + [line(row) for row in rows]
    )
