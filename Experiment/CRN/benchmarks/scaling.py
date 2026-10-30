"""Scaling benchmark for the :mod:`synkit.CRN` analysis stack.

Measures wall-clock time against network size for parameterized synthetic
network families. The results locate practical runtime limits for the selected
families. They do not establish asymptotic complexity.

Network families
----------------

``chain``
    ``S0 -> S1 -> ... -> Sn``. A linear pathway with one irreversible
    reaction between each adjacent pair of species.
``reversible_chain``
    The same chain with every reaction reversed as well. Weakly reversible, so
    exercises analyses that can terminate early on non-reversible networks.
``cycle``
    A closed chain ``S0 -> ... -> Sn -> S0``, which is weakly reversible and has
    a single conservation law.
``random_sparse``
    Random bimolecular reactions at a fixed reaction-to-species ratio. The
    densest of the four benchmark families.

.. rubric:: Example

.. code-block:: python

    from Experiment.CRN.benchmarks import run_scaling_benchmark, scaling_table

    records = run_scaling_benchmark(sizes=(10, 20), families=("chain",))
    print(scaling_table(records))
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from multiprocessing import get_all_start_methods, get_context
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from synkit.CRN.Petrinet.structure import find_siphons
from synkit.CRN.Props.deficiency import crnt_summary
from synkit.CRN.Props.stoich import conserved_moieties, stoichiometric_rank
from synkit.CRN.Structure.syncrn import SynCRN

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
    return len(find_siphons(crn, strict_limit=True))


def _task_canonical(crn: SynCRN) -> Any:
    """Time whole-network canonicalization.

    :param crn: Network to analyse.
    :type crn: SynCRN
    :return: Node count of the canonical graph.
    :rtype: Any
    """
    from synkit.CRN.Symmetry import canonical

    return canonical(crn).number_of_nodes()


def _task_sbml(crn: SynCRN) -> Any:
    """Time SBML export.

    :param crn: Network to analyse.
    :type crn: SynCRN
    :return: Length of the SBML document.
    :rtype: Any
    """
    from synkit.CRN.IO import crn_to_sbml

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
# Timing records and workers
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

    :param timed_out:
        Whether the task was terminated at the measurement budget.
    :type timed_out: bool
    """

    family: str
    size: int
    n_species: int
    n_reactions: int
    task: str
    seconds: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    timed_out: bool = False
    backend: str = "python"
    termination_reason: str = "complete"
    elapsed_seconds: Optional[float] = None

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
            "timed_out": self.timed_out,
            "backend": self.backend,
            "termination_reason": self.termination_reason,
            "elapsed_seconds": self.elapsed_seconds,
            "complete": self.termination_reason == "complete"
            and self.error is None and self.seconds is not None,
        }


def _measure_task(task, crn, repeats, native_library=None):
    """Keep completed timings, elapsed time and search-stop reasons distinct."""
    from synkit.CRN.native_analysis import NativeSearchLimit
    from synkit.CRN.Petrinet.minimal_semiflows import SemiflowSearchLimit
    from synkit.CRN.Petrinet.siphon_search import SiphonSearchLimit

    started = time.perf_counter()
    best = None
    result = None
    error = None
    reason = "complete"
    try:
        for _ in range(max(1, repeats)):
            repeat_started = time.perf_counter()
            if native_library is not None:
                from synkit.CRN import native_backend
                with native_backend(native_library):
                    result = task(crn)
            else:
                result = task(crn)
            elapsed = time.perf_counter() - repeat_started
            best = elapsed if best is None else min(best, elapsed)
    except (NativeSearchLimit, SemiflowSearchLimit, SiphonSearchLimit) as exc:
        details = getattr(exc, "result", None)
        reason = getattr(details, "termination_reason", "node_limit")
        best, result = None, None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        reason = "error"
        best, result = None, None
    return {
        "seconds": best,
        "result": result,
        "error": error,
        "timed_out": reason == "time_limit",
        "termination_reason": reason,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _time_task_worker(task, crn, repeats, connection, native_library=None):
    """Send a compact task measurement to the parent process."""
    try:
        connection.send(_measure_task(task, crn, repeats, native_library))
    finally:
        connection.close()


def _time_task(task, crn, *, repeats, time_budget):
    """Measure a task with an optional parent-enforced wall-clock limit."""
    if time_budget is None:
        return _measure_task(task, crn, repeats)

    from synkit.CRN.backend import resolve_backend

    start_method = "fork" if "fork" in get_all_start_methods() else "spawn"
    context = get_context(start_method)
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_time_task_worker,
        args=(task, crn, repeats, sender, resolve_backend()[1]),
    )
    started = time.perf_counter()
    process.start()
    sender.close()
    process.join(max(0.0, time_budget))
    if process.is_alive():
        process.terminate()
        process.join()
        receiver.close()
        return {
            "seconds": None, "result": None, "error": None,
            "timed_out": True, "termination_reason": "time_limit",
            "elapsed_seconds": time.perf_counter() - started,
        }
    if receiver.poll():
        measurement = receiver.recv()
        receiver.close()
        return measurement
    receiver.close()
    return {
        "seconds": None, "result": None,
        "error": f"worker exited with code {process.exitcode}",
        "timed_out": False, "termination_reason": "error",
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_scaling_benchmark(
    sizes: Sequence[int] = (10, 20, 40, 80),
    *,
    families: Sequence[str] = ("chain", "reversible_chain", "random_sparse"),
    tasks: Sequence[str] = (
        "rank",
        "crnt_summary",
        "conserved_moieties",
        "minimal_siphons",
    ),
    repeats: int = 1,
    seed: int = 0,
    time_budget: Optional[float] = 30.0,
) -> List[ScalingRecord]:
    """Time each analysis across network families and sizes.

    A task is terminated when it reaches ``time_budget`` and is skipped for
    every larger size of that family, so one slow analysis cannot make the
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

                measurement = _time_task(
                    TASKS[task],
                    crn,
                    repeats=repeats,
                    time_budget=time_budget,
                )
                records.append(
                    ScalingRecord(
                        family=family,
                        size=size,
                        n_species=crn.n_species,
                        n_reactions=crn.n_reactions,
                        task=task,
                        **measurement,
                        backend=(
                            "python_io"
                            if task == "sbml_roundtrip"
                            else __import__(
                                "synkit.CRN.backend", fromlist=["resolve_backend"]
                            ).resolve_backend()[0]
                        ),
                    )
                )

                if measurement["termination_reason"] != "complete":
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
            else (record.termination_reason if record.seconds is None else f"{record.seconds:.4f}")
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
        return "\n".join([",".join(headers)] + [",".join(row) for row in rows])

    widths = [
        (
            max(len(headers[i]), *(len(row[i]) for row in rows))
            if rows
            else len(headers[i])
        )
        for i in range(len(headers))
    ]

    def line(values: Sequence[str]) -> str:
        return (
            "| "
            + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values))
            + " |"
        )

    return "\n".join(
        [line(headers), line(["-" * w for w in widths])] + [line(row) for row in rows]
    )
