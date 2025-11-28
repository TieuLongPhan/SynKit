from __future__ import annotations

from dataclasses import dataclass

from . import CRNLike
from .deficiency import compute_deficiency_summary, deficiency_zero_theorem_applicable
from .petri import siphon_persistence_condition
from .injectivity import count_sr_cycles, is_sr_graph_acyclic
from .thermo import compute_thermo_summary


@dataclass
class DynamicTheoremSummary:
    """
    High-level theorem-based conclusions about a CRN.

    This intentionally stays qualitative, aggregating separate checks.

    :param dz_theorem_applies: Whether the structural hypotheses of the
        Deficiency Zero Theorem are satisfied (weakly reversible,
        deficiency zero).
    :type dz_theorem_applies: bool
    :param sr_graph_cycles: Number of cycles detected in the
        species–reaction graph (simple cycles).
    :type sr_graph_cycles: int
    :param sr_graph_acyclic: Whether the SR graph is acyclic; if so,
        simple Craciun–Feinberg-type injectivity obstructions are absent.
    :type sr_graph_acyclic: bool
    :param persistence_condition: Whether the Angeli–De Leenheer–Sontag
        siphon-P-semiflow sufficient condition for persistence holds.
    :type persistence_condition: bool
    :param conservative: Whether the network is conservative.
    :type conservative: bool
    :param irreversible_futile_cycles: Whether irreversible futile cycles
        are detected via non-trivial nullspace of :math:`N`.
    :type irreversible_futile_cycles: bool
    """

    dz_theorem_applies: bool
    sr_graph_cycles: int
    sr_graph_acyclic: bool
    persistence_condition: bool
    conservative: bool
    irreversible_futile_cycles: bool


def summarize_dynamics(crn: CRNLike) -> DynamicTheoremSummary:
    """
    Summarise several structural/dynamical theorems for a CRN.

    This does **not** simulate the network; it only inspects algebraic and
    graph-theoretic properties and translates them into qualitative
    conclusions.

    :param crn: Network-like object.
    :type crn: CRNLike
    :returns: Aggregated dynamic theorem summary.
    :rtype: DynamicTheoremSummary

    **Example**

    .. code-block:: python

        from synkit.CRN.Props.dynamics import summarize_dynamics

        summary = summarize_dynamics(H)
        if summary.dz_theorem_applies:
            print("Complex-balanced & monostationary under mass-action.")
    """
    # deficiency zero theorem
    dz = deficiency_zero_theorem_applicable(crn)

    # SR graph injectivity-style info
    n_cycles = count_sr_cycles(crn, max_cycles=10_000)
    sr_acyclic = n_cycles == 0 or is_sr_graph_acyclic(crn)

    # persistence sufficient condition via siphons and P-semiflows
    persistence_ok = siphon_persistence_condition(crn)

    # thermo summary
    thermo = compute_thermo_summary(crn)

    return DynamicTheoremSummary(
        dz_theorem_applies=dz,
        sr_graph_cycles=n_cycles,
        sr_graph_acyclic=sr_acyclic,
        persistence_condition=persistence_ok,
        conservative=thermo.conservative,
        irreversible_futile_cycles=thermo.irreversible_futile_cycles,
    )
