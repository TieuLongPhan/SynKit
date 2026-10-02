"""Chemical Reaction Network Theory: complexes, linkage classes and deficiency.

This module supplies the structural quantities that Chemical Reaction Network
Theory (CRNT) is built on, and the two classical theorems that turn them into
dynamical verdicts.

A reaction network is read as a directed graph on *complexes* — the multisets of
species that appear on either side of a reaction. Writing ``n`` for the number
of distinct complexes, ``l`` for the number of linkage classes (connected
components of that graph) and ``s`` for the rank of the stoichiometric matrix,
the **deficiency** is

.. math::

    \\delta = n - l - s

and is always non-negative. It measures how far the network's reaction vectors
are from being independent given its complex-graph structure, and it is the
first number a CRNT reviewer looks for.

Two theorems are implemented on top of it:

- **Deficiency Zero Theorem** (Feinberg / Horn / Jackson). A weakly reversible
  network of deficiency zero has, for *every* choice of positive mass-action
  rate constants, exactly one positive steady state in each positive
  stoichiometric compatibility class, and it is locally asymptotically stable.
  A deficiency-zero network that is *not* weakly reversible admits no positive
  steady state at all, again regardless of rate constants.
- **Deficiency One Theorem** (Feinberg). If every linkage class has deficiency
  at most one, the linkage-class deficiencies sum to the network deficiency,
  and each linkage class contains exactly one terminal strong linkage class,
  then there is *at most* one positive steady state per positive stoichiometric
  compatibility class — and exactly one when the network is also weakly
  reversible.

Both verdicts are structural: they hold for all positive rate constants and need
no kinetic parameters.

All ranks are computed exactly over :class:`~fractions.Fraction`, so an integer
network never gets a floating-point deficiency.

.. rubric:: Example

.. code-block:: python

    from synkit.CRN import SynCRN
    from synkit.CRN.Props import crnt_summary

    crn = SynCRN.from_reaction_strings(["A+B>>C", "C>>A+B", "C>>D", "D>>C"])
    report = crnt_summary(crn)
    print(report.deficiency, report.is_weakly_reversible)
    print(report)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np

from .helper import _as_graph, _species_and_rule_order
from .stoich import build_S_minus_plus

__all__ = [
    "Complex",
    "CRNTSummary",
    "complexes",
    "complex_graph",
    "crnt_summary",
    "deficiency",
    "deficiency_one_verdict",
    "deficiency_zero_verdict",
    "is_deficiency_one_applicable",
    "is_deficiency_zero_applicable",
    "is_reversible",
    "is_weakly_reversible",
    "linkage_class_deficiencies",
    "linkage_classes",
    "strong_linkage_classes",
    "terminal_strong_linkage_classes",
]


# ---------------------------------------------------------------------------
# Complexes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Complex:
    """One complex: a multiset of species with non-negative coefficients.

    :param coefficients:
        Coefficient vector in canonical species order.
    :type coefficients: Tuple[int, ...]

    :param species_labels:
        Species labels in the same order as ``coefficients``.
    :type species_labels: Tuple[str, ...]

    .. rubric:: Example

    .. code-block:: python

        print(str(Complex((2, 1, 0), ("A", "B", "C"))))
        # '2A + B'
    """

    coefficients: Tuple[int, ...]
    species_labels: Tuple[str, ...] = field(compare=False, default=())

    @property
    def is_zero(self) -> bool:
        """Return whether this is the zero complex (an in- or outflow).

        :return:
            ``True`` when every coefficient is zero.
        :rtype: bool
        """
        return not any(self.coefficients)

    @property
    def support(self) -> Tuple[str, ...]:
        """Return the labels of the species occurring in this complex.

        :return:
            Species labels with a nonzero coefficient.
        :rtype: Tuple[str, ...]
        """
        return tuple(
            label
            for label, coeff in zip(self.species_labels, self.coefficients)
            if coeff
        )

    def to_dict(self) -> Dict[str, int]:
        """Return the complex as a ``species label -> coefficient`` mapping.

        :return:
            Nonzero coefficients keyed by species label.
        :rtype: Dict[str, int]
        """
        return {
            label: int(coeff)
            for label, coeff in zip(self.species_labels, self.coefficients)
            if coeff
        }

    def __str__(self) -> str:
        """Return the usual chemical rendering, such as ``2A + B``.

        :return:
            Human-readable complex.
        :rtype: str
        """
        if self.is_zero:
            return "0"
        parts = []
        for label, coeff in zip(self.species_labels, self.coefficients):
            if not coeff:
                continue
            parts.append(f"{label}" if coeff == 1 else f"{_fmt_coeff(coeff)}{label}")
        return " + ".join(parts)


def _fmt_coeff(coeff: Any) -> str:
    """Render a stoichiometric coefficient without a trailing ``.0``.

    :param coeff:
        Coefficient value.
    :type coeff: Any

    :return:
        Compact string form.
    :rtype: str
    """
    value = float(coeff)
    return str(int(value)) if value.is_integer() else str(value)


def _species_labels(crn: Any) -> List[str]:
    """Return display labels for the canonical species order.

    :param crn:
        CRN-like input.
    :type crn: Any

    :return:
        One label per species, falling back to the node id.
    :rtype: List[str]
    """
    graph = _as_graph(crn)
    species_order, _, _, _ = _species_and_rule_order(graph)
    labels: List[str] = []
    for node in species_order:
        data = graph.nodes[node]
        label = data.get("label") or data.get("smiles") or node
        labels.append(str(label))
    return labels


def _complex_columns(crn: Any) -> Tuple[List[Complex], List[Tuple[int, int]]]:
    """Extract the distinct complexes and the reaction arrows between them.

    :param crn:
        CRN-like input.
    :type crn: Any

    :return:
        Pair ``(complexes, arrows)`` where ``arrows[j]`` is the
        ``(reactant complex index, product complex index)`` of reaction ``j``.
    :rtype: Tuple[List[Complex], List[Tuple[int, int]]]
    """
    _, _, s_minus, s_plus = build_S_minus_plus(crn)
    labels = tuple(_species_labels(crn))

    order: List[Complex] = []
    index: Dict[Tuple[int, ...], int] = {}

    def _intern(column: np.ndarray) -> int:
        key = tuple(
            int(v) if float(v).is_integer() else float(v)  # type: ignore[misc]
            for v in column
        )
        if key not in index:
            index[key] = len(order)
            order.append(Complex(coefficients=key, species_labels=labels))
        return index[key]

    arrows: List[Tuple[int, int]] = []
    for j in range(s_minus.shape[1]):
        arrows.append((_intern(s_minus[:, j]), _intern(s_plus[:, j])))

    return order, arrows


def complexes(crn: Any) -> List[Complex]:
    """Return the distinct complexes of a network, in first-appearance order.

    The zero complex is included when the network has an inflow or outflow, as
    CRNT requires.

    :param crn:
        CRN-like input.
    :type crn: Any

    :return:
        Distinct complexes.
    :rtype: List[Complex]

    .. rubric:: Example

    .. code-block:: python

        crn = SynCRN.from_reaction_strings(["A+B>>C", "C>>A+B"])
        print([str(c) for c in complexes(crn)])
        # ['A + B', 'C']
    """
    order, _ = _complex_columns(crn)
    return order


def complex_graph(crn: Any) -> nx.DiGraph:
    """Return the directed complex graph of a network.

    Nodes are complex indices carrying a ``complex`` attribute (a
    :class:`Complex`) and a ``label`` attribute (its rendering). One edge is
    added per reaction, from its reactant complex to its product complex.

    :param crn:
        CRN-like input.
    :type crn: Any

    :return:
        Complex graph.
    :rtype: networkx.DiGraph

    .. rubric:: Example

    .. code-block:: python

        g = complex_graph(SynCRN.from_reaction_strings(["A>>B", "B>>A"]))
        print(g.number_of_nodes(), g.number_of_edges())
        # 2 2
    """
    order, arrows = _complex_columns(crn)

    graph = nx.DiGraph()
    for i, cx in enumerate(order):
        graph.add_node(i, complex=cx, label=str(cx))
    for src, dst in arrows:
        if src == dst:
            continue
        graph.add_edge(src, dst)

    return graph


# ---------------------------------------------------------------------------
# Exact linear algebra
# ---------------------------------------------------------------------------


def _exact_rank(rows: Sequence[Sequence[Any]]) -> int:
    """Compute the rank of a matrix exactly, over the rationals.

    :param rows:
        Matrix rows.
    :type rows: Sequence[Sequence[Any]]

    :return:
        Exact rank.
    :rtype: int
    """
    work = [[Fraction(str(v)) if not isinstance(v, int) else Fraction(v) for v in row] for row in rows]
    if not work or not work[0]:
        return 0

    n_cols = len(work[0])
    rank = 0
    pivot_row = 0

    for col in range(n_cols):
        pivot = None
        for r in range(pivot_row, len(work)):
            if work[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            continue

        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        pivot_value = work[pivot_row][col]
        for r in range(pivot_row + 1, len(work)):
            factor = work[r][col] / pivot_value
            if factor == 0:
                continue
            work[r] = [a - factor * b for a, b in zip(work[r], work[pivot_row])]

        rank += 1
        pivot_row += 1
        if pivot_row == len(work):
            break

    return rank


def _reaction_vectors(crn: Any) -> List[List[Any]]:
    """Return the reaction vectors of a network, one per reaction.

    :param crn:
        CRN-like input.
    :type crn: Any

    :return:
        Reaction vectors as rows (reactions x species).
    :rtype: List[List[Any]]
    """
    _, _, s_minus, s_plus = build_S_minus_plus(crn)
    net = s_plus - s_minus
    return [
        [int(v) if float(v).is_integer() else float(v) for v in net[:, j]]
        for j in range(net.shape[1])
    ]


# ---------------------------------------------------------------------------
# Linkage classes
# ---------------------------------------------------------------------------


def linkage_classes(crn: Any) -> List[List[int]]:
    """Return the linkage classes as sorted lists of complex indices.

    A linkage class is a connected component of the complex graph, taken
    without regard to arrow direction.

    :param crn:
        CRN-like input.
    :type crn: Any

    :return:
        Linkage classes, each a sorted list of complex indices.
    :rtype: List[List[int]]

    .. rubric:: Example

    .. code-block:: python

        crn = SynCRN.from_reaction_strings(["A>>B", "C>>D"])
        print(linkage_classes(crn))
        # [[0, 1], [2, 3]]
    """
    graph = complex_graph(crn)
    components = [sorted(c) for c in nx.weakly_connected_components(graph)]
    return sorted(components, key=lambda comp: comp[0] if comp else -1)


def strong_linkage_classes(crn: Any) -> List[List[int]]:
    """Return the strong linkage classes as sorted lists of complex indices.

    A strong linkage class is a strongly connected component of the complex
    graph: a maximal set of complexes each reachable from every other.

    :param crn:
        CRN-like input.
    :type crn: Any

    :return:
        Strong linkage classes, each a sorted list of complex indices.
    :rtype: List[List[int]]
    """
    graph = complex_graph(crn)
    components = [sorted(c) for c in nx.strongly_connected_components(graph)]
    return sorted(components, key=lambda comp: comp[0] if comp else -1)


def terminal_strong_linkage_classes(crn: Any) -> List[List[int]]:
    """Return the terminal strong linkage classes.

    A strong linkage class is *terminal* when no reaction leads out of it: once
    the network's complex graph enters it, it cannot leave.

    :param crn:
        CRN-like input.
    :type crn: Any

    :return:
        Terminal strong linkage classes, each a sorted list of complex indices.
    :rtype: List[List[int]]
    """
    graph = complex_graph(crn)
    terminal: List[List[int]] = []

    for component in nx.strongly_connected_components(graph):
        leaves = any(
            successor not in component
            for node in component
            for successor in graph.successors(node)
        )
        if not leaves:
            terminal.append(sorted(component))

    return sorted(terminal, key=lambda comp: comp[0] if comp else -1)


def is_weakly_reversible(crn: Any) -> bool:
    """Return whether every linkage class is strongly connected.

    Weak reversibility means that whenever complex ``y`` can reach complex
    ``y'`` through a sequence of reactions, ``y'`` can reach ``y`` as well. It is
    the hypothesis both classical deficiency theorems turn on.

    :param crn:
        CRN-like input.
    :type crn: Any

    :return:
        ``True`` when the network is weakly reversible.
    :rtype: bool

    .. rubric:: Example

    .. code-block:: python

        is_weakly_reversible(SynCRN.from_reaction_strings(["A>>B", "B>>A"]))
        # True
        is_weakly_reversible(SynCRN.from_reaction_strings(["A>>B"]))
        # False
    """
    graph = complex_graph(crn)
    if graph.number_of_nodes() == 0:
        return True

    strong = {frozenset(c) for c in nx.strongly_connected_components(graph)}
    weak = {frozenset(c) for c in nx.weakly_connected_components(graph)}
    return strong == weak


def is_reversible(crn: Any) -> bool:
    """Return whether every reaction arrow has a matching reverse arrow.

    This is the strict notion: ``y -> y'`` requires ``y' -> y`` to be present as
    a reaction. Reversibility implies weak reversibility, not the converse.

    :param crn:
        CRN-like input.
    :type crn: Any

    :return:
        ``True`` when the network is reversible.
    :rtype: bool
    """
    _, arrows = _complex_columns(crn)
    arrow_set = {(src, dst) for src, dst in arrows if src != dst}
    return all((dst, src) in arrow_set for src, dst in arrow_set)


# ---------------------------------------------------------------------------
# Deficiency
# ---------------------------------------------------------------------------


def deficiency(crn: Any) -> int:
    """Compute the deficiency ``delta = n - l - s`` of a network.

    ``n`` counts distinct complexes, ``l`` counts linkage classes and ``s`` is
    the exact rank of the stoichiometric matrix. The result is always
    non-negative.

    :param crn:
        CRN-like input.
    :type crn: Any

    :return:
        Network deficiency.
    :rtype: int

    .. rubric:: Example

    .. code-block:: python

        deficiency(SynCRN.from_reaction_strings(["A>>B", "B>>A"]))
        # 0
        deficiency(SynCRN.from_reaction_strings(["A+B>>C", "C>>A", "C>>B"]))
        # 1
    """
    order, _ = _complex_columns(crn)
    n_complexes = len(order)
    n_linkage = len(linkage_classes(crn))
    rank = _exact_rank(_reaction_vectors(crn))
    return n_complexes - n_linkage - rank


def linkage_class_deficiencies(crn: Any) -> List[int]:
    """Compute the deficiency of each linkage class.

    For a linkage class with ``n_l`` complexes and reaction-vector rank ``s_l``
    the linkage-class deficiency is ``n_l - 1 - s_l``. Their sum never exceeds
    the network deficiency; equality is one of the Deficiency One Theorem's
    hypotheses.

    :param crn:
        CRN-like input.
    :type crn: Any

    :return:
        One deficiency per linkage class, in the order of
        :func:`linkage_classes`.
    :rtype: List[int]
    """
    order, arrows = _complex_columns(crn)
    if not order:
        return []

    vectors = _reaction_vectors(crn)
    classes = linkage_classes(crn)
    member_of = {node: i for i, comp in enumerate(classes) for node in comp}

    per_class: List[List[List[Any]]] = [[] for _ in classes]
    for j, (src, _dst) in enumerate(arrows):
        per_class[member_of[src]].append(vectors[j])

    return [
        len(comp) - 1 - _exact_rank(rows) for comp, rows in zip(classes, per_class)
    ]


# ---------------------------------------------------------------------------
# Theorem verdicts
# ---------------------------------------------------------------------------


def is_deficiency_zero_applicable(crn: Any) -> bool:
    """Return whether the Deficiency Zero Theorem applies to this network.

    The theorem applies exactly when the deficiency is zero; weak reversibility
    then decides *which* of its two conclusions holds. Use
    :func:`deficiency_zero_verdict` for the conclusion itself.

    :param crn:
        CRN-like input.
    :type crn: Any

    :return:
        ``True`` when the network has deficiency zero.
    :rtype: bool
    """
    return deficiency(crn) == 0


def deficiency_zero_verdict(crn: Any) -> Dict[str, Any]:
    """Apply the Deficiency Zero Theorem and report its conclusion.

    :param crn:
        CRN-like input.
    :type crn: Any

    :return:
        Mapping with ``applicable``, ``deficiency``, ``weakly_reversible``,
        ``conclusion`` (one of ``"unique_stable_equilibrium"``,
        ``"no_positive_equilibrium"`` or ``"inconclusive"``) and a
        human-readable ``statement``.
    :rtype: Dict[str, Any]

    .. rubric:: Example

    .. code-block:: python

        crn = SynCRN.from_reaction_strings(["A>>B", "B>>A"])
        print(deficiency_zero_verdict(crn)["conclusion"])
        # 'unique_stable_equilibrium'
    """
    delta = deficiency(crn)
    weakly_reversible = is_weakly_reversible(crn)

    if delta != 0:
        return {
            "applicable": False,
            "deficiency": delta,
            "weakly_reversible": weakly_reversible,
            "conclusion": "inconclusive",
            "statement": (
                f"Deficiency is {delta}, not zero; the Deficiency Zero Theorem "
                "says nothing about this network."
            ),
        }

    if weakly_reversible:
        return {
            "applicable": True,
            "deficiency": 0,
            "weakly_reversible": True,
            "conclusion": "unique_stable_equilibrium",
            "statement": (
                "Deficiency zero and weakly reversible: for every choice of "
                "positive mass-action rate constants there is exactly one "
                "positive steady state in each positive stoichiometric "
                "compatibility class, it is locally asymptotically stable, and "
                "there is no nontrivial periodic orbit."
            ),
        }

    return {
        "applicable": True,
        "deficiency": 0,
        "weakly_reversible": False,
        "conclusion": "no_positive_equilibrium",
        "statement": (
            "Deficiency zero but not weakly reversible: no positive steady "
            "state and no cyclic composition trajectory through a positive "
            "composition exists, for any positive mass-action rate constants."
        ),
    }


def is_deficiency_one_applicable(crn: Any) -> bool:
    """Return whether the Deficiency One Theorem's hypotheses hold.

    The three hypotheses are: every linkage-class deficiency is at most one,
    the linkage-class deficiencies sum to the network deficiency, and each
    linkage class contains exactly one terminal strong linkage class.

    :param crn:
        CRN-like input.
    :type crn: Any

    :return:
        ``True`` when all three hypotheses hold.
    :rtype: bool
    """
    return bool(deficiency_one_verdict(crn)["applicable"])


def deficiency_one_verdict(crn: Any) -> Dict[str, Any]:
    """Apply the Deficiency One Theorem and report its conclusion.

    :param crn:
        CRN-like input.
    :type crn: Any

    :return:
        Mapping with ``applicable``, the three hypothesis flags
        (``all_linkage_deficiencies_at_most_one``, ``deficiencies_sum_to_total``,
        ``one_terminal_class_per_linkage_class``), ``deficiency``,
        ``linkage_class_deficiencies``, ``weakly_reversible``, ``conclusion``
        (one of ``"exactly_one_equilibrium"``, ``"at_most_one_equilibrium"`` or
        ``"inconclusive"``) and a human-readable ``statement``.
    :rtype: Dict[str, Any]

    .. rubric:: Example

    .. code-block:: python

        crn = SynCRN.from_reaction_strings(["A+B>>C", "C>>A+B", "C>>A", "A>>C"])
        print(deficiency_one_verdict(crn)["conclusion"])
    """
    delta = deficiency(crn)
    per_class = linkage_class_deficiencies(crn)
    weakly_reversible = is_weakly_reversible(crn)

    classes = linkage_classes(crn)
    terminal = terminal_strong_linkage_classes(crn)
    terminal_counts = []
    for comp in classes:
        members = set(comp)
        terminal_counts.append(sum(1 for t in terminal if set(t) <= members))

    hypotheses = {
        "all_linkage_deficiencies_at_most_one": all(d <= 1 for d in per_class),
        "deficiencies_sum_to_total": sum(per_class) == delta,
        "one_terminal_class_per_linkage_class": all(c == 1 for c in terminal_counts),
    }
    applicable = all(hypotheses.values())

    result: Dict[str, Any] = {
        "applicable": applicable,
        "deficiency": delta,
        "linkage_class_deficiencies": per_class,
        "weakly_reversible": weakly_reversible,
        **hypotheses,
    }

    if not applicable:
        failed = [name for name, ok in hypotheses.items() if not ok]
        result["conclusion"] = "inconclusive"
        result["statement"] = (
            "The Deficiency One Theorem does not apply; failed hypotheses: "
            + ", ".join(failed)
            + "."
        )
        return result

    if weakly_reversible:
        result["conclusion"] = "exactly_one_equilibrium"
        result["statement"] = (
            "Deficiency One hypotheses hold and the network is weakly "
            "reversible: for every choice of positive mass-action rate "
            "constants there is exactly one positive steady state in each "
            "positive stoichiometric compatibility class."
        )
        return result

    result["conclusion"] = "at_most_one_equilibrium"
    result["statement"] = (
        "Deficiency One hypotheses hold: for every choice of positive "
        "mass-action rate constants there is at most one positive steady state "
        "in each positive stoichiometric compatibility class."
    )
    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


@dataclass
class CRNTSummary:
    """Structural CRNT report for one network.

    :param n_species:
        Number of species.
    :type n_species: int

    :param n_reactions:
        Number of reactions.
    :type n_reactions: int

    :param n_complexes:
        Number of distinct complexes.
    :type n_complexes: int

    :param n_linkage_classes:
        Number of linkage classes.
    :type n_linkage_classes: int

    :param n_terminal_strong_linkage_classes:
        Number of terminal strong linkage classes.
    :type n_terminal_strong_linkage_classes: int

    :param rank:
        Rank of the stoichiometric matrix.
    :type rank: int

    :param deficiency:
        Network deficiency ``n - l - s``.
    :type deficiency: int

    :param linkage_class_deficiencies:
        Deficiency of each linkage class.
    :type linkage_class_deficiencies: List[int]

    :param is_weakly_reversible:
        Whether every linkage class is strongly connected.
    :type is_weakly_reversible: bool

    :param is_reversible:
        Whether every reaction arrow has a matching reverse arrow.
    :type is_reversible: bool

    :param deficiency_zero:
        Verdict from :func:`deficiency_zero_verdict`.
    :type deficiency_zero: Dict[str, Any]

    :param deficiency_one:
        Verdict from :func:`deficiency_one_verdict`.
    :type deficiency_one: Dict[str, Any]

    :param complexes:
        Rendered complexes in canonical order.
    :type complexes: List[str]
    """

    n_species: int
    n_reactions: int
    n_complexes: int
    n_linkage_classes: int
    n_terminal_strong_linkage_classes: int
    rank: int
    deficiency: int
    linkage_class_deficiencies: List[int]
    is_weakly_reversible: bool
    is_reversible: bool
    deficiency_zero: Dict[str, Any] = field(default_factory=dict)
    deficiency_one: Dict[str, Any] = field(default_factory=dict)
    complexes: List[str] = field(default_factory=list)

    @classmethod
    def from_crn(cls, crn: Any) -> "CRNTSummary":
        """Build a CRNT summary for a network.

        :param crn:
            CRN-like input.
        :type crn: Any

        :return:
            Populated summary.
        :rtype: CRNTSummary
        """
        order, arrows = _complex_columns(crn)
        labels = _species_labels(crn)
        classes = linkage_classes(crn)
        rank = _exact_rank(_reaction_vectors(crn))

        return cls(
            n_species=len(labels),
            n_reactions=len(arrows),
            n_complexes=len(order),
            n_linkage_classes=len(classes),
            n_terminal_strong_linkage_classes=len(
                terminal_strong_linkage_classes(crn)
            ),
            rank=rank,
            deficiency=len(order) - len(classes) - rank,
            linkage_class_deficiencies=linkage_class_deficiencies(crn),
            is_weakly_reversible=is_weakly_reversible(crn),
            is_reversible=is_reversible(crn),
            deficiency_zero=deficiency_zero_verdict(crn),
            deficiency_one=deficiency_one_verdict(crn),
            complexes=[str(c) for c in order],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the summary as a plain dictionary.

        :return:
            Serializable mapping of every field.
        :rtype: Dict[str, Any]
        """
        return {
            "n_species": self.n_species,
            "n_reactions": self.n_reactions,
            "n_complexes": self.n_complexes,
            "n_linkage_classes": self.n_linkage_classes,
            "n_terminal_strong_linkage_classes": (
                self.n_terminal_strong_linkage_classes
            ),
            "rank": self.rank,
            "deficiency": self.deficiency,
            "linkage_class_deficiencies": list(self.linkage_class_deficiencies),
            "is_weakly_reversible": self.is_weakly_reversible,
            "is_reversible": self.is_reversible,
            "deficiency_zero": dict(self.deficiency_zero),
            "deficiency_one": dict(self.deficiency_one),
            "complexes": list(self.complexes),
        }

    def __str__(self) -> str:
        """Return a compact multi-line report.

        :return:
            Human-readable summary.
        :rtype: str
        """
        lines = [
            f"CRNT summary: {self.n_species} species, "
            f"{self.n_reactions} reactions, {self.n_complexes} complexes",
            f"  linkage classes n={self.n_linkage_classes} "
            f"(terminal strong: {self.n_terminal_strong_linkage_classes})",
            f"  rank s={self.rank}, deficiency delta={self.deficiency}",
            f"  weakly reversible: {self.is_weakly_reversible}, "
            f"reversible: {self.is_reversible}",
        ]
        if self.deficiency_zero.get("applicable"):
            lines.append(f"  deficiency-zero: {self.deficiency_zero['conclusion']}")
        if self.deficiency_one.get("applicable"):
            lines.append(f"  deficiency-one: {self.deficiency_one['conclusion']}")
        return "\n".join(lines)


def crnt_summary(crn: Any) -> CRNTSummary:
    """Compute the full structural CRNT report for a network.

    :param crn:
        CRN-like input.
    :type crn: Any

    :return:
        Populated summary.
    :rtype: CRNTSummary

    .. rubric:: Example

    .. code-block:: python

        crn = SynCRN.from_reaction_strings(["A>>B", "B>>A", "B>>C", "C>>B"])
        print(crnt_summary(crn))
    """
    return CRNTSummary.from_crn(crn)
