"""Independent recomputation of every quantity the benchmark asserts.

A validation table that checks a module against itself proves nothing. Each
function here recomputes one quantity by a **different route** from the one
:mod:`synkit.CRN` uses in production, so agreement between the two is evidence
and not tautology:

=================== ======================================== =========================================
Quantity            Production route                          Independent route used here
=================== ======================================== =========================================
deficiency          ``n - l``, ``l`` from graph components    ``rank(I_a) - rank(Y I_a)``, no components
rank                exact Fraction elimination               ``numpy.linalg.matrix_rank`` (SVD)
conservation laws   integer kernel of ``S^T``                verify ``S^T y = 0`` and integrality
minimal semiflows   Farkas / Colom-Silva elimination         verify ``S x = 0``, sign and minimality
minimal siphons     closure operator, branch and bound       brute-force enumeration of all subsets
persistence         siphons from the fast search             the same test over brute-forced siphons
=================== ======================================== =========================================

The deficiency cross-check is the substantive one. The identity

.. math::

    \\delta = n - l - s = \\operatorname{rank}(I_a) - \\operatorname{rank}(Y I_a)

holds because ``rank(I_a) = n - l`` for the incidence matrix of the complex
graph and ``S = Y I_a``. Computing the right-hand side never counts connected
components, so a bug in linkage-class detection cannot hide.

Brute-force siphon enumeration is exponential, so
:func:`brute_force_siphons` refuses networks above ``max_species``.

.. rubric:: Example

.. code-block:: python

    from synkit.CRN import SynCRN
    from synkit.CRN.Benchmark.crosschecks import deficiency_by_rank_identity

    crn = SynCRN.from_reaction_strings(["A>>B", "B>>A"])
    print(deficiency_by_rank_identity(crn))
    # 0
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set

import numpy as np

from ..Petrinet.minimal_semiflows import minimal_semiflows
from ..Props.deficiency import _complex_columns
from ..Props.stoich import build_S_minus_plus

__all__ = [
    "brute_force_siphons",
    "check_conservation_laws",
    "check_semiflows",
    "deficiency_by_rank_identity",
    "persistence_by_brute_force",
    "rank_by_svd",
]


def _net_matrix(crn: Any) -> np.ndarray:
    """Return the net stoichiometric matrix as a float array.

    :param crn:
        CRN-like input.
    :type crn: Any

    :return:
        Species-by-reaction matrix ``S``.
    :rtype: numpy.ndarray
    """
    _, _, s_minus, s_plus = build_S_minus_plus(crn)
    return s_plus - s_minus


def _matrix_species_order(crn: Any) -> List[Any]:
    """Return the species order of the rows of ``S``.

    Row order comes from the bipartite graph traversal in
    :func:`~synkit.CRN.Props.stoich.build_S_minus_plus`, which is *not* the
    ``SynCRN`` insertion order returned by ``to_petrinet()["places"]``. Semiflow
    indices must be resolved against this order, or a siphon and a semiflow
    support end up compared in two different coordinate systems.

    :param crn:
        CRN-like input.
    :type crn: Any

    :return:
        Species node ids in matrix row order.
    :rtype: List[Any]
    """
    from ..Props.helper import _species_and_rule_order

    species_order, _, _, _ = _species_and_rule_order(crn)
    return list(species_order)


def rank_by_svd(crn: Any) -> int:
    """Recompute the stoichiometric rank numerically.

    :param crn:
        CRN-like input.
    :type crn: Any

    :return:
        Rank of ``S`` from a singular-value decomposition.
    :rtype: int
    """
    matrix = _net_matrix(crn)
    if matrix.size == 0:
        return 0
    return int(np.linalg.matrix_rank(matrix))


def deficiency_by_rank_identity(crn: Any) -> int:
    """Recompute the deficiency without counting linkage classes.

    Uses ``delta = rank(I_a) - rank(Y I_a)``, where ``Y`` is the complex
    composition matrix and ``I_a`` the incidence matrix of the complex graph.

    :param crn:
        CRN-like input.
    :type crn: Any

    :return:
        Network deficiency.
    :rtype: int
    """
    order, arrows = _complex_columns(crn)
    if not order or not arrows:
        return 0

    n_species = len(order[0].coefficients)
    n_complexes = len(order)
    n_reactions = len(arrows)

    y = np.zeros((n_species, n_complexes), dtype=float)
    for j, cx in enumerate(order):
        for i, coeff in enumerate(cx.coefficients):
            y[i, j] = float(coeff)

    incidence = np.zeros((n_complexes, n_reactions), dtype=float)
    for k, (src, dst) in enumerate(arrows):
        incidence[src, k] -= 1.0
        incidence[dst, k] += 1.0

    return int(np.linalg.matrix_rank(incidence)) - int(
        np.linalg.matrix_rank(y @ incidence)
    )


def check_conservation_laws(
    crn: Any,
    laws: Sequence[Sequence[int]],
    *,
    expected_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Verify that reported conservation laws really are conservation laws.

    A conservation law is a vector ``y`` with ``S^T y = 0``. This function
    checks that property directly, plus integrality and non-triviality, and
    compares the count against ``n_species - rank``.

    :param crn:
        CRN-like input.
    :type crn: Any

    :param laws:
        Candidate conservation laws, one per row.
    :type laws: Sequence[Sequence[int]]

    :param expected_count:
        Optional expected number of independent laws.
    :type expected_count: Optional[int]

    :return:
        Mapping with ``ok`` and the individual checks.
    :rtype: Dict[str, Any]
    """
    matrix = _net_matrix(crn)
    n_species = matrix.shape[0]

    in_left_kernel = True
    integral = True
    nontrivial = True

    for law in laws:
        vector = np.asarray(law, dtype=float)
        if vector.size != n_species:
            in_left_kernel = False
            continue
        if not np.allclose(vector @ matrix, 0.0, atol=1e-9):
            in_left_kernel = False
        if not all(float(v).is_integer() for v in vector):
            integral = False
        if not np.any(vector):
            nontrivial = False

    rank = rank_by_svd(crn)
    independent_count = n_species - rank

    checks = {
        "in_left_kernel": in_left_kernel,
        "integral": integral,
        "nontrivial": nontrivial,
        "count_matches_nullity": len(laws) == independent_count,
        "nullity": independent_count,
        "n_reported": len(laws),
    }
    if expected_count is not None:
        checks["count_matches_expected"] = len(laws) == expected_count

    checks["ok"] = all(v for k, v in checks.items() if isinstance(v, bool))
    return checks


def check_semiflows(crn: Any, *, kind: str = "p") -> Dict[str, Any]:
    """Verify the defining properties of minimal semiflows.

    A P-semiflow is a non-negative integer vector ``y`` with ``y^T S = 0``; a
    T-semiflow is a non-negative integer ``x`` with ``S x = 0``. Minimality is
    checked as inclusion-minimality of supports.

    :param crn:
        CRN-like input.
    :type crn: Any

    :param kind:
        ``"p"`` for place semiflows or ``"t"`` for transition semiflows.
    :type kind: str

    :return:
        Mapping with ``ok`` and the individual checks.
    :rtype: Dict[str, Any]

    :raises ValueError:
        If ``kind`` is neither ``"p"`` nor ``"t"``.
    """
    if kind not in {"p", "t"}:
        raise ValueError(f"kind must be 'p' or 't', got {kind!r}")

    matrix = _net_matrix(crn)
    vectors = minimal_semiflows(matrix, kind=kind)

    nonnegative = all(all(v >= 0 for v in vec) for vec in vectors)
    integral = all(all(float(v).is_integer() for v in vec) for vec in vectors)

    in_kernel = True
    for vec in vectors:
        array = np.asarray(vec, dtype=float)
        product = array @ matrix if kind == "p" else matrix @ array
        if not np.allclose(product, 0.0, atol=1e-9):
            in_kernel = False

    supports = [frozenset(i for i, v in enumerate(vec) if v) for vec in vectors]
    minimal = all(
        not any(other < support for other in supports) for support in supports
    )

    checks = {
        "nonnegative": nonnegative,
        "integral": integral,
        "in_kernel": in_kernel,
        "supports_minimal": minimal,
        "n_semiflows": len(vectors),
    }
    checks["ok"] = all(v for k, v in checks.items() if isinstance(v, bool))
    return checks


def brute_force_siphons(
    crn: Any,
    *,
    max_species: int = 14,
) -> List[FrozenSet[str]]:
    """Enumerate all minimal siphons by exhaustive subset search.

    A siphon is a set of species ``S`` such that every reaction producing a
    member of ``S`` also consumes a member of ``S``: once empty, ``S`` stays
    empty. This is the definition applied directly to every subset — correct,
    exponential, and therefore only a reference implementation.

    :param crn:
        CRN-like input.
    :type crn: Any

    :param max_species:
        Refuse networks with more species than this, since the search is
        ``2**n``.
    :type max_species: int

    :return:
        Minimal siphons as frozensets of species ids.
    :rtype: List[FrozenSet[str]]

    :raises ValueError:
        If the network has more than ``max_species`` species.
    """
    petri = crn.to_petrinet()
    places = list(petri["places"])

    if len(places) > max_species:
        raise ValueError(
            f"brute_force_siphons refuses {len(places)} species (limit "
            f"{max_species}); it enumerates 2**n subsets by design."
        )

    pre = petri["pre"]
    post = petri["post"]

    consumers: Dict[str, Set[str]] = {p: set(pre.get(p, {})) for p in places}
    producers: Dict[str, Set[str]] = {p: set(post.get(p, {})) for p in places}

    def is_siphon(subset: FrozenSet[str]) -> bool:
        for place in subset:
            for transition in producers[place]:
                if not any(transition in consumers[other] for other in subset):
                    return False
        return True

    siphons: List[FrozenSet[str]] = []
    for size in range(1, len(places) + 1):
        for combo in combinations(places, size):
            subset = frozenset(combo)
            if not is_siphon(subset):
                continue
            if any(existing < subset for existing in siphons):
                continue
            siphons.append(subset)

    return [s for s in siphons if not any(other < s for other in siphons)]


def persistence_by_brute_force(
    crn: Any,
    *,
    max_species: int = 14,
) -> bool:
    """Decide structural persistence using brute-forced siphons.

    Applies the Angeli-De Leenheer-Sontag sufficient condition: a network is
    structurally persistent when every minimal siphon contains the support of a
    P-semiflow. Siphons come from :func:`brute_force_siphons` and semiflows from
    the exact Farkas routine, so this shares no code path with
    :mod:`synkit.CRN.Petrinet.persistence`.

    :param crn:
        CRN-like input.
    :type crn: Any

    :param max_species:
        Species limit forwarded to :func:`brute_force_siphons`.
    :type max_species: int

    :return:
        ``True`` when the condition holds for every minimal siphon.
    :rtype: bool
    """
    siphons = brute_force_siphons(crn, max_species=max_species)
    if not siphons:
        return True

    matrix = _net_matrix(crn)
    row_order = _matrix_species_order(crn)
    supports = [
        frozenset(row_order[i] for i, v in enumerate(vec) if v)
        for vec in minimal_semiflows(matrix, kind="p")
    ]

    return all(
        any(support <= siphon for support in supports) for siphon in siphons
    )
