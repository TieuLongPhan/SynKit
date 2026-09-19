from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.linalg import null_space as scipy_null_space  # type: ignore

    _SCIPY_AVAILABLE = True
except Exception:
    scipy_null_space = None  # type: ignore
    _SCIPY_AVAILABLE = False

from .helper import (
    _as_graph,
    _edge_coeff,
    _is_rule_node,
    _is_species_node,
    _normalize_role,
    _species_and_rule_order,
)

__all__ = [
    "build_S_minus_plus",
    "build_S",
    "stoichiometric_matrix",
    "stoichiometric_rank",
    "left_nullspace",
    "right_nullspace",
    "left_right_kernels",
    "integer_conservation_laws",
    "conserved_moieties",
    "StoichSummary",
    "summary",
]

# ---------------------------------------------------------------------------
# S⁻, S⁺ and S = S⁺ − S⁻
# ---------------------------------------------------------------------------


def _resolve_species_rule_incidence(
    G: Any,
    u: Any,
    v: Any,
) -> Optional[Tuple[Any, Any]]:
    """Resolve an edge endpoint pair into a ``(species_node, rule_node)`` pair.

    This helper inspects node metadata and accepts either orientation
    ``species -> rule`` or ``rule -> species``. If the endpoints do not form
    a valid species-rule incidence, ``None`` is returned.

    :param G: Graph containing the edge endpoints.
    :type G: Any
    :param u: First endpoint.
    :type u: Any
    :param v: Second endpoint.
    :type v: Any
    :return:
        A pair ``(species_node, rule_node)`` when the endpoints define a valid
        species-rule incidence, otherwise ``None``.
    :rtype: Optional[Tuple[Any, Any]]
    """
    u_data = G.nodes[u]
    v_data = G.nodes[v]

    if _is_species_node(u_data) and _is_rule_node(v_data):
        return u, v
    if _is_rule_node(u_data) and _is_species_node(v_data):
        return v, u
    return None


def _accumulate_stoich_entry(
    S_minus: np.ndarray,
    S_plus: np.ndarray,
    i: int,
    j: int,
    role: Optional[str],
    coeff: float,
) -> None:
    """Accumulate one stoichiometric coefficient into ``S^-`` or ``S^+``.

    :param S_minus: Reactant stoichiometric matrix.
    :type S_minus: np.ndarray
    :param S_plus: Product stoichiometric matrix.
    :type S_plus: np.ndarray
    :param i: Species row index.
    :type i: int
    :param j: Rule column index.
    :type j: int
    :param role:
        Normalized edge role. Expected values are ``"reactant"``,
        ``"product"``, or ``None``.
    :type role: Optional[str]
    :param coeff: Stoichiometric coefficient to add.
    :type coeff: float
    :return: ``None``. The matrices are modified in place.
    :rtype: None
    """
    if role == "reactant":
        S_minus[i, j] += coeff
    elif role == "product":
        S_plus[i, j] += coeff


def _iter_graph_edges_with_data(G: Any) -> Iterable[Tuple[Any, Any, Dict[str, Any]]]:
    """Yield graph edges as ``(u, v, data)`` triples for simple and multigraphs.

    For multigraphs, parallel edges are yielded individually while discarding
    the internal edge key.

    :param G: NetworkX graph-like object.
    :type G: Any
    :return:
        Iterable of edge triples ``(u, v, data)`` where ``data`` is the edge
        attribute mapping.
    :rtype: Iterable[Tuple[Any, Any, Dict[str, Any]]]
    """
    if G.is_multigraph():
        for u, v, _k, data in G.edges(data=True, keys=True):
            yield u, v, data
    else:
        for u, v, data in G.edges(data=True):
            yield u, v, data


def build_S_minus_plus(
    crn: Any,
) -> Tuple[List[Any], List[Any], np.ndarray, np.ndarray]:
    """Build the reactant matrix ``S^-`` and product matrix ``S^+`` from a SynCRN.

    The input is interpreted as a bipartite species-rule incidence graph. Rows
    correspond to species nodes and columns correspond to rule nodes.

    Graph conventions
    -----------------
    Nodes:
      - species nodes have ``kind="species"``
      - rule nodes have ``kind="rule"``

    Edges:
      - ``role``: ``"reactant"`` or ``"product"``
      - ``stoich``: stoichiometric coefficient, default ``1.0``

    Orientation
    -----------
    The graph may be directed or undirected. The edge ``role`` attribute is
    treated as the source of truth, and endpoint order is not assumed to encode
    reactant/product direction.

    :param crn:
        A NetworkX bipartite graph or a SynCRN-like object containing one.
    :type crn: Any
    :return:
        A 4-tuple ``(species_order, rule_order, S_minus, S_plus)`` where
        ``species_order`` defines row order, ``rule_order`` defines column
        order, ``S_minus`` contains reactant stoichiometries, and ``S_plus``
        contains product stoichiometries.
    :rtype: Tuple[List[Any], List[Any], np.ndarray, np.ndarray]
    """
    G = _as_graph(crn)

    species_order, rule_order, species_index, rule_index = _species_and_rule_order(G)

    n_species = len(species_order)
    n_rules = len(rule_order)

    S_minus = np.zeros((n_species, n_rules), dtype=float)
    S_plus = np.zeros((n_species, n_rules), dtype=float)

    for u, v, data in _iter_graph_edges_with_data(G):
        role = _normalize_role(data.get("role"))
        if role is None:
            continue

        incidence = _resolve_species_rule_incidence(G, u, v)
        if incidence is None:
            continue

        s_node, r_node = incidence
        i = species_index[s_node]
        j = rule_index[r_node]
        coeff = _edge_coeff(data)

        _accumulate_stoich_entry(S_minus, S_plus, i, j, role, coeff)

    return species_order, rule_order, S_minus, S_plus


def build_S(crn: Any) -> Tuple[List[Any], List[Any], np.ndarray]:
    """Build the stoichiometric matrix ``S = S^+ - S^-``.

    :param crn: A NetworkX bipartite graph or a SynCRN-like object containing one.
    :type crn: Any
    :return:
        A 3-tuple ``(species_order, rule_order, S)`` where ``S`` is the species
        x rule stoichiometric matrix.
    :rtype: Tuple[List[Any], List[Any], np.ndarray]
    """
    species_order, rule_order, S_minus, S_plus = build_S_minus_plus(crn)
    S = S_plus - S_minus
    return species_order, rule_order, S


def stoichiometric_matrix(crn: Any) -> np.ndarray:
    """Return the species x rule stoichiometric matrix ``S``.

    :param crn: A NetworkX bipartite graph or a SynCRN-like object containing one.
    :type crn: Any
    :return: Stoichiometric matrix with species as rows and rule nodes as columns.
    :rtype: np.ndarray
    """
    _, _, S = build_S(crn)
    return S


def stoichiometric_rank(crn: Any, *, tol: float = 1e-10) -> int:
    """Compute the numerical rank of the stoichiometric matrix.

    :param crn: A NetworkX bipartite graph or a SynCRN-like object containing one.
    :type crn: Any
    :param tol: Numerical tolerance passed to ``numpy.linalg.matrix_rank``.
    :type tol: float
    :return: Rank of the stoichiometric matrix.
    :rtype: int
    """
    S = stoichiometric_matrix(crn)
    return int(np.linalg.matrix_rank(S, tol=tol))


# ---------------------------------------------------------------------------
# Nullspaces: left and right kernels
# ---------------------------------------------------------------------------


def _svd_null_space(A: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    """Compute a null-space basis using SVD.

    This is used as a fallback when SciPy is unavailable.

    :param A: Input matrix.
    :type A: np.ndarray
    :param rtol: Relative singular-value threshold for rank detection.
    :type rtol: float
    :return:
        Matrix whose columns form a basis of the null space of ``A``.
    :rtype: np.ndarray
    """
    _u, s, vh = np.linalg.svd(A, full_matrices=True)
    if s.size == 0:
        return np.eye(A.shape[1])

    tol = rtol * s[0]
    rank = int((s > tol).sum())
    ns = vh[rank:].T

    if ns.size == 0:
        return np.zeros((A.shape[1], 0))

    return ns


def _null_space(A: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    """Compute a null-space basis using SciPy when available, otherwise SVD.

    :param A: Input matrix.
    :type A: np.ndarray
    :param rtol: Relative threshold used to determine the numerical null space.
    :type rtol: float
    :return:
        Matrix whose columns form a basis of the null space of ``A``.
    :rtype: np.ndarray
    """
    if _SCIPY_AVAILABLE and scipy_null_space is not None:
        return scipy_null_space(A, rcond=rtol)
    return _svd_null_space(A, rtol=rtol)


def left_nullspace(crn: Any, *, rtol: float = 1e-12) -> np.ndarray:
    """Compute a basis for the left null space ``ker(S^T)``.

    In CRN language, these directions correspond to conservation-law vectors
    over species.

    :param crn: A NetworkX bipartite graph or a SynCRN-like object containing one.
    :type crn: Any
    :param rtol: Relative tolerance used in null-space computation.
    :type rtol: float
    :return:
        Matrix whose columns form a basis of ``ker(S^T)``.
    :rtype: np.ndarray
    """
    S = stoichiometric_matrix(crn)
    return _null_space(S.T, rtol=rtol)


def right_nullspace(crn: Any, *, rtol: float = 1e-12) -> np.ndarray:
    """Compute a basis for the right null space ``ker(S)``.

    In CRN or Petri-net language, these directions correspond to rule-flux
    modes or T-semiflows.

    :param crn: A NetworkX bipartite graph or a SynCRN-like object containing one.
    :type crn: Any
    :param rtol: Relative tolerance used in null-space computation.
    :type rtol: float
    :return:
        Matrix whose columns form a basis of ``ker(S)``.
    :rtype: np.ndarray
    """
    S = stoichiometric_matrix(crn)
    return _null_space(S, rtol=rtol)


def left_right_kernels(
    crn: Any,
    *,
    rtol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute both left and right kernels of the stoichiometric matrix.

    :param crn: A NetworkX bipartite graph or a SynCRN-like object containing one.
    :type crn: Any
    :param rtol: Relative tolerance used in null-space computation.
    :type rtol: float
    :return:
        Pair ``(left_basis, right_basis)`` where ``left_basis`` spans
        ``ker(S^T)`` and ``right_basis`` spans ``ker(S)``.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    left_basis = left_nullspace(crn, rtol=rtol)
    right_basis = right_nullspace(crn, rtol=rtol)
    return left_basis, right_basis


# ---------------------------------------------------------------------------
# Integer scaling helpers
# ---------------------------------------------------------------------------


def _lcm(a: int, b: int) -> int:
    """Compute the least common multiple of two integers.

    :param a: First integer.
    :type a: int
    :param b: Second integer.
    :type b: int
    :return: Least common multiple of ``a`` and ``b``.
    :rtype: int
    """
    return abs(a // gcd(a, b) * b) if a and b else abs(a or b)


def _vector_to_minimal_integer(vec: np.ndarray, *, tol: float = 1e-12) -> List[int]:
    """Scale a floating vector to a minimal integer vector.

    The routine first normalizes the vector, then attempts rational
    reconstruction with bounded denominators. If that becomes unstable, a
    rounded fallback is used.

    :param vec: Input floating-point vector.
    :type vec: np.ndarray
    :param tol: Threshold below which entries are treated as zero.
    :type tol: float
    :return:
        Integer vector reduced by the greatest common divisor of its entries.
    :rtype: List[int]
    """
    if np.all(np.abs(vec) <= tol):
        return [0] * int(vec.size)

    v = np.array(vec, dtype=float)
    max_abs = float(np.max(np.abs(v)))
    if max_abs <= tol:
        return [0] * int(v.size)
    v = v / max_abs

    fracs: List[Fraction] = []
    for x in v:
        if abs(x) <= tol:
            fracs.append(Fraction(0, 1))
        else:
            fracs.append(Fraction(float(x)).limit_denominator(10**6))

    den_lcm = 1
    for f in fracs:
        den_lcm = _lcm(den_lcm, f.denominator)
        if den_lcm > 10**6:
            break

    if den_lcm <= 10**6:
        ints = [int(f.numerator * (den_lcm // f.denominator)) for f in fracs]
    else:
        scale = 10**3
        ints = [int(round(float(x) * scale)) for x in v]

    g = 0
    for val in ints:
        g = gcd(g, abs(val))

    if g == 0:
        nonzeros = [abs(float(x)) for x in v if abs(float(x)) > tol]
        if not nonzeros:
            return [0] * int(v.size)
        scale = 1.0 / min(nonzeros)
        ints = [int(round(float(x) * scale)) for x in v]
        g = 0
        for val in ints:
            g = gcd(g, abs(val))
        if g == 0:
            g = 1

    ints = [val // g for val in ints]
    return ints


def _primitive_integer_vector(vec: Sequence[Fraction]) -> List[int]:
    """Scale an exact rational vector to a primitive integer vector.

    The vector is multiplied by the least common multiple of its denominators,
    divided by the greatest common divisor of the result, and sign-normalized so
    that the first nonzero entry is positive.

    :param vec: Exact rational vector.
    :type vec: Sequence[Fraction]
    :return: Primitive integer vector.
    :rtype: List[int]
    """
    den_lcm = 1
    for value in vec:
        den_lcm = _lcm(den_lcm, value.denominator)

    ints = [int(value * den_lcm) for value in vec]

    divisor = 0
    for value in ints:
        divisor = gcd(divisor, abs(value))
    if divisor > 1:
        ints = [value // divisor for value in ints]

    for value in ints:
        if value != 0:
            if value < 0:
                ints = [-x for x in ints]
            break

    return ints


def _exact_null_space(rows: Sequence[Sequence[Any]], n_cols: int) -> List[List[int]]:
    """Compute an exact integer basis of the null space of a rational matrix.

    Row-reduces over :class:`~fractions.Fraction`, so the result is exact: every
    returned vector ``v`` satisfies ``M v = 0`` identically rather than to
    within a tolerance.

    :param rows: Matrix rows.
    :type rows: Sequence[Sequence[Any]]
    :param n_cols: Number of columns.
    :type n_cols: int
    :return: Primitive integer basis vectors of the null space.
    :rtype: List[List[int]]
    """
    if n_cols == 0:
        return []

    mat: List[List[Fraction]] = [
        [Fraction(str(value)) if not isinstance(value, int) else Fraction(value)
         for value in row]
        for row in rows
    ]

    pivot_cols: List[int] = []
    pivot_row = 0
    for col in range(n_cols):
        if pivot_row >= len(mat):
            break
        candidate = next(
            (r for r in range(pivot_row, len(mat)) if mat[r][col] != 0), None
        )
        if candidate is None:
            continue

        mat[pivot_row], mat[candidate] = mat[candidate], mat[pivot_row]
        pivot_value = mat[pivot_row][col]
        mat[pivot_row] = [value / pivot_value for value in mat[pivot_row]]

        for r in range(len(mat)):
            if r != pivot_row and mat[r][col] != 0:
                factor = mat[r][col]
                mat[r] = [a - factor * b for a, b in zip(mat[r], mat[pivot_row])]

        pivot_cols.append(col)
        pivot_row += 1

    free_cols = [c for c in range(n_cols) if c not in pivot_cols]

    basis: List[List[int]] = []
    for free_col in free_cols:
        vector = [Fraction(0)] * n_cols
        vector[free_col] = Fraction(1)
        for row_index, pivot_col in enumerate(pivot_cols):
            vector[pivot_col] = -mat[row_index][free_col]
        basis.append(_primitive_integer_vector(vector))

    return basis


def integer_conservation_laws(crn: Any, *, rtol: float = 1e-12) -> List[List[int]]:
    """Return an exact integer basis for ``ker(S^T)``, the conservation laws.

    A conservation law is a vector ``y`` over species with ``y^T S = 0``: the
    quantity ``y . c`` is constant along every trajectory, whatever the kinetics.
    Basis vectors are exact primitive integers obtained by rational row
    reduction, so ``y^T S`` is identically zero rather than zero to within a
    tolerance.

    Basis vectors may contain negative entries. For the non-negative,
    inclusion-minimal laws — the *conserved moieties*, which are the ones with a
    chemical reading such as "total enzyme" or "total phosphate" — use
    :func:`conserved_moieties`.

    :param crn: A NetworkX bipartite graph or a SynCRN-like object containing one.
    :type crn: Any
    :param rtol:
        Retained for backward compatibility and ignored; the computation is
        exact and needs no tolerance.
    :type rtol: float
    :return:
        Exact integer basis of the left kernel, one law per row.
    :rtype: List[List[int]]

    .. rubric:: Example

    .. code-block:: python

        crn = SynCRN.from_reaction_strings(["A>>B", "B>>A"])
        print(integer_conservation_laws(crn))
        # [[1, 1]]
    """
    S = stoichiometric_matrix(crn)
    if S.size == 0:
        return []

    n_species, _ = S.shape
    rows = [[int(v) if float(v).is_integer() else float(v) for v in S[:, j]]
            for j in range(S.shape[1])]
    return _exact_null_space(rows, n_species)


def conserved_moieties(crn: Any) -> List[List[int]]:
    """Return the minimal non-negative integer conservation laws.

    These are the P-semiflows of the network's Petri-net view: vectors ``y >= 0``
    with ``y^T S = 0`` and inclusion-minimal support. Each one reads as a
    conserved pool — total enzyme, total phosphate, a carbon backbone — which is
    what makes them the useful form in a metabolic or signalling context.

    Unlike :func:`integer_conservation_laws`, this is not a basis: a network can
    have more minimal moieties than the dimension of its left kernel, and a
    network with only sign-mixed conservation laws has none at all.

    :param crn: A NetworkX bipartite graph or a SynCRN-like object containing one.
    :type crn: Any
    :return:
        Minimal non-negative integer conservation laws, one per row, in the
        canonical species order.
    :rtype: List[List[int]]

    .. rubric:: Example

    .. code-block:: python

        crn = SynCRN.from_reaction_strings(["E+S>>ES", "ES>>E+S", "ES>>E+P"])
        print(conserved_moieties(crn))
    """
    from ..Petrinet.minimal_semiflows import minimal_semiflows

    S = stoichiometric_matrix(crn)
    if S.size == 0:
        return []
    return minimal_semiflows(S, kind="p")


# ---------------------------------------------------------------------------
# Lightweight structural summary
# ---------------------------------------------------------------------------


@dataclass
class StoichSummary:
    """
    Lightweight stoichiometric summary of a SynCRN.

    The field name ``n_reactions`` is retained for backward compatibility even
    though the current SynCRN representation uses rule nodes as process
    columns.

    :param n_species: Number of species rows in the stoichiometric matrix.
    :type n_species: int
    :param n_reactions:
        Number of process columns in the stoichiometric matrix. In the current
        representation this corresponds to the number of rule nodes.
    :type n_reactions: int
    :param rank: Numerical rank of the stoichiometric matrix.
    :type rank: int
    """

    n_species: int
    n_reactions: int
    rank: int

    dim_left_kernel: int = field(init=False)
    dim_right_kernel: int = field(init=False)

    def __post_init__(self) -> None:
        """
        Validate dimensions and derive kernel dimensions.

        :raises ValueError:
            If dimensions are negative or if ``rank`` exceeds
            ``min(n_species, n_reactions)``.
        """
        if self.n_species < 0 or self.n_reactions < 0:
            raise ValueError("n_species and n_reactions must be non-negative.")
        if self.rank < 0:
            raise ValueError("rank must be non-negative.")
        if self.rank > min(self.n_species, self.n_reactions):
            raise ValueError(
                f"rank={self.rank} cannot exceed min(n_species, n_reactions) = "
                f"{min(self.n_species, self.n_reactions)}"
            )

        self.dim_left_kernel = max(self.n_species - self.rank, 0)
        self.dim_right_kernel = max(self.n_reactions - self.rank, 0)

    @property
    def is_full_rank(self) -> bool:
        """Whether the stoichiometric matrix has full rank.

        :return:
            ``True`` when ``rank == min(n_species, n_reactions)``,
            otherwise ``False``.
        :rtype: bool
        """
        return self.rank == min(self.n_species, self.n_reactions)

    @property
    def is_underdetermined(self) -> bool:
        """Whether the right kernel is non-trivial.

        Equivalently, this checks whether ``rank < n_reactions``.

        :return: ``True`` if ``dim_right_kernel > 0``, else ``False``.
        :rtype: bool
        """
        return self.rank < self.n_reactions

    def to_dict(self) -> Dict[str, Any]:
        """Convert the summary into a plain dictionary.

        :return: Dictionary representation of the summary.
        :rtype: Dict[str, Any]
        """
        return {
            "n_species": self.n_species,
            "n_reactions": self.n_reactions,
            "rank": self.rank,
            "dim_left_kernel": self.dim_left_kernel,
            "dim_right_kernel": self.dim_right_kernel,
        }

    @classmethod
    def from_crn(cls, crn: Any) -> "StoichSummary":
        """Construct a summary directly from a CRN object or graph.

        :param crn: A NetworkX bipartite graph or a SynCRN-like object containing one.
        :type crn: Any
        :return: Stoichiometric summary derived from the CRN.
        :rtype: StoichSummary
        """
        S = stoichiometric_matrix(crn)
        n_species, n_reactions = S.shape
        # ``matrix_rank`` raises on a zero-size array, which a network with no
        # species or no reactions legitimately produces.
        rank = 0 if S.size == 0 else int(np.linalg.matrix_rank(S))
        return cls(
            n_species=n_species,
            n_reactions=n_reactions,
            rank=rank,
        )

    def __str__(self) -> str:
        """Return a human-readable multi-line representation.

        :return: Formatted summary string.
        :rtype: str
        """
        lines = [
            "StoichSummary(",
            f"  n_species        = {self.n_species}",
            f"  n_reactions      = {self.n_reactions}",
            f"  rank             = {self.rank}",
            f"  dim_left_kernel  = {self.dim_left_kernel}",
            f"  dim_right_kernel = {self.dim_right_kernel}",
            ")",
        ]
        return "\n".join(lines)


def summary(crn: Any) -> StoichSummary:
    """Compute a quick stoichiometric summary.

    :param crn: A NetworkX bipartite graph or a SynCRN-like object containing one.
    :type crn: Any
    :return: Lightweight stoichiometric summary of the CRN.
    :rtype: StoichSummary
    """
    return StoichSummary.from_crn(crn)
