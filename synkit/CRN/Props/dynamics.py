from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from networkx.algorithms import bipartite as nx_bipartite

try:
    import sympy as sp  # type: ignore

    _SYMPY_AVAILABLE = True
except Exception:
    sp = None  # type: ignore
    _SYMPY_AVAILABLE = False

from .stoich import build_S, build_S_minus_plus

__all__ = [
    "StructuralSingularitySummary",
    "symbolic_reactivity_matrix",
    "symbolic_jacobian",
    "jacobian_sparsity",
    "jacobian_sign_pattern",
    "species_influence_graph",
    "structural_singularity_summary",
]


def _require_sympy() -> None:
    """
    Ensure SymPy is available for symbolic CRN dynamics helpers.

    :raises ImportError:
        If :mod:`sympy` is not installed.
    """
    if not _SYMPY_AVAILABLE or sp is None:
        raise ImportError(
            "sympy is required for symbolic CRN dynamics helpers. "
            "Please install it, e.g. `pip install sympy`."
        )


def _safe_symbol_token(value: Any) -> str:
    """
    Convert an arbitrary object to a safe symbol token.

    Non-alphanumeric characters are replaced by underscores.

    :param value: Object to convert.
    :type value: Any
    :returns: Safe token suitable for SymPy symbol names.
    :rtype: str
    """
    text = str(value)
    token = "".join(ch if ch.isalnum() else "_" for ch in text)
    if not token:
        token = "x"
    if token[0].isdigit():
        token = f"_{token}"
    return token


def _sympy_matrix_from_numpy(A: np.ndarray) -> "sp.Matrix":
    """
    Convert a numeric NumPy matrix to a SymPy matrix with simplified entries.

    :param A: Input numeric matrix.
    :type A: numpy.ndarray
    :returns: SymPy matrix with entries converted via :func:`sympy.nsimplify`.
    :rtype: sympy.Matrix
    """
    _require_sympy()
    return sp.Matrix([[sp.nsimplify(x) for x in row] for row in np.asarray(A)])


def _structural_sign_pattern(
    S: np.ndarray,
    S_minus: np.ndarray,
    *,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Internal helper to compute the sign pattern of the symbolic Jacobian.

    For the symbolic Jacobian :math:`G = S R`, entry :math:`G_{ik}` is a sum
    over reactions :math:`j` of terms

    .. math::

        S_{ij} \\cdot r'_{jk},

    where :math:`r'_{jk}` is a positive symbol if species :math:`k` is a
    reactant of reaction :math:`j`, and zero otherwise.

    The structural sign is therefore determined only by the signs of the
    contributing :math:`S_{ij}` values among reactions for which species
    :math:`k` is a reactant.

    Returned values are one of:
      - ``"0"``
      - ``"+"``
      - ``"-"``
      - ``"mixed"``

    :param S: Stoichiometric matrix of shape ``(n_species, n_reactions)``.
    :type S: numpy.ndarray
    :param S_minus: Reactant matrix of shape ``(n_species, n_reactions)``.
    :type S_minus: numpy.ndarray
    :param tol: Small tolerance for zero testing.
    :type tol: float
    :returns: Sign-pattern matrix of shape ``(n_species, n_species)``.
    :rtype: numpy.ndarray
    """
    n_species, n_reactions = S.shape
    out = np.full((n_species, n_species), "0", dtype=object)

    for i in range(n_species):  # affected species / row of G
        for k in range(n_species):  # source species / column of G
            has_pos = False
            has_neg = False

            for j in range(n_reactions):
                if S_minus[k, j] > tol:
                    coeff = float(S[i, j])
                    if coeff > tol:
                        has_pos = True
                    elif coeff < -tol:
                        has_neg = True

                    if has_pos and has_neg:
                        out[i, k] = "mixed"
                        break
            else:
                if has_pos and has_neg:
                    out[i, k] = "mixed"
                elif has_pos:
                    out[i, k] = "+"
                elif has_neg:
                    out[i, k] = "-"
                else:
                    out[i, k] = "0"

    return out


def _jacobian_pattern_bipartite(A: np.ndarray) -> Tuple[nx.Graph, List[str], List[str]]:
    """
    Build the row/column bipartite graph for a boolean Jacobian pattern.

    :param A: Boolean or truthy matrix of shape ``(n, n)``.
    :type A: numpy.ndarray
    :returns:
        Tuple ``(B, row_nodes, col_nodes)`` where ``B`` is a bipartite graph.
    :rtype: Tuple[networkx.Graph, List[str], List[str]]
    """
    n_rows, n_cols = A.shape
    B = nx.Graph()

    row_nodes = [f"row:{i}" for i in range(n_rows)]
    col_nodes = [f"col:{j}" for j in range(n_cols)]

    B.add_nodes_from(row_nodes, bipartite=0)
    B.add_nodes_from(col_nodes, bipartite=1)

    for i in range(n_rows):
        for j in range(n_cols):
            if bool(A[i, j]):
                B.add_edge(row_nodes[i], col_nodes[j])

    return B, row_nodes, col_nodes


@dataclass
class StructuralSingularitySummary:
    """
    Summary of structural singularity diagnostics for the symbolic Jacobian.

    The summary combines:

    - a sparsity-pattern structural-rank check via bipartite matching, and
    - an optional exact symbolic determinant check for small systems.

    Pattern-level checks can prove singularity, but they cannot detect
    cancellations in the determinant. The exact symbolic determinant can.

    :param n_species: Number of species.
    :type n_species: int
    :param structural_rank: Structural rank of the Jacobian sparsity pattern.
    :type structural_rank: int
    :param has_perfect_matching:
        Whether the Jacobian sparsity pattern admits a perfect matching.
    :type has_perfect_matching: bool
    :param pattern_singular:
        Whether the sparsity pattern is singular, i.e. structural rank is
        strictly less than ``n_species``.
    :type pattern_singular: bool
    :param determinant_checked:
        Whether an exact symbolic determinant was computed.
    :type determinant_checked: bool
    :param determinant_expr:
        Exact symbolic determinant expression when computed, else ``None``.
    :type determinant_expr: Any or None
    :param determinant_is_zero:
        Whether the exact determinant simplifies to zero. ``None`` if not
        checked.
    :type determinant_is_zero: bool or None
    """

    n_species: int
    structural_rank: int
    has_perfect_matching: bool
    pattern_singular: bool
    determinant_checked: bool
    determinant_expr: Optional[Any] = None
    determinant_is_zero: Optional[bool] = None

    @property
    def classification(self) -> str:
        """
        Return a concise structural-singularity classification string.

        Possible values are:

        - ``"singular_by_pattern"``
        - ``"singular_by_exact_determinant"``
        - ``"structurally_nonsingular"``
        - ``"pattern_nonsingular_exact_unchecked"``
        """
        if self.pattern_singular:
            return "singular_by_pattern"
        if self.determinant_is_zero is True:
            return "singular_by_exact_determinant"
        if self.determinant_is_zero is False:
            return "structurally_nonsingular"
        return "pattern_nonsingular_exact_unchecked"

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a plain dictionary representation.
        """
        return {
            "n_species": self.n_species,
            "structural_rank": self.structural_rank,
            "has_perfect_matching": self.has_perfect_matching,
            "pattern_singular": self.pattern_singular,
            "determinant_checked": self.determinant_checked,
            "determinant_expr": (
                None if self.determinant_expr is None else str(self.determinant_expr)
            ),
            "determinant_is_zero": self.determinant_is_zero,
            "classification": self.classification,
        }

    def __str__(self) -> str:
        """
        Human-readable multi-line summary.
        """
        lines = [
            "StructuralSingularitySummary(",
            f"  n_species           = {self.n_species}",
            f"  structural_rank     = {self.structural_rank}",
            f"  has_perfect_matching= {self.has_perfect_matching}",
            f"  pattern_singular    = {self.pattern_singular}",
            f"  determinant_checked = {self.determinant_checked}",
            f"  determinant_is_zero = {self.determinant_is_zero}",
            f"  classification      = {self.classification}",
            ")",
        ]
        return "\n".join(lines)


def symbolic_reactivity_matrix(
    crn: Any,
    *,
    symbol_prefix: str = "rprime",
    tol: float = 1e-12,
) -> Tuple[List[str], List[str], "sp.Matrix"]:
    """
    Build the symbolic reactivity matrix :math:`R`.

    For a CRN with species set :math:`M` and reaction set :math:`E`,
    the matrix :math:`R` has shape ``(|E|, |M|)``. Entry ``R[j, i]`` is a
    positive symbolic variable if species ``i`` is a reactant of reaction
    ``j``, and zero otherwise.

    This encodes the structural local dependence of reaction rates on species
    concentrations under broad monotone kinetics.

    :param crn: Hypergraph or bipartite NetworkX graph.
    :type crn: Any
    :param symbol_prefix: Prefix used when naming positive symbolic variables.
    :type symbol_prefix: str
    :param tol: Small tolerance used to test whether an entry of ``S_minus`` is
                structurally positive.
    :type tol: float
    :returns:
        Tuple ``(species_order, reaction_order, R)`` where ``R`` is a SymPy
        matrix of shape ``(n_reactions, n_species)``.
    :rtype: Tuple[List[str], List[str], sympy.Matrix]

    .. code-block:: python

        from synkit.CRN.Props.dynamics import symbolic_reactivity_matrix

        sp_order, rxn_order, R = symbolic_reactivity_matrix(G)
        print(R)
    """
    _require_sympy()

    species_order, reaction_order, S_minus, _S_plus = build_S_minus_plus(crn)
    n_species = len(species_order)
    n_reactions = len(reaction_order)

    rows = []
    for j in range(n_reactions):
        r_tok = _safe_symbol_token(reaction_order[j])
        row = []
        for i in range(n_species):
            s_tok = _safe_symbol_token(species_order[i])
            if float(S_minus[i, j]) > tol:
                row.append(sp.Symbol(f"{symbol_prefix}_{r_tok}_{s_tok}", positive=True))
            else:
                row.append(sp.Integer(0))
        rows.append(row)

    R = sp.Matrix(rows)
    return species_order, reaction_order, R


def symbolic_jacobian(
    crn: Any,
    *,
    symbol_prefix: str = "rprime",
    tol: float = 1e-12,
) -> Tuple[List[str], List[str], "sp.Matrix"]:
    """
    Build the symbolic Jacobian :math:`G = S R`.

    Here :math:`S` is the stoichiometric matrix and :math:`R` is the symbolic
    reactivity matrix induced by reactant incidence. The result captures the
    local species-to-species interaction structure implied by the CRN under
    broad monotone kinetics, without requiring a fully specified rate law.

    :param crn: Hypergraph or bipartite NetworkX graph.
    :type crn: Any
    :param symbol_prefix: Prefix used for symbolic reactivity variables.
    :type symbol_prefix: str
    :param tol: Small tolerance forwarded to
                :func:`symbolic_reactivity_matrix`.
    :type tol: float
    :returns:
        Tuple ``(species_order, reaction_order, G)`` where ``G`` is a SymPy
        matrix of shape ``(n_species, n_species)``.
    :rtype: Tuple[List[str], List[str], sympy.Matrix]

    .. code-block:: python

        from synkit.CRN.Props.dynamics import symbolic_jacobian

        sp_order, rxn_order, Gsym = symbolic_jacobian(G)
        print(Gsym)
    """
    _require_sympy()

    species_order, reaction_order, S = build_S(crn)
    _sp_order, _rxn_order, R = symbolic_reactivity_matrix(
        crn,
        symbol_prefix=symbol_prefix,
        tol=tol,
    )

    S_sym = _sympy_matrix_from_numpy(S)
    G = S_sym * R
    return species_order, reaction_order, G


def jacobian_sparsity(
    crn: Any,
    *,
    tol: float = 1e-12,
) -> Tuple[List[str], np.ndarray]:
    """
    Return the boolean sparsity pattern of the symbolic Jacobian.

    Entry ``A[i, k]`` is ``True`` if species ``k`` can structurally influence
    species ``i`` through the local linearized dynamics implied by the CRN.

    :param crn: Hypergraph or bipartite NetworkX graph.
    :type crn: Any
    :param tol: Small tolerance for zero testing.
    :type tol: float
    :returns:
        Tuple ``(species_order, A)`` where ``A`` is a boolean matrix of shape
        ``(n_species, n_species)``.
    :rtype: Tuple[List[str], numpy.ndarray]

    .. code-block:: python

        from synkit.CRN.Props.dynamics import jacobian_sparsity

        sp_order, A = jacobian_sparsity(G)
        print(A.astype(int))
    """
    species_order, _reaction_order, S_minus, S_plus = build_S_minus_plus(crn)
    S = S_plus - S_minus
    sign_pat = _structural_sign_pattern(S, S_minus, tol=tol)
    A = sign_pat != "0"
    return species_order, A


def jacobian_sign_pattern(
    crn: Any,
    *,
    tol: float = 1e-12,
) -> Tuple[List[str], np.ndarray]:
    """
    Compute the structural sign pattern of the symbolic Jacobian.

    Returned entries are one of:

    - ``"0"``
    - ``"+"``
    - ``"-"``
    - ``"mixed"``

    The sign is determined structurally from the stoichiometric matrix and the
    reactant incidence pattern, assuming positive local rate sensitivities with
    respect to reactants.

    :param crn: Hypergraph or bipartite NetworkX graph.
    :type crn: Any
    :param tol: Small tolerance for zero testing.
    :type tol: float
    :returns:
        Tuple ``(species_order, P)`` where ``P`` is an object array of shape
        ``(n_species, n_species)`` containing sign strings.
    :rtype: Tuple[List[str], numpy.ndarray]

    .. code-block:: python

        from synkit.CRN.Props.dynamics import jacobian_sign_pattern

        sp_order, P = jacobian_sign_pattern(G)
        print(P)
    """
    species_order, _reaction_order, S_minus, S_plus = build_S_minus_plus(crn)
    S = S_plus - S_minus
    P = _structural_sign_pattern(S, S_minus, tol=tol)
    return species_order, P


def species_influence_graph(
    crn: Any,
    *,
    tol: float = 1e-12,
) -> nx.DiGraph:
    """
    Build the species influence graph induced by the symbolic Jacobian.

    Nodes are species. A directed edge ``u -> v`` is added when species ``u``
    can structurally influence species ``v`` in the local linearized dynamics.
    Edge attribute ``sign`` is one of:

    - ``"+"``
    - ``"-"``
    - ``"mixed"``

    Diagonal self-loops are included when the Jacobian has structurally nonzero
    diagonal entries.

    :param crn: Hypergraph or bipartite NetworkX graph.
    :type crn: Any
    :param tol: Small tolerance for zero testing.
    :type tol: float
    :returns: Directed species influence graph.
    :rtype: networkx.DiGraph

    .. code-block:: python

        from synkit.CRN.Props.dynamics import species_influence_graph

        Gi = species_influence_graph(G)
        print(Gi.edges(data=True))
    """
    species_order, P = jacobian_sign_pattern(crn, tol=tol)

    G_inf = nx.DiGraph()
    for s in species_order:
        G_inf.add_node(s)

    n_species = len(species_order)
    for i in range(n_species):  # target row
        for k in range(n_species):  # source column
            sign = str(P[i, k])
            if sign != "0":
                G_inf.add_edge(
                    species_order[k],
                    species_order[i],
                    sign=sign,
                )

    return G_inf


def structural_singularity_summary(
    crn: Any,
    *,
    tol: float = 1e-12,
    max_exact_size: int = 7,
    symbol_prefix: str = "rprime",
) -> StructuralSingularitySummary:
    """
    Diagnose structural singularity of the symbolic Jacobian.

    This function performs:

    1. a structural-rank / perfect-matching check on the Jacobian sparsity
       pattern, and
    2. an optional exact symbolic determinant check for small systems.

    The exact determinant step is useful because a Jacobian can be
    pattern-nonsingular yet still have an identically zero determinant due to
    symbolic cancellation.

    :param crn: Hypergraph or bipartite NetworkX graph.
    :type crn: Any
    :param tol: Small tolerance for zero testing.
    :type tol: float
    :param max_exact_size:
        Maximum number of species for which the exact symbolic determinant is
        computed.
    :type max_exact_size: int
    :param symbol_prefix: Prefix used for symbolic reactivity variables.
    :type symbol_prefix: str
    :returns: Structural singularity summary.
    :rtype: StructuralSingularitySummary

    .. code-block:: python

        from synkit.CRN.Props.dynamics import structural_singularity_summary

        summary = structural_singularity_summary(G)
        print(summary.classification)
    """
    species_order, A = jacobian_sparsity(crn, tol=tol)
    n_species = len(species_order)

    B, row_nodes, _col_nodes = _jacobian_pattern_bipartite(A)
    matching = nx_bipartite.maximum_matching(B, top_nodes=set(row_nodes))
    structural_rank = sum(1 for u in row_nodes if u in matching)
    has_perfect_matching = structural_rank == n_species
    pattern_singular = structural_rank < n_species

    determinant_checked = False
    determinant_expr: Optional[Any] = None
    determinant_is_zero: Optional[bool] = None

    if n_species <= max_exact_size:
        _require_sympy()
        determinant_checked = True
        _sp_order, _rxn_order, G = symbolic_jacobian(
            crn,
            symbol_prefix=symbol_prefix,
            tol=tol,
        )
        determinant_expr = sp.simplify(G.det())
        determinant_is_zero = bool(determinant_expr == 0)

    return StructuralSingularitySummary(
        n_species=n_species,
        structural_rank=structural_rank,
        has_perfect_matching=has_perfect_matching,
        pattern_singular=pattern_singular,
        determinant_checked=determinant_checked,
        determinant_expr=determinant_expr,
        determinant_is_zero=determinant_is_zero,
    )
