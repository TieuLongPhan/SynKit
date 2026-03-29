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
    Ensure that SymPy is available before using symbolic CRN dynamics helpers.

    This helper centralizes the optional dependency check used by symbolic
    routines such as symbolic Jacobian construction and exact determinant
    evaluation.

    :raises ImportError:
        If :mod:`sympy` is not installed or could not be imported.

    Example
    -------
    .. code-block:: python

        try:
            _require_sympy()
        except ImportError as exc:
            print(exc)
    """
    if not _SYMPY_AVAILABLE or sp is None:
        raise ImportError(
            "sympy is required for symbolic CRN dynamics helpers. "
            "Please install it, e.g. `pip install sympy`."
        )


def _safe_symbol_token(value: Any) -> str:
    """
    Convert an arbitrary object into a SymPy-safe symbol token.

    Non-alphanumeric characters are replaced by underscores. If the resulting
    token is empty, ``\"x\"`` is used. If the token starts with a digit, a
    leading underscore is prepended.

    :param value:
        Arbitrary object to convert into a symbol-safe token.
    :type value:
        Any

    :returns:
        Sanitized token suitable for building SymPy symbol names.
    :rtype:
        str

    Example
    -------
    .. code-block:: python

        _safe_symbol_token("A")
        _safe_symbol_token("species-1")
        _safe_symbol_token("3-node")

        # possible outputs:
        # "A"
        # "species_1"
        # "_3_node"
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
    Convert a numeric NumPy array into a SymPy matrix.

    Each entry is passed through :func:`sympy.nsimplify` so that integer and
    rational values are preserved when possible.

    :param A:
        Numeric array to convert.
    :type A:
        numpy.ndarray

    :returns:
        SymPy matrix with simplified symbolic entries.
    :rtype:
        sp.Matrix

    :raises ImportError:
        If :mod:`sympy` is not available.

    Example
    -------
    .. code-block:: python

        import numpy as np

        A = np.array([[1.0, -1.0], [0.0, 2.0]])
        M = _sympy_matrix_from_numpy(A)
        print(M)
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
    Compute the structural sign pattern of the symbolic Jacobian.

    For a symbolic Jacobian of the form ``G = S R``, the sign of entry
    ``G[i, k]`` depends on the stoichiometric effects in row ``i`` and on
    whether species ``k`` participates as a reactant in the corresponding
    rule columns. This routine derives the sign pattern directly from the
    stoichiometric matrices without constructing the full symbolic Jacobian.

    Returned entries are one of:

    - ``"0"``
    - ``"+"``
    - ``"-"``
    - ``"mixed"``

    :param S:
        Net stoichiometric matrix with shape ``(n_species, n_rules)``.
    :type S:
        numpy.ndarray
    :param S_minus:
        Reactant stoichiometric matrix with shape ``(n_species, n_rules)``.
    :type S_minus:
        numpy.ndarray
    :param tol:
        Numerical tolerance used to test structural positivity or negativity.
    :type tol:
        float

    :returns:
        Matrix of sign labels with shape ``(n_species, n_species)``.
    :rtype:
        numpy.ndarray

    Example
    -------
    .. code-block:: python

        import numpy as np

        S = np.array([
            [-1,  0],
            [ 1, -1],
            [ 0,  1],
        ], dtype=float)

        S_minus = np.array([
            [1, 0],
            [0, 1],
            [0, 0],
        ], dtype=float)

        P = _structural_sign_pattern(S, S_minus)
        print(P)
    """
    n_species, n_rules = S.shape
    out = np.full((n_species, n_species), "0", dtype=object)

    for i in range(n_species):  # affected species / row of G
        for k in range(n_species):  # source species / column of G
            has_pos = False
            has_neg = False

            for j in range(n_rules):
                if float(S_minus[k, j]) > tol:
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
    Build the row/column bipartite graph for a Jacobian sparsity pattern.

    The bipartite graph is used for structural-rank diagnostics via maximum
    matching. Row nodes correspond to Jacobian rows and column nodes correspond
    to Jacobian columns. An edge is added when the corresponding entry in the
    sparsity matrix is nonzero.

    :param A:
        Boolean or truthy/falsy Jacobian sparsity matrix.
    :type A:
        numpy.ndarray

    :returns:
        Tuple containing:

        - bipartite graph,
        - ordered row-node names,
        - ordered column-node names.
    :rtype:
        Tuple[nx.Graph, List[str], List[str]]

    Example
    -------
    .. code-block:: python

        import numpy as np

        A = np.array([
            [True, False],
            [True, True],
        ], dtype=bool)

        B, row_nodes, col_nodes = _jacobian_pattern_bipartite(A)
        print(row_nodes)
        print(col_nodes)
        print(B.number_of_edges())
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
    Summary of structural singularity diagnostics for a symbolic Jacobian.

    The diagnostics are species-level because the Jacobian is a
    species-by-species object, even though the underlying SynCRN graph is
    bipartite with species nodes and rule nodes.

    :param n_species:
        Number of species, i.e. Jacobian dimension.
    :type n_species:
        int
    :param structural_rank:
        Structural rank inferred from the Jacobian sparsity pattern.
    :type structural_rank:
        int
    :param has_perfect_matching:
        Whether the bipartite sparsity graph admits a perfect matching.
    :type has_perfect_matching:
        bool
    :param pattern_singular:
        Whether the Jacobian is singular at the structural-pattern level.
    :type pattern_singular:
        bool
    :param determinant_checked:
        Whether an exact symbolic determinant was computed.
    :type determinant_checked:
        bool
    :param determinant_expr:
        Exact symbolic determinant expression, if evaluated.
    :type determinant_expr:
        Optional[Any]
    :param determinant_is_zero:
        Whether the exact symbolic determinant simplified to zero.
    :type determinant_is_zero:
        Optional[bool]

    Example
    -------
    .. code-block:: python

        summary = StructuralSingularitySummary(
            n_species=3,
            structural_rank=2,
            has_perfect_matching=False,
            pattern_singular=True,
            determinant_checked=False,
        )

        print(summary.classification)
        print(summary.to_dict())
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
        Return a concise structural-singularity classification label.

        Possible values include pattern-level singularity, exact symbolic
        singularity, exact symbolic nonsingularity, or the case where only
        the sparsity-pattern analysis was performed.

        :returns:
            Classification string summarizing the diagnostic outcome.
        :rtype:
            str

        Example
        -------
        .. code-block:: python

            summary = StructuralSingularitySummary(
                n_species=4,
                structural_rank=4,
                has_perfect_matching=True,
                pattern_singular=False,
                determinant_checked=True,
                determinant_is_zero=False,
            )

            print(summary.classification)
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
        Convert the summary to a plain dictionary.

        Symbolic determinant expressions are stringified so the result is easier
        to serialize or log.

        :returns:
            Plain dictionary representation of the summary.
        :rtype:
            Dict[str, Any]

        Example
        -------
        .. code-block:: python

            d = summary.to_dict()
            print(d["classification"])
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
        Return a readable multiline summary.

        :returns:
            Human-readable diagnostic summary.
        :rtype:
            str

        Example
        -------
        .. code-block:: python

            print(summary)
        """
        lines = [
            "StructuralSingularitySummary(",
            f"  n_species            = {self.n_species}",
            f"  structural_rank      = {self.structural_rank}",
            f"  has_perfect_matching = {self.has_perfect_matching}",
            f"  pattern_singular     = {self.pattern_singular}",
            f"  determinant_checked  = {self.determinant_checked}",
            f"  determinant_is_zero  = {self.determinant_is_zero}",
            f"  classification       = {self.classification}",
            ")",
        ]
        return "\n".join(lines)


def symbolic_reactivity_matrix(
    crn: Any,
    *,
    symbol_prefix: str = "rprime",
    tol: float = 1e-12,
) -> Tuple[List[Any], List[Any], "sp.Matrix"]:
    """
    Build the symbolic reactivity matrix ``R``.

    For a SynCRN with ``n_species`` species nodes and ``n_rules`` rule nodes,
    the matrix ``R`` has shape ``(n_rules, n_species)``. Entry ``R[j, i]`` is
    a positive symbolic variable if species ``i`` is a reactant of rule ``j``,
    and zero otherwise.

    This matrix captures the structural dependence of reaction rates on species
    concentrations without assuming a specific kinetic law beyond positive
    reactant sensitivity.

    :param crn:
        SynCRN graph or SynCRN-like object in the species/rule representation.
    :type crn:
        Any
    :param symbol_prefix:
        Prefix used when naming symbolic reactivity variables.
    :type symbol_prefix:
        str
    :param tol:
        Tolerance used to decide whether a reactant stoichiometric entry is
        structurally positive.
    :type tol:
        float

    :returns:
        Tuple containing:

        - species node order,
        - rule node order,
        - symbolic reactivity matrix ``R``.
    :rtype:
        Tuple[List[Any], List[Any], sp.Matrix]

    :raises ImportError:
        If :mod:`sympy` is not available.

    Example
    -------
    .. code-block:: python

        species_order, rule_order, R = symbolic_reactivity_matrix(crn)
        print(species_order)
        print(rule_order)
        print(R)
    """
    _require_sympy()

    species_order, rule_order, S_minus, _S_plus = build_S_minus_plus(crn)
    n_species = len(species_order)
    n_rules = len(rule_order)

    rows = []
    for j in range(n_rules):
        r_tok = _safe_symbol_token(rule_order[j])
        row = []
        for i in range(n_species):
            s_tok = _safe_symbol_token(species_order[i])
            if float(S_minus[i, j]) > tol:
                row.append(sp.Symbol(f"{symbol_prefix}_{r_tok}_{s_tok}", positive=True))
            else:
                row.append(sp.Integer(0))
        rows.append(row)

    R = sp.Matrix(rows)
    return species_order, rule_order, R


def symbolic_jacobian(
    crn: Any,
    *,
    symbol_prefix: str = "rprime",
    tol: float = 1e-12,
) -> Tuple[List[Any], List[Any], "sp.Matrix"]:
    """
    Build the symbolic Jacobian ``G = S R``.

    Here ``S`` is the species-by-rule stoichiometric matrix and ``R`` is the
    rule-by-species symbolic reactivity matrix induced by reactant incidence.
    The resulting Jacobian is a species-by-species symbolic matrix describing
    local structural influence in the CRN dynamics.

    :param crn:
        SynCRN graph or SynCRN-like object in the species/rule representation.
    :type crn:
        Any
    :param symbol_prefix:
        Prefix used for symbolic reactivity variables.
    :type symbol_prefix:
        str
    :param tol:
        Tolerance forwarded to :func:`symbolic_reactivity_matrix`.
    :type tol:
        float

    :returns:
        Tuple containing:

        - species node order,
        - rule node order,
        - symbolic Jacobian ``G``.
    :rtype:
        Tuple[List[Any], List[Any], sp.Matrix]

    :raises ImportError:
        If :mod:`sympy` is not available.

    Example
    -------
    .. code-block:: python

        species_order, rule_order, G = symbolic_jacobian(crn)
        print(G.shape)
        print(G)
    """
    _require_sympy()

    species_order, rule_order, S = build_S(crn)
    _sp_order, _rule_order, R = symbolic_reactivity_matrix(
        crn,
        symbol_prefix=symbol_prefix,
        tol=tol,
    )

    S_sym = _sympy_matrix_from_numpy(S)
    G = S_sym * R
    return species_order, rule_order, G


def jacobian_sparsity(
    crn: Any,
    *,
    tol: float = 1e-12,
) -> Tuple[List[Any], np.ndarray]:
    """
    Return the boolean sparsity pattern of the symbolic Jacobian.

    Entry ``A[i, k]`` is ``True`` if species ``k`` can structurally influence
    species ``i`` through at least one rule under the local linearized
    dynamics.

    :param crn:
        SynCRN graph or SynCRN-like object in the species/rule representation.
    :type crn:
        Any
    :param tol:
        Tolerance used for structural zero testing.
    :type tol:
        float

    :returns:
        Tuple containing:

        - species node order,
        - boolean Jacobian sparsity matrix.
    :rtype:
        Tuple[List[Any], numpy.ndarray]

    Example
    -------
    .. code-block:: python

        species_order, A = jacobian_sparsity(crn)
        print(species_order)
        print(A.astype(int))
    """
    species_order, _rule_order, S_minus, S_plus = build_S_minus_plus(crn)
    S = S_plus - S_minus
    sign_pat = _structural_sign_pattern(S, S_minus, tol=tol)
    A = sign_pat != "0"
    return species_order, A


def jacobian_sign_pattern(
    crn: Any,
    *,
    tol: float = 1e-12,
) -> Tuple[List[Any], np.ndarray]:
    """
    Compute the structural sign pattern of the symbolic Jacobian.

    Returned entries are one of:

    - ``"0"``
    - ``"+"``
    - ``"-"``
    - ``"mixed"``

    The value ``"mixed"`` means that multiple structurally valid paths exist
    with conflicting positive and negative net effects.

    :param crn:
        SynCRN graph or SynCRN-like object in the species/rule representation.
    :type crn:
        Any
    :param tol:
        Tolerance used for structural zero testing.
    :type tol:
        float

    :returns:
        Tuple containing:

        - species node order,
        - Jacobian sign-pattern matrix.
    :rtype:
        Tuple[List[Any], numpy.ndarray]

    Example
    -------
    .. code-block:: python

        species_order, P = jacobian_sign_pattern(crn)
        print(species_order)
        print(P)
    """
    species_order, _rule_order, S_minus, S_plus = build_S_minus_plus(crn)
    S = S_plus - S_minus
    P = _structural_sign_pattern(S, S_minus, tol=tol)
    return species_order, P


def species_influence_graph(
    crn: Any,
    *,
    tol: float = 1e-12,
    use_labels: bool = False,
) -> nx.DiGraph:
    """
    Build the species influence graph induced by the symbolic Jacobian.

    Nodes represent species. A directed edge ``u -> v`` is added when species
    ``u`` can structurally influence species ``v`` in the local linearized
    dynamics.

    Edge attribute ``sign`` is one of:

    - ``"+"``
    - ``"-"``
    - ``"mixed"``

    Additional edge attributes record the original source and target species
    node identifiers.

    :param crn:
        SynCRN graph or SynCRN-like object in the species/rule representation.
    :type crn:
        Any
    :param tol:
        Tolerance used for structural zero testing.
    :type tol:
        float
    :param use_labels:
        If ``True`` and ``crn`` is a NetworkX graph, use node labels when
        available; otherwise use original node identifiers.
    :type use_labels:
        bool

    :returns:
        Directed species influence graph.
    :rtype:
        nx.DiGraph

    Example
    -------
    .. code-block:: python

        G_inf = species_influence_graph(crn, use_labels=True)

        print(G_inf.nodes(data=True))
        print(G_inf.edges(data=True))
    """
    species_order, P = jacobian_sign_pattern(crn, tol=tol)

    label_map: Dict[Any, Any] = {}
    if use_labels and isinstance(
        crn, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)
    ):
        for s in species_order:
            label_map[s] = crn.nodes[s].get("label", s)
    else:
        for s in species_order:
            label_map[s] = s

    G_inf = nx.DiGraph()
    for s in species_order:
        G_inf.add_node(label_map[s], source_node=s)

    n_species = len(species_order)
    for i in range(n_species):  # target row
        for k in range(n_species):  # source column
            sign = str(P[i, k])
            if sign != "0":
                G_inf.add_edge(
                    label_map[species_order[k]],
                    label_map[species_order[i]],
                    sign=sign,
                    source_species=species_order[k],
                    target_species=species_order[i],
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

    This routine performs two levels of analysis:

    1. structural-rank and perfect-matching diagnostics on the Jacobian
       sparsity pattern, and
    2. optional exact symbolic determinant evaluation for sufficiently small
       systems.

    Pattern-level singularity indicates that the Jacobian is singular for all
    admissible parameter values consistent with the sparsity structure. If the
    pattern is not singular, an exact determinant test may still detect
    symbolic cancellation and prove singularity for small systems.

    :param crn:
        SynCRN graph or SynCRN-like object in the species/rule representation.
    :type crn:
        Any
    :param tol:
        Tolerance used for structural zero testing.
    :type tol:
        float
    :param max_exact_size:
        Maximum number of species for which the exact symbolic determinant is
        computed.
    :type max_exact_size:
        int
    :param symbol_prefix:
        Prefix used for symbolic reactivity variables.
    :type symbol_prefix:
        str

    :returns:
        Structured summary of Jacobian structural-singularity diagnostics.
    :rtype:
        StructuralSingularitySummary

    :raises ImportError:
        If exact symbolic determinant evaluation is requested but
        :mod:`sympy` is not available.

    Example
    -------
    .. code-block:: python

        summary = structural_singularity_summary(
            crn,
            max_exact_size=6,
            symbol_prefix="rprime",
        )

        print(summary)
        print(summary.to_dict())
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
        _sp_order, _rule_order, G = symbolic_jacobian(
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
