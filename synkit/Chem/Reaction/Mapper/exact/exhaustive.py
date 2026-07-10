"""
Exhaustive DFS/branch-and-bound exact mapper for small labeled graphs.

This module provides a standalone solver (:class:`ExactMapper`) that is
independent of the SLAP heuristic. It is most useful for very small graphs
(up to ~10 atoms) or for unit-testing the kernel-based solvers, where the
solution space is small enough to enumerate directly.

For larger reactions, extract the uncertainty-region kernel first
(:func:`mapper.exact.kernel.extract_kernel`) and solve it with the faster
orbital-branching solver (:func:`mapper.exact.branching.solve_kernel`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from scipy.optimize import linear_sum_assignment

from ..graph.labeled_graph import LabeledGraph


@dataclass
class ExactResult:
    """Container for an exact mapping result.

    Attributes
    ----------
    mapping : list[int]
        ``mapping[i]`` gives the product atom index for reactant atom ``i``.
    cost : float
        Total absolute difference between adjacency matrices and loop counts.
    """

    mapping: List[int]
    cost: float


class ExactMapper:
    """Exhaustive exact mapper for pairs of labelled graphs.

    Performs a depth-first search over all bijective atom assignments.
    Three enhancements over naive enumeration keep it practical on kernels:

    1. **Branch-and-bound** — prunes branches whose current cost already
       exceeds the best known solution.
    2. **Admissible lower bound** — at each node, solves a small LAP on the
       loop (diagonal) differences of unassigned atoms to get a tighter
       pruning threshold.
    3. **Lex-leader symmetry breaking** — atoms with identical local
       fingerprints are assigned in non-decreasing product-index order,
       avoiding equivalent permutations.

    Intended for small graphs (reaction-center kernels up to ~10 atoms).
    """

    def __init__(self):
        self.best_result: Optional[ExactResult] = None
        self._fingerprint: Optional[List[Tuple]] = None

    def _compute_partial_cost(
        self,
        react: LabeledGraph,
        prod: LabeledGraph,
        assignment: List[Optional[int]],
        depth: int,
    ) -> float:
        """Incremental cost added by assigning reactant atom ``depth``."""
        i = depth
        p = assignment[i]
        cost = 0.0
        loop_react = react.graph.get(i, {}).get(i, 0)
        loop_prod = prod.graph.get(p, {}).get(p, 0)
        cost += abs(loop_react - loop_prod)
        for j in range(i):
            q = assignment[j]
            if q is None:
                continue
            w_react = react.graph.get(i, {}).get(j, 0)
            w_prod = prod.graph.get(p, {}).get(q, 0)
            cost += abs(w_react - w_prod)
        return cost

    def _search(  # noqa: C901
        self,
        react: LabeledGraph,
        prod: LabeledGraph,
        assignment: List[Optional[int]],
        used_prod: List[bool],
        depth: int,
        current_cost: float,
    ) -> None:
        n = len(react.labels)
        if self.best_result is not None and current_cost >= self.best_result.cost:
            return

        if self.best_result is not None:
            lb = 0.0
            label_to_unreact: dict = {}
            label_to_unprod: dict = {}
            for i_un in range(depth, n):
                lbl = react.labels[i_un]
                label_to_unreact.setdefault(lbl, []).append(i_un)
            for p_un in range(n):
                if not used_prod[p_un]:
                    lbl = prod.labels[p_un]
                    label_to_unprod.setdefault(lbl, []).append(p_un)

            infeasible = False
            for lbl, rs in label_to_unreact.items():
                ps = label_to_unprod.get(lbl, [])
                if len(rs) != len(ps):
                    infeasible = True
                    break
                if not rs:
                    continue
                m = len(rs)
                cost_mat = [[0.0] * m for _ in range(m)]
                for ii, i_idx in enumerate(rs):
                    loop_i = react.graph.get(i_idx, {}).get(i_idx, 0)
                    for jj, p_idx in enumerate(ps):
                        loop_p = prod.graph.get(p_idx, {}).get(p_idx, 0)
                        cost_mat[ii][jj] = abs(loop_i - loop_p)
                row_ind, col_ind = linear_sum_assignment(cost_mat)
                lb += sum(cost_mat[r][c] for r, c in zip(row_ind, col_ind))

            if infeasible or current_cost + lb >= self.best_result.cost:
                return

        if depth == n:
            if self.best_result is None or current_cost < self.best_result.cost:
                self.best_result = ExactResult(
                    mapping=list(assignment), cost=current_cost
                )
            return

        i = depth
        for p in range(n):
            if used_prod[p]:
                continue
            if react.labels[i] != prod.labels[p]:
                continue
            skip_sym = False
            for j in range(i):
                if (
                    assignment[j] is not None
                    and self._fingerprint[i] == self._fingerprint[j]
                ):
                    if p < assignment[j]:
                        skip_sym = True
                        break
            if skip_sym:
                continue

            assignment[i] = p
            used_prod[p] = True
            incr = self._compute_partial_cost(react, prod, assignment, depth)
            self._search(
                react, prod, assignment, used_prod, depth + 1, current_cost + incr
            )
            assignment[i] = None
            used_prod[p] = False

    def solve(self, react: LabeledGraph, prod: LabeledGraph) -> ExactResult:
        """Compute the optimal bijective mapping between two labelled graphs.

        Parameters
        ----------
        react, prod : LabeledGraph
            Must have the same number of nodes and the same multiset of labels.

        Returns
        -------
        ExactResult
            Optimal mapping and its cost.

        Raises
        ------
        ValueError
            If the graphs have incompatible sizes or label multisets.
        RuntimeError
            If no valid mapping is found (should never happen for compatible graphs).
        """
        n = len(react.labels)
        if len(prod.labels) != n:
            raise ValueError("Reactant and product must have the same number of atoms")
        if sorted(react.labels) != sorted(prod.labels):
            raise ValueError(
                "Reactant and product must have the same multiset of labels"
            )

        self.best_result = None
        self._fingerprint = []
        for i in range(n):
            label = react.labels[i]
            loop = react.graph.get(i, {}).get(i, 0)
            neighbours = react.graph.get(i, {})
            deg = len(neighbours) - (1 if i in neighbours else 0)
            bonds = sorted(w for j, w in neighbours.items() if j != i)
            self._fingerprint.append((label, loop, deg, tuple(bonds)))

        assignment: List[Optional[int]] = [None] * n
        used_prod: List[bool] = [False] * n
        self._search(react, prod, assignment, used_prod, 0, 0.0)

        if self.best_result is None:
            raise RuntimeError("No valid mapping found")
        return self.best_result
