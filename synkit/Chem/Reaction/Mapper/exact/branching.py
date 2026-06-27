"""
Orbital-branching exact solver for the uncertainty-region kernel.

The SLAP engine identifies a small *kernel* of atoms whose mapping it cannot
resolve unambiguously (:mod:`mapper.exact.kernel`). This module exhaustively
searches over those atoms using two complementary pruning strategies:

* **Reactant lex-leader** — atoms in the same reactant automorphism orbit must
  receive product positions in non-decreasing order, eliminating assignments
  that differ only by a reactant symmetry.
* **Cost lower bound** — an admissible bound is computed at each node from the
  fixed-atom interactions already determined and an LAP-relaxed estimate for
  the remaining atoms; branches that cannot beat the best known cost are pruned.

Block decomposition (:func:`solve_kernel_blockwise`) further splits independent
coupling-components into separate sub-problems, turning one O(k!) search into
a sum of much smaller searches.
"""

from __future__ import annotations

import itertools as _itertools
from collections import defaultdict as _defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..graph.automorphism import node_orbits, orbit_id_map
from ..slap.lap import _adjacency_and_elements, chemical_distance
from .kernel import Kernel, apply_kernel_solution

_MAX_ENUMERATE_BRANCH_SIZE = 15
_MAX_ENUMERATED_SOLUTIONS = 10000


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass
class KernelSolution:
    """Exact solution of a kernel sub-problem.

    Attributes
    ----------
    cost : float
        Optimal chemical distance (cd units) of the full mapping.
    sub_mappings : list[dict[int, int]]
        Optimal kernel sub-mappings (position ``a`` → position ``b``).
        Multiple entries when ``enumerate_all=True``.
    proven_optimal : bool
        Always ``True``: the orbital-branching search is exhaustive over the
        kernel (pruning only removes symmetric or dominated branches).
    """

    cost: float
    sub_mappings: List[Dict[int, int]]
    proven_optimal: bool = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _reference_assignment(kernel: Kernel) -> Dict[int, int]:
    """Return a valid element-compatible kernel assignment (position a → b)."""
    if kernel.candidate_images is not None:
        assigned: Dict[int, int] = {}
        used: set = set()
        p_pos = {p: b for b, p in enumerate(kernel.p_idx)}

        def dfs(a: int) -> bool:
            if a == kernel.size:
                return True
            for p in kernel.candidate_images[a]:
                if p not in p_pos:
                    continue
                b = p_pos[p]
                if b in used or kernel.p_colors[b] != kernel.r_colors[a]:
                    continue
                assigned[a] = b
                used.add(b)
                if dfs(a + 1):
                    return True
                used.remove(b)
                del assigned[a]
            return False

        if dfs(0):
            return assigned

    rpos: dict = _defaultdict(list)
    ppos: dict = _defaultdict(list)
    for a in range(kernel.size):
        rpos[kernel.r_colors[a]].append(a)
    for b in range(kernel.size):
        ppos[kernel.p_colors[b]].append(b)
    ref: Dict[int, int] = {}
    for col, ras in rpos.items():
        for a, b in zip(ras, ppos[col]):
            ref[a] = b
    return ref


def _coupling_components(
    kernel: Kernel, A: np.ndarray, B: np.ndarray
) -> List[List[int]]:
    """Partition kernel positions into mutually independent coupling-components.

    Two positions are *coupled* when their cost terms cannot all be zero
    regardless of assignment: their reactant atoms are adjacent, their candidate
    product sets overlap, or some candidate product pair is adjacent.
    """
    n = kernel.size
    p_pos = {p: b for b, p in enumerate(kernel.p_idx)}
    cand = []
    for a in range(n):
        if kernel.candidate_images is None:
            allowed: list = list(range(n))
        else:
            allowed = [p_pos[p] for p in kernel.candidate_images[a] if p in p_pos]
        cand.append({b for b in allowed if kernel.p_colors[b] == kernel.r_colors[a]})

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for a in range(n):
        i = kernel.r_idx[a]
        for a2 in range(a + 1, n):
            i2 = kernel.r_idx[a2]
            coupled = A[i, i2] != 0 or bool(cand[a] & cand[a2])
            if not coupled:
                for b in cand[a]:
                    pb = kernel.p_idx[b]
                    if any(B[pb, kernel.p_idx[b2]] != 0 for b2 in cand[a2]):
                        coupled = True
                        break
            if coupled:
                union(a, a2)

    groups: dict = _defaultdict(list)
    for a in range(n):
        groups[find(a)].append(a)
    return list(groups.values())


def _subkernel_for_group(kernel: Kernel, comp: List[int], ref: Dict[int, int]) -> tuple:
    """Build a sub-kernel optimising ``comp`` with all other atoms fixed at ``ref``."""
    comp_set = set(comp)
    if kernel.candidate_images is None:
        elems = {kernel.r_colors[a] for a in comp}
        pos_b = [b for b in range(kernel.size) if kernel.p_colors[b] in elems]
    else:
        p_pos = {p: b for b, p in enumerate(kernel.p_idx)}
        pos_b = sorted(
            {p_pos[p] for a in comp for p in kernel.candidate_images[a] if p in p_pos}
        )
    sub_p_idx = [kernel.p_idx[b] for b in pos_b]
    sub_p_set = set(sub_p_idx)

    fixed = dict(kernel.fixed_mapping)
    for a in range(kernel.size):
        if a not in comp_set:
            fixed[kernel.r_idx[a]] = kernel.p_idx[ref[a]]

    subk = Kernel(
        r_idx=[kernel.r_idx[a] for a in comp],
        p_idx=[kernel.p_idx[b] for b in pos_b],
        r_colors=[kernel.r_colors[a] for a in comp],
        p_colors=[kernel.p_colors[b] for b in pos_b],
        fixed_mapping=fixed,
        candidate_images=(
            None
            if kernel.candidate_images is None
            else [
                [p for p in kernel.candidate_images[a] if p in sub_p_set] for a in comp
            ]
        ),
        lgp=kernel.lgp,
        binary=kernel.binary,
    )
    return subk, pos_b


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


def solve_kernel(  # noqa: C901
    kernel: Kernel, enumerate_all: bool = False
) -> KernelSolution:
    """Solve a :class:`~mapper.exact.kernel.Kernel` exactly by orbital branching.

    Assigns each uncertain reactant atom to an element-compatible uncertain
    product atom, pruned by a **reactant lex-leader** rule (atoms in the same
    automorphism orbit take product positions in increasing order) and an
    admissible cost lower bound.

    Parameters
    ----------
    kernel : Kernel
    enumerate_all : bool, optional
        Collect all symmetry-distinct optimal sub-mappings, not just one.

    Returns
    -------
    KernelSolution
    """
    lgp = kernel.lgp
    binary = kernel.binary
    A, _ = _adjacency_and_elements(lgp[0], binary)
    B, _ = _adjacency_and_elements(lgp[1], binary)
    N = A.shape[0]

    base_map = [-1] * N
    for i, p in kernel.fixed_mapping.items():
        base_map[i] = p

    n = kernel.size
    if n == 0:
        # base_map is complete when the kernel is trivial (no uncertain atoms).
        cost = chemical_distance(lgp, base_map, binary)
        return KernelSolution(cost=cost, sub_mappings=[{}])
    if len(kernel.p_idx) != n:
        return KernelSolution(cost=float("inf"), sub_mappings=[])

    p_pos = {p: b for b, p in enumerate(kernel.p_idx)}
    if kernel.candidate_images is None:
        candidate_b = [
            [b for b in range(n) if kernel.p_colors[b] == kernel.r_colors[a]]
            for a in range(n)
        ]
    else:
        candidate_b = []
        for a, images in enumerate(kernel.candidate_images):
            allowed = [
                p_pos[p]
                for p in images
                if p in p_pos and kernel.p_colors[p_pos[p]] == kernel.r_colors[a]
            ]
            candidate_b.append(allowed)
    if any(not candidates for candidates in candidate_b):
        return KernelSolution(cost=float("inf"), sub_mappings=[])
    if (
        enumerate_all
        and n > _MAX_ENUMERATE_BRANCH_SIZE
        and max(len(candidates) for candidates in candidate_b) > 4
    ):
        return KernelSolution(cost=float("inf"), sub_mappings=[])

    r_orbit = orbit_id_map(node_orbits(lgp[0], binary))
    r_pos_orbit = [r_orbit.get(i, i) for i in kernel.r_idx]

    mp = list(base_map)
    assigned_pos = [-1] * n
    assigned_stack: List[int] = []
    used = [False] * n
    fixed_atoms = [i for i in range(N) if mp[i] != -1]

    acc0 = 0.0
    for i in fixed_atoms:
        for j in fixed_atoms:
            acc0 += abs(A[i, j] - B[mp[i], mp[j]])

    best: Dict = {"cost": float("inf"), "sols": [], "aborted": False}

    try:
        ref = _reference_assignment(kernel)
        best["cost"] = chemical_distance(
            lgp,
            apply_kernel_solution(kernel, ref),
            binary,
        )
        if not enumerate_all:
            best["sols"] = [dict(ref)]
    except Exception:
        pass

    fixed_add = [[0.0] * n for _ in range(n)]
    for a in range(n):
        i = kernel.r_idx[a]
        for b in candidate_b[a]:
            p = kernel.p_idx[b]
            c = sum(
                abs(A[i, j] - B[p, mp[j]]) + abs(A[j, i] - B[mp[j], p])
                for j in fixed_atoms
            )
            fixed_add[a][b] = c

    pair_add = [[0.0] * (n * n) for _ in range(n * n)]
    for a in range(n):
        i = kernel.r_idx[a]
        for b in candidate_b[a]:
            p = kernel.p_idx[b]
            row = pair_add[a * n + b]
            for a2 in range(n):
                j = kernel.r_idx[a2]
                for b2 in candidate_b[a2]:
                    q = kernel.p_idx[b2]
                    row[a2 * n + b2] = abs(A[i, j] - B[p, q]) + abs(A[j, i] - B[q, p])

    if max((len(c) for c in candidate_b), default=0) <= 3 and n >= 20:
        search_order = sorted(
            range(n),
            key=lambda a: (
                len(candidate_b[a]),
                -sum(
                    1
                    for j in fixed_atoms
                    if A[kernel.r_idx[a], j] or A[j, kernel.r_idx[a]]
                ),
                kernel.r_colors[a],
                kernel.r_idx[a],
            ),
        )
    else:
        search_order = list(range(n))

    use_remaining_lb = n >= 16

    def incr_cost(a: int, b: int) -> float:
        c = fixed_add[a][b]
        row = pair_add[a * n + b]
        for a2 in assigned_stack:
            c += row[a2 * n + assigned_pos[a2]]
        return c

    def remaining_assignment_lb(depth: int) -> float:
        remaining_a = search_order[depth:]
        if not remaining_a:
            return 0.0
        remaining_b = [b for b in range(n) if not used[b]]
        costs = np.full((len(remaining_a), len(remaining_b)), 1e9, dtype=float)
        for ia, a in enumerate(remaining_a):
            for ib, b in enumerate(remaining_b):
                if b not in candidate_b[a]:
                    continue
                c = fixed_add[a][b]
                row = pair_add[a * n + b]
                for a2 in assigned_stack:
                    c += row[a2 * n + assigned_pos[a2]]
                costs[ia, ib] = c
        row_ind, col_ind = linear_sum_assignment(costs)
        lb = float(costs[row_ind, col_ind].sum())
        if lb >= 1e8:
            return float("inf")

        rem_b_set = set(remaining_b)
        pair_lb = 0.0
        for ia, a in enumerate(remaining_a):
            cand_a = [b for b in candidate_b[a] if b in rem_b_set]
            if not cand_a:
                return float("inf")
            for a2 in remaining_a[ia + 1 :]:
                i = kernel.r_idx[a]
                j = kernel.r_idx[a2]
                if A[i, j] == 0 and A[j, i] == 0:
                    continue
                cand_a2 = [b for b in candidate_b[a2] if b in rem_b_set]
                best_pair = float("inf")
                for b in cand_a:
                    row = pair_add[a * n + b]
                    for b2 in cand_a2:
                        if b == b2:
                            continue
                        c = row[a2 * n + b2]
                        if c < best_pair:
                            best_pair = c
                            if best_pair == 0:
                                break
                    if best_pair == 0:
                        break
                if best_pair == float("inf"):
                    return float("inf")
                pair_lb += best_pair
        return lb + pair_lb

    def dfs(depth: int, acc: float) -> None:
        if best["aborted"]:
            return
        half = 0.5 * acc
        if half > best["cost"] + 1e-9:
            return
        if use_remaining_lb and depth < n and best["cost"] < float("inf"):
            if 0.5 * (acc + remaining_assignment_lb(depth)) > best["cost"] + 1e-9:
                return
        if depth == n:
            if half < best["cost"] - 1e-9:
                best["cost"] = half
                best["sols"] = [dict(enumerate(assigned_pos))]
            elif enumerate_all and abs(half - best["cost"]) <= 1e-9:
                best["sols"].append(dict(enumerate(assigned_pos)))
                if len(best["sols"]) > _MAX_ENUMERATED_SOLUTIONS:
                    best["aborted"] = True
            return

        a = search_order[depth]
        lo = -1
        for a2 in search_order[:depth]:
            if r_pos_orbit[a2] == r_pos_orbit[a] and set(candidate_b[a2]) == set(
                candidate_b[a]
            ):
                lo = max(lo, assigned_pos[a2])

        for b in candidate_b[a]:
            if used[b] or b <= lo:
                continue
            p = kernel.p_idx[b]
            add = incr_cost(a, b)
            mp[kernel.r_idx[a]] = p
            used[b] = True
            assigned_pos[a] = b
            assigned_stack.append(a)

            dfs(depth + 1, acc + add)

            assigned_stack.pop()
            assigned_pos[a] = -1
            used[b] = False
            mp[kernel.r_idx[a]] = -1

    dfs(0, acc0)
    if best["aborted"]:
        return KernelSolution(cost=float("inf"), sub_mappings=[])
    return KernelSolution(cost=best["cost"], sub_mappings=best["sols"])


# ---------------------------------------------------------------------------
# Block-decomposed solver
# ---------------------------------------------------------------------------


def solve_kernel_blockwise(
    kernel: Kernel,
    solver=None,
    enumerate_all: bool = False,
) -> KernelSolution:
    """Solve a kernel by decomposing it into independent coupling-components.

    When the uncertainty region splits into mutually independent groups (often
    along block-cut-tree boundaries), each group is solved separately and the
    results are combined—turning one O(k!) search into a sum of smaller ones.
    When no decomposition is possible the call delegates to ``solver``.

    Parameters
    ----------
    kernel : Kernel
    solver : callable, optional
        Per-group exact solver (defaults to :func:`solve_kernel`).
    enumerate_all : bool, optional
        Enumerate all (symmetry-distinct) optima as the product over groups.

    Returns
    -------
    KernelSolution
    """
    if solver is None:
        solver = solve_kernel

    n = kernel.size
    if n <= 1:
        return solver(kernel, enumerate_all=enumerate_all)

    A, _ = _adjacency_and_elements(kernel.lgp[0], kernel.binary)
    B, _ = _adjacency_and_elements(kernel.lgp[1], kernel.binary)
    comps = _coupling_components(kernel, A, B)
    if len(comps) <= 1:
        return solver(kernel, enumerate_all=enumerate_all)

    ref = _reference_assignment(kernel)
    base_ref = chemical_distance(
        kernel.lgp, apply_kernel_solution(kernel, ref), kernel.binary
    )

    total = base_ref
    proven = True
    group_options: List[List[Dict[int, int]]] = []
    option_count = 1
    for comp in comps:
        subk, pos_b = _subkernel_for_group(kernel, comp, ref)
        sub = solver(subk, enumerate_all=enumerate_all)
        if not sub.sub_mappings:
            return KernelSolution(cost=float("inf"), sub_mappings=[])
        total += sub.cost - base_ref
        proven = proven and sub.proven_optimal
        opts = [
            {comp[a_sub]: pos_b[b_sub] for a_sub, b_sub in sm.items()}
            for sm in sub.sub_mappings
        ]
        group_options.append(opts)
        if enumerate_all:
            option_count *= len(opts)
            if option_count > _MAX_ENUMERATED_SOLUTIONS:
                return KernelSolution(cost=float("inf"), sub_mappings=[])

    combos = (
        _itertools.product(*group_options)
        if enumerate_all
        else [tuple(opts[0] for opts in group_options)]
    )
    sub_mappings = []
    for combo in combos:
        merged = dict(ref)
        for part in combo:
            merged.update(part)
        sub_mappings.append(merged)

    return KernelSolution(cost=total, sub_mappings=sub_mappings, proven_optimal=proven)
