"""
Exact MILP/QAP formulation of the kernel mapping problem (PuLP + CBC backend).

The chemical-distance objective is a quadratic assignment problem (QAP). Over the
small uncertainty-region kernel (:mod:`mapper.exact.kernel`) we write it as a
Koopmans-Beckmann QAP and linearise it to a mixed-integer linear program:

* binary ``x[a,b] = 1`` iff kernel reactant atom ``a`` maps to kernel product
  atom ``b`` (created only for element-compatible pairs);
* assignment constraints ``sum_b x[a,b] = 1`` and ``sum_a x[a,b] = 1``;
* products ``y[a,b,a',b'] = x[a,b] * x[a',b']`` linearised by
  ``y <= x[a,b]``, ``y <= x[a',b']``, ``y >= x[a,b] + x[a',b'] - 1``.

Writing the chemical distance ``1/2 * sum_{i,j} |A_R[i,j] - A_P[m(i),m(j)]|`` and
splitting atoms into certain/kernel, the certain-certain part is a constant, the
kernel-certain part becomes linear coefficients on ``x``, and the kernel-kernel
part becomes coefficients on ``y`` (the factor 1/2 and the two ordered directions
of each undirected edge cancel). CBC solves the MILP to proven optimality, so the
returned cost certifies the kernel optimum (see :mod:`mapper.exact.certificate`).

The solver is isolated behind :func:`solve_qap`; :func:`solve_kernel_milp` adapts
a :class:`~mapper.exact.kernel.Kernel` to it and returns the same
:class:`~mapper.exact.branching.KernelSolution` type as the orbital-branching
solver, so callers can use either interchangeably.
"""

from __future__ import annotations

from typing import Dict, Tuple

import warnings

from ..slap.lap import _adjacency_and_elements
from .branching import KernelSolution

try:
    import pulp

    HAS_PULP = True
except Exception:  # pragma: no cover - exercised only without pulp installed
    HAS_PULP = False


def _cbc_solver():
    """Return a silent CBC solver, preferring the non-deprecated entry point."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            solver = pulp.COIN_CMD(msg=False)
            if solver.available():
                return solver
        except Exception:
            pass
        return pulp.PULP_CBC_CMD(msg=False)


def solve_qap(lin, quad, n, compatible, const=0.0):
    """Solve a linearised QAP assignment with PuLP/CBC.

    Parameters
    ----------
    lin : dict[(int, int), float]
        Linear cost ``lin[a, b]`` for assigning ``a -> b``.
    quad : dict[(int, int, int, int), float]
        Quadratic cost ``quad[a, b, a', b']`` (defined for ``a < a'``) added when
        both ``a -> b`` and ``a' -> b'`` hold.
    n : int
        Number of reactant positions (== product positions).
    compatible : set[(int, int)]
        Allowed ``(a, b)`` assignments (element-compatible pairs).
    const : float
        Constant added to the objective.

    Returns
    -------
    (cost, mapping, proven) : (float, dict[int, int], bool)
        Optimal cost, the assignment ``a -> b``, and whether CBC proved
        optimality.
    """
    if not HAS_PULP:
        raise ImportError("pulp is required for the MILP/QAP solver")

    prob = pulp.LpProblem("kernel_qap", pulp.LpMinimize)

    # PuLP 3.x emits 4.0-migration DeprecationWarnings from the supported
    # LpVariable API; silence that noise around variable construction.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        x = {
            (a, b): pulp.LpVariable(f"x_{a}_{b}", cat="Binary") for (a, b) in compatible
        }
        y = {}
        for (a, b, a2, b2), c in quad.items():
            if (a, b) in x and (a2, b2) in x:
                y[(a, b, a2, b2)] = pulp.LpVariable(
                    f"y_{a}_{b}_{a2}_{b2}", lowBound=0, upBound=1, cat="Continuous"
                )

    # Objective.
    obj = [lin[(a, b)] * x[(a, b)] for (a, b) in x if lin.get((a, b))]
    obj += [quad[k] * y[k] for k in y if quad.get(k)]
    prob += pulp.lpSum(obj) + const

    # Assignment constraints.
    for a in range(n):
        prob += pulp.lpSum(x[(a, b)] for b in range(n) if (a, b) in x) == 1
    for b in range(n):
        prob += pulp.lpSum(x[(a, b)] for a in range(n) if (a, b) in x) == 1

    # McCormick linearisation of the products.
    for (a, b, a2, b2), yv in y.items():
        prob += yv <= x[(a, b)]
        prob += yv <= x[(a2, b2)]
        prob += yv >= x[(a, b)] + x[(a2, b2)] - 1

    prob.solve(_cbc_solver())

    proven = pulp.LpStatus[prob.status] == "Optimal"
    mapping = {}
    for (a, b), var in x.items():
        if var.value() is not None and var.value() > 0.5:
            mapping[a] = b
    cost = pulp.value(prob.objective)
    if cost is None:
        raise RuntimeError(
            "MILP solver returned no objective value — problem may be infeasible. "
            "Verify that element-compatible assignments exist for all kernel atoms."
        )
    return float(cost), mapping, proven


def solve_kernel_milp(kernel):  # noqa: C901
    """Solve a :class:`~mapper.exact.kernel.Kernel` exactly via MILP/QAP.

    Returns
    -------
    KernelSolution
        With ``proven_optimal`` reflecting CBC's status.
    """
    lgp = kernel.lgp
    binary = kernel.binary
    A, _ = _adjacency_and_elements(lgp[0], binary)
    B, _ = _adjacency_and_elements(lgp[1], binary)
    N = A.shape[0]
    n = kernel.size

    base_map = [-1] * N
    for i, p in kernel.fixed_mapping.items():
        base_map[i] = p
    certain = [i for i in range(N) if base_map[i] != -1]

    # Certain-certain constant (0.5 * ordered sum).
    const = 0.0
    for i in certain:
        for j in certain:
            const += abs(A[i, j] - B[base_map[i], base_map[j]])
    const *= 0.5

    if n == 0:
        return KernelSolution(cost=const, sub_mappings=[{}], proven_optimal=True)

    compatible = set()
    lin: Dict[Tuple[int, int], float] = {}
    for a in range(n):
        i = kernel.r_idx[a]
        for b in range(n):
            if kernel.r_colors[a] != kernel.p_colors[b]:
                continue
            p = kernel.p_idx[b]
            compatible.add((a, b))
            # Kernel-certain: 0.5 * (ordered i,j + ordered j,i) = sum_j |A-B|.
            lin[(a, b)] = sum(abs(A[i, j] - B[p, base_map[j]]) for j in certain)

    quad: Dict[Tuple[int, int, int, int], float] = {}
    for a in range(n):
        i = kernel.r_idx[a]
        for a2 in range(a + 1, n):
            i2 = kernel.r_idx[a2]
            for b in range(n):
                if (a, b) not in compatible:
                    continue
                p = kernel.p_idx[b]
                for b2 in range(n):
                    if (a2, b2) not in compatible or b2 == b:
                        continue
                    p2 = kernel.p_idx[b2]
                    # 0.5 * (ordered (i,i') + (i',i)) = |A_R[i,i'] - A_P[p,p']|.
                    c = abs(A[i, i2] - B[p, p2])
                    if c:
                        quad[(a, b, a2, b2)] = c

    cost, mapping, proven = solve_qap(lin, quad, n, compatible, const=const)
    return KernelSolution(cost=cost, sub_mappings=[mapping], proven_optimal=proven)
