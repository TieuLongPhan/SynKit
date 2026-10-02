"""Independent global all-solutions bond-order assignment MILP (SciPy/HiGHS).

This is a new reference formulation, not a reproduction of a published mapper.
It does not import Synister's search, bounds, symmetry or objective evaluation.
Completion relies on the numerical MILP solver's optimal/infeasible statuses;
it is not a machine-checked certificate. Every returned map is integer-rescored.
"""

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import math
import time
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csc_matrix, vstack


@dataclass
class AssignmentModel:
    pairs: tuple
    pair_index: dict
    products: tuple
    objective: np.ndarray
    integrality: np.ndarray
    matrix: csc_matrix
    lower: np.ndarray
    upper: np.ndarray
    constant: int


def build_model(reactant, product):
    """Use |a-b|=a+b-2 min(a,b); auxiliaries only for two present bonds."""
    reactant.__post_init__()
    product.__post_init__()
    if Counter(reactant.atomic_numbers) != Counter(product.atomic_numbers):
        raise ValueError("Global MILP requires equal element inventories")
    n = len(reactant.atomic_numbers)
    pairs = tuple((i, p) for i in range(n) for p in range(n)
                  if reactant.atomic_numbers[i] == product.atomic_numbers[p])
    indexes = {pair: index for index, pair in enumerate(pairs)}
    products = []
    coefficients = [0] * len(pairs)
    for i, j, a in reactant.bonds:
        for p, q, b in product.bonds:
            for first, second in ((p, q), (q, p)):
                if (i, first) in indexes and (j, second) in indexes:
                    products.append((indexes[i, first], indexes[j, second]))
                    coefficients.append(-2 * min(a, b))
    rows, columns, values, lower, upper = [], [], [], [], []
    def row(entries, lo, hi):
        index = len(lower)
        for col, value in entries:
            rows.append(index)
            columns.append(col)
            values.append(value)
        lower.append(lo)
        upper.append(hi)
    for i in range(n):
        row(((index, 1) for (atom, _), index in indexes.items() if atom == i), 1, 1)
    for p in range(n):
        row(((index, 1) for (_, image), index in indexes.items() if image == p), 1, 1)
    for y, (first, second) in enumerate(products, start=len(pairs)):
        row(((y, 1), (first, -1)), -np.inf, 0)
        row(((y, 1), (second, -1)), -np.inf, 0)
        row(((y, 1), (first, -1), (second, -1)), -1, np.inf)
    matrix = coo_matrix((np.asarray(values, dtype=float), (rows, columns)),
                        shape=(len(lower), len(coefficients))).tocsc()
    return AssignmentModel(pairs, indexes, tuple(products), np.array(coefficients, dtype=float),
                           np.array([1] * len(pairs) + [0] * len(products)), matrix,
                           np.array(lower), np.array(upper),
                           sum(w for _, _, w in reactant.bonds) + sum(w for _, _, w in product.bonds))


def doubled_distance(reactant, product, mapping):
    """Literal integer witness rescoring, separate from MILP expression."""
    n = len(reactant.atomic_numbers)
    if len(mapping) != n or sorted(mapping) != list(range(n)):
        raise ValueError("Returned mapping is not a bijection")
    if any(reactant.atomic_numbers[i] != product.atomic_numbers[p] for i, p in enumerate(mapping)):
        raise ValueError("Returned mapping does not preserve elements")
    rb = {(i, j): w for i, j, w in reactant.bonds}
    pb = {(i, j): w for i, j, w in product.bonds}
    return sum(abs(rb.get((i, j), 0) - pb.get(tuple(sorted((mapping[i], mapping[j]))), 0))
               for i, j in combinations(range(n), 2))


def decode(model, solution, n):
    if solution is None or not np.isfinite(solution).all():
        raise ValueError("No finite MILP solution")
    assignment = solution[:len(model.pairs)]
    if np.any(np.abs(assignment - np.rint(assignment)) > 1e-6):
        raise ValueError("MILP assignment is not integral")
    mapping = [-1] * n
    for (i, p), value in zip(model.pairs, assignment):
        if value > 0.5:
            if mapping[i] != -1:
                raise ValueError("Multiple images in MILP assignment")
            mapping[i] = p
    if sorted(mapping) != list(range(n)):
        raise ValueError("MILP assignment is not a permutation")
    return tuple(mapping)


def enumerate_milp(reactant, product, *, target="minimal", seconds=60, max_maps=100_000,
                   initial_mapping=None, _solve=milp):
    """Enumerate indexed maps at a doubled-CD target; enforce a total deadline.

    Setup, repeated optimization and map checking count toward `seconds`.
    A supervising process must enforce a hard wall/memory limit, because native
    solver preprocessing and Python model construction are not interruptible here.
    A supplied feasible mapping adds a non-strict objective upper bound only
    for minimum queries. SciPy's interface does not receive a native warm start.
    """
    if target != "minimal" and (isinstance(target, bool) or not isinstance(target, int) or target < 0):
        raise ValueError("Target must be 'minimal' or a nonnegative doubled-CD integer")
    if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Time limit must be finite and nonnegative")
    if isinstance(max_maps, bool) or not isinstance(max_maps, int) or max_maps < 1:
        raise ValueError("Map cap must be a positive integer")
    started = time.perf_counter()
    deadline = started + seconds
    seed_cost = None
    if initial_mapping is not None:
        if target != 'minimal':
            raise ValueError('A seed cutoff is only defined for minimum queries')
        seed_cost = doubled_distance(reactant, product, tuple(initial_mapping))
    result = {"method": "independent_global_milp_highs", "target_doubled_cd": target,
              "complete": False, "minimum_proved": False, "minimum_doubled_cd": None,
              "minimum_proof_seconds": None, "first_map_seconds": None,
              "mappings": [], "solver_calls": [], "termination": "time_limit",
              "seed_doubled_cd": seed_cost,
              "seed_interface": "feasible_cost_cutoff_not_native_warm_start" if seed_cost is not None else "none",
              "guarantee": "Numerical MILP status, not independently verified infeasibility/optimality"}
    def finish(reason, complete=False):
        result.update(termination=reason, complete=complete,
                      elapsed_seconds=time.perf_counter() - started)
        return result
    if seconds == 0:
        return finish("time_limit")
    model = build_model(reactant, product)
    n = len(reactant.atomic_numbers)
    result.update(setup_seconds=time.perf_counter()-started, assignment_variables=len(model.pairs),
                  product_variables=len(model.products), base_constraints=len(model.lower))
    if time.perf_counter() >= deadline:
        return finish("setup_time_limit")
    matrix, lower, upper = model.matrix, model.lower, model.upper
    if seed_cost is not None:
        matrix = vstack((matrix, csc_matrix(model.objective.reshape(1, -1))), format='csc')
        lower = np.append(lower, -np.inf)
        upper = np.append(upper, seed_cost-model.constant)
    variable_bounds = Bounds(np.zeros(len(model.objective)), np.ones(len(model.objective)))
    def solve(phase, objective):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return None
        before = time.perf_counter()
        # SciPy forwards extra options to HiGHS; set one solver thread explicitly.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected.*", category=RuntimeWarning)
            answer = _solve(objective, integrality=model.integrality, bounds=variable_bounds,
                            constraints=LinearConstraint(matrix, lower, upper),
                            options={"time_limit": remaining, "mip_rel_gap": 0,
                                     "threads": 1, "mip_abs_gap": 0, "presolve": True})
        gap = getattr(answer, "mip_gap", None)
        result["solver_calls"].append({"phase": phase, "status": int(answer.status),
                                      "message": str(answer.message),
                                      "seconds": time.perf_counter()-before,
                                      "mip_gap": float(gap) if gap is not None and math.isfinite(gap) else None,
                                      "nodes": int(getattr(answer, "mip_node_count", 0) or 0)})
        return answer
    def checked_map(answer):
        mapping = decode(model, answer.x, n)
        actual = doubled_distance(reactant, product, mapping)
        represented = model.constant + float(model.objective @ answer.x)
        if abs(actual - represented) > 1e-5:
            raise ValueError("MILP objective and integer witness cost disagree")
        return mapping, actual
    optimum_mapping = None
    if target == "minimal":
        answer = solve("minimum", model.objective)
        if answer is None or answer.status == 1:
            return finish("minimum_time_limit")
        if answer.status != 0 or (getattr(answer, "mip_gap", 0) or 0) > 0:
            return finish("minimum_solver_failure")
        try:
            optimum_mapping, target = checked_map(answer)
        except ValueError as exc:
            result["error"] = str(exc)
            return finish("invalid_solver_witness")
        result.update(minimum_proved=True, minimum_doubled_cd=target,
                      minimum_proof_seconds=time.perf_counter()-started)
    # Equality is safe because every auxiliary is linked to its binary product
    # on BOTH sides, not just lower-bounded by an absolute-value relaxation.
    matrix = vstack((matrix, csc_matrix(model.objective.reshape(1, -1))), format="csc")
    lower = np.append(lower, target-model.constant)
    upper = np.append(upper, target-model.constant)
    seen = set()
    while time.perf_counter() < deadline:
        if optimum_mapping is not None:
            mapping, actual = optimum_mapping, target
            optimum_mapping = None
        else:
            answer = solve("enumeration", np.zeros_like(model.objective))
            if answer is None or answer.status == 1:
                return finish("enumeration_time_limit")
            if answer.status == 2:
                return finish("exhausted" if seen else "proved_empty", complete=True)
            if answer.status != 0:
                return finish("enumeration_solver_failure")
            try:
                mapping, actual = checked_map(answer)
            except ValueError as exc:
                result["error"] = str(exc)
                return finish("invalid_solver_witness")
        if actual != target or mapping in seen:
            return finish("wrong_distance_or_duplicate")
        if result["first_map_seconds"] is None:
            result["first_map_seconds"] = time.perf_counter()-started
        seen.add(mapping)
        result["mappings"].append(mapping)
        if len(seen) >= max_maps:
            return finish("output_limit")
        indexes = [model.pair_index[i, p] for i, p in enumerate(mapping)]
        exclusion = csc_matrix((np.ones(n), (np.zeros(n, dtype=int), indexes)),
                               shape=(1, len(model.objective)))
        matrix = vstack((matrix, exclusion), format="csc")
        lower = np.append(lower, -np.inf)
        upper = np.append(upper, n-1)
    return finish("enumeration_time_limit")
