"""Bounded typed quadratic-overlap relaxation for feasible search seeds."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix


def improve_relaxed_seed_mapping(a, b, er, ep, mapping, *, iterations=40):
    """Find a better typed bijection using four bounded relaxed starts.

    At permutation vertices, layered same-sign bond overlap equals a constant
    minus chemical distance. Sparse matrix products give its quadratic gradient;
    element-blocked LAP steps and quadratic line searches optimize a doubly
    stochastic relaxation. Only a fully evaluated feasible mapping is returned.
    The relaxation is a heuristic, never an optimality certificate or a domain
    restriction. Size/layer limits bound this optional preprocessing.
    """
    n = len(mapping)
    if sorted(mapping) != list(range(len(er))) or any(
        er[i] != ep[image] for i, image in enumerate(mapping)
    ):
        raise ValueError("seed must be a complete element-compatible bijection")
    if n > 256 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return list(mapping), {"iterations": 0, "skipped": "size_or_nonfinite_input"}
    if len(np.unique(np.concatenate((a.ravel(), b.ravel())))) > 16:
        return list(mapping), {"iterations": 0, "skipped": "weight_layer_limit"}
    blocks = [
        (np.flatnonzero(np.asarray(er) == e), np.flatnonzero(np.asarray(ep) == e))
        for e in sorted(set(er))
    ]
    layers = []
    for sign in (1, -1):
        values = np.unique(np.concatenate(((sign * a).ravel(), (sign * b).ravel())))
        previous = 0.0
        for value in values[values > 0]:
            ar = csr_matrix((sign * a >= value).astype(float))
            bp = csr_matrix((sign * b >= value).astype(float))
            layers.append((float(value - previous), ar, bp, ar.T.tocsr(), bp.T.tocsr()))
            previous = value

    def gradient(p):
        g = np.zeros_like(p)
        for weight, ar, bp, at, bt in layers:
            g += weight * ((bp @ (ar @ p).T).T + (bt @ (at @ p).T).T)
        return g

    def project(p):
        result = np.empty(n, dtype=int)
        for rows, cols in blocks:
            i, j = linear_sum_assignment(-p[np.ix_(rows, cols)])
            result[rows[i]] = cols[j]
        return result

    def cost(m):
        return 0.5 * float(np.abs(a - b[np.ix_(m, m)]).sum())

    best = np.asarray(mapping).copy()
    best_cost = cost(best)
    identity = np.zeros((n, n))
    identity[np.arange(n), best] = 1
    uniform = np.zeros((n, n))
    for rows, cols in blocks:
        uniform[np.ix_(rows, cols)] = 1 / len(cols)
    count = 0
    for mixing in (0.95, 0.7, 0.3, 0.0):
        p = mixing * identity + (1 - mixing) * uniform
        for _ in range(iterations):
            g = gradient(p)
            candidate = project(g)
            candidate_cost = cost(candidate)
            if candidate_cost < best_cost:
                best, best_cost = candidate.copy(), candidate_cost
            q = np.zeros_like(p)
            q[np.arange(n), candidate] = 1
            d = q - p
            slope = float((g * d).sum())
            curve = 0.5 * float((d * gradient(d)).sum())
            t = (
                min(1.0, max(0.0, -slope / (2 * curve)))
                if curve < 0
                else float(slope + curve > 0)
            )
            count += 1
            if t < 1e-8 or np.linalg.norm(d) < 1e-8:
                break
            p += t * d
        candidate = project(p)
        candidate_cost = cost(candidate)
        if candidate_cost < best_cost:
            best, best_cost = candidate.copy(), candidate_cost
    return best.tolist(), {
        "cost": best_cost,
        "iterations": count,
        "max_iterations_per_start": iterations,
        "starts": 4,
    }
