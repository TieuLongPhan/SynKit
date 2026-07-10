"""
Symmetry-distinct enumeration of exact kernel optima.

The exact kernel solvers can enumerate every optimal assignment over the small
uncertainty region. This module stitches those kernel assignments back into full
atom mappings and removes duplicates under graph automorphisms. For mapped
reaction SMILES, ITS canonical hashes provide the chemistry-aware canonical form;
for plain graph matching, mappings are canonicalised under ``Aut(R) x Aut(P)``.
"""

from __future__ import annotations

import itertools
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..chem.its import (
    dedup_mapped_rxns,
    mapped_rxn_is_electron_balanced,
    reaction_center_atom_maps,
)
from ..chem.smiles import get_numbered_rxn_smiles, expand_reaction_center_hydrogens
from ..graph.automorphism import to_nx
from ..slap.lap import _adjacency_and_elements, chemical_distance, recover_mapping
from .branching import solve_kernel_blockwise
from .kernel import apply_kernel_solution

_MAX_MAPPED_ENUMERATION_RESULTS = 1000
_MAX_SYNKIT_DEDUP_RESULTS = 64
_MAX_AUTOMORPHISM_DEDUP_RESULTS = 64
_MAX_PAIR_REPAIR_ATOMS = 80
_PAIR_REPAIR_MIN_CD = 10.0
_MAX_PAIR_REPAIR_BOND_PAIRS = 120
_MAX_PAIR_REPAIR_DESCENT_STEPS = 3
_MAX_HCOUNT_PERMUTE_BLOCK = 6
_MAX_HCOUNT_PERMUTATION_PRODUCTS = 5000
_LOCAL_SWAP_DEFAULT_MAX_ALL_ATOMS = 48
_LOCAL_SWAP_DEFAULT_MAX_POOL_PER_ELEMENT = 14
_LOCAL_SWAP_DEFAULT_MAX_DESCENT_STEPS = 8


@dataclass
class EnumerationResult:
    """Symmetry-distinct exact optima for a kernel."""

    cost: float
    mappings: List[List[int]]
    sub_mappings: List[Dict[int, int]]
    results: List[dict]
    proven_optimal: bool = True
    enumeration_complete: bool = True


def mapping_to_lgp(lgp, mapping):
    """Return a resolved graph pair whose labels encode ``mapping``."""
    out = [lgp[0].copy(), lgp[1].copy()]
    labels0 = list(range(1, len(mapping) + 1))
    labels1 = [0] * len(mapping)
    for i, p in enumerate(mapping):
        labels1[p] = i + 1
    out[0].labels = labels0
    out[1].labels = labels1
    out[0].build_label2idxs()
    out[1].build_label2idxs()
    return out


def improve_results_by_pair_swaps(lgp, results, binary=False, valfactor=2):
    """Append greedy coupled same-element bond-pair repairs."""
    if (
        not results
        or len(lgp[0].labels) > _MAX_PAIR_REPAIR_ATOMS
        or not _has_balancing_dummies(lgp)
        or _best_result_cd(results, valfactor) < _PAIR_REPAIR_MIN_CD
    ):
        return results

    improved = []
    seen = set()
    for result in results:
        mapping = complete_mapping(
            lgp,
            recover_mapping(result["lgp"]),
            binary=binary,
        )
        key = tuple(mapping)
        if key not in seen:
            seen.add(key)
            original = dict(result)
            original.setdefault("mapping", list(mapping))
            improved.append(original)

        repaired, cd = _pair_swap_descent(lgp, mapping, binary=binary)
        key = tuple(repaired)
        if key in seen:
            continue
        seen.add(key)
        updated = dict(result)
        updated["lgp"] = mapping_to_lgp(lgp, repaired)
        updated["mapping"] = repaired
        updated["cd"] = cd
        updated["val"] = cd * valfactor
        updated["repair"] = "pair-swap"
        improved.append(updated)

    improved.sort(
        key=lambda r: (
            float(r.get("cd", r.get("val", 0) / valfactor)),
            r.get("repair") or "",
        )
    )
    return improved


def expand_results_by_local_swaps(
    lgp,
    results,
    binary=False,
    valfactor=2,
    depth=0,
    cap=128,
    slack=0.0,
    min_cd=4.0,
    max_all_atoms=_LOCAL_SWAP_DEFAULT_MAX_ALL_ATOMS,
    max_pool_per_element=_LOCAL_SWAP_DEFAULT_MAX_POOL_PER_ELEMENT,
    max_descent_steps=_LOCAL_SWAP_DEFAULT_MAX_DESCENT_STEPS,
):
    """Add same-or-better CD maps from bounded same-element swaps."""
    if (
        depth <= 0
        or cap <= 0
        or not results
        or not _has_balancing_dummies(lgp)
        or _best_result_cd(results, valfactor) < min_cd
    ):
        return results

    seeds = []
    seed_seen = set()
    for result in results:
        mapping = tuple(
            complete_mapping(
                lgp,
                recover_mapping(result["lgp"]),
                binary=binary,
            )
        )
        if mapping in seed_seen:
            continue
        seed_seen.add(mapping)
        cd = float(result.get("cd", chemical_distance(lgp, mapping, binary)))
        seeds.append((list(mapping), cd))
    if not seeds:
        return results

    original_keys = {tuple(mapping) for mapping, _ in seeds}
    records = _bounded_local_swap_records(
        lgp,
        seeds,
        binary=binary,
        depth=depth,
        slack=slack,
        cap=cap,
        max_all_atoms=max_all_atoms,
        max_pool_per_element=max_pool_per_element,
        max_descent_steps=max_descent_steps,
    )
    if not records:
        return results

    best = min(cd for _, cd in records)
    filtered = [(mapping, cd) for mapping, cd in records if cd <= best + slack + 1e-9][
        :cap
    ]
    if not filtered:
        return results
    filtered_keys = {tuple(mapping) for mapping, _ in filtered}
    if filtered_keys <= original_keys:
        return results

    expanded = []
    seen = set()
    for result in results:
        mapping = tuple(
            complete_mapping(
                lgp,
                recover_mapping(result["lgp"]),
                binary=binary,
            )
        )
        seen.add(mapping)
        updated = dict(result)
        updated.setdefault("mapping", list(mapping))
        expanded.append(updated)

    for mapping, cd in filtered:
        key = tuple(mapping)
        if key in seen:
            continue
        seen.add(key)
        expanded.append(
            {
                "lgp": mapping_to_lgp(lgp, mapping),
                "mapping": list(mapping),
                "cd": cd,
                "val": cd * valfactor,
                "repair": "local-swap",
            }
        )
    return expanded or results


def improve_results_by_hcount_permutations(
    lgp,
    results,
    binary=False,
    valfactor=2,
    hcount_weight=0.0,
):
    """Append small-block optima ranked by CD plus H-count change."""
    if hcount_weight <= 0 or not results:
        return results

    context = _hcount_scoring_context(lgp, binary)
    improved = []
    seen = set()
    for result in results:
        mapping = complete_mapping(
            lgp,
            recover_mapping(result["lgp"]),
            binary=binary,
        )
        base_mapping, cd, hdelta, role, score = _hcount_mapping_record(
            lgp,
            mapping,
            binary,
            hcount_weight,
            context=context,
        )
        base = dict(result)
        base["mapping"] = base_mapping
        base["cd"] = cd
        base["val"] = cd * valfactor
        base["hcount_delta"] = hdelta
        base["hcount_score"] = score
        base["role_delta"] = role
        key = tuple(base_mapping)
        if key not in seen:
            seen.add(key)
            improved.append(base)

        optima = _hcount_permutation_optima(
            lgp,
            mapping,
            binary=binary,
            hcount_weight=hcount_weight,
            context=context,
        )
        for repaired, cd, hdelta, role, score in optima:
            key = tuple(repaired)
            if key in seen:
                continue
            seen.add(key)
            updated = dict(result)
            updated["lgp"] = mapping_to_lgp(lgp, repaired)
            updated["mapping"] = repaired
            updated["cd"] = cd
            updated["val"] = cd * valfactor
            updated["hcount_delta"] = hdelta
            updated["hcount_score"] = score
            updated["role_delta"] = role
            improved.append(updated)

    improved.sort(
        key=lambda r: (
            float(r.get("hcount_score", 0.0)),
            float(r.get("cd", 0.0)),
            float(r.get("role_delta", 0.0)),
            r.get("repair") or "",
        )
    )
    return improved


def annotate_hcount_scores(lgp, results, binary=False, hcount_weight=0.0):
    """Attach H-count score fields to final result dictionaries."""
    if hcount_weight <= 0:
        return
    for result in results:
        mapping = result.get("mapping")
        if mapping is None:
            mapping = recover_mapping(result["lgp"])
        cd = result.get("cd")
        if cd is None:
            cd = chemical_distance(lgp, mapping, binary)
            result["cd"] = cd
        hdelta = _hcount_delta(lgp, mapping)
        result["hcount_delta"] = hdelta
        result["hcount_score"] = cd + hcount_weight * hdelta
        result["role_delta"] = _role_delta(lgp, mapping)


def _best_result_cd(results, valfactor):
    best = float("inf")
    for result in results:
        cd = result.get("cd")
        if cd is None and "val" in result:
            cd = result["val"] / valfactor
        if cd is not None and cd < best:
            best = cd
    return best


def _has_balancing_dummies(lgp):
    for graph in lgp:
        slices = graph.props.get("natoms slices")
        if slices and len(slices) >= 2 and slices[-1] != slices[-2]:
            return True
    return False


def _hcount_permutation_optima(
    lgp,
    mapping,
    binary=False,
    hcount_weight=1.0,
    context=None,
):
    elements = _elements(lgp[0])
    prod_elements = _elements(lgp[1])
    blocks = _small_element_blocks(elements, prod_elements)
    option_lists = []
    option_count = 1
    for block in blocks:
        product_pool = [mapping[i] for i in block]
        options = list(itertools.permutations(product_pool))
        option_count *= len(options)
        if option_count > _MAX_HCOUNT_PERMUTATION_PRODUCTS:
            return [
                _hcount_mapping_record(
                    lgp,
                    mapping,
                    binary,
                    hcount_weight,
                    context=context,
                )
            ]
        option_lists.append((block, options))

    if context is None:
        context = _hcount_scoring_context(lgp, binary)
    best_score = None
    optima = []
    combos = (
        itertools.product(*(options for _, options in option_lists))
        if option_lists
        else [()]
    )
    for combo in combos:
        candidate = list(mapping)
        for (block, _), perm in zip(option_lists, combo):
            for i, p in zip(block, perm):
                candidate[i] = p
        if len(set(candidate)) != len(candidate):
            continue
        record = _hcount_mapping_record(
            lgp,
            candidate,
            binary,
            hcount_weight,
            context=context,
        )
        score = record[-1]
        if best_score is None or score < best_score - 1e-9:
            best_score = score
            optima = [record]
        elif abs(score - best_score) <= 1e-9:
            optima.append(record)
    return optima or [
        _hcount_mapping_record(
            lgp,
            mapping,
            binary,
            hcount_weight,
            context=context,
        )
    ]


def _hcount_scoring_context(lgp, binary):
    A, _ = _adjacency_and_elements(lgp[0], binary=binary)
    B, _ = _adjacency_and_elements(lgp[1], binary=binary)
    h_r, h_p = _atom_hcounts(lgp)
    return A, B, np.asarray(h_r, dtype=int), np.asarray(h_p, dtype=int)


def _hcount_mapping_record(lgp, mapping, binary, hcount_weight, context=None):
    if context is None:
        context = _hcount_scoring_context(lgp, binary)
    A, B, h_r, h_p = context
    cd = _chemical_distance_arrays(A, B, mapping)
    hdelta = _hcount_delta_arrays(h_r, h_p, mapping)
    role = _role_delta_arrays(A, B, mapping)
    score = cd + hcount_weight * hdelta
    return list(mapping), cd, hdelta, role, score


def _chemical_distance_arrays(A, B, mapping):
    mapping = np.asarray(mapping, dtype=int)
    return 0.5 * float(np.abs(A - B[np.ix_(mapping, mapping)]).sum())


def _hcount_delta_arrays(h_r, h_p, mapping):
    mapping = np.asarray(mapping, dtype=int)
    return int(np.abs(h_r - h_p[mapping]).sum())


def _role_delta_arrays(A, B, mapping):
    total = 0.0
    for i, p in enumerate(mapping):
        left = sorted(float(w) for w in A[i] if w != 0)
        right = sorted(float(w) for w in B[p] if w != 0)
        n = max(len(left), len(right))
        left.extend([0.0] * (n - len(left)))
        right.extend([0.0] * (n - len(right)))
        total += sum(abs(a - b) for a, b in zip(left, right))
    return total


def _small_element_blocks(elements_r, elements_p):
    blocks = []
    for element in sorted(set(elements_r)):
        r_block = [i for i, e in enumerate(elements_r) if e == element]
        p_count = sum(1 for e in elements_p if e == element)
        if len(r_block) == p_count and 1 < len(r_block) <= _MAX_HCOUNT_PERMUTE_BLOCK:
            blocks.append(r_block)
    return blocks


def _atom_hcounts(lgp):
    out = []
    for side in (0, 1):
        mol = lgp[side].props["sources"][-1]
        base_natoms = lgp[side].props["natoms slices"][0]
        hcounts = []
        for i, atom in enumerate(mol.GetAtoms()):
            hcounts.append(
                atom.GetTotalNumHs(includeNeighbors=True) if i < base_natoms else 0
            )
        out.append(hcounts)
    return out


def _hcount_delta(lgp, mapping):
    h_r, h_p = _atom_hcounts(lgp)
    return sum(abs(h_r[i] - h_p[p]) for i, p in enumerate(mapping))


def _role_delta(lgp, mapping):
    A, _ = _adjacency_and_elements(lgp[0], binary=False)
    B, _ = _adjacency_and_elements(lgp[1], binary=False)
    return _role_delta_arrays(A, B, mapping)


def _pair_swap_descent(lgp, mapping, binary=False):
    A, _ = _adjacency_and_elements(lgp[0], binary=binary)
    B, _ = _adjacency_and_elements(lgp[1], binary=binary)
    elements = _elements(lgp[0])
    current = list(mapping)
    current_cd = chemical_distance(lgp, current, binary)
    pairs = _budgeted_oriented_bond_pairs(A, B, current, elements)

    for _ in range(_MAX_PAIR_REPAIR_DESCENT_STEPS):
        best = None
        for x, (i, j, ei, ej) in enumerate(pairs):
            for k, ell, ek, el in pairs[x + 1 :]:
                if ei != ek or ej != el or len({i, j, k, ell}) < 4:
                    continue
                cd = _pair_swap_cd(A, B, current, current_cd, i, j, k, ell)
                if cd < current_cd - 1e-9 and (best is None or cd < best[0]):
                    best = (cd, i, j, k, ell)
        if best is None:
            return current, current_cd
        current_cd, i, j, k, ell = best
        current[i], current[k] = current[k], current[i]
        current[j], current[ell] = current[ell], current[j]
    return current, current_cd


def _budgeted_oriented_bond_pairs(A, B, mapping, elements):
    pairs = _oriented_bond_pairs(A, elements)
    if len(pairs) <= _MAX_PAIR_REPAIR_BOND_PAIRS:
        return pairs

    scores = _mismatch_scores(A, B, mapping)
    pool = _expanded_swap_pool(A, B, mapping, scores)
    focused = [pair for pair in pairs if pair[0] in pool or pair[1] in pool]
    if len(focused) <= _MAX_PAIR_REPAIR_BOND_PAIRS:
        return focused

    focused.sort(
        key=lambda pair: (-(scores[pair[0]] + scores[pair[1]]), pair[0], pair[1])
    )
    return focused[:_MAX_PAIR_REPAIR_BOND_PAIRS]


def _pair_swap_cd(A, B, mapping, current_cd, i, j, k, ell):
    changed = (i, j, k, ell)
    image = {
        i: mapping[k],
        k: mapping[i],
        j: mapping[ell],
        ell: mapping[j],
    }

    def mapped(atom):
        return image.get(atom, mapping[atom])

    old = 0.0
    new = 0.0
    n = len(mapping)
    changed_set = set(changed)
    for u in changed:
        mu = mapping[u]
        nmu = mapped(u)
        for v in range(n):
            mv = mapping[v]
            nmv = mapped(v)
            old += abs(A[u, v] - B[mu, mv])
            new += abs(A[u, v] - B[nmu, nmv])
            if v not in changed_set:
                old += abs(A[v, u] - B[mv, mu])
                new += abs(A[v, u] - B[nmv, nmu])
    return current_cd + 0.5 * (new - old)


def _oriented_bond_pairs(adjacency, elements):
    pairs = []
    n = len(elements)
    for i in range(n):
        for j in range(n):
            if i != j and adjacency[i, j] != 0:
                pairs.append((i, j, elements[i], elements[j]))
    return pairs


def _swap_cd(A, B, mapping, current_cd, i, k):
    image = {i: mapping[k], k: mapping[i]}
    changed = (i, k)
    changed_set = set(changed)
    old = 0.0
    new = 0.0
    n = len(mapping)

    def mapped(atom):
        return image.get(atom, mapping[atom])

    for u in changed:
        mu = mapping[u]
        nmu = mapped(u)
        for v in range(n):
            mv = mapping[v]
            nmv = mapped(v)
            old += abs(A[u, v] - B[mu, mv])
            new += abs(A[u, v] - B[nmu, nmv])
            if v not in changed_set:
                old += abs(A[v, u] - B[mv, mu])
                new += abs(A[v, u] - B[nmv, nmu])
    return current_cd + 0.5 * (new - old)


def _mismatch_scores(A, B, mapping):
    mapped_product = B[np.ix_(mapping, mapping)]
    return np.abs(A - mapped_product).sum(axis=1).tolist()


def _expanded_swap_pool(A, B, mapping, scores):
    inv = [-1] * len(mapping)
    for i, p in enumerate(mapping):
        if 0 <= p < len(inv):
            inv[p] = i

    pool = {i for i, score in enumerate(scores) if score > 0}
    for i in list(pool):
        pool.update(j for j, value in enumerate(A[i]) if value)
        p = mapping[i]
        for q, value in enumerate(B[p]):
            if value and inv[q] >= 0:
                pool.add(inv[q])
    return pool


def _local_swap_pool(lgp, mapping, binary, max_all_atoms, max_pool_per_element):
    A, _ = _adjacency_and_elements(lgp[0], binary)
    B, _ = _adjacency_and_elements(lgp[1], binary)
    elements = _elements(lgp[0])
    n = len(mapping)
    if n <= max_all_atoms:
        return set(range(n)), A, B

    scores = _mismatch_scores(A, B, mapping)
    pool = _expanded_swap_pool(A, B, mapping, scores)
    by_element = defaultdict(list)
    for i in pool:
        by_element[elements[i]].append(i)

    trimmed = set()
    for idxs in by_element.values():
        idxs.sort(key=lambda i: (-scores[i], i))
        trimmed.update(idxs[:max_pool_per_element])
    return trimmed, A, B


def _same_element_swaps(mapping, pool, elements, product_elements):
    pool = sorted(pool)
    for pos, i in enumerate(pool):
        element = elements[i]
        if product_elements[mapping[i]] != element:
            continue
        for k in pool[pos + 1 :]:
            if elements[k] == element and product_elements[mapping[k]] == element:
                yield i, k


def _greedy_local_swap_descent(lgp, mapping, pool, A, B, binary, max_steps):
    elements = _elements(lgp[0])
    product_elements = _elements(lgp[1])
    current = list(mapping)
    current_cd = chemical_distance(lgp, current, binary)
    out = [(tuple(current), current_cd)]

    for _ in range(max_steps):
        best = None
        for i, k in _same_element_swaps(current, pool, elements, product_elements):
            cd = _swap_cd(A, B, current, current_cd, i, k)
            if cd < current_cd - 1e-9 and (best is None or cd < best[0]):
                best = (cd, i, k)
        if best is None:
            break
        current_cd, i, k = best
        current[i], current[k] = current[k], current[i]
        out.append((tuple(current), current_cd))
    return out


def _bounded_local_swap_records(
    lgp,
    seeds,
    binary,
    depth,
    slack,
    cap,
    max_all_atoms,
    max_pool_per_element,
    max_descent_steps,
):
    start_best = min(cd for _, cd in seeds)
    elements = _elements(lgp[0])
    product_elements = _elements(lgp[1])

    seen = set()
    records = []
    queue = deque()
    for mapping, cd in seeds:
        mapping_tuple = tuple(mapping)
        if cd > start_best + slack + 1e-9 or mapping_tuple in seen:
            continue
        seen.add(mapping_tuple)
        pool, A, B = _local_swap_pool(
            lgp,
            mapping,
            binary,
            max_all_atoms,
            max_pool_per_element,
        )
        queue.append((list(mapping), cd, 0, pool, A, B))
        records.append((mapping_tuple, cd))

        for descended, descent_cd in _greedy_local_swap_descent(
            lgp,
            mapping,
            pool,
            A,
            B,
            binary,
            max_descent_steps,
        ):
            if descent_cd > start_best + slack + 1e-9 or descended in seen:
                continue
            seen.add(descended)
            records.append((descended, descent_cd))
            if len(records) >= cap:
                break

    while queue and len(records) < cap:
        mapping, cd, current_depth, pool, A, B = queue.popleft()
        if current_depth >= depth:
            continue

        scored = []
        for i, k in _same_element_swaps(mapping, pool, elements, product_elements):
            next_cd = _swap_cd(A, B, mapping, cd, i, k)
            if next_cd <= start_best + slack + 1e-9:
                scored.append((next_cd, i, k))
        scored.sort(key=lambda item: (item[0], item[1], item[2]))

        for next_cd, i, k in scored:
            next_mapping = list(mapping)
            next_mapping[i], next_mapping[k] = next_mapping[k], next_mapping[i]
            next_tuple = tuple(next_mapping)
            if next_tuple in seen:
                continue
            seen.add(next_tuple)
            records.append((next_tuple, next_cd))
            queue.append((next_mapping, next_cd, current_depth + 1, pool, A, B))
            if len(records) >= cap:
                break

    records.sort(key=lambda item: (item[1], item[0]))
    return records


def _elements(lg):
    elements = lg.props.get("atomic numbers")
    if elements is None:
        elements = list(lg.labels)
    return list(elements)


def complete_mapping(lgp, mapping, binary=False):  # noqa: C901
    """
    Complete a partial/duplicate mapping to an element-compatible bijection.

    SLAP's default chemistry front-end breaks heavy-atom symmetry only, so
    hydrogens can remain tied by identical labels. Kernel enumeration fixes the
    chemically relevant atoms; this helper assigns any unresolved duplicate
    atoms, typically hydrogens, to the remaining same-element product atoms.
    """
    mapping = list(mapping)
    n = len(mapping)
    elem_r = _elements(lgp[0])
    elem_p = _elements(lgp[1])
    out = [-1] * n
    used = set()

    for i, p in enumerate(mapping):
        if not isinstance(p, int) or p < 0 or p >= n or p in used:
            continue
        if elem_r[i] != elem_p[p]:
            continue
        out[i] = p
        used.add(p)

    A, _ = _adjacency_and_elements(lgp[0], binary=binary)
    B, _ = _adjacency_and_elements(lgp[1], binary=binary)

    remaining_by_element = {}
    for p in range(n):
        if p not in used:
            remaining_by_element.setdefault(elem_p[p], []).append(p)

    unresolved_by_element = {}
    for i, p in enumerate(out):
        if p == -1:
            unresolved_by_element.setdefault(elem_r[i], []).append(i)

    fixed = [i for i, p in enumerate(out) if p != -1]
    for element, idxs in unresolved_by_element.items():
        candidates = remaining_by_element.get(element, [])
        if len(candidates) != len(idxs):
            raise ValueError("Could not complete element-compatible mapping")

        cost = np.zeros((len(idxs), len(candidates)), dtype=float)
        for row, i in enumerate(idxs):
            for col, p in enumerate(candidates):
                val = 0.0
                for j in fixed:
                    pj = out[j]
                    val += abs(A[i, j] - B[p, pj])
                    val += abs(A[j, i] - B[pj, p])
                cost[row, col] = val

        rows, cols = linear_sum_assignment(cost)
        for row, col in zip(rows, cols):
            out[idxs[row]] = candidates[col]
            used.add(candidates[col])
        fixed.extend(idxs)

    return out


def _identity_permutation(n):
    return tuple(range(n))


def _automorphism_permutations(lg, binary=False, limit=10000):
    """Enumerate graph automorphisms as permutations ``old_index -> new_index``."""
    n = len(lg.labels)
    try:
        import networkx as nx
        from networkx.algorithms import isomorphism as iso

        g = to_nx(lg, binary=binary)
        node_match = iso.categorical_node_match("element", None)
        edge_match = iso.categorical_edge_match("order", None)
        matcher = nx.algorithms.isomorphism.GraphMatcher(
            g, g, node_match=node_match, edge_match=edge_match
        )
    except Exception:
        return [_identity_permutation(n)]

    perms = []
    for mapping in matcher.isomorphisms_iter():
        perms.append(tuple(mapping[i] for i in range(n)))
        if len(perms) >= limit:
            break
    return perms or [_identity_permutation(n)]


def canonical_mapping_key(lgp, mapping, binary=False, max_automorphisms=10000):
    """
    Canonical tuple for a full mapping modulo ``Aut(R) x Aut(P)``.

    If ``alpha`` is a reactant automorphism and ``beta`` is a product
    automorphism, the equivalent mapping sends ``alpha(i)`` to
    ``beta(mapping[i])``. The lexicographically smallest transformed mapping is
    used as the canonical representative.
    """
    mapping = list(mapping)
    r_aut = _automorphism_permutations(lgp[0], binary, max_automorphisms)
    p_aut = _automorphism_permutations(lgp[1], binary, max_automorphisms)

    best = None
    for alpha in r_aut:
        for beta in p_aut:
            transformed = [None] * len(mapping)
            for i, p in enumerate(mapping):
                transformed[alpha[i]] = beta[p]
            key = tuple(transformed)
            if best is None or key < best:
                best = key
    return best if best is not None else tuple(mapping)


def canonical_partial_mapping_key(lgp, pairs, binary=False, max_automorphisms=10000):
    """
    Canonical key for fixed mapping pairs modulo ``Aut(R) x Aut(P)``.

    ``pairs`` may be a dict or iterable of ``(reactant_index, product_index)``
    assignments. This is useful for partially resolved SLAP/GraphMatcher states
    where only the symmetry-breaking choices, rather than every atom, are unique.
    """
    if isinstance(pairs, dict):
        pairs = list(pairs.items())
    else:
        pairs = list(pairs)
    if not pairs:
        return ()

    r_aut = _automorphism_permutations(lgp[0], binary, max_automorphisms)
    p_aut = _automorphism_permutations(lgp[1], binary, max_automorphisms)

    best = None
    for alpha in r_aut:
        for beta in p_aut:
            key = tuple(sorted((alpha[i], beta[p]) for i, p in pairs))
            if best is None or key < best:
                best = key
    return best if best is not None else tuple(sorted(pairs))


def dedup_mappings_by_automorphism(lgp, mappings, binary=False):
    """Keep one representative per ``Aut(R) x Aut(P)`` mapping class."""
    seen = set()
    distinct = []
    for mapping in mappings:
        key = canonical_mapping_key(lgp, mapping, binary=binary)
        if key in seen:
            continue
        seen.add(key)
        distinct.append(list(mapping))
    return distinct


def _mapping_to_rxn_smiles(
    rxn_smiles,
    mapping,
    explicit_hs=False,
    explicit_h_atoms_pair=None,
):
    react_nums = list(range(1, len(mapping) + 1))
    prod_nums = [0] * len(mapping)
    for i, p in enumerate(mapping):
        prod_nums[p] = i + 1
    return get_numbered_rxn_smiles(
        rxn_smiles,
        [react_nums, prod_nums],
        explicit_hs=explicit_hs,
        explicit_h_atoms_pair=explicit_h_atoms_pair,
    )


def _dedup_result_dicts_by_mapping(lgp, results, binary):
    seen = set()
    distinct = []
    use_canonical = len(results) <= _MAX_AUTOMORPHISM_DEDUP_RESULTS
    for result in results:
        if use_canonical:
            key = canonical_mapping_key(lgp, result["mapping"], binary=binary)
        else:
            key = tuple(result["mapping"])
        if key in seen:
            continue
        seen.add(key)
        distinct.append(result)
    return distinct


def enumerate_kernel_optima(  # noqa: C901
    kernel,
    rxn_smiles: Optional[str] = None,
    unique: bool = True,
    electron_balance: bool = False,
    explicit_hs: bool = False,
    reaction_center_hs: bool = False,
):
    """
    Enumerate exact, symmetry-distinct optima for a kernel.

    Parameters
    ----------
    kernel : Kernel
        Uncertainty-region kernel.
    rxn_smiles : str, optional
        Original reaction SMILES. If supplied, each result receives mapped
        reaction SMILES and ITS-hash deduplication is used when available.
    unique : bool, optional
        Collapse automorphism/ITS-equivalent optima.
    electron_balance : bool, optional
        If True, prefer maps not rejected by the explicit-H ITS heuristic.
    explicit_hs : bool, optional
        If True, include explicit mapped hydrogens in output reaction SMILES.
    reaction_center_hs : bool, optional
        If True, expand hydrogens only on heavy atoms incident to changed ITS
        edges. This is intended for heavy-atom optimization with lightweight
        reaction-center hydrogen display.
    """
    solution = solve_kernel_blockwise(kernel, enumerate_all=True)
    enumeration_complete = True
    if not solution.sub_mappings:
        fallback = solve_kernel_blockwise(kernel, enumerate_all=False)
        if fallback.sub_mappings:
            solution = fallback
            enumeration_complete = False
        else:
            enumeration_complete = False

    if (
        rxn_smiles is not None
        and len(solution.sub_mappings) > _MAX_MAPPED_ENUMERATION_RESULTS
    ):
        fallback = solve_kernel_blockwise(kernel, enumerate_all=False)
        if fallback.sub_mappings:
            solution = fallback
            enumeration_complete = False
        else:
            return EnumerationResult(
                cost=solution.cost,
                mappings=[],
                sub_mappings=[],
                results=[],
                proven_optimal=solution.proven_optimal,
                enumeration_complete=False,
            )

    results = []
    rejected_results = []
    for sub_mapping in solution.sub_mappings:
        mapping = complete_mapping(
            kernel.lgp,
            apply_kernel_solution(kernel, sub_mapping),
            binary=kernel.binary,
        )
        result_lgp = mapping_to_lgp(kernel.lgp, mapping)
        result = {
            "lgp": result_lgp,
            "mapping": mapping,
            "sub_mapping": dict(sub_mapping),
            "cd": chemical_distance(kernel.lgp, mapping, kernel.binary),
        }
        if rxn_smiles is not None:
            result["its_smiles"] = _mapping_to_rxn_smiles(
                rxn_smiles,
                mapping,
                explicit_hs=False,
            )
            if reaction_center_hs:
                selected_maps = reaction_center_atom_maps(result["its_smiles"])
                react_nums = list(range(1, len(mapping) + 1))
                prod_nums = [0] * len(mapping)
                for i, p in enumerate(mapping):
                    prod_nums[p] = i + 1
                expanded_rxn = expand_reaction_center_hydrogens(
                    rxn_smiles,
                    [react_nums, prod_nums],
                    selected_maps,
                )
                try:
                    from ..chem.aam import AAMapper

                    center_mapper = AAMapper(binary=kernel.binary)
                    center_mapper.map_smiles(
                        expanded_rxn,
                        add_Hs=False,
                        break_sym="all",
                        unique=True,
                        certify=False,
                        electron_balance=False,
                        enumerate_exact=True,
                    )
                    center_smiles = [r["smiles"] for r in center_mapper.results]
                except Exception:
                    center_smiles = []
                if electron_balance:
                    ok = mapped_rxn_is_electron_balanced(result["its_smiles"])
                    result["electron_balanced"] = ok
                else:
                    ok = None
                target = (
                    rejected_results if electron_balance and ok is False else results
                )
                for mapped_smiles in center_smiles or [expanded_rxn]:
                    rr = dict(result)
                    rr["smiles"] = mapped_smiles
                    target.append(rr)
                continue
            else:
                result["smiles"] = _mapping_to_rxn_smiles(
                    rxn_smiles,
                    mapping,
                    explicit_hs=explicit_hs,
                )
            if electron_balance:
                ok = mapped_rxn_is_electron_balanced(result["its_smiles"])
                result["electron_balanced"] = ok
                if ok is False:
                    rejected_results.append(result)
                    continue
        results.append(result)

    if electron_balance and not results and rejected_results:
        results = rejected_results

    if (
        unique
        and rxn_smiles is not None
        and 1 < len(results) <= _MAX_SYNKIT_DEDUP_RESULTS
    ):
        results = dedup_mapped_rxns(results, smiles_key="its_smiles")
    elif unique and len(results) > 1:
        results = _dedup_result_dicts_by_mapping(kernel.lgp, results, kernel.binary)

    return EnumerationResult(
        cost=solution.cost,
        mappings=[r["mapping"] for r in results],
        sub_mappings=[r["sub_mapping"] for r in results],
        results=results,
        proven_optimal=solution.proven_optimal,
        enumeration_complete=enumeration_complete,
    )


def enumerate_symmetry_distinct_optima(*args, **kwargs):
    """Alias for :func:`enumerate_kernel_optima`."""
    return enumerate_kernel_optima(*args, **kwargs)
