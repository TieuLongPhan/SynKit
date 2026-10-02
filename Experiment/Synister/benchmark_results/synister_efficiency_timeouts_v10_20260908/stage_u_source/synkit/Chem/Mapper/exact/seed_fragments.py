"""Bounded common-fragment anchors used only for feasible seed construction."""

import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS

from .seed import improve_seed_mapping
from .seed_relaxation import improve_relaxed_seed_mapping

_MAX_FRAGMENT_SEARCH_SECONDS = 0.025


class _FragmentProgress(rdFMCS.MCSProgress):
    def __init__(self, deadline):
        super().__init__()
        self.deadline = deadline

    def __call__(self, progress, parameters):
        return time.perf_counter() < self.deadline


def _molecule(matrix, elements):
    molecule = Chem.RWMol()
    for element in elements:
        molecule.AddAtom(Chem.Atom(int(element)))
    orders = {
        1: Chem.BondType.SINGLE,
        1.5: Chem.BondType.AROMATIC,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }
    for i, j in zip(*np.triu_indices(len(elements), 1)):
        if matrix[i, j]:
            molecule.AddBond(int(i), int(j), orders[matrix[i, j]])
    result = molecule.GetMol()
    result.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(result)
    return result


def improve_fragment_seed(
    a, b, er, ep, mapping, *, max_fragments=16, budget_seconds=0.5
):
    """Anchor a greedy common-fragment cover, then polish a feasible bijection.

    Anchors constrain only this optional heuristic. They are never passed to
    exact search. The fragment finder has a shared cooperative budget, and
    unsupported matrix semantics simply retain the existing seed.
    """
    if len(mapping) > 256 or not np.array_equal(a, a.T) or not np.array_equal(b, b.T):
        return list(mapping), {"skipped": "size_or_directed_matrix"}
    values = set(np.unique(a)) | set(np.unique(b))
    if not values <= {0, 1, 1.5, 2, 3}:
        return list(mapping), {"skipped": "unsupported_bond_orders"}
    if sorted(mapping) != list(range(len(er))) or any(
        er[i] != ep[p] for i, p in enumerate(mapping)
    ):
        raise ValueError("seed must be a complete element-compatible bijection")
    fixed = {}
    working = list(mapping)
    orientation_evaluations = 0
    sizes = []
    canceled = False
    deadline = time.perf_counter() + budget_seconds
    for _ in range(max_fragments):
        if time.perf_counter() >= deadline:
            break
        rows = [i for i in range(len(er)) if i not in fixed]
        used = set(fixed.values())
        columns = [i for i in range(len(ep)) if i not in used]
        if len(rows) < 3:
            break
        reactant = _molecule(a[np.ix_(rows, rows)], [er[i] for i in rows])
        product = _molecule(b[np.ix_(columns, columns)], [ep[i] for i in columns])
        parameters = rdFMCS.MCSParameters()
        parameters.Timeout = 1
        parameters.BondCompareParameters.RingMatchesRingOnly = True
        parameters.AtomCompareParameters.RingMatchesRingOnly = True
        parameters.ProgressCallback = _FragmentProgress(
            min(deadline, time.perf_counter() + _MAX_FRAGMENT_SEARCH_SECONDS)
        )
        result = rdFMCS.FindMCS([reactant, product], parameters)
        canceled |= result.canceled
        if result.numAtoms < 3:
            break
        query = Chem.MolFromSmarts(result.smartsString)
        left_matches = reactant.GetSubstructMatches(query, uniquify=False, maxMatches=8)
        right_matches = product.GetSubstructMatches(
            query, uniquify=False, maxMatches=32
        )
        best = None
        for left in left_matches:
            for right in right_matches:
                if len(left) != len(right) or len(left) < 3:
                    continue
                anchors = {rows[i]: columns[j] for i, j in zip(left, right)}
                if any(er[i] != ep[j] for i, j in anchors.items()):
                    continue
                trial = working.copy()
                for atom, image in anchors.items():
                    owner = trial.index(image)
                    trial[atom], trial[owner] = trial[owner], trial[atom]
                trial_cost = 0.5 * float(np.abs(a - b[np.ix_(trial, trial)]).sum())
                orientation_evaluations += 1
                if best is None or trial_cost < best[0]:
                    best = trial_cost, trial, anchors
                if time.perf_counter() >= deadline:
                    break
            if time.perf_counter() >= deadline:
                break
        if best is None:
            break
        _, working, anchors = best
        fixed.update(anchors)
        sizes.append(len(anchors))
    candidate = working
    if fixed:
        candidate, _ = improve_relaxed_seed_mapping(
            a, b, er, ep, candidate, fixed_mapping=fixed
        )
        candidate, _ = improve_seed_mapping(a, b, er, ep, candidate)
    old_cost = 0.5 * float(np.abs(a - b[np.ix_(mapping, mapping)]).sum())
    new_cost = 0.5 * float(np.abs(a - b[np.ix_(candidate, candidate)]).sum())
    if new_cost >= old_cost:
        candidate, new_cost = list(mapping), old_cost
    return candidate, {
        "fragment_sizes": sizes,
        "orientation_evaluations": orientation_evaluations,
        "anchored_atoms": len(fixed),
        "canceled": canceled,
        "cost": new_cost,
        "budget_seconds": budget_seconds,
        "fragment_search_seconds": _MAX_FRAGMENT_SEARCH_SECONDS,
    }
