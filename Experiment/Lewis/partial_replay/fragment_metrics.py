"""Exact molecular-fragment recovery with explicit treatment of extra outputs."""

from collections import Counter
from functools import lru_cache

from rdkit import Chem


@lru_cache(maxsize=65536)
def heavy_atoms(fragment):
    molecule = Chem.MolFromSmiles(fragment)
    if molecule is None:
        raise ValueError("Cannot rank an invalid molecular fragment")
    return molecule.GetNumHeavyAtoms()


def maximal_fragments(fragments):
    """All fragments tied for maximum heavy-atom count, independent of order."""
    if not fragments:
        return frozenset()
    maximum = max(heavy_atoms(fragment) for fragment in fragments)
    return frozenset(fragment for fragment in fragments if heavy_atoms(fragment) == maximum)


METRICS = ("exact_set", "all_gt_molecules", "any_gt_molecule", "largest_gt_recovered",
           "max_fragment_match")


def score_candidate(actual, expected):
    """Score one canonical component multiset, never a union of applications.

    ``largest_gt_recovered`` permits any output molecule to equal a largest GT
    molecule. ``max_fragment_match`` requires that molecule to be maximal in
    the output as well. A tie is successful if any maximal identity agrees.
    No wildcard is expanded and no substructure match counts as identity.
    """
    actual_counts, expected_counts = Counter(actual), Counter(expected)
    shared = actual_counts & expected_counts
    largest_gt, largest_actual = maximal_fragments(expected), maximal_fragments(actual)
    return {
        "exact_set": actual_counts == expected_counts,
        "all_gt_molecules": bool(expected) and not (expected_counts - actual_counts),
        "any_gt_molecule": bool(shared),
        "largest_gt_recovered": bool(largest_gt & set(actual)),
        "max_fragment_match": bool(largest_gt & largest_actual),
        "gt_molecule_recall": sum(shared.values()) / sum(expected_counts.values()) if expected else 0.0,
    }


def score_outputs(outcomes, expected):
    """Unranked recovery: each successful score has one candidate witness."""
    result = {name: False for name in METRICS}
    result["gt_molecule_recall"] = 0.0
    for actual in outcomes:
        values = score_candidate(actual, expected)
        for name in METRICS:
            result[name] |= values[name]
        result["gt_molecule_recall"] = max(result["gt_molecule_recall"], values["gt_molecule_recall"])
    return result
