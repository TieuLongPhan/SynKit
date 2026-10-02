"""Seeded tiny-graph checks independent of branch-and-bound pruning."""

from fractions import Fraction
from itertools import permutations
import random

import numpy as np

from synkit.Chem.Mapper.identifiability import Endpoint, extract_label
from synkit.Chem.Mapper.evaluation import ExactBondEvaluator
from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings
from synkit.Chem.Mapper.spectrum import exact_its_and_template_codes


def test_seeded_literal_minima_labels_and_compressed_search():
    randomizer = random.Random(192026)
    for _ in range(32):
        n = randomizer.randrange(2, 6)
        elements = tuple(randomizer.choice((6, 6, 8)) for _ in range(n))
        def endpoint():
            return Endpoint(elements, (0,) * n, (0,) * n,
                            tuple((i, j, randomizer.choice((2, 3, 4)))
                                  for i in range(n) for j in range(i + 1, n)
                                  if randomizer.random() < 0.4))
        r, p = endpoint(), endpoint()
        ar = {(i, j): w for i, j, w in r.bonds}
        ap = {(i, j): w for i, j, w in p.bonds}
        oracle = {}
        for m in permutations(range(n)):
            if any(elements[i] != elements[m[i]] for i in range(n)):
                continue
            cost = sum(abs(ar.get((i, j), 0) - ap.get(tuple(sorted((m[i], m[j]))), 0))
                       for i in range(n) for j in range(i + 1, n))
            oracle[m] = cost
        best = min(oracle.values())
        expected = {m for m, v in oracle.items() if v == best}
        expected_labels = {extract_label(r, p, m).changed_bonds for m in expected}
        for compressed in (False, True):
            result = enumerate_distance_mappings(
                [r.graph(), p.graph()], CD="minimal", binary=False,
                symmetry_pruning=compressed, symmetry_node_properties=("hcounts", "charges"),
            )
            assert result.complete and result.cost == Fraction(best, 2)
            observed = {tuple(m) for m in result.mappings}
            assert {extract_label(r, p, m).changed_bonds for m in observed} == expected_labels
            if not compressed:
                assert observed == expected
        # A deliberately worst feasible incumbent must not restrict the set.
        seeded = enumerate_distance_mappings(
            [r.graph(), p.graph()], CD="minimal", binary=False,
            initial_mapping=max(oracle, key=oracle.get), symmetry_pruning=False,
        )
        assert seeded.complete and seeded.cost == Fraction(best, 2)
        assert {tuple(m) for m in seeded.mappings} == expected


def test_distinct_its_can_have_identical_primary_bond_label():
    # Formal attributed-graph control, not a valence-valid chemistry example.
    r = p = Endpoint((6, 6), (0, 1), (4, 3), ())
    labels = [extract_label(r, p, m) for m in ((0, 1), (1, 0))]
    assert labels[0].changed_bonds == labels[1].changed_bonds == frozenset()
    assert labels[0].unary_changes != labels[1].unary_changes
    matrix = np.zeros((2, 2))
    properties = {"charges": (r.charges, p.charges), "hcounts": (r.hcounts, p.hcounts)}
    codes = [exact_its_and_template_codes(matrix, matrix, r.atomic_numbers, properties, m)
             for m in ((0, 1), (1, 0))]
    assert all(code[2] is None for code in codes)
    assert codes[0][0] != codes[1][0]
    scorer = ExactBondEvaluator(r)
    lo, hi = scorer.paired_envelope([], [], [x.changed_bonds for x in labels], labels_complete=True)
    assert lo.difference == hi.difference == 0
