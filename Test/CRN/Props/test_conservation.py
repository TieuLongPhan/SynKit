import unittest

import numpy as np

from synkit.CRN.Props.stoich import (
    _exact_null_space,
    _primitive_integer_vector,
    conserved_moieties,
    integer_conservation_laws,
    stoichiometric_matrix,
)
from synkit.CRN.Props.helper import _species_and_rule_order
from synkit.CRN.Structure.syncrn import SynCRN
from fractions import Fraction

# Distributive two-site phosphorylation. Its left kernel is where the old
# float-kernel path produced vectors that were not conservation laws at all.
TWO_SITE = [
    "S0+E>>S0E",
    "S0E>>S0+E",
    "S0E>>S1+E",
    "S1+E>>S1E",
    "S1E>>S1+E",
    "S1E>>S2+E",
    "S2+F>>S2F",
    "S2F>>S2+F",
    "S2F>>S1+F",
    "S1+F>>S1F2",
    "S1F2>>S1+F",
    "S1F2>>S0+F",
]


def labels_of(crn):
    order, _, _, _ = _species_and_rule_order(crn)
    return [crn.species[node].label for node in order]


class TestPrimitiveIntegerVector(unittest.TestCase):
    def test_clears_denominators(self):
        self.assertEqual(
            _primitive_integer_vector([Fraction(1, 2), Fraction(1, 3)]), [3, 2]
        )

    def test_reduces_by_gcd(self):
        self.assertEqual(
            _primitive_integer_vector([Fraction(4), Fraction(6)]), [2, 3]
        )

    def test_normalizes_leading_sign(self):
        self.assertEqual(
            _primitive_integer_vector([Fraction(-1), Fraction(2)]), [1, -2]
        )

    def test_zero_vector(self):
        self.assertEqual(
            _primitive_integer_vector([Fraction(0), Fraction(0)]), [0, 0]
        )


class TestExactNullSpace(unittest.TestCase):
    def test_full_rank_has_empty_kernel(self):
        self.assertEqual(_exact_null_space([[1, 0], [0, 1]], 2), [])

    def test_rank_deficient(self):
        basis = _exact_null_space([[1, 1]], 2)
        self.assertEqual(len(basis), 1)
        self.assertEqual(sum(a * b for a, b in zip([1, 1], basis[0])), 0)

    def test_zero_columns(self):
        self.assertEqual(_exact_null_space([[1]], 0), [])

    def test_fractional_input_is_exact(self):
        basis = _exact_null_space([[0.5, 0.25]], 2)
        self.assertEqual(len(basis), 1)
        self.assertEqual(0.5 * basis[0][0] + 0.25 * basis[0][1], 0)


class TestIntegerConservationLaws(unittest.TestCase):
    """Every reported law must satisfy ``y^T S = 0`` exactly."""

    NETWORKS = {
        "reversible pair": ["A>>B", "B>>A"],
        "michaelis menten": ["E+S>>ES", "ES>>E+S", "ES>>E+P"],
        "two-site futile cycle": TWO_SITE,
        "stoichiometric": ["A>>2B", "2B>>A"],
        "disjoint blocks": ["A>>B", "B>>A", "C>>D", "D>>C"],
    }

    def test_laws_are_in_the_left_kernel(self):
        for name, rxns in self.NETWORKS.items():
            with self.subTest(network=name):
                crn = SynCRN.from_reaction_strings(rxns)
                matrix = stoichiometric_matrix(crn)
                for law in integer_conservation_laws(crn):
                    np.testing.assert_allclose(
                        np.array(law, dtype=float) @ matrix, 0.0, atol=1e-12
                    )

    def test_laws_are_integers(self):
        for name, rxns in self.NETWORKS.items():
            with self.subTest(network=name):
                for law in integer_conservation_laws(SynCRN.from_reaction_strings(rxns)):
                    self.assertTrue(all(isinstance(x, int) for x in law))

    def test_basis_size_is_the_nullity(self):
        for name, rxns in self.NETWORKS.items():
            with self.subTest(network=name):
                crn = SynCRN.from_reaction_strings(rxns)
                matrix = stoichiometric_matrix(crn)
                nullity = matrix.shape[0] - int(np.linalg.matrix_rank(matrix))
                self.assertEqual(len(integer_conservation_laws(crn)), nullity)

    def test_simple_pair(self):
        self.assertEqual(
            integer_conservation_laws(SynCRN.from_reaction_strings(["A>>B", "B>>A"])),
            [[1, 1]],
        )

    def test_non_unit_stoichiometry(self):
        # A <-> 2B conserves 2A + B, not A + B.
        crn = SynCRN.from_reaction_strings(["A>>2B", "2B>>A"])
        self.assertEqual(integer_conservation_laws(crn), [[2, 1]])

    def test_empty_network(self):
        self.assertEqual(
            integer_conservation_laws(SynCRN.from_reaction_strings([])), []
        )


class TestConservedMoieties(unittest.TestCase):
    def test_two_site_futile_cycle_pools(self):
        crn = SynCRN.from_reaction_strings(TWO_SITE)
        labels = labels_of(crn)
        pools = {
            frozenset(label for label, coeff in zip(labels, moiety) if coeff)
            for moiety in conserved_moieties(crn)
        }
        self.assertEqual(
            pools,
            {
                frozenset({"E", "S0E", "S1E"}),
                frozenset({"F", "S2F", "S1F2"}),
                frozenset({"S0", "S1", "S2", "S0E", "S1E", "S2F", "S1F2"}),
            },
        )

    def test_moieties_are_non_negative(self):
        crn = SynCRN.from_reaction_strings(TWO_SITE)
        for moiety in conserved_moieties(crn):
            self.assertTrue(all(coeff >= 0 for coeff in moiety))

    def test_moieties_are_conservation_laws(self):
        crn = SynCRN.from_reaction_strings(TWO_SITE)
        matrix = stoichiometric_matrix(crn)
        for moiety in conserved_moieties(crn):
            np.testing.assert_allclose(
                np.array(moiety, dtype=float) @ matrix, 0.0, atol=1e-12
            )

    def test_michaelis_menten_pools(self):
        crn = SynCRN.from_reaction_strings(["E+S>>ES", "ES>>E+S", "ES>>E+P"])
        labels = labels_of(crn)
        pools = {
            frozenset(label for label, coeff in zip(labels, moiety) if coeff)
            for moiety in conserved_moieties(crn)
        }
        self.assertIn(frozenset({"E", "ES"}), pools)
        self.assertIn(frozenset({"S", "ES", "P"}), pools)

    def test_empty_network(self):
        self.assertEqual(conserved_moieties(SynCRN.from_reaction_strings([])), [])


if __name__ == "__main__":
    unittest.main()
