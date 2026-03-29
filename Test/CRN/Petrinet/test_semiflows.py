from __future__ import annotations

import unittest

import numpy as np

from synkit.CRN.Structure import SynCRN
from synkit.CRN.Petrinet.semiflows import (
    _basis_column_support,
    _select_semiflow_basis,
    find_p_semiflows,
    find_t_semiflows,
    semiflow_supports,
    stoichiometric_matrix,
)
from synkit.CRN.Props.stoich import (
    left_nullspace,
    right_nullspace,
    stoichiometric_matrix as props_stoichiometric_matrix,
)


class TestSemiflows(unittest.TestCase):
    """Unit tests for semiflow utilities built on top of Props.stoich."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.simple = SynCRN.from_reaction_strings(["A>>B"])
        cls.cycle = SynCRN.from_reaction_strings(["A>>B", "B>>A"])
        cls.medium = SynCRN.from_reaction_strings(
            [
                "2A>>B+3C",
                "2B>>D",
                "C+D>>E",
            ]
        )

    def test_stoichiometric_matrix_matches_props_matrix(self) -> None:
        """
        Wrapper should return the same S matrix as Props.stoich, plus orders.
        """
        species_order, reaction_order, s = stoichiometric_matrix(self.simple)
        s_props = props_stoichiometric_matrix(self.simple)

        self.assertEqual(s.shape, s_props.shape)
        self.assertTrue(np.allclose(s, s_props))
        self.assertEqual(len(species_order), s.shape[0])
        self.assertEqual(len(reaction_order), s.shape[1])

    def test_stoichiometric_matrix_on_digraph(self) -> None:
        """
        Wrapper should also work on SynCRN bipartite digraph input.
        """
        g = self.simple.to_digraph()

        species_order, reaction_order, s = stoichiometric_matrix(g)
        s_props = props_stoichiometric_matrix(g)

        self.assertTrue(np.allclose(s, s_props))
        self.assertEqual(len(species_order), s.shape[0])
        self.assertEqual(len(reaction_order), s.shape[1])

    def test_find_p_semiflows_matches_props_left_nullspace(self) -> None:
        """
        P-semiflows should be exactly the left nullspace basis from Props.stoich.
        """
        basis = find_p_semiflows(self.simple)
        basis_props = left_nullspace(self.simple)

        self.assertEqual(basis.shape, basis_props.shape)
        self.assertTrue(np.allclose(basis, basis_props))

    def test_find_t_semiflows_matches_props_right_nullspace(self) -> None:
        """
        T-semiflows should be exactly the right nullspace basis from Props.stoich.
        """
        basis = find_t_semiflows(self.cycle)
        basis_props = right_nullspace(self.cycle)

        self.assertEqual(basis.shape, basis_props.shape)
        self.assertTrue(np.allclose(basis, basis_props))

    def test_find_p_semiflows_satisfies_kernel_equation(self) -> None:
        """
        P-semiflow basis should satisfy S^T x = 0.
        """
        _, _, s = stoichiometric_matrix(self.simple)
        basis = find_p_semiflows(self.simple)

        self.assertEqual(basis.shape[0], s.shape[0])
        self.assertTrue(np.allclose(s.T @ basis, 0.0, atol=1e-10))

    def test_find_t_semiflows_satisfies_kernel_equation(self) -> None:
        """
        T-semiflow basis should satisfy S x = 0.
        """
        _, _, s = stoichiometric_matrix(self.cycle)
        basis = find_t_semiflows(self.cycle)

        self.assertEqual(basis.shape[0], s.shape[1])
        self.assertTrue(np.allclose(s @ basis, 0.0, atol=1e-10))

    def test_find_t_semiflows_empty_for_irreversible_step(self) -> None:
        """
        A single irreversible step should have trivial right kernel.
        """
        _, reaction_order, _ = stoichiometric_matrix(self.simple)
        basis = find_t_semiflows(self.simple)

        self.assertEqual(basis.shape[0], len(reaction_order))
        self.assertEqual(basis.shape[1], 0)

    def test_select_semiflow_basis_p(self) -> None:
        """
        _select_semiflow_basis(kind='p') should return species order and P basis.
        """
        species_order, _, _ = stoichiometric_matrix(self.simple)
        order, basis = _select_semiflow_basis(self.simple, kind="p", rtol=1e-12)

        self.assertEqual(order, species_order)
        self.assertEqual(basis.shape[0], len(order))

    def test_select_semiflow_basis_t(self) -> None:
        """
        _select_semiflow_basis(kind='t') should return reaction order and T basis.
        """
        _, reaction_order, _ = stoichiometric_matrix(self.cycle)
        order, basis = _select_semiflow_basis(self.cycle, kind="t", rtol=1e-12)

        self.assertEqual(order, reaction_order)
        self.assertEqual(basis.shape[0], len(order))

    def test_select_semiflow_basis_invalid_kind(self) -> None:
        with self.assertRaises(ValueError):
            _select_semiflow_basis(self.simple, kind="x", rtol=1e-12)

    def test_basis_column_support(self) -> None:
        """
        Support extraction should drop entries below tolerance.
        """
        vec = np.array([1.0, 1e-10, -2.0, 0.0])
        order = ["A", "B", "C", "D"]

        supp = _basis_column_support(vec, order, support_tol=1e-8)

        self.assertEqual(supp, {"A": 1.0, "C": -2.0})

    def test_semiflow_supports_p(self) -> None:
        """
        P-supports should return sparse dicts keyed by species ids.
        """
        supports = semiflow_supports(self.simple, kind="p")

        self.assertGreaterEqual(len(supports), 1)
        for supp in supports:
            self.assertIsInstance(supp, dict)
            self.assertTrue(supp)

    def test_semiflow_supports_t_cycle(self) -> None:
        """
        T-supports should be non-empty for a reversible 2-step cycle.
        """
        supports = semiflow_supports(self.cycle, kind="t")

        self.assertGreaterEqual(len(supports), 1)
        for supp in supports:
            self.assertIsInstance(supp, dict)
            self.assertTrue(supp)

    def test_semiflow_supports_t_empty_for_no_t_invariant(self) -> None:
        """
        T-supports should be empty when the right kernel is trivial.
        """
        supports = semiflow_supports(self.simple, kind="t")
        self.assertEqual(supports, [])

    def test_semiflow_supports_invalid_kind(self) -> None:
        with self.assertRaises(ValueError):
            semiflow_supports(self.simple, kind="x")

    def test_medium_input_smoke(self) -> None:
        """
        Medium network should run without errors and return consistent shapes.
        """
        species_order, reaction_order, s = stoichiometric_matrix(self.medium)
        p_basis = find_p_semiflows(self.medium)
        t_basis = find_t_semiflows(self.medium)

        self.assertEqual(s.shape, (len(species_order), len(reaction_order)))
        self.assertEqual(p_basis.shape[0], len(species_order))
        self.assertEqual(t_basis.shape[0], len(reaction_order))


if __name__ == "__main__":
    unittest.main()
