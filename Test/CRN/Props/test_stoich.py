import unittest
from typing import List

import numpy as np
import networkx as nx

from synkit.CRN.Props import stoich


# ---------------------------------------------------------------------------
# Helpers to build small test networks
# ---------------------------------------------------------------------------


def build_chain_example_graph() -> nx.DiGraph:
    """
    Build bipartite graph for

        2A + B >> C
        C + D >> E
        E + F >> D + G

    using species nodes A..G and reaction nodes r1, r2, r3.
    """
    G = nx.DiGraph()

    # Species nodes
    species = ["A", "B", "C", "D", "E", "F", "G"]
    for s in species:
        # label == node id to match typical _species_and_reaction_order expectations
        G.add_node(s, kind="species", label=s)

    # Reaction nodes
    reactions = ["r1", "r2", "r3"]
    for r in reactions:
        G.add_node(r, kind="reaction")

    # 2A + B >> C
    G.add_edge("A", "r1", role="reactant", stoich=2.0)
    G.add_edge("B", "r1", role="reactant", stoich=1.0)
    G.add_edge("r1", "C", role="product", stoich=1.0)

    # C + D >> E
    G.add_edge("C", "r2", role="reactant", stoich=1.0)
    G.add_edge("D", "r2", role="reactant", stoich=1.0)
    G.add_edge("r2", "E", role="product", stoich=1.0)

    # E + F >> D + G
    G.add_edge("E", "r3", role="reactant", stoich=1.0)
    G.add_edge("F", "r3", role="reactant", stoich=1.0)
    G.add_edge("r3", "D", role="product", stoich=1.0)
    G.add_edge("r3", "G", role="product", stoich=1.0)

    return G


def build_simple_A_to_B_graph() -> nx.DiGraph:
    """
    Build bipartite graph for a single reaction:

        A >> B
    """
    G = nx.DiGraph()
    for s in ("A", "B"):
        G.add_node(s, kind="species", label=s)
    G.add_node("r1", kind="reaction")
    G.add_edge("A", "r1", role="reactant", stoich=1.0)
    G.add_edge("r1", "B", role="product", stoich=1.0)
    return G


def build_decay_graph() -> nx.DiGraph:
    """
    Build bipartite graph for a decay reaction:

        A >> ∅   (no explicit product species)
    """
    G = nx.DiGraph()
    G.add_node("A", kind="species", label="A")
    G.add_node("r1", kind="reaction")
    G.add_edge("A", "r1", role="reactant", stoich=1.0)
    return G


def build_cycle_graph() -> nx.DiGraph:
    """
    Build bipartite graph for cycle:

        A >> B
        B >> A

    which should admit a strictly positive steady-state flux.
    """
    G = nx.DiGraph()
    for s in ("A", "B"):
        G.add_node(s, kind="species", label=s)
    G.add_node("r1", kind="reaction")
    G.add_node("r2", kind="reaction")
    # A -> B
    G.add_edge("A", "r1", role="reactant", stoich=1.0)
    G.add_edge("r1", "B", role="product", stoich=1.0)
    # B -> A
    G.add_edge("B", "r2", role="reactant", stoich=1.0)
    G.add_edge("r2", "A", role="product", stoich=1.0)
    return G


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStoichiometricMatrices(unittest.TestCase):
    def setUp(self) -> None:
        # Example corresponding to rxns = ["2A+B>>C", "C+D>>E", "E+F>>D+G"]
        self.G = build_chain_example_graph()

    def test_build_S_minus_plus_entries(self) -> None:
        species_order, reaction_order, S_minus, S_plus = stoich.build_S_minus_plus(
            self.G
        )

        # Build index lookup independent of internal ordering
        sp_idx = {s: i for i, s in enumerate(species_order)}
        rx_idx = {r: j for j, r in enumerate(reaction_order)}

        # Shapes: 7 species × 3 reactions
        self.assertEqual(S_minus.shape, (7, 3))
        self.assertEqual(S_plus.shape, (7, 3))

        # 2A + B >> C
        self.assertAlmostEqual(S_minus[sp_idx["A"], rx_idx["r1"]], 2.0)
        self.assertAlmostEqual(S_minus[sp_idx["B"], rx_idx["r1"]], 1.0)
        self.assertAlmostEqual(S_plus[sp_idx["C"], rx_idx["r1"]], 1.0)

        # C + D >> E
        self.assertAlmostEqual(S_minus[sp_idx["C"], rx_idx["r2"]], 1.0)
        self.assertAlmostEqual(S_minus[sp_idx["D"], rx_idx["r2"]], 1.0)
        self.assertAlmostEqual(S_plus[sp_idx["E"], rx_idx["r2"]], 1.0)

        # E + F >> D + G
        self.assertAlmostEqual(S_minus[sp_idx["E"], rx_idx["r3"]], 1.0)
        self.assertAlmostEqual(S_minus[sp_idx["F"], rx_idx["r3"]], 1.0)
        self.assertAlmostEqual(S_plus[sp_idx["D"], rx_idx["r3"]], 1.0)
        self.assertAlmostEqual(S_plus[sp_idx["G"], rx_idx["r3"]], 1.0)

    def test_build_S_and_stoichiometric_matrix(self) -> None:
        species_order, reaction_order, S = stoich.build_S(self.G)
        S2 = stoich.stoichiometric_matrix(self.G)

        # build_S and stoichiometric_matrix must agree
        self.assertTrue(np.allclose(S, S2))

        # Check a few key entries using known stoichiometry
        sp_idx = {s: i for i, s in enumerate(species_order)}
        rx_idx = {r: j for j, r in enumerate(reaction_order)}

        # For 2A + B >> C: ΔA = -2, ΔB = -1, ΔC = +1
        self.assertAlmostEqual(S[sp_idx["A"], rx_idx["r1"]], -2.0)
        self.assertAlmostEqual(S[sp_idx["B"], rx_idx["r1"]], -1.0)
        self.assertAlmostEqual(S[sp_idx["C"], rx_idx["r1"]], 1.0)

    def test_stoichiometric_rank(self) -> None:
        S = stoich.stoichiometric_matrix(self.G)
        expected_rank = int(np.linalg.matrix_rank(S))
        self.assertEqual(stoich.stoichiometric_rank(self.G), expected_rank)
        # For this example network we know rank should be 3
        self.assertEqual(expected_rank, 3)


class TestNullspacesAndSemiflows(unittest.TestCase):
    def setUp(self) -> None:
        self.G = build_chain_example_graph()
        self.S = stoich.stoichiometric_matrix(self.G)

    def test_left_nullspace_properties(self) -> None:
        L = stoich.left_nullspace(self.G)
        n_species, n_reactions = self.S.shape

        # Shape: n_species × k
        self.assertEqual(L.shape[0], n_species)

        # Each column m must satisfy m^T S = 0  → S^T m = 0
        self.assertTrue(np.allclose(self.S.T @ L, 0.0, atol=1e-8))

        # Dimension should be n_species - rank
        rank = int(np.linalg.matrix_rank(self.S))
        self.assertEqual(L.shape[1], n_species - rank)

    def test_right_nullspace_properties(self) -> None:
        R = stoich.right_nullspace(self.G)
        n_species, n_reactions = self.S.shape

        # Shape: n_reactions × k
        self.assertEqual(R.shape[0], n_reactions)

        # Each column v must satisfy S v = 0
        if R.size > 0:
            self.assertTrue(np.allclose(self.S @ R, 0.0, atol=1e-8))

        # Dimension should be n_reactions - rank
        rank = int(np.linalg.matrix_rank(self.S))
        self.assertEqual(R.shape[1], n_reactions - rank)

    def test_left_right_kernels(self) -> None:
        L, R = stoich.left_right_kernels(self.G)

        self.assertTrue(np.allclose(self.S.T @ L, 0.0, atol=1e-8))
        if R.size > 0:
            self.assertTrue(np.allclose(self.S @ R, 0.0, atol=1e-8))


class TestIntegerHelpers(unittest.TestCase):
    def test_lcm_basic_cases(self) -> None:
        self.assertEqual(stoich._lcm(4, 6), 12)
        self.assertEqual(stoich._lcm(0, 5), 5)
        self.assertEqual(stoich._lcm(7, 0), 7)
        self.assertEqual(stoich._lcm(0, 0), 0)

    def test_vector_to_minimal_integer(self) -> None:
        v = np.array([0.5, 1.0, 1.5], dtype=float)
        ints = stoich._vector_to_minimal_integer(v)
        self.assertEqual(ints, [1, 2, 3])

        # All zeros
        v_zero = np.zeros(4, dtype=float)
        ints_zero = stoich._vector_to_minimal_integer(v_zero)
        self.assertEqual(ints_zero, [0, 0, 0, 0])

    def test_integer_conservation_laws(self) -> None:
        G = build_chain_example_graph()
        S = stoich.stoichiometric_matrix(G)
        n_species, _ = S.shape
        rank = int(np.linalg.matrix_rank(S))
        expected_dim = n_species - rank

        laws: List[List[int]] = stoich.integer_conservation_laws(G)  # type: ignore[assignment]

        # Dimension of integer laws should match dim ker(S^T)
        self.assertEqual(len(laws), expected_dim)


class TestConservativityAndConsistency(unittest.TestCase):
    def test_is_conservative_simple_A_to_B(self) -> None:
        # A >> B has a conservation law m = (1, 1) > 0 → conservative
        G = build_simple_A_to_B_graph()
        res = stoich.is_conservative(G)
        self.assertIs(res, True)

    def test_is_conservative_decay_false(self) -> None:
        # A >> ∅ has S = [-1]; no nontrivial conservation law
        G = build_decay_graph()
        res = stoich.is_conservative(G)
        self.assertIs(res, False)

    def test_is_consistent_cycle_true(self) -> None:
        # A >> B, B >> A admits v = (1, 1) > 0 such that S v = 0
        G = build_cycle_graph()
        res = stoich.is_consistent(G)
        self.assertIs(res, True)

    def test_is_consistent_decay_false(self) -> None:
        # A >> ∅ has no v > 0 with S v = 0
        G = build_decay_graph()
        res = stoich.is_consistent(G)
        self.assertIs(res, False)


class TestSummary(unittest.TestCase):
    def test_summary_chain_example(self) -> None:
        G = build_chain_example_graph()
        S = stoich.stoichiometric_matrix(G)
        n_species, n_reactions = S.shape

        s = stoich.summary(G)

        self.assertEqual(s.n_species, n_species)
        self.assertEqual(s.n_reactions, n_reactions)
        self.assertEqual(s.rank, int(np.linalg.matrix_rank(S)))
        self.assertEqual(s.dim_left_kernel, 4)
        self.assertEqual(s.dim_right_kernel, 0)
        self.assertFalse(s.is_conservative)
        self.assertFalse(s.is_consistent)


if __name__ == "__main__":
    unittest.main()
