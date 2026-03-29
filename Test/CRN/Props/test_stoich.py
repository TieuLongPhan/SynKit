import unittest
from typing import List

import networkx as nx
import numpy as np

from synkit.CRN.Props import stoich

# ---------------------------------------------------------------------------
# Helpers to build small test CRNs
# ---------------------------------------------------------------------------


def build_chain_example_graph() -> nx.DiGraph:
    """
    Build bipartite graph for:

        2A + B >> C
        C + D >> E
        E + F >> D + G

    using species nodes A..G and rule nodes r1, r2, r3.
    """
    G = nx.DiGraph()

    species = ["A", "B", "C", "D", "E", "F", "G"]
    for s in species:
        G.add_node(s, kind="species", label=s)

    rules = ["r1", "r2", "r3"]
    for r in rules:
        G.add_node(r, kind="rule", label=r)

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


def build_conserved_cycle_graph() -> nx.DiGraph:
    """
    Build a tiny CRN with an exact conservation law:

        A >> B
        B >> A

    Stoichiometric matrix:
        [-1,  1]
        [ 1, -1]

    So A + B is conserved and the left kernel has dimension 1.
    """
    G = nx.DiGraph()

    for s in ["A", "B"]:
        G.add_node(s, kind="species", label=s)

    for r in ["r1", "r2"]:
        G.add_node(r, kind="rule", label=r)

    # A >> B
    G.add_edge("A", "r1", role="reactant", stoich=1.0)
    G.add_edge("r1", "B", role="product", stoich=1.0)

    # B >> A
    G.add_edge("B", "r2", role="reactant", stoich=1.0)
    G.add_edge("r2", "A", role="product", stoich=1.0)

    return G


def build_flux_cycle_graph() -> nx.DiGraph:
    """
    Build a CRN with a non-trivial right kernel:

        A >> B
        B >> A

    The flux vector [1, 1]^T lies in ker(S).
    """
    return build_conserved_cycle_graph()


def build_undirected_example_graph() -> nx.Graph:
    """
    Same chemistry as:

        A >> B

    but encoded in an undirected graph. The edge role must still drive
    S^- / S^+ assignment.
    """
    G = nx.Graph()
    G.add_node("A", kind="species")
    G.add_node("B", kind="species")
    G.add_node("r1", kind="rule")

    G.add_edge("A", "r1", role="reactant", stoich=1.0)
    G.add_edge("r1", "B", role="product", stoich=1.0)
    return G


def build_multigraph_example() -> nx.MultiDiGraph:
    """
    Build a multigraph where parallel edges should be accumulated:

        A --(reactant,1)--> r1
        A --(reactant,2)--> r1
        r1 --(product,1)--> B

    This should give:
        S^- [A, r1] = 3
        S^+ [B, r1] = 1
    """
    G = nx.MultiDiGraph()
    G.add_node("A", kind="species")
    G.add_node("B", kind="species")
    G.add_node("r1", kind="rule")

    G.add_edge("A", "r1", role="reactant", stoich=1.0)
    G.add_edge("A", "r1", role="reactant", stoich=2.0)
    G.add_edge("r1", "B", role="product", stoich=1.0)

    return G


def build_graph_with_noise_edges() -> nx.DiGraph:
    """
    Build a graph that contains valid species-rule incidences plus edges that
    should be ignored:
      - species-species edge
      - rule-rule edge
      - missing role
      - invalid role
    """
    G = nx.DiGraph()

    for s in ["A", "B"]:
        G.add_node(s, kind="species")
    for r in ["r1", "r2"]:
        G.add_node(r, kind="rule")

    # valid reaction: A >> B
    G.add_edge("A", "r1", role="reactant", stoich=1.0)
    G.add_edge("r1", "B", role="product", stoich=1.0)

    # noise that should be ignored
    G.add_edge("A", "B", role="reactant", stoich=99.0)  # species-species
    G.add_edge("r1", "r2", role="product", stoich=99.0)  # rule-rule
    G.add_edge("A", "r2", stoich=5.0)  # missing role
    G.add_edge("r2", "B", role="catalyst", stoich=7.0)  # invalid role

    return G


# ---------------------------------------------------------------------------
# Tests for low-level helpers
# ---------------------------------------------------------------------------


class TestLowLevelHelpers(unittest.TestCase):
    def test_resolve_species_rule_incidence_species_to_rule(self) -> None:
        G = nx.DiGraph()
        G.add_node("A", kind="species")
        G.add_node("r1", kind="rule")

        self.assertEqual(
            stoich._resolve_species_rule_incidence(G, "A", "r1"),
            ("A", "r1"),
        )

    def test_resolve_species_rule_incidence_rule_to_species(self) -> None:
        G = nx.DiGraph()
        G.add_node("A", kind="species")
        G.add_node("r1", kind="rule")

        self.assertEqual(
            stoich._resolve_species_rule_incidence(G, "r1", "A"),
            ("A", "r1"),
        )

    def test_resolve_species_rule_incidence_invalid_pair(self) -> None:
        G = nx.DiGraph()
        G.add_node("A", kind="species")
        G.add_node("B", kind="species")

        self.assertIsNone(stoich._resolve_species_rule_incidence(G, "A", "B"))

    def test_accumulate_stoich_entry(self) -> None:
        S_minus = np.zeros((2, 2), dtype=float)
        S_plus = np.zeros((2, 2), dtype=float)

        stoich._accumulate_stoich_entry(S_minus, S_plus, 0, 1, "reactant", 2.0)
        stoich._accumulate_stoich_entry(S_minus, S_plus, 1, 0, "product", 3.0)
        stoich._accumulate_stoich_entry(S_minus, S_plus, 1, 1, None, 5.0)

        self.assertAlmostEqual(S_minus[0, 1], 2.0)
        self.assertAlmostEqual(S_plus[1, 0], 3.0)
        self.assertAlmostEqual(S_minus[1, 1], 0.0)
        self.assertAlmostEqual(S_plus[1, 1], 0.0)

    def test_iter_graph_edges_with_data_simple_graph(self) -> None:
        G = nx.DiGraph()
        G.add_node("A", kind="species")
        G.add_node("r1", kind="rule")
        G.add_edge("A", "r1", role="reactant", stoich=1.0)

        edges = list(stoich._iter_graph_edges_with_data(G))
        self.assertEqual(len(edges), 1)
        u, v, data = edges[0]
        self.assertEqual((u, v), ("A", "r1"))
        self.assertEqual(data["role"], "reactant")
        self.assertEqual(data["stoich"], 1.0)

    def test_iter_graph_edges_with_data_multigraph(self) -> None:
        G = nx.MultiDiGraph()
        G.add_node("A", kind="species")
        G.add_node("r1", kind="rule")
        G.add_edge("A", "r1", role="reactant", stoich=1.0)
        G.add_edge("A", "r1", role="reactant", stoich=2.0)

        edges = list(stoich._iter_graph_edges_with_data(G))
        self.assertEqual(len(edges), 2)
        coeffs = sorted(data["stoich"] for _, _, data in edges)
        self.assertEqual(coeffs, [1.0, 2.0])


# ---------------------------------------------------------------------------
# Tests for stoichiometric matrix construction
# ---------------------------------------------------------------------------


class TestStoichiometricMatrices(unittest.TestCase):
    def setUp(self) -> None:
        self.G = build_chain_example_graph()

    def test_build_S_minus_plus_entries(self) -> None:
        species_order, rule_order, S_minus, S_plus = stoich.build_S_minus_plus(self.G)

        sp_idx = {s: i for i, s in enumerate(species_order)}
        rx_idx = {r: j for j, r in enumerate(rule_order)}

        self.assertEqual(S_minus.shape, (7, 3))
        self.assertEqual(S_plus.shape, (7, 3))

        # r1: 2A + B >> C
        self.assertAlmostEqual(S_minus[sp_idx["A"], rx_idx["r1"]], 2.0)
        self.assertAlmostEqual(S_minus[sp_idx["B"], rx_idx["r1"]], 1.0)
        self.assertAlmostEqual(S_plus[sp_idx["C"], rx_idx["r1"]], 1.0)

        # r2: C + D >> E
        self.assertAlmostEqual(S_minus[sp_idx["C"], rx_idx["r2"]], 1.0)
        self.assertAlmostEqual(S_minus[sp_idx["D"], rx_idx["r2"]], 1.0)
        self.assertAlmostEqual(S_plus[sp_idx["E"], rx_idx["r2"]], 1.0)

        # r3: E + F >> D + G
        self.assertAlmostEqual(S_minus[sp_idx["E"], rx_idx["r3"]], 1.0)
        self.assertAlmostEqual(S_minus[sp_idx["F"], rx_idx["r3"]], 1.0)
        self.assertAlmostEqual(S_plus[sp_idx["D"], rx_idx["r3"]], 1.0)
        self.assertAlmostEqual(S_plus[sp_idx["G"], rx_idx["r3"]], 1.0)

        # all other unspecified entries should remain zero in a few spot checks
        self.assertAlmostEqual(S_minus[sp_idx["G"], rx_idx["r1"]], 0.0)
        self.assertAlmostEqual(S_plus[sp_idx["A"], rx_idx["r2"]], 0.0)

    def test_build_S_and_stoichiometric_matrix(self) -> None:
        species_order, rule_order, S = stoich.build_S(self.G)
        S2 = stoich.stoichiometric_matrix(self.G)

        self.assertTrue(np.allclose(S, S2))

        sp_idx = {s: i for i, s in enumerate(species_order)}
        rx_idx = {r: j for j, r in enumerate(rule_order)}

        # r1: 2A + B >> C
        self.assertAlmostEqual(S[sp_idx["A"], rx_idx["r1"]], -2.0)
        self.assertAlmostEqual(S[sp_idx["B"], rx_idx["r1"]], -1.0)
        self.assertAlmostEqual(S[sp_idx["C"], rx_idx["r1"]], 1.0)

        # r2: C + D >> E
        self.assertAlmostEqual(S[sp_idx["C"], rx_idx["r2"]], -1.0)
        self.assertAlmostEqual(S[sp_idx["D"], rx_idx["r2"]], -1.0)
        self.assertAlmostEqual(S[sp_idx["E"], rx_idx["r2"]], 1.0)

        # r3: E + F >> D + G
        self.assertAlmostEqual(S[sp_idx["E"], rx_idx["r3"]], -1.0)
        self.assertAlmostEqual(S[sp_idx["F"], rx_idx["r3"]], -1.0)
        self.assertAlmostEqual(S[sp_idx["D"], rx_idx["r3"]], 1.0)
        self.assertAlmostEqual(S[sp_idx["G"], rx_idx["r3"]], 1.0)

    def test_stoichiometric_rank(self) -> None:
        S = stoich.stoichiometric_matrix(self.G)
        expected_rank = int(np.linalg.matrix_rank(S))
        self.assertEqual(stoich.stoichiometric_rank(self.G), expected_rank)
        self.assertEqual(expected_rank, 3)

    def test_ignores_noise_edges(self) -> None:
        G = build_graph_with_noise_edges()
        species_order, rule_order, S_minus, S_plus = stoich.build_S_minus_plus(G)

        sp_idx = {s: i for i, s in enumerate(species_order)}
        rx_idx = {r: j for j, r in enumerate(rule_order)}

        # only valid reaction should contribute: A >> B via r1
        self.assertAlmostEqual(S_minus[sp_idx["A"], rx_idx["r1"]], 1.0)
        self.assertAlmostEqual(S_plus[sp_idx["B"], rx_idx["r1"]], 1.0)

        # r2 should remain all zeros because all its incidences are invalid / ignored
        self.assertTrue(np.allclose(S_minus[:, rx_idx["r2"]], 0.0))
        self.assertTrue(np.allclose(S_plus[:, rx_idx["r2"]], 0.0))

    def test_undirected_graph_is_supported(self) -> None:
        G = build_undirected_example_graph()
        species_order, rule_order, S = stoich.build_S(G)

        sp_idx = {s: i for i, s in enumerate(species_order)}
        rx_idx = {r: j for j, r in enumerate(rule_order)}

        self.assertEqual(S.shape, (2, 1))
        self.assertAlmostEqual(S[sp_idx["A"], rx_idx["r1"]], -1.0)
        self.assertAlmostEqual(S[sp_idx["B"], rx_idx["r1"]], 1.0)

    def test_multigraph_parallel_edges_accumulate(self) -> None:
        G = build_multigraph_example()
        species_order, rule_order, S_minus, S_plus = stoich.build_S_minus_plus(G)

        sp_idx = {s: i for i, s in enumerate(species_order)}
        rx_idx = {r: j for j, r in enumerate(rule_order)}

        self.assertAlmostEqual(S_minus[sp_idx["A"], rx_idx["r1"]], 3.0)
        self.assertAlmostEqual(S_plus[sp_idx["B"], rx_idx["r1"]], 1.0)


# ---------------------------------------------------------------------------
# Tests for nullspaces
# ---------------------------------------------------------------------------


class TestNullspacesAndSemiflows(unittest.TestCase):
    def setUp(self) -> None:
        self.G = build_chain_example_graph()
        self.S = stoich.stoichiometric_matrix(self.G)

    def test_left_nullspace_properties(self) -> None:
        L = stoich.left_nullspace(self.G)
        n_species, _ = self.S.shape
        rank = int(np.linalg.matrix_rank(self.S))

        self.assertEqual(L.shape[0], n_species)
        self.assertEqual(L.shape[1], n_species - rank)

        if L.size > 0:
            self.assertTrue(np.allclose(self.S.T @ L, 0.0, atol=1e-8))

    def test_right_nullspace_properties(self) -> None:
        R = stoich.right_nullspace(self.G)
        _, n_rules = self.S.shape
        rank = int(np.linalg.matrix_rank(self.S))

        self.assertEqual(R.shape[0], n_rules)
        self.assertEqual(R.shape[1], n_rules - rank)

        if R.size > 0:
            self.assertTrue(np.allclose(self.S @ R, 0.0, atol=1e-8))

    def test_left_right_kernels(self) -> None:
        L, R = stoich.left_right_kernels(self.G)

        if L.size > 0:
            self.assertTrue(np.allclose(self.S.T @ L, 0.0, atol=1e-8))
        if R.size > 0:
            self.assertTrue(np.allclose(self.S @ R, 0.0, atol=1e-8))

    def test_known_left_nullspace_dimension_on_conserved_cycle(self) -> None:
        G = build_conserved_cycle_graph()
        S = stoich.stoichiometric_matrix(G)
        L = stoich.left_nullspace(G)

        self.assertEqual(S.shape, (2, 2))
        self.assertEqual(int(np.linalg.matrix_rank(S)), 1)
        self.assertEqual(L.shape, (2, 1))
        self.assertTrue(np.allclose(S.T @ L, 0.0, atol=1e-8))

    def test_known_right_nullspace_dimension_on_flux_cycle(self) -> None:
        G = build_flux_cycle_graph()
        S = stoich.stoichiometric_matrix(G)
        R = stoich.right_nullspace(G)

        self.assertEqual(S.shape, (2, 2))
        self.assertEqual(int(np.linalg.matrix_rank(S)), 1)
        self.assertEqual(R.shape, (2, 1))
        self.assertTrue(np.allclose(S @ R, 0.0, atol=1e-8))


# ---------------------------------------------------------------------------
# Tests for integer helpers
# ---------------------------------------------------------------------------


class TestIntegerHelpers(unittest.TestCase):
    def test_lcm_basic_cases(self) -> None:
        self.assertEqual(stoich._lcm(4, 6), 12)
        self.assertEqual(stoich._lcm(0, 5), 5)
        self.assertEqual(stoich._lcm(7, 0), 7)
        self.assertEqual(stoich._lcm(0, 0), 0)
        self.assertEqual(stoich._lcm(-4, 6), 12)

    def test_vector_to_minimal_integer_simple_ratio(self) -> None:
        v = np.array([0.5, 1.0, 1.5], dtype=float)
        ints = stoich._vector_to_minimal_integer(v)
        self.assertEqual(ints, [1, 2, 3])

    def test_vector_to_minimal_integer_zero_vector(self) -> None:
        v = np.zeros(4, dtype=float)
        ints = stoich._vector_to_minimal_integer(v)
        self.assertEqual(ints, [0, 0, 0, 0])

    def test_vector_to_minimal_integer_sign_and_reduction(self) -> None:
        v = np.array([-0.5, 1.0, -1.5], dtype=float)
        ints = stoich._vector_to_minimal_integer(v)
        self.assertEqual(ints, [-1, 2, -3])

    def test_integer_conservation_laws_chain_example_shape_only(self) -> None:
        G = build_chain_example_graph()
        S = stoich.stoichiometric_matrix(G)
        n_species, _ = S.shape
        rank = int(np.linalg.matrix_rank(S))

        self.assertEqual(n_species - rank, 4)

        laws: List[List[int]] = stoich.integer_conservation_laws(G)
        self.assertEqual(len(laws), 4)

        for law in laws:
            self.assertEqual(len(law), n_species)
            self.assertTrue(all(isinstance(x, int) for x in law))
            self.assertFalse(all(x == 0 for x in law))

    def test_integer_conservation_laws_known_cycle(self) -> None:
        G = build_conserved_cycle_graph()
        S = stoich.stoichiometric_matrix(G)

        laws = stoich.integer_conservation_laws(G)
        self.assertEqual(len(laws), 1)

        law = np.array(laws[0], dtype=float).reshape(-1, 1)
        self.assertEqual(law.shape, (2, 1))
        self.assertTrue(np.allclose(S.T @ law, 0.0, atol=1e-8))
        self.assertTrue(
            np.array_equal(np.abs(law.flatten()).astype(int), np.array([1, 1]))
        )


# ---------------------------------------------------------------------------
# Tests for summary dataclass
# ---------------------------------------------------------------------------


class TestStoichSummary(unittest.TestCase):
    def test_summary_from_crn(self) -> None:
        G = build_chain_example_graph()
        S = stoich.stoichiometric_matrix(G)
        n_species, n_rules = S.shape
        rank = int(np.linalg.matrix_rank(S))

        s = stoich.summary(G)

        self.assertEqual(s.n_species, n_species)
        self.assertEqual(s.n_reactions, n_rules)
        self.assertEqual(s.rank, rank)
        self.assertEqual(s.dim_left_kernel, n_species - rank)
        self.assertEqual(s.dim_right_kernel, n_rules - rank)

    def test_summary_is_full_rank_and_is_underdetermined(self) -> None:
        s1 = stoich.StoichSummary(n_species=3, n_reactions=3, rank=3)
        self.assertTrue(s1.is_full_rank)
        self.assertFalse(s1.is_underdetermined)

        s2 = stoich.StoichSummary(n_species=5, n_reactions=3, rank=2)
        self.assertFalse(s2.is_full_rank)
        self.assertTrue(s2.is_underdetermined)

    def test_summary_to_dict(self) -> None:
        s = stoich.StoichSummary(n_species=5, n_reactions=3, rank=2)
        d = s.to_dict()

        self.assertEqual(
            d,
            {
                "n_species": 5,
                "n_reactions": 3,
                "rank": 2,
                "dim_left_kernel": 3,
                "dim_right_kernel": 1,
            },
        )

    def test_summary_str(self) -> None:
        s = stoich.StoichSummary(n_species=5, n_reactions=3, rank=2)
        text = str(s)

        self.assertIn("StoichSummary(", text)
        self.assertIn("n_species", text)
        self.assertIn("n_reactions", text)
        self.assertIn("rank", text)
        self.assertIn("dim_left_kernel", text)
        self.assertIn("dim_right_kernel", text)

    def test_summary_invalid_negative_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            stoich.StoichSummary(n_species=-1, n_reactions=3, rank=1)

        with self.assertRaises(ValueError):
            stoich.StoichSummary(n_species=2, n_reactions=-3, rank=1)

        with self.assertRaises(ValueError):
            stoich.StoichSummary(n_species=2, n_reactions=3, rank=-1)

    def test_summary_invalid_rank_too_large(self) -> None:
        with self.assertRaises(ValueError):
            stoich.StoichSummary(n_species=2, n_reactions=3, rank=4)

    def test_from_crn_classmethod(self) -> None:
        G = build_conserved_cycle_graph()
        s = stoich.StoichSummary.from_crn(G)

        self.assertEqual(s.n_species, 2)
        self.assertEqual(s.n_reactions, 2)
        self.assertEqual(s.rank, 1)
        self.assertEqual(s.dim_left_kernel, 1)
        self.assertEqual(s.dim_right_kernel, 1)


if __name__ == "__main__":
    unittest.main()
