import unittest

import networkx as nx
import numpy as np

from synkit.CRN.Props import thermo

# ---------------------------------------------------------------------------
# Helpers to build small test networks
# ---------------------------------------------------------------------------


def build_simple_A_to_B_graph() -> nx.DiGraph:
    """
    Build bipartite graph for:

        A >> B
    """
    G = nx.DiGraph()
    for s in ("A", "B"):
        G.add_node(s, kind="species", label=s)
    G.add_node("r1", kind="rule", label="r1")

    G.add_edge("A", "r1", role="reactant", stoich=1.0)
    G.add_edge("r1", "B", role="product", stoich=1.0)
    return G


def build_decay_graph() -> nx.DiGraph:
    """
    Build bipartite graph for:

        A >> ∅
    """
    G = nx.DiGraph()
    G.add_node("A", kind="species", label="A")
    G.add_node("r1", kind="rule", label="r1")

    G.add_edge("A", "r1", role="reactant", stoich=1.0)
    return G


def build_cycle_graph() -> nx.DiGraph:
    """
    Build bipartite graph for:

        A >> B
        B >> A
    """
    G = nx.DiGraph()
    for s in ("A", "B"):
        G.add_node(s, kind="species", label=s)

    G.add_node("r1", kind="rule", label="r1")
    G.add_node("r2", kind="rule", label="r2")

    G.add_edge("A", "r1", role="reactant", stoich=1.0)
    G.add_edge("r1", "B", role="product", stoich=1.0)

    G.add_edge("B", "r2", role="reactant", stoich=1.0)
    G.add_edge("r2", "A", role="product", stoich=1.0)
    return G


def build_cycle_with_branch_graph() -> nx.DiGraph:
    """
    Build:

        A >> B
        B >> A
        B >> C
        C >> B
        C >> D
    """
    G = nx.DiGraph()
    for s in ("A", "B", "C", "D"):
        G.add_node(s, kind="species", label=s)

    for r in ("r1", "r2", "r3", "r4", "r5"):
        G.add_node(r, kind="rule", label=r)

    # A >> B
    G.add_edge("A", "r1", role="reactant", stoich=1.0)
    G.add_edge("r1", "B", role="product", stoich=1.0)

    # B >> A
    G.add_edge("B", "r2", role="reactant", stoich=1.0)
    G.add_edge("r2", "A", role="product", stoich=1.0)

    # B >> C
    G.add_edge("B", "r3", role="reactant", stoich=1.0)
    G.add_edge("r3", "C", role="product", stoich=1.0)

    # C >> B
    G.add_edge("C", "r4", role="reactant", stoich=1.0)
    G.add_edge("r4", "B", role="product", stoich=1.0)

    # C >> D
    G.add_edge("C", "r5", role="reactant", stoich=1.0)
    G.add_edge("r5", "D", role="product", stoich=1.0)

    return G


def build_partial_conservation_graph() -> nx.DiGraph:
    """
    Build:

        A >> B
        B >> A
        X >> ∅

    The A/B subsystem is conservative, but the whole network is not
    conservative because X decays.
    """
    G = nx.DiGraph()
    for s in ("A", "B", "X"):
        G.add_node(s, kind="species", label=s)

    for r in ("r1", "r2", "r3"):
        G.add_node(r, kind="rule", label=r)

    # A >> B
    G.add_edge("A", "r1", role="reactant", stoich=1.0)
    G.add_edge("r1", "B", role="product", stoich=1.0)

    # B >> A
    G.add_edge("B", "r2", role="reactant", stoich=1.0)
    G.add_edge("r2", "A", role="product", stoich=1.0)

    # X >> ∅
    G.add_edge("X", "r3", role="reactant", stoich=1.0)

    return G


def build_two_independent_cycles_graph() -> nx.DiGraph:
    """
    Build:

        A >> B
        B >> A
        C >> D
        D >> C
    """
    G = nx.DiGraph()
    for s in ("A", "B", "C", "D"):
        G.add_node(s, kind="species", label=s)

    for r in ("r1", "r2", "r3", "r4"):
        G.add_node(r, kind="rule", label=r)

    # A >> B
    G.add_edge("A", "r1", role="reactant", stoich=1.0)
    G.add_edge("r1", "B", role="product", stoich=1.0)

    # B >> A
    G.add_edge("B", "r2", role="reactant", stoich=1.0)
    G.add_edge("r2", "A", role="product", stoich=1.0)

    # C >> D
    G.add_edge("C", "r3", role="reactant", stoich=1.0)
    G.add_edge("r3", "D", role="product", stoich=1.0)

    # D >> C
    G.add_edge("D", "r4", role="reactant", stoich=1.0)
    G.add_edge("r4", "C", role="product", stoich=1.0)

    return G


def build_large_network_graph() -> nx.DiGraph:
    """
    Build:

        A >> B
        B >> C
        C >> A
        D >> E
        E >> F
        F >> D
        X >> Y
        C + E >> Z
        Z >> A + F
    """
    G = nx.DiGraph()
    for s in ("A", "B", "C", "D", "E", "F", "X", "Y", "Z"):
        G.add_node(s, kind="species", label=s)

    for r in ("r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9"):
        G.add_node(r, kind="rule", label=r)

    # A >> B
    G.add_edge("A", "r1", role="reactant", stoich=1.0)
    G.add_edge("r1", "B", role="product", stoich=1.0)

    # B >> C
    G.add_edge("B", "r2", role="reactant", stoich=1.0)
    G.add_edge("r2", "C", role="product", stoich=1.0)

    # C >> A
    G.add_edge("C", "r3", role="reactant", stoich=1.0)
    G.add_edge("r3", "A", role="product", stoich=1.0)

    # D >> E
    G.add_edge("D", "r4", role="reactant", stoich=1.0)
    G.add_edge("r4", "E", role="product", stoich=1.0)

    # E >> F
    G.add_edge("E", "r5", role="reactant", stoich=1.0)
    G.add_edge("r5", "F", role="product", stoich=1.0)

    # F >> D
    G.add_edge("F", "r6", role="reactant", stoich=1.0)
    G.add_edge("r6", "D", role="product", stoich=1.0)

    # X >> Y
    G.add_edge("X", "r7", role="reactant", stoich=1.0)
    G.add_edge("r7", "Y", role="product", stoich=1.0)

    # C + E >> Z
    G.add_edge("C", "r8", role="reactant", stoich=1.0)
    G.add_edge("E", "r8", role="reactant", stoich=1.0)
    G.add_edge("r8", "Z", role="product", stoich=1.0)

    # Z >> A + F
    G.add_edge("Z", "r9", role="reactant", stoich=1.0)
    G.add_edge("r9", "A", role="product", stoich=1.0)
    G.add_edge("r9", "F", role="product", stoich=1.0)

    return G


# ---------------------------------------------------------------------------
# Tests for low-level helpers
# ---------------------------------------------------------------------------


class TestNormalizePositive(unittest.TestCase):
    def test_normalize_positive_accepts_positive(self) -> None:
        v = np.array([2.0, 3.0, 5.0], dtype=float)
        out = thermo._normalize_positive(v)

        self.assertIsNotNone(out)
        assert out is not None
        self.assertTrue(np.all(out > 0.0))
        self.assertAlmostEqual(float(np.sum(out)), 1.0)

    def test_normalize_positive_accepts_negative_by_sign_flip(self) -> None:
        v = np.array([-2.0, -3.0, -5.0], dtype=float)
        out = thermo._normalize_positive(v)

        self.assertIsNotNone(out)
        assert out is not None
        self.assertTrue(np.all(out > 0.0))
        self.assertAlmostEqual(float(np.sum(out)), 1.0)

    def test_normalize_positive_rejects_mixed_sign(self) -> None:
        v = np.array([1.0, -2.0, 3.0], dtype=float)
        out = thermo._normalize_positive(v)
        self.assertIsNone(out)

    def test_normalize_positive_rejects_zero_entries(self) -> None:
        v = np.array([1.0, 0.0, 2.0], dtype=float)
        out = thermo._normalize_positive(v, eps=1e-8)
        self.assertIsNone(out)


class TestLeftNullspaceFromMatrix(unittest.TestCase):
    def test_left_nullspace_from_matrix_simple_reversible_pair(self) -> None:
        S = np.array(
            [
                [-1.0, 1.0],
                [1.0, -1.0],
            ],
            dtype=float,
        )

        L = thermo.left_nullspace_from_matrix(S)

        self.assertEqual(L.shape[0], 2)
        self.assertEqual(L.shape[1], 1)
        self.assertTrue(np.allclose(S.T @ L, 0.0, atol=1e-8))

    def test_left_nullspace_from_matrix_full_rank(self) -> None:
        S = np.array(
            [
                [-1.0],
                [1.0],
            ],
            dtype=float,
        )

        L = thermo.left_nullspace_from_matrix(S)
        self.assertEqual(L.shape[0], 2)
        self.assertEqual(L.shape[1], 1)
        self.assertTrue(np.allclose(S.T @ L, 0.0, atol=1e-8))


class TestPositiveLeftKernelSearch(unittest.TestCase):
    def test_find_positive_left_kernel_vector_no_reactions(self) -> None:
        S = np.zeros((3, 0), dtype=float)
        flag, m = thermo._find_positive_left_kernel_vector(S)

        self.assertIs(flag, True)
        self.assertIsNotNone(m)
        assert m is not None
        self.assertEqual(m.shape, (3,))
        self.assertTrue(np.all(m > 0.0))
        self.assertAlmostEqual(float(np.sum(m)), 1.0)

    def test_find_positive_left_kernel_vector_empty_matrix(self) -> None:
        S = np.zeros((0, 0), dtype=float)
        flag, m = thermo._find_positive_left_kernel_vector(S)

        self.assertIs(flag, True)
        self.assertIsNone(m)

    def test_find_positive_left_kernel_vector_conservative_case(self) -> None:
        S = np.array(
            [
                [-1.0, 1.0],
                [1.0, -1.0],
            ],
            dtype=float,
        )

        flag, m = thermo._find_positive_left_kernel_vector(S)

        self.assertIs(flag, True)
        self.assertIsNotNone(m)
        assert m is not None
        self.assertTrue(np.all(m > 0.0))
        self.assertAlmostEqual(float(np.sum(m)), 1.0)
        self.assertTrue(np.allclose(S.T @ m.reshape(-1, 1), 0.0, atol=1e-8))

    def test_find_positive_left_kernel_vector_nonconservative_case(self) -> None:
        S = np.array([[-1.0]], dtype=float)
        flag, m = thermo._find_positive_left_kernel_vector(S)

        self.assertIs(flag, False)
        self.assertIsNone(m)


# ---------------------------------------------------------------------------
# Tests for public conservativity / consistency API
# ---------------------------------------------------------------------------


class TestConservativityAndConsistency(unittest.TestCase):
    def test_is_conservative_simple_A_to_B_true(self) -> None:
        G = build_simple_A_to_B_graph()
        res = thermo.is_conservative(G)
        self.assertIs(res, True)

    def test_compute_conservativity_simple_A_to_B_returns_vector(self) -> None:
        G = build_simple_A_to_B_graph()
        flag, m = thermo.compute_conservativity(G)

        self.assertIs(flag, True)
        self.assertIsNotNone(m)
        assert m is not None
        self.assertEqual(m.shape, (2,))
        self.assertTrue(np.all(m > 0.0))
        self.assertAlmostEqual(float(np.sum(m)), 1.0)

    def test_is_conservative_decay_false(self) -> None:
        G = build_decay_graph()
        res = thermo.is_conservative(G)
        self.assertIs(res, False)

    def test_compute_conservativity_decay_false(self) -> None:
        G = build_decay_graph()
        flag, m = thermo.compute_conservativity(G)

        self.assertIs(flag, False)
        self.assertIsNone(m)

    def test_is_consistent_cycle_true(self) -> None:
        G = build_cycle_graph()
        res = thermo.is_consistent(G)
        self.assertIs(res, True)

    def test_is_consistent_decay_false(self) -> None:
        G = build_decay_graph()
        res = thermo.is_consistent(G)
        self.assertIs(res, False)

    def test_is_consistent_single_A_to_B_false(self) -> None:
        G = build_simple_A_to_B_graph()
        res = thermo.is_consistent(G)
        self.assertIs(res, False)

    def test_is_consistent_no_reactions_cases(self) -> None:
        G1 = nx.DiGraph()
        G1.add_node("A", kind="species")
        self.assertIs(thermo.is_consistent(G1), False)

        G2 = nx.DiGraph()
        self.assertIs(thermo.is_consistent(G2), True)


# ---------------------------------------------------------------------------
# Tests for futile-cycle proxy
# ---------------------------------------------------------------------------


class TestFutileCycleProxy(unittest.TestCase):
    def test_has_irreversible_futile_cycles_cycle_true(self) -> None:
        G = build_cycle_graph()
        self.assertTrue(thermo.has_irreversible_futile_cycles(G))

    def test_has_irreversible_futile_cycles_simple_A_to_B_false(self) -> None:
        G = build_simple_A_to_B_graph()
        self.assertFalse(thermo.has_irreversible_futile_cycles(G))

    def test_has_irreversible_futile_cycles_decay_false(self) -> None:
        G = build_decay_graph()
        self.assertFalse(thermo.has_irreversible_futile_cycles(G))


# ---------------------------------------------------------------------------
# Tests for summary object
# ---------------------------------------------------------------------------


class TestThermoSummary(unittest.TestCase):
    def test_compute_thermo_summary_cycle(self) -> None:
        G = build_cycle_graph()
        summary = thermo.compute_thermo_summary(G)

        self.assertIsInstance(summary, thermo.ThermoSummary)
        self.assertIs(summary.conservative, True)
        self.assertIs(summary.consistent, True)
        self.assertIs(summary.irreversible_futile_cycles, True)
        self.assertIsNotNone(summary.example_conservation_law)

        m = summary.example_conservation_law
        assert m is not None
        self.assertEqual(m.shape, (2,))
        self.assertTrue(np.all(m > 0.0))
        self.assertAlmostEqual(float(np.sum(m)), 1.0)

    def test_compute_thermo_summary_decay(self) -> None:
        G = build_decay_graph()
        summary = thermo.compute_thermo_summary(G)

        self.assertIsInstance(summary, thermo.ThermoSummary)
        self.assertIs(summary.conservative, False)
        self.assertIs(summary.consistent, False)
        self.assertIs(summary.irreversible_futile_cycles, False)
        self.assertIsNone(summary.example_conservation_law)

    def test_compute_thermo_summary_simple_A_to_B(self) -> None:
        G = build_simple_A_to_B_graph()
        summary = thermo.compute_thermo_summary(G)

        self.assertIs(summary.conservative, True)
        self.assertIs(summary.consistent, False)
        self.assertIs(summary.irreversible_futile_cycles, False)
        self.assertIsNotNone(summary.example_conservation_law)


# ---------------------------------------------------------------------------
# More complex structural tests
# ---------------------------------------------------------------------------


class TestThermoComplex(unittest.TestCase):
    def test_cycle_with_branch(self) -> None:
        G = build_cycle_with_branch_graph()
        summary = thermo.compute_thermo_summary(G)

        self.assertTrue(summary.irreversible_futile_cycles)
        self.assertTrue(summary.conservative)
        self.assertFalse(summary.consistent)

        self.assertIsNotNone(summary.example_conservation_law)
        m = summary.example_conservation_law
        assert m is not None
        self.assertTrue(np.all(m > 0.0))
        self.assertAlmostEqual(float(np.sum(m)), 1.0)

    def test_partial_conservation(self) -> None:
        G = build_partial_conservation_graph()
        summary = thermo.compute_thermo_summary(G)

        self.assertFalse(summary.conservative)
        self.assertTrue(summary.irreversible_futile_cycles)
        self.assertIsNone(summary.example_conservation_law)

        # consistency should be False because reaction X >> ∅ cannot participate
        # in a strictly positive steady-state flux vector over all rules
        self.assertIs(summary.consistent, False)

    def test_two_independent_cycles(self) -> None:
        G = build_two_independent_cycles_graph()
        summary = thermo.compute_thermo_summary(G)

        self.assertTrue(summary.irreversible_futile_cycles)
        self.assertTrue(summary.conservative)
        self.assertTrue(summary.consistent)
        self.assertIsNotNone(summary.example_conservation_law)

        m = summary.example_conservation_law
        assert m is not None
        self.assertEqual(m.shape, (4,))
        self.assertTrue(np.all(m > 0.0))
        self.assertAlmostEqual(float(np.sum(m)), 1.0)

    def test_large_network(self) -> None:
        G = build_large_network_graph()
        summary = thermo.compute_thermo_summary(G)

        self.assertIsInstance(summary.conservative, (bool, type(None)))
        self.assertIsInstance(summary.consistent, (bool, type(None)))
        self.assertIsInstance(summary.irreversible_futile_cycles, bool)

        self.assertTrue(summary.irreversible_futile_cycles)

        if summary.conservative is True:
            self.assertIsNotNone(summary.example_conservation_law)
            m = summary.example_conservation_law
            assert m is not None
            self.assertGreater(float(np.sum(m)), 0.0)
            self.assertTrue(np.all(m > 0.0))
        elif summary.conservative is False:
            self.assertIsNone(summary.example_conservation_law)


if __name__ == "__main__":
    unittest.main()
