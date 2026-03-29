from __future__ import annotations

import unittest

import networkx as nx
import numpy as np

from synkit.CRN.Props import dynamics


def _add_species(G: nx.DiGraph, name: str, *, label: str | None = None) -> None:
    G.add_node(
        name,
        kind="species",
        bipartite=0,
        label=name if label is None else label,
    )


def _add_rule(G: nx.DiGraph, name: str, *, label: str | None = None) -> None:
    G.add_node(
        name,
        kind="rule",
        bipartite=1,
        label=name if label is None else label,
    )


def _add_reactant_edge(
    G: nx.DiGraph,
    species: str,
    rule: str,
    stoich: float = 1.0,
) -> None:
    G.add_edge(
        species,
        rule,
        role="reactant",
        stoich=stoich,
    )


def _add_product_edge(
    G: nx.DiGraph,
    rule: str,
    species: str,
    stoich: float = 1.0,
) -> None:
    G.add_edge(
        rule,
        species,
        role="product",
        stoich=stoich,
    )


def make_cycle_crn() -> nx.DiGraph:
    """
    A >> B
    B >> C
    C >> A
    """
    G = nx.DiGraph()

    for s in ("A", "B", "C"):
        _add_species(G, s)
    for r in ("r1", "r2", "r3"):
        _add_rule(G, r)

    _add_reactant_edge(G, "A", "r1")
    _add_product_edge(G, "r1", "B")

    _add_reactant_edge(G, "B", "r2")
    _add_product_edge(G, "r2", "C")

    _add_reactant_edge(G, "C", "r3")
    _add_product_edge(G, "r3", "A")

    return G


def make_single_step_crn() -> nx.DiGraph:
    """
    A >> B
    """
    G = nx.DiGraph()
    for s in ("A", "B"):
        _add_species(G, s)
    _add_rule(G, "r1")

    _add_reactant_edge(G, "A", "r1")
    _add_product_edge(G, "r1", "B")
    return G


def make_two_decay_crn() -> nx.DiGraph:
    """
    A >> ∅
    B >> ∅
    """
    G = nx.DiGraph()
    for s in ("A", "B"):
        _add_species(G, s)
    for r in ("dA", "dB"):
        _add_rule(G, r)

    _add_reactant_edge(G, "A", "dA")
    _add_reactant_edge(G, "B", "dB")
    return G


def make_autocatalytic_mixed_sign_crn() -> nx.DiGraph:
    """
    A >> B
    A + B >> 2A

    This gives a mixed-sign effect of A on A:
    - reaction 1 consumes A  -> negative contribution
    - reaction 2 produces A and depends on A -> positive contribution
    """
    G = nx.DiGraph()
    for s in ("A", "B"):
        _add_species(G, s)
    for r in ("r1", "r2"):
        _add_rule(G, r)

    # r1: A >> B
    _add_reactant_edge(G, "A", "r1", 1.0)
    _add_product_edge(G, "r1", "B", 1.0)

    # r2: A + B >> 2A
    _add_reactant_edge(G, "A", "r2", 1.0)
    _add_reactant_edge(G, "B", "r2", 1.0)
    _add_product_edge(G, "r2", "A", 2.0)

    return G


class TestLowLevelHelpers(unittest.TestCase):
    def test_safe_symbol_token(self) -> None:
        self.assertEqual(dynamics._safe_symbol_token("A"), "A")
        self.assertEqual(dynamics._safe_symbol_token("species-1"), "species_1")
        self.assertEqual(dynamics._safe_symbol_token("3-node"), "_3_node")
        self.assertEqual(dynamics._safe_symbol_token(""), "x")

    def test_structural_sign_pattern_cycle(self) -> None:
        S = np.array(
            [
                [-1.0, 0.0, 1.0],
                [1.0, -1.0, 0.0],
                [0.0, 1.0, -1.0],
            ]
        )
        S_minus = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        P = dynamics._structural_sign_pattern(S, S_minus)

        expected = np.array(
            [
                ["-", "0", "+"],
                ["+", "-", "0"],
                ["0", "+", "-"],
            ],
            dtype=object,
        )
        self.assertTrue(np.array_equal(P, expected))

    def test_structural_sign_pattern_mixed(self) -> None:
        S = np.array(
            [
                [-1.0, 1.0],
                [1.0, -1.0],
            ]
        )
        S_minus = np.array(
            [
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )

        P = dynamics._structural_sign_pattern(S, S_minus)
        self.assertEqual(P[0, 0], "mixed")

    def test_jacobian_pattern_bipartite(self) -> None:
        A = np.array(
            [
                [True, False],
                [True, True],
            ],
            dtype=bool,
        )
        B, row_nodes, col_nodes = dynamics._jacobian_pattern_bipartite(A)

        self.assertEqual(row_nodes, ["row:0", "row:1"])
        self.assertEqual(col_nodes, ["col:0", "col:1"])
        self.assertEqual(B.number_of_edges(), 3)
        self.assertTrue(B.has_edge("row:0", "col:0"))
        self.assertTrue(B.has_edge("row:1", "col:0"))
        self.assertTrue(B.has_edge("row:1", "col:1"))


class TestDynamicsMatrices(unittest.TestCase):
    def test_symbolic_reactivity_matrix_cycle(self) -> None:
        G = make_cycle_crn()

        species_order, rule_order, R = dynamics.symbolic_reactivity_matrix(G)

        iA = species_order.index("A")
        iB = species_order.index("B")
        iC = species_order.index("C")

        jr1 = rule_order.index("r1")
        jr2 = rule_order.index("r2")
        jr3 = rule_order.index("r3")

        self.assertNotEqual(R[jr1, iA], 0)
        self.assertEqual(R[jr1, iB], 0)
        self.assertEqual(R[jr1, iC], 0)

        self.assertEqual(R[jr2, iA], 0)
        self.assertNotEqual(R[jr2, iB], 0)
        self.assertEqual(R[jr2, iC], 0)

        self.assertEqual(R[jr3, iA], 0)
        self.assertEqual(R[jr3, iB], 0)
        self.assertNotEqual(R[jr3, iC], 0)

    def test_symbolic_reactivity_matrix_uses_safe_symbol_names(self) -> None:
        G = nx.DiGraph()
        _add_species(G, "A-1")
        _add_species(G, "2B")
        _add_rule(G, "r-1")

        _add_reactant_edge(G, "A-1", "r-1")
        _add_reactant_edge(G, "2B", "r-1")

        species_order, rule_order, R = dynamics.symbolic_reactivity_matrix(G)
        self.assertEqual(len(species_order), 2)
        self.assertEqual(len(rule_order), 1)

        symbols = [str(R[0, i]) for i in range(R.shape[1]) if R[0, i] != 0]
        self.assertTrue(any("rprime_r_1_A_1" == s for s in symbols))
        self.assertTrue(any("rprime_r_1__2B" == s for s in symbols))

    def test_symbolic_jacobian_cycle_shape_and_entries(self) -> None:
        G = make_cycle_crn()

        species_order, _rule_order, J = dynamics.symbolic_jacobian(G)
        idx = {s: i for i, s in enumerate(species_order)}

        self.assertEqual(J.shape, (3, 3))

        self.assertNotEqual(J[idx["A"], idx["A"]], 0)
        self.assertNotEqual(J[idx["A"], idx["C"]], 0)
        self.assertEqual(J[idx["A"], idx["B"]], 0)

        self.assertNotEqual(J[idx["B"], idx["A"]], 0)
        self.assertNotEqual(J[idx["B"], idx["B"]], 0)
        self.assertEqual(J[idx["B"], idx["C"]], 0)

        self.assertEqual(J[idx["C"], idx["A"]], 0)
        self.assertNotEqual(J[idx["C"], idx["B"]], 0)
        self.assertNotEqual(J[idx["C"], idx["C"]], 0)


class TestDynamicsPatterns(unittest.TestCase):
    def test_jacobian_sign_pattern_cycle(self) -> None:
        G = make_cycle_crn()

        species_order, P = dynamics.jacobian_sign_pattern(G)
        idx = {s: i for i, s in enumerate(species_order)}

        self.assertEqual(P[idx["A"], idx["A"]], "-")
        self.assertEqual(P[idx["A"], idx["B"]], "0")
        self.assertEqual(P[idx["A"], idx["C"]], "+")

        self.assertEqual(P[idx["B"], idx["A"]], "+")
        self.assertEqual(P[idx["B"], idx["B"]], "-")
        self.assertEqual(P[idx["B"], idx["C"]], "0")

        self.assertEqual(P[idx["C"], idx["A"]], "0")
        self.assertEqual(P[idx["C"], idx["B"]], "+")
        self.assertEqual(P[idx["C"], idx["C"]], "-")

    def test_jacobian_sign_pattern_mixed(self) -> None:
        G = make_autocatalytic_mixed_sign_crn()

        species_order, P = dynamics.jacobian_sign_pattern(G)
        idx = {s: i for i, s in enumerate(species_order)}

        self.assertEqual(P[idx["A"], idx["A"]], "mixed")
        self.assertEqual(P[idx["A"], idx["B"]], "+")
        self.assertEqual(P[idx["B"], idx["A"]], "mixed")
        self.assertEqual(P[idx["B"], idx["B"]], "-")

    def test_jacobian_sparsity_matches_sign_pattern(self) -> None:
        G = make_cycle_crn()

        species_order_1, P = dynamics.jacobian_sign_pattern(G)
        species_order_2, A = dynamics.jacobian_sparsity(G)

        self.assertEqual(species_order_1, species_order_2)

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                self.assertEqual(bool(A[i, j]), P[i, j] != "0")


class TestSpeciesInfluenceGraph(unittest.TestCase):
    def test_species_influence_graph_cycle(self) -> None:
        G = make_cycle_crn()
        Gi = dynamics.species_influence_graph(G)

        self.assertTrue(Gi.has_node("A"))
        self.assertTrue(Gi.has_node("B"))
        self.assertTrue(Gi.has_node("C"))

        self.assertTrue(Gi.has_edge("A", "A"))
        self.assertEqual(Gi["A"]["A"]["sign"], "-")

        self.assertTrue(Gi.has_edge("C", "A"))
        self.assertEqual(Gi["C"]["A"]["sign"], "+")

        self.assertTrue(Gi.has_edge("A", "B"))
        self.assertEqual(Gi["A"]["B"]["sign"], "+")

        self.assertTrue(Gi.has_edge("B", "B"))
        self.assertEqual(Gi["B"]["B"]["sign"], "-")

        self.assertTrue(Gi.has_edge("B", "C"))
        self.assertEqual(Gi["B"]["C"]["sign"], "+")

        self.assertTrue(Gi.has_edge("C", "C"))
        self.assertEqual(Gi["C"]["C"]["sign"], "-")

        self.assertFalse(Gi.has_edge("A", "C"))
        self.assertFalse(Gi.has_edge("B", "A"))
        self.assertFalse(Gi.has_edge("C", "B"))

    def test_species_influence_graph_use_labels(self) -> None:
        G = make_single_step_crn()
        G.nodes["A"]["label"] = "Alpha"
        G.nodes["B"]["label"] = "Beta"

        Gi = dynamics.species_influence_graph(G, use_labels=True)

        self.assertTrue(Gi.has_node("Alpha"))
        self.assertTrue(Gi.has_node("Beta"))
        self.assertTrue(Gi.has_edge("Alpha", "Alpha"))
        self.assertTrue(Gi.has_edge("Alpha", "Beta"))

        self.assertEqual(Gi.nodes["Alpha"]["source_node"], "A")
        self.assertEqual(Gi.nodes["Beta"]["source_node"], "B")
        self.assertEqual(Gi["Alpha"]["Beta"]["source_species"], "A")
        self.assertEqual(Gi["Alpha"]["Beta"]["target_species"], "B")


class TestStructuralSingularitySummaryDataclass(unittest.TestCase):
    def test_classification_singular_by_pattern(self) -> None:
        s = dynamics.StructuralSingularitySummary(
            n_species=2,
            structural_rank=1,
            has_perfect_matching=False,
            pattern_singular=True,
            determinant_checked=True,
            determinant_is_zero=True,
        )
        self.assertEqual(s.classification, "singular_by_pattern")

    def test_classification_singular_by_exact_determinant(self) -> None:
        s = dynamics.StructuralSingularitySummary(
            n_species=3,
            structural_rank=3,
            has_perfect_matching=True,
            pattern_singular=False,
            determinant_checked=True,
            determinant_is_zero=True,
        )
        self.assertEqual(s.classification, "singular_by_exact_determinant")

    def test_classification_structurally_nonsingular(self) -> None:
        s = dynamics.StructuralSingularitySummary(
            n_species=2,
            structural_rank=2,
            has_perfect_matching=True,
            pattern_singular=False,
            determinant_checked=True,
            determinant_is_zero=False,
        )
        self.assertEqual(s.classification, "structurally_nonsingular")

    def test_classification_pattern_nonsingular_exact_unchecked(self) -> None:
        s = dynamics.StructuralSingularitySummary(
            n_species=2,
            structural_rank=2,
            has_perfect_matching=True,
            pattern_singular=False,
            determinant_checked=False,
            determinant_is_zero=None,
        )
        self.assertEqual(s.classification, "pattern_nonsingular_exact_unchecked")

    def test_to_dict_and_str(self) -> None:
        s = dynamics.StructuralSingularitySummary(
            n_species=2,
            structural_rank=2,
            has_perfect_matching=True,
            pattern_singular=False,
            determinant_checked=True,
            determinant_expr="x + y",
            determinant_is_zero=False,
        )

        d = s.to_dict()
        self.assertEqual(d["n_species"], 2)
        self.assertEqual(d["structural_rank"], 2)
        self.assertEqual(d["classification"], "structurally_nonsingular")
        self.assertEqual(d["determinant_expr"], "x + y")

        text = str(s)
        self.assertIn("StructuralSingularitySummary(", text)
        self.assertIn("classification", text)


class TestStructuralSingularitySummary(unittest.TestCase):
    def test_single_step_is_singular_by_pattern(self) -> None:
        G = make_single_step_crn()

        summary = dynamics.structural_singularity_summary(G, max_exact_size=5)

        self.assertEqual(summary.n_species, 2)
        self.assertEqual(summary.structural_rank, 1)
        self.assertFalse(summary.has_perfect_matching)
        self.assertTrue(summary.pattern_singular)
        self.assertTrue(summary.determinant_checked)
        self.assertTrue(summary.determinant_is_zero)
        self.assertEqual(summary.classification, "singular_by_pattern")

    def test_cycle_is_pattern_nonsingular_but_exactly_singular(self) -> None:
        G = make_cycle_crn()

        summary = dynamics.structural_singularity_summary(G, max_exact_size=5)

        self.assertEqual(summary.n_species, 3)
        self.assertEqual(summary.structural_rank, 3)
        self.assertTrue(summary.has_perfect_matching)
        self.assertFalse(summary.pattern_singular)
        self.assertTrue(summary.determinant_checked)
        self.assertTrue(summary.determinant_is_zero)
        self.assertEqual(summary.classification, "singular_by_exact_determinant")

    def test_two_decay_is_structurally_nonsingular(self) -> None:
        G = make_two_decay_crn()

        summary = dynamics.structural_singularity_summary(G, max_exact_size=5)

        self.assertEqual(summary.n_species, 2)
        self.assertEqual(summary.structural_rank, 2)
        self.assertTrue(summary.has_perfect_matching)
        self.assertFalse(summary.pattern_singular)
        self.assertTrue(summary.determinant_checked)
        self.assertFalse(summary.determinant_is_zero)
        self.assertEqual(summary.classification, "structurally_nonsingular")

    def test_exact_check_skipped_when_size_too_large(self) -> None:
        G = make_cycle_crn()

        summary = dynamics.structural_singularity_summary(G, max_exact_size=2)

        self.assertEqual(summary.n_species, 3)
        self.assertEqual(summary.structural_rank, 3)
        self.assertTrue(summary.has_perfect_matching)
        self.assertFalse(summary.pattern_singular)
        self.assertFalse(summary.determinant_checked)
        self.assertIsNone(summary.determinant_expr)
        self.assertIsNone(summary.determinant_is_zero)
        self.assertEqual(summary.classification, "pattern_nonsingular_exact_unchecked")


if __name__ == "__main__":
    unittest.main()
