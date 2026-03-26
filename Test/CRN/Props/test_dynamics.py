from __future__ import annotations

import unittest

import networkx as nx

from synkit.CRN.Props.dynamics import (
    jacobian_sign_pattern,
    jacobian_sparsity,
    species_influence_graph,
    structural_singularity_summary,
    symbolic_jacobian,
    symbolic_reactivity_matrix,
)


def _add_species(G: nx.DiGraph, name: str) -> None:
    G.add_node(
        name,
        kind="species",
        bipartite=0,
        label=name,
    )


def _add_reaction(G: nx.DiGraph, name: str) -> None:
    G.add_node(
        name,
        kind="reaction",
        bipartite=1,
        label=name,
    )


def _add_reactant_edge(
    G: nx.DiGraph, species: str, reaction: str, stoich: float = 1.0
) -> None:
    G.add_edge(
        species,
        reaction,
        role="reactant",
        stoich=stoich,
    )


def _add_product_edge(
    G: nx.DiGraph, reaction: str, species: str, stoich: float = 1.0
) -> None:
    G.add_edge(
        reaction,
        species,
        role="product",
        stoich=stoich,
    )


def make_cycle_crn() -> nx.DiGraph:
    """
    A -> B -> C -> A
    """
    G = nx.DiGraph()
    for s in ("A", "B", "C"):
        _add_species(G, s)
    for r in ("r1", "r2", "r3"):
        _add_reaction(G, r)

    _add_reactant_edge(G, "A", "r1")
    _add_product_edge(G, "r1", "B")

    _add_reactant_edge(G, "B", "r2")
    _add_product_edge(G, "r2", "C")

    _add_reactant_edge(G, "C", "r3")
    _add_product_edge(G, "r3", "A")

    return G


def make_single_step_crn() -> nx.DiGraph:
    """
    A -> B
    """
    G = nx.DiGraph()
    for s in ("A", "B"):
        _add_species(G, s)
    _add_reaction(G, "r1")

    _add_reactant_edge(G, "A", "r1")
    _add_product_edge(G, "r1", "B")
    return G


def make_two_decay_crn() -> nx.DiGraph:
    """
    A -> ∅
    B -> ∅
    """
    G = nx.DiGraph()
    for s in ("A", "B"):
        _add_species(G, s)
    for r in ("dA", "dB"):
        _add_reaction(G, r)

    _add_reactant_edge(G, "A", "dA")
    _add_reactant_edge(G, "B", "dB")
    return G


class TestDynamicsMatrices(unittest.TestCase):
    def test_symbolic_reactivity_matrix_cycle(self) -> None:
        G = make_cycle_crn()

        species_order, reaction_order, R = symbolic_reactivity_matrix(G)

        iA = species_order.index("A")
        iB = species_order.index("B")
        iC = species_order.index("C")

        jr1 = reaction_order.index("r1")
        jr2 = reaction_order.index("r2")
        jr3 = reaction_order.index("r3")

        self.assertNotEqual(R[jr1, iA], 0)
        self.assertEqual(R[jr1, iB], 0)
        self.assertEqual(R[jr1, iC], 0)

        self.assertEqual(R[jr2, iA], 0)
        self.assertNotEqual(R[jr2, iB], 0)
        self.assertEqual(R[jr2, iC], 0)

        self.assertEqual(R[jr3, iA], 0)
        self.assertEqual(R[jr3, iB], 0)
        self.assertNotEqual(R[jr3, iC], 0)

    def test_symbolic_jacobian_cycle_shape_and_entries(self) -> None:
        G = make_cycle_crn()

        species_order, _reaction_order, J = symbolic_jacobian(G)
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

        species_order, P = jacobian_sign_pattern(G)
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

    def test_jacobian_sparsity_matches_sign_pattern(self) -> None:
        G = make_cycle_crn()

        species_order_1, P = jacobian_sign_pattern(G)
        species_order_2, A = jacobian_sparsity(G)

        self.assertEqual(species_order_1, species_order_2)

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                self.assertEqual(bool(A[i, j]), P[i, j] != "0")


class TestSpeciesInfluenceGraph(unittest.TestCase):
    def test_species_influence_graph_cycle(self) -> None:
        G = make_cycle_crn()

        Gi = species_influence_graph(G)

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


class TestStructuralSingularitySummary(unittest.TestCase):
    def test_single_step_is_singular_by_pattern(self) -> None:
        G = make_single_step_crn()

        summary = structural_singularity_summary(G, max_exact_size=5)

        self.assertEqual(summary.n_species, 2)
        self.assertEqual(summary.structural_rank, 1)
        self.assertFalse(summary.has_perfect_matching)
        self.assertTrue(summary.pattern_singular)
        self.assertTrue(summary.determinant_checked)
        self.assertTrue(summary.determinant_is_zero)
        self.assertEqual(summary.classification, "singular_by_pattern")

    def test_cycle_is_pattern_nonsingular_but_exactly_singular(self) -> None:
        G = make_cycle_crn()

        summary = structural_singularity_summary(G, max_exact_size=5)

        self.assertEqual(summary.n_species, 3)
        self.assertEqual(summary.structural_rank, 3)
        self.assertTrue(summary.has_perfect_matching)
        self.assertFalse(summary.pattern_singular)
        self.assertTrue(summary.determinant_checked)
        self.assertTrue(summary.determinant_is_zero)
        self.assertEqual(summary.classification, "singular_by_exact_determinant")

    def test_two_decay_is_structurally_nonsingular(self) -> None:
        G = make_two_decay_crn()

        summary = structural_singularity_summary(G, max_exact_size=5)

        self.assertEqual(summary.n_species, 2)
        self.assertEqual(summary.structural_rank, 2)
        self.assertTrue(summary.has_perfect_matching)
        self.assertFalse(summary.pattern_singular)
        self.assertTrue(summary.determinant_checked)
        self.assertFalse(summary.determinant_is_zero)
        self.assertEqual(summary.classification, "structurally_nonsingular")


if __name__ == "__main__":
    unittest.main()
