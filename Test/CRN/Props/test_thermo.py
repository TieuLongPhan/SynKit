import unittest

import networkx as nx

from synkit.CRN.Structure.conversion import rxns_to_hypergraph
from synkit.CRN.Props import thermo

# ---------------------------------------------------------------------------
# Helpers to build small test networks
# ---------------------------------------------------------------------------


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

        A >> ∅
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
    """
    G = nx.DiGraph()
    for s in ("A", "B"):
        G.add_node(s, kind="species", label=s)
    G.add_node("r1", kind="reaction")
    G.add_node("r2", kind="reaction")

    G.add_edge("A", "r1", role="reactant", stoich=1.0)
    G.add_edge("r1", "B", role="product", stoich=1.0)

    G.add_edge("B", "r2", role="reactant", stoich=1.0)
    G.add_edge("r2", "A", role="product", stoich=1.0)
    return G


# -------------------------------------------------------------------
# Additional complex CRN examples
# -------------------------------------------------------------------


def crn_cycle_with_branch():
    rxns = ["A>>B", "B>>A", "B>>C", "C>>B", "C>>D"]
    return rxns_to_hypergraph(rxns)


def crn_partial_conservation():
    rxns = ["A>>B", "B>>A", "X>>"]
    return rxns_to_hypergraph(rxns)


def crn_two_independent_cycles():
    rxns = ["A>>B", "B>>A", "C>>D", "D>>C"]
    return rxns_to_hypergraph(rxns)


def crn_large_network():
    rxns = [
        "A>>B",
        "B>>C",
        "C>>A",
        "D>>E",
        "E>>F",
        "F>>D",
        "X>>Y",
        "C+E>>Z",
        "Z>>A+F",
    ]
    return rxns_to_hypergraph(rxns)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConservativityAndConsistency(unittest.TestCase):
    def test_is_conservative_simple_A_to_B(self) -> None:
        G = build_simple_A_to_B_graph()
        res = thermo.is_conservative(G)
        self.assertIs(res, True)

    def test_is_conservative_decay_false(self) -> None:
        G = build_decay_graph()
        res = thermo.is_conservative(G)
        self.assertIs(res, False)

    def test_is_consistent_cycle_true(self) -> None:
        G = build_cycle_graph()
        res = thermo.is_consistent(G)
        self.assertIs(res, True)

    def test_is_consistent_decay_false(self) -> None:
        G = build_decay_graph()
        res = thermo.is_consistent(G)
        self.assertIs(res, False)


class TestThermoComplex(unittest.TestCase):
    def test_cycle_with_branch(self):
        H = crn_cycle_with_branch()
        summary = thermo.compute_thermo_summary(H)

        self.assertTrue(summary.irreversible_futile_cycles)
        self.assertTrue(summary.conservative)

    def test_partial_conservation(self):
        H = crn_partial_conservation()
        summary = thermo.compute_thermo_summary(H)

        self.assertFalse(summary.conservative)
        self.assertTrue(summary.irreversible_futile_cycles)

    def test_two_independent_cycles(self):
        H = crn_two_independent_cycles()
        summary = thermo.compute_thermo_summary(H)

        self.assertTrue(summary.irreversible_futile_cycles)
        self.assertTrue(summary.conservative)
        self.assertIsNotNone(summary.example_conservation_law)

    def test_large_network(self):
        H = crn_large_network()
        summary = thermo.compute_thermo_summary(H)

        self.assertIsInstance(summary.conservative, bool)
        self.assertIsInstance(summary.irreversible_futile_cycles, bool)

        self.assertTrue(summary.irreversible_futile_cycles)

        if summary.conservative:
            self.assertIsNotNone(summary.example_conservation_law)
            self.assertGreater(summary.example_conservation_law.sum(), 0)
        else:
            self.assertIsNone(summary.example_conservation_law)


if __name__ == "__main__":
    unittest.main()
