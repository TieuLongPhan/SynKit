import unittest

import networkx as nx

from synkit.CRN.Structure.syncrn import SynCRN
from synkit.CRN.Visualize.validation import (
    CRNGraphInfo,
    node_sort_key,
    validate_crn_graph,
)


def linear_graph(reaction_kind="rule"):
    return SynCRN.from_reaction_strings(["A>>B", "B>>C"]).to_digraph(
        reaction_kind=reaction_kind
    )


class TestValidateCrnGraph(unittest.TestCase):
    def test_returns_partitioned_nodes(self):
        info = validate_crn_graph(linear_graph())
        self.assertIsInstance(info, CRNGraphInfo)
        self.assertEqual(len(info.species_nodes), 3)
        self.assertEqual(len(info.rule_nodes), 2)

    def test_detects_dag(self):
        self.assertTrue(validate_crn_graph(linear_graph()).is_dag)

    def test_detects_cycle(self):
        cyclic = SynCRN.from_reaction_strings(["A>>B", "B>>A"]).to_digraph()
        self.assertFalse(validate_crn_graph(cyclic).is_dag)

    def test_requires_digraph(self):
        with self.assertRaises(TypeError):
            validate_crn_graph(nx.Graph())

    def test_strict_rejects_unknown_kind(self):
        graph = linear_graph()
        graph.add_node("mystery", kind="widget")
        with self.assertRaises(ValueError):
            validate_crn_graph(graph, strict=True)

    def test_non_strict_tolerates_unknown_kind(self):
        graph = linear_graph()
        graph.add_node("mystery", kind="widget")
        info = validate_crn_graph(graph, strict=False)
        self.assertEqual(len(info.species_nodes), 3)

    def test_strict_rejects_edge_contradicting_its_role(self):
        graph = linear_graph()
        species = next(
            n for n, d in graph.nodes(data=True) if d.get("kind") == "species"
        )
        graph.add_edge("bogus", species, role="reactant")
        graph.nodes["bogus"]["kind"] = "species"
        with self.assertRaises(ValueError):
            validate_crn_graph(graph, strict=True)

    def test_empty_graph_is_accepted(self):
        info = validate_crn_graph(nx.DiGraph())
        self.assertEqual(info.species_nodes, [])
        self.assertEqual(info.rule_nodes, [])


class TestNodeSortKey(unittest.TestCase):
    """Reaction nodes must sort after species regardless of their spelling."""

    def test_reaction_kind_sorts_like_rule_kind(self):
        as_rule = validate_crn_graph(linear_graph("rule"))
        as_reaction = validate_crn_graph(linear_graph("reaction"))
        self.assertEqual(as_rule.species_nodes, as_reaction.species_nodes)
        self.assertEqual(as_rule.rule_nodes, as_reaction.rule_nodes)

    def test_species_rank_before_reactions(self):
        graph = linear_graph("reaction")
        species = next(
            n for n, d in graph.nodes(data=True) if d.get("kind") == "species"
        )
        reaction = next(
            n for n, d in graph.nodes(data=True) if d.get("kind") == "reaction"
        )
        self.assertLess(
            node_sort_key(graph, species)[0], node_sort_key(graph, reaction)[0]
        )

    def test_is_deterministic(self):
        graph = linear_graph()
        first = validate_crn_graph(graph).rule_nodes
        second = validate_crn_graph(graph).rule_nodes
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
