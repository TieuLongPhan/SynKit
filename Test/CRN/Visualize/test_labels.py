import unittest

from synkit.CRN.Structure.syncrn import SynCRN
from synkit.CRN.Visualize.labels import (
    _truncate,
    _wrap,
    build_edge_labels,
    build_node_labels,
)
from synkit.CRN.Visualize.validation import validate_crn_graph


def graph_and_info(rxns=("A>>B", "B>>C")):
    graph = SynCRN.from_reaction_strings(list(rxns)).to_digraph()
    return graph, validate_crn_graph(graph)


class TestTextHelpers(unittest.TestCase):
    def test_truncate_shortens_with_ellipsis(self):
        self.assertEqual(_truncate("abcdefgh", 4), "a...")

    def test_truncate_leaves_short_text(self):
        self.assertEqual(_truncate("ab", 4), "ab")

    def test_truncate_below_ellipsis_width_hard_cuts(self):
        self.assertEqual(_truncate("abcdefgh", 2), "ab")

    def test_truncate_without_limit(self):
        self.assertEqual(_truncate("abcdefgh", None), "abcdefgh")

    def test_wrap_inserts_newlines(self):
        self.assertIn("\n", _wrap("aaaa bbbb cccc", 5))

    def test_wrap_without_limit(self):
        self.assertEqual(_wrap("aaaa bbbb", None), "aaaa bbbb")


class TestBuildNodeLabels(unittest.TestCase):
    def test_labels_every_node(self):
        graph, info = graph_and_info()
        labels = build_node_labels(
            graph,
            species_nodes=info.species_nodes,
            rule_nodes=info.rule_nodes,
        )
        self.assertEqual(set(labels), set(graph.nodes))

    def test_species_labels_use_the_label_attribute(self):
        graph, info = graph_and_info()
        labels = build_node_labels(
            graph,
            species_nodes=info.species_nodes,
            rule_nodes=info.rule_nodes,
        )
        self.assertEqual(
            {labels[n] for n in info.species_nodes}, {"A", "B", "C"}
        )

    def test_hiding_labels_falls_back_to_node_ids(self):
        graph, info = graph_and_info()
        labels = build_node_labels(
            graph,
            species_nodes=info.species_nodes,
            rule_nodes=info.rule_nodes,
            show_species_labels=False,
            show_rule_labels=False,
        )
        self.assertEqual(
            {labels[n] for n in info.species_nodes}, {"s_1", "s_2", "s_3"}
        )
        self.assertEqual({labels[n] for n in info.rule_nodes}, {"r_1", "r_2"})

    def test_max_chars_is_respected(self):
        graph, info = graph_and_info(["ABCDEFGHIJ>>B"])
        labels = build_node_labels(
            graph,
            species_nodes=info.species_nodes,
            rule_nodes=info.rule_nodes,
            max_chars=4,
        )
        self.assertTrue(all(len(text) <= 4 for text in labels.values()))

    def test_unsupported_rule_label_mode_raises(self):
        graph, info = graph_and_info()
        with self.assertRaises(ValueError):
            build_node_labels(
                graph,
                species_nodes=info.species_nodes,
                rule_nodes=info.rule_nodes,
                rule_label="hieroglyph",
            )


class TestBuildEdgeLabels(unittest.TestCase):
    def test_none_mode_yields_nothing(self):
        graph, _ = graph_and_info()
        self.assertEqual(build_edge_labels(graph, mode="none"), {})

    def test_role_mode(self):
        graph, _ = graph_and_info()
        labels = build_edge_labels(graph, mode="role")
        self.assertEqual(set(labels.values()), {"reactant", "product"})

    def test_stoich_mode(self):
        graph, _ = graph_and_info(["2A>>B"])
        labels = build_edge_labels(graph, mode="stoich")
        self.assertIn("2", set(labels.values()))

    def test_role_stoich_mode(self):
        graph, _ = graph_and_info(["2A>>B"])
        labels = build_edge_labels(graph, mode="role_stoich")
        self.assertIn("reactant:2", set(labels.values()))

    def test_labels_cover_every_edge(self):
        graph, _ = graph_and_info()
        labels = build_edge_labels(graph, mode="role")
        self.assertEqual(len(labels), graph.number_of_edges())

    def test_unsupported_mode_raises(self):
        graph, _ = graph_and_info()
        with self.assertRaises(ValueError):
            build_edge_labels(graph, mode="semaphore")


if __name__ == "__main__":
    unittest.main()
