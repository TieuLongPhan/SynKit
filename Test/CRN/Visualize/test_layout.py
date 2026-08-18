import unittest

import networkx as nx
import numpy as np

from synkit.CRN.Structure.syncrn import SynCRN
from synkit.CRN.Visualize.layout import (
    available_layouts,
    choose_auto_layout,
    compute_layout,
)
from synkit.CRN.Visualize.validation import validate_crn_graph


def crn_graph(rxns):
    graph = SynCRN.from_reaction_strings(list(rxns)).to_digraph()
    return graph, validate_crn_graph(graph)


CHAIN = ["A>>B", "B>>C", "C>>D"]


class TestAvailableLayouts(unittest.TestCase):
    def test_includes_auto(self):
        self.assertIn("auto", available_layouts())

    def test_includes_the_domain_layouts(self):
        names = set(available_layouts())
        self.assertTrue({"step", "bipartite", "spring"} <= names)


class TestComputeLayout(unittest.TestCase):
    def test_every_named_layout_positions_every_node(self):
        graph, info = crn_graph(CHAIN)
        for name in available_layouts():
            with self.subTest(layout=name):
                pos = compute_layout(
                    graph,
                    species_nodes=info.species_nodes,
                    rule_nodes=info.rule_nodes,
                    layout=name,
                )
                self.assertEqual(set(pos), set(graph.nodes))

    def test_positions_are_two_dimensional_and_finite(self):
        graph, info = crn_graph(CHAIN)
        pos = compute_layout(
            graph,
            species_nodes=info.species_nodes,
            rule_nodes=info.rule_nodes,
            layout="step",
        )
        for node, coords in pos.items():
            with self.subTest(node=node):
                self.assertEqual(len(coords), 2)
                self.assertTrue(all(isinstance(float(c), float) for c in coords))

    def test_unknown_layout_raises_and_lists_options(self):
        graph, info = crn_graph(CHAIN)
        with self.assertRaises(ValueError) as ctx:
            compute_layout(
                graph,
                species_nodes=info.species_nodes,
                rule_nodes=info.rule_nodes,
                layout="hyperbolic_origami",
            )
        self.assertIn("step", str(ctx.exception))

    def test_stochastic_layouts_are_seed_reproducible(self):
        graph, info = crn_graph(CHAIN)
        kwargs = dict(
            species_nodes=info.species_nodes,
            rule_nodes=info.rule_nodes,
            layout="spring",
            seed=7,
        )
        first = compute_layout(graph, **kwargs)
        second = compute_layout(graph, **kwargs)
        self.assertEqual(set(first), set(second))
        for node in first:
            with self.subTest(node=node):
                np.testing.assert_allclose(first[node], second[node])

    def test_bipartite_orientation_transposes(self):
        graph, info = crn_graph(CHAIN)
        vertical = compute_layout(
            graph,
            species_nodes=info.species_nodes,
            rule_nodes=info.rule_nodes,
            layout="bipartite",
            orientation="vertical",
        )
        horizontal = compute_layout(
            graph,
            species_nodes=info.species_nodes,
            rule_nodes=info.rule_nodes,
            layout="bipartite",
            orientation="horizontal",
        )
        self.assertNotEqual(vertical, horizontal)

    def test_empty_graph_yields_empty_positions(self):
        self.assertEqual(
            compute_layout(nx.DiGraph(), species_nodes=[], rule_nodes=[], layout="step"),
            {},
        )

    def test_spacing_scales_coordinates(self):
        graph, info = crn_graph(CHAIN)
        tight = compute_layout(
            graph,
            species_nodes=info.species_nodes,
            rule_nodes=info.rule_nodes,
            layout="step",
            layer_spacing=1.0,
        )
        loose = compute_layout(
            graph,
            species_nodes=info.species_nodes,
            rule_nodes=info.rule_nodes,
            layout="step",
            layer_spacing=10.0,
        )

        def layer_spread(pos):
            xs = [x for x, _ in pos.values()]
            return max(xs) - min(xs)

        self.assertGreater(layer_spread(loose), layer_spread(tight))


class TestChooseAutoLayout(unittest.TestCase):
    def test_returns_a_registered_layout_name(self):
        graph, info = crn_graph(CHAIN)
        chosen = choose_auto_layout(
            graph,
            species_nodes=info.species_nodes,
            rule_nodes=info.rule_nodes,
        )
        self.assertIn(chosen, available_layouts())
        self.assertNotEqual(chosen, "auto")

    def test_auto_resolves_to_a_usable_layout(self):
        graph, info = crn_graph(CHAIN)
        pos = compute_layout(
            graph,
            species_nodes=info.species_nodes,
            rule_nodes=info.rule_nodes,
            layout="auto",
        )
        self.assertEqual(set(pos), set(graph.nodes))

    def test_cyclic_network_is_handled(self):
        graph, info = crn_graph(["A>>B", "B>>A"])
        pos = compute_layout(
            graph,
            species_nodes=info.species_nodes,
            rule_nodes=info.rule_nodes,
            layout="auto",
        )
        self.assertEqual(set(pos), set(graph.nodes))


if __name__ == "__main__":
    unittest.main()
