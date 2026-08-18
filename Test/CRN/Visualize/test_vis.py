import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

from synkit.CRN.Structure.syncrn import SynCRN  # noqa: E402
from synkit.CRN.Visualize.palette import ColorPalette, get_palette  # noqa: E402
from synkit.CRN.Visualize.vis import CRNStyle, CRNVis, draw_crn  # noqa: E402

CHAIN = ["A>>B", "B>>C", "C>>D"]


def chain_graph(rxns=CHAIN, **kwargs):
    return SynCRN.from_reaction_strings(list(rxns)).to_digraph(**kwargs)


class TestCRNVisConstruction(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_default_construction_succeeds(self):
        # Regression: the default palette name must actually be registered.
        self.assertIsInstance(CRNVis(graph=chain_graph()).palette, ColorPalette)

    def test_palette_name_is_resolved(self):
        vis = CRNVis(graph=chain_graph(), palette="paper_sage")
        self.assertEqual(vis.palette, get_palette("paper_sage"))

    def test_palette_instance_is_accepted(self):
        palette = get_palette("paper_sage")
        self.assertEqual(CRNVis(graph=chain_graph(), palette=palette).palette, palette)

    def test_palette_overrides_apply(self):
        vis = CRNVis(
            graph=chain_graph(),
            palette="paper_sage",
            palette_overrides={"background": "#010203"},
        )
        self.assertEqual(vis.palette.background, "#010203")

    def test_unknown_palette_raises(self):
        with self.assertRaises(ValueError):
            CRNVis(graph=chain_graph(), palette="does_not_exist")

    def test_requires_a_digraph(self):
        with self.assertRaises(TypeError):
            CRNVis(graph=nx.Graph())

    def test_strict_rejects_unknown_node_kind(self):
        graph = chain_graph()
        graph.add_node("mystery", kind="widget")
        with self.assertRaises(ValueError):
            CRNVis(graph=graph, strict=True)

    def test_non_strict_accepts_unknown_node_kind(self):
        graph = chain_graph()
        graph.add_node("mystery", kind="widget")
        self.assertEqual(len(CRNVis(graph=graph, strict=False).species_nodes), 4)


class TestCRNVisAccessors(unittest.TestCase):
    def setUp(self):
        self.vis = CRNVis(graph=chain_graph())

    def tearDown(self):
        plt.close("all")

    def test_node_partition(self):
        self.assertEqual(len(self.vis.species_nodes), 4)
        self.assertEqual(len(self.vis.rule_nodes), 3)

    def test_is_dag(self):
        self.assertTrue(self.vis.is_dag)
        self.assertFalse(CRNVis(graph=chain_graph(["A>>B", "B>>A"])).is_dag)

    def test_positions_cover_every_node(self):
        self.assertEqual(set(self.vis.positions()), set(self.vis.graph.nodes))

    def test_node_labels_cover_every_node(self):
        self.assertEqual(set(self.vis.node_labels()), set(self.vis.graph.nodes))

    def test_edge_labels_default_to_empty(self):
        self.assertEqual(self.vis.edge_labels(), {})

    def test_edge_labels_role_mode(self):
        labels = self.vis.edge_labels(mode="role")
        self.assertEqual(set(labels.values()), {"reactant", "product"})

    def test_strongly_connected_species_on_a_cycle(self):
        vis = CRNVis(graph=chain_graph(["A>>B", "B>>A"]))
        components = vis.strongly_connected_species()
        self.assertTrue(any(len(component) > 1 for component in components))

    def test_chain_has_no_species_cycle(self):
        self.assertEqual(self.vis.strongly_connected_species(), [])

    def test_subgraph_keeps_the_selected_nodes(self):
        keep = self.vis.species_nodes[:2] + self.vis.rule_nodes[:1]
        sub = self.vis.subgraph(keep)
        self.assertIsInstance(sub, CRNVis)
        self.assertEqual(set(sub.graph.nodes), set(keep))

    def test_subgraph_inherits_style_and_palette(self):
        sub = self.vis.subgraph(self.vis.species_nodes[:2])
        self.assertEqual(sub.palette, self.vis.palette)


class TestCRNVisDraw(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_draw_returns_figure_axes_positions(self):
        fig, ax, pos = CRNVis(graph=chain_graph()).draw()
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        self.assertEqual(set(pos), set(chain_graph().nodes))

    def test_draw_onto_existing_axes(self):
        fig, ax = plt.subplots()
        out_fig, out_ax, _ = CRNVis(graph=chain_graph()).draw(ax=ax)
        self.assertIs(out_ax, ax)
        self.assertIs(out_fig, fig)

    def test_draw_with_title(self):
        _, ax, _ = CRNVis(graph=chain_graph()).draw(title="My network")
        self.assertEqual(ax.get_title(), "My network")

    def test_draw_every_layout(self):
        from synkit.CRN.Visualize.layout import available_layouts

        for name in available_layouts():
            with self.subTest(layout=name):
                fig, _, pos = CRNVis(graph=chain_graph(), layout=name).draw()
                self.assertEqual(set(pos), set(chain_graph().nodes))
                plt.close(fig)

    def test_draw_with_edge_labels(self):
        fig, _, _ = CRNVis(graph=chain_graph()).draw(edge_label_mode="role_stoich")
        self.assertIsNotNone(fig)

    def test_draw_without_legend(self):
        fig, _, _ = CRNVis(graph=chain_graph()).draw(with_legend=False)
        self.assertIsNotNone(fig)

    def test_draw_with_highlights(self):
        vis = CRNVis(graph=chain_graph())
        nodes = vis.species_nodes[:1]
        edges = list(vis.graph.edges)[:1]
        fig, _, _ = vis.draw(highlight_nodes=nodes, highlight_edges=edges)
        self.assertIsNotNone(fig)

    def test_draw_with_cycle_highlighting(self):
        vis = CRNVis(graph=chain_graph(["A>>B", "B>>A"]))
        fig, _, _ = vis.draw(highlight_cycles=True)
        self.assertIsNotNone(fig)

    def test_draw_reaction_kind_graph(self):
        fig, _, pos = CRNVis(graph=chain_graph(reaction_kind="reaction")).draw()
        self.assertEqual(len(pos), 7)

    def test_draw_empty_graph(self):
        fig, _, pos = CRNVis(graph=nx.DiGraph()).draw()
        self.assertEqual(pos, {})

    def test_draw_crn_helper_matches_class(self):
        fig, ax, pos = draw_crn(chain_graph())
        self.assertEqual(set(pos), set(chain_graph().nodes))

    def test_draw_with_stoichiometric_coefficients(self):
        fig, _, _ = draw_crn(chain_graph(["2A>>3B"]))
        self.assertIsNotNone(fig)


class TestCRNVisSave(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_save_writes_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = CRNVis(graph=chain_graph()).save(Path(tmp) / "crn.png", dpi=72)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)

    def test_save_accepts_draw_kwargs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = CRNVis(graph=chain_graph()).save(
                Path(tmp) / "crn.png", dpi=72, title="t", with_legend=False
            )
            self.assertTrue(out.exists())


class TestCRNStyle(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_defaults_are_usable(self):
        style = CRNStyle()
        self.assertEqual(len(style.figsize), 2)

    def test_custom_style_is_applied(self):
        style = CRNStyle(figsize=(4.0, 3.0))
        fig, _, _ = CRNVis(graph=chain_graph(), style=style).draw()
        self.assertAlmostEqual(fig.get_figwidth(), 4.0)
        self.assertAlmostEqual(fig.get_figheight(), 3.0)


if __name__ == "__main__":
    unittest.main()
