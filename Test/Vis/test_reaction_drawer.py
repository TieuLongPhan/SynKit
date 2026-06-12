import unittest

import matplotlib

matplotlib.use("Agg")

# flake8: noqa: E402
from synkit.IO.chem_converter import rsmi_to_graph  # noqa: E402
from synkit.Vis.molecule import (  # noqa: E402
    draw_reaction_graph,
    draw_reaction_graphs,
    find_reaction_highlights,
)


class TestReactionDrawer(unittest.TestCase):
    def test_find_reaction_highlights_detects_broken_and_formed_bonds(self):
        rsmi = "[CH3:1][Cl:2].[NH3:3]>>[CH3:1][NH3+:3].[Cl-:2]"
        reactant, product = rsmi_to_graph(
            rsmi,
            drop_non_aam=False,
            use_index_as_atom_map=True,
        )

        highlights = find_reaction_highlights(reactant, product)

        self.assertIn(frozenset({1, 2}), highlights.broken_bonds)
        self.assertIn(frozenset({1, 3}), highlights.formed_bonds)
        self.assertEqual(highlights.changed_atoms, frozenset({1, 2, 3}))

    def test_draw_reaction_graph_from_rsmi(self):
        rsmi = "[CH3:1][Cl:2].[NH3:3]>>[CH3:1][NH3+:3].[Cl-:2]"

        fig, axes = draw_reaction_graph(rsmi, title="SN2")

        self.assertIs(fig, axes[0].figure)
        self.assertEqual(len(axes), 5)

    def test_draw_reaction_graphs_accepts_prebuilt_graphs(self):
        rsmi = "[C:1]=[O:2].[O:3]>>[C:1]([O:2])[O:3]"
        reactant, product = rsmi_to_graph(
            rsmi,
            drop_non_aam=False,
            use_index_as_atom_map=True,
        )

        fig, axes = draw_reaction_graphs(reactant, product, title="addition")

        self.assertIs(fig, axes[-1].figure)
        self.assertEqual(len(axes), 4)


if __name__ == "__main__":
    unittest.main()
