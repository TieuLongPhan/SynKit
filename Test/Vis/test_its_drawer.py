import unittest

import matplotlib

matplotlib.use("Agg")

from synkit.IO import rsmi_to_its  # noqa: E402
from synkit.Vis.its import (  # noqa: E402
    draw_its_from_rsmi,
    draw_its_graph,
    draw_its_only,
)


class TestITSDrawer(unittest.TestCase):
    rsmi = (
        "[Cl:1][Cl:2].[H:9][c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1"
        ">>"
        "[Cl:1][H:9].[Cl:2][c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1"
    )

    def test_draw_tuple_its_from_rsmi(self):
        fig, axes = draw_its_from_rsmi(
            self.rsmi,
            format="tuple",
            core=False,
            title="chlorination ITS",
        )

        self.assertIs(fig, axes[0].figure)
        self.assertEqual(len(axes), 1)
        self.assertEqual(axes[0].get_title(), "chlorination ITS")

    def test_draw_tuple_its_graph(self):
        its = rsmi_to_its(self.rsmi, core=False, format="tuple")

        fig, axes = draw_its_graph(its, title="tuple ITS")

        self.assertIs(fig, axes[0].figure)
        self.assertEqual(len(axes), 1)
        self.assertEqual(axes[0].get_title(), "tuple ITS")

    def test_draw_its_only_can_show_changed_edge_labels(self):
        its = rsmi_to_its(self.rsmi, core=False, format="tuple")

        ax = draw_its_only(
            its,
            title="pretty ITS",
            show_edge_labels=True,
            edge_label_mode="kekule",
        )

        self.assertEqual(ax.get_title(), "pretty ITS")

    def test_draw_its_only_supports_sigma_pi_labels(self):
        its = rsmi_to_its(self.rsmi, core=False, format="tuple")

        ax = draw_its_only(
            its,
            title="sigma/pi ITS",
            show_edge_labels=True,
            edge_label_mode="sigma_pi",
        )

        self.assertEqual(ax.get_title(), "sigma/pi ITS")

    def test_sigma_pi_labels_only_include_changed_components(self):
        from synkit.Vis.its.drawer import _its_display_graph

        its = rsmi_to_its(self.rsmi, core=False, format="tuple")
        display = _its_display_graph(its)
        labels = [
            attrs["its_label_sigma_pi"]
            for _, _, attrs in display.edges(data=True)
            if attrs["its_state"] != "unchanged"
        ]

        self.assertTrue(labels)
        self.assertTrue(all(label.startswith("σ") for label in labels))
        self.assertTrue(all("π" not in label for label in labels))

    def test_electron_labels_capture_charge_and_lone_pair_changes_separately(self):
        from synkit.Vis.its.drawer import _its_display_graph

        rsmi = "[CH3:1][Cl:2].[NH3:3]>>[CH3:1][NH3+:3].[Cl-:2]"
        its = rsmi_to_its(rsmi, core=False, format="tuple")
        display = _its_display_graph(its)

        self.assertEqual(display.nodes[2]["its_electron_label_charge"], "q0→-1")
        self.assertEqual(display.nodes[2]["its_electron_label_lone_pair"], "λ3→4")
        self.assertEqual(display.nodes[3]["its_electron_label_charge"], "q0→+1")
        self.assertEqual(display.nodes[3]["its_electron_label_lone_pair"], "λ1→0")

    def test_draw_its_only_can_show_electron_labels(self):
        rsmi = "[CH3:1][Cl:2].[NH3:3]>>[CH3:1][NH3+:3].[Cl-:2]"
        its = rsmi_to_its(rsmi, core=False, format="tuple")

        ax = draw_its_only(
            its,
            title="SN2 ITS",
            show_electron_labels=True,
            electron_label_mode="lone_pair",
        )

        self.assertEqual(ax.get_title(), "SN2 ITS")

    def test_invalid_its_label_modes_raise(self):
        its = rsmi_to_its(self.rsmi, core=False, format="tuple")

        with self.assertRaises(ValueError):
            draw_its_only(its, edge_label_mode="verbose")
        with self.assertRaises(ValueError):
            draw_its_only(its, show_electron_labels=True, electron_label_mode="both")

    def test_draw_tuple_its_projection_when_requested(self):
        its = rsmi_to_its(self.rsmi, core=False, format="tuple")

        fig, axes = draw_its_graph(its, title="tuple ITS", projection=True)

        self.assertIs(fig, axes[-1].figure)
        self.assertGreaterEqual(len(axes), 4)
        self.assertEqual(axes[-1].get_title(), "ITS delta")

    def test_draw_legacy_its_graph_without_delta(self):
        its = rsmi_to_its(self.rsmi, core=False, format="typesGH")

        fig, axes = draw_its_graph(its, include_delta_panel=False, projection=True)

        self.assertIs(fig, axes[0].figure)
        self.assertGreaterEqual(len(axes), 4)


if __name__ == "__main__":
    unittest.main()
