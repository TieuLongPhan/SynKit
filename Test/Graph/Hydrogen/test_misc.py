import unittest

from synkit.Graph.Hyrogen._misc import (
    has_XH,
    h_to_implicit,
    h_to_explicit,
    check_explicit_hydrogen,
    check_hcount_change,
)
from synkit.IO.chem_converter import rsmi_to_its
from synkit.Graph.ITS.its_decompose import its_decompose


class TestHydrogenUtilities(unittest.TestCase):

    def setUp(self):
        # Explicit hydrogen example
        self.rsmi_explicit = "[C:1]=[C:2].[H:3][H:4]>>[C:1]([H:3])[C:2][H:4]"
        self.its_explicit = rsmi_to_its(self.rsmi_explicit)
        self.rc_explicit = self.its_explicit
        self.r_explicit, self.p_explicit = its_decompose(self.rc_explicit)

        # Implicit hydrogen example
        self.rsmi_implicit = "[C:1]=[C:2].[H:3][H:4]>>[CH:1][CH:2]"
        self.its_implicit = rsmi_to_its(self.rsmi_implicit)
        self.rc_implicit = self.its_implicit
        self.r_implicit, self.p_implicit = its_decompose(self.rc_implicit)

    def test_has_XH_explicit(self):
        self.assertTrue(
            has_XH(self.rc_explicit), "Explicit hydrogen bonds should be detected."
        )

    def test_has_XH_implicit(self):
        self.assertFalse(
            has_XH(self.rc_implicit),
            "Implicit hydrogen representation should have no explicit X-H bonds.",
        )

    def test_h_to_implicit_removes_H_nodes(self):
        G = self.rc_explicit
        heavy_atoms = [n for n, d in G.nodes(data=True) if d.get("element") != "H"]
        G_implicit = h_to_implicit(G)
        h_nodes = [n for n, d in G_implicit.nodes(data=True) if d.get("element") == "H"]
        self.assertEqual(len(h_nodes), 0, "All hydrogen nodes should be removed.")
        for n in heavy_atoms:
            self.assertGreaterEqual(
                G_implicit.nodes[n].get("hcount", 0),
                1,
                "Heavy atoms should have hcount ≥ 1.",
            )

    def test_h_to_explicit_adds_H_nodes(self):
        G = self.rc_explicit
        G_implicit = h_to_implicit(G)
        heavy_atoms = [
            n for n, d in G_implicit.nodes(data=True) if d.get("element") != "H"
        ]
        G_expanded = h_to_explicit(G_implicit, heavy_atoms)
        h_nodes = [n for n, d in G_expanded.nodes(data=True) if d.get("element") == "H"]
        self.assertGreaterEqual(len(h_nodes), 2, "Hydrogen nodes should be added back.")

    def test_roundtrip_hydrogen_conversion(self):
        G = self.rc_explicit
        heavy_atoms = [n for n, d in G.nodes(data=True) if d.get("element") != "H"]
        G1 = h_to_implicit(G)
        G2 = h_to_explicit(G1, heavy_atoms)
        h_nodes = [n for n, d in G2.nodes(data=True) if d.get("element") == "H"]
        self.assertGreaterEqual(
            len(h_nodes),
            2,
            "Hydrogens should be recoverable after roundtrip conversion.",
        )

    def test_check_explicit_hydrogen(self):
        count, ids = check_explicit_hydrogen(self.rc_explicit)
        self.assertEqual(count, 2, "Should detect 2 explicit hydrogens.")
        self.assertEqual(len(ids), 2)
        for node_id in ids:
            self.assertEqual(self.rc_explicit.nodes[node_id]["element"], "H")

    def test_check_hcount_change_explicit(self):
        delta = check_hcount_change(self.r_explicit, self.p_explicit)
        self.assertEqual(delta, 2, "Expected hydrogen movement in explicit graph.")

    def test_check_hcount_change_implicit(self):
        delta = check_hcount_change(self.r_implicit, self.p_implicit)
        self.assertEqual(delta, 2, "Expected hydrogen movement in implicit graph.")


if __name__ == "__main__":
    unittest.main()
