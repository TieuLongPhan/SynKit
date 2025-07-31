import unittest
import networkx as nx

from synkit.IO.combinatorial.smarts_to_graph import SMARTSToGraph


class TestSMARTSToGraph(unittest.TestCase):

    def setUp(self):
        self.stg = SMARTSToGraph()

    def test_smarts_to_graph_simple(self):
        g = self.stg.smarts_to_graph("[C:1]-[O:2]")
        self.assertIsInstance(g, nx.Graph)
        self.assertEqual(set(g.nodes), {1, 2})
        self.assertEqual(g.nodes[1]["element"], "C")
        self.assertEqual(g.nodes[2]["element"], "O")
        self.assertIsNone(g.nodes[1]["constraint"])

    def test_smarts_to_graph_constraint(self):
        g = self.stg.smarts_to_graph("[C,N,O:1]-[N:2]")
        # Node 1 should be placeholder
        self.assertEqual(g.nodes[1]["element"], "*")
        self.assertIsInstance(g.nodes[1]["constraint"], list)
        self.assertIn("C", g.nodes[1]["constraint"])
        self.assertEqual(g.nodes[2]["element"], "N")
        self.assertIsNone(g.nodes[2]["constraint"])

    def test_smarts_to_graph_hcount(self):
        g = self.stg.smarts_to_graph("[CH3:1]-[O:2]")
        # For SMARTS as written, RDKit returns 0 hydrogens for both
        self.assertEqual(g.nodes[1]["hcount"], 0)
        self.assertEqual(g.nodes[2]["hcount"], 0)

    def test_invalid_smarts(self):
        with self.assertRaises(ValueError):
            self.stg.smarts_to_graph("[C:1]-[N")

    def test_missing_atom_map(self):
        with self.assertRaises(ValueError):
            self.stg.smarts_to_graph("[C]-[O:2]")

    def test_rxn_smarts_to_graphs(self):
        rxn = (
            "[H+:6].[C:7](-[O:8](-[H:12]))(-[C,N,O,P,S:9])(-[C,N,O,P,S:10])(-[H:11])."
            "[C:2](-[S:4](-[C,N,O,P,S:5]))(-[C,N,O,P,S:1])(=[O:3])>>"
            "[S:4](-[H:6])(-[C,N,O,P,S:5]).[H+:12]."
            "[C:7](-[O:8](-[C:2](-[C,N,O,P,S:1])(=[O:3])))(-[C,N,O,P,S:9])(-[C,N,O,P,S:10])(-[H:11])"
        )
        g_react, _ = self.stg.rxn_smarts_to_graphs(rxn)

        # These are the atom_map indices that should have constraint (from SMARTS [C,N,O,P,S:idx])
        expected_constraint_nodes = {1, 5, 9, 10}
        for idx in expected_constraint_nodes:
            self.assertIn(idx, g_react.nodes)
            self.assertIsNotNone(
                g_react.nodes[idx]["constraint"],
                f"Node {idx} should have a constraint list but does not",
            )
            self.assertEqual(
                set(g_react.nodes[idx]["constraint"]),
                {"C", "N", "O", "P", "S"},
                f"Node {idx} has incorrect constraint list",
            )
        # All other nodes should NOT have a constraint
        for idx in set(g_react.nodes) - expected_constraint_nodes:
            self.assertIsNone(
                g_react.nodes[idx]["constraint"],
                f"Node {idx} should NOT have a constraint list",
            )

    def test_rxn_separator(self):
        with self.assertRaises(ValueError):
            self.stg.rxn_smarts_to_graphs("[C:1]-[O:2]")  # no '>>'

    def test_repr_and_describe(self):
        r = repr(self.stg)
        self.assertIn("placeholders", r)
        desc = self.stg.describe()
        self.assertIn("smarts_to_graph", desc)
        self.assertIn("rxn_smarts_to_graphs", desc)


if __name__ == "__main__":
    unittest.main()
