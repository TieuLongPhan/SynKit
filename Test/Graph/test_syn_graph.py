import unittest
import networkx as nx
from synkit.IO.chem_converter import rsmi_to_its
from synkit.Graph.syn_graph import SynGraph
from synkit.Graph.canon_graph import GraphCanonicaliser


class TestSynGraph(unittest.TestCase):
    def setUp(self):
        smart = (
            "[Br:1][CH2:2][c:3]1[cH:4][cH:6][c:7]([Br:8])[cH:9][cH:5]1."
            + "[CH3:10][O:11][CH2:12][CH2:13][O:14][H:15]>>[Br:1][H:15]."
            + "[CH2:2]([c:3]1[cH:4][cH:6][c:7]([Br:8])[cH:9][cH:5]1)"
            + "[O:14][CH2:13][CH2:12][O:11][CH3:10]"
        )
        self.G = rsmi_to_its(smart)
        self.canon = GraphCanonicaliser()
        self.SG = SynGraph(self.G, canonicaliser=self.canon)

    def test_get_nodes_and_edges(self):
        # get_nodes returns tuple pairs
        nodes = list(self.SG.get_nodes(data=True))
        print(nodes)
        self.assertTrue((1, self.G.nodes[1]) in nodes)
        self.assertTrue((15, self.G.nodes[15]) in nodes)

        edges = list(self.SG.get_edges(data=True))
        self.assertTrue((1, 2, self.G.edges[1, 2]) in edges)
        self.assertTrue((14, 15, self.G.edges[14, 15]) in edges)

    def test_raw_and_repr(self):
        # raw property
        self.assertIs(self.SG.raw, self.G)
        # repr contains correct counts and signature prefix
        rep = repr(self.SG)
        self.assertIn("|V|=15", rep)
        self.assertIn("|E|=16", rep)
        self.assertTrue(self.SG.signature[:8] in rep)

    def test_canonical_and_signature(self):
        # signature matches canon.canonical_signature
        expected = self.canon.canonical_signature(self.G)
        self.assertEqual(self.SG.signature, expected)
        # canonical graph has nodes relabelled 1..15
        C = self.SG.canonical
        self.assertEqual(sorted(C.nodes()), list(range(1, 16)))
        # attributes preserved for a sample node
        self.assertEqual(C.nodes[1]["element"], "Br")

    def test_eq_and_hash(self):
        # Create relabelled raw graph should be equal
        G2 = nx.relabel_nodes(self.G, {n: n + 100 for n in self.G.nodes()})
        SG2 = SynGraph(G2, canonicaliser=self.canon)
        self.assertEqual(self.SG, SG2)
        self.assertEqual(hash(self.SG), hash(SG2))
        # in a set only one remains
        s = {self.SG, SG2}
        self.assertEqual(len(s), 1)

    def test_help_runs(self):
        # help should not error
        try:
            self.SG.help()
        except Exception as e:
            self.fail(f"help() raised {e}")


if __name__ == "__main__":
    unittest.main()
