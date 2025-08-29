import unittest
import importlib

try:
    import networkx as nx  # type: ignore

    NX_AVAILABLE = True
except Exception:
    NX_AVAILABLE = False

GRAPH_ANNOTATOR = None
if NX_AVAILABLE:
    try:
        GRAPH_ANNOTATOR = importlib.import_module(
            "synkit.Chem.Molecule.graph_annotator"
        )
    except Exception:
        GRAPH_ANNOTATOR = None


@unittest.skipIf(
    not NX_AVAILABLE or GRAPH_ANNOTATOR is None,
    "networkx or graph_annotator module not available",
)
class TestGraphAnnotator(unittest.TestCase):
    def _build_test_graph_cycle_with_halogen(self):
        G = nx.Graph()
        # cycle nodes (0..5)
        for i in range(6):
            G.add_node(i, element="C", aromatic=True)
        # halogen node
        G.add_node(6, element="Cl", aromatic=False, is_halogen=True)
        # cycle edges (mark conjugated True)
        for i in range(6):
            j = (i + 1) % 6
            G.add_edge(i, j, order=1.5, conjugated=True)
        # attach halogen to node 0
        G.add_edge(0, 6, order=1.0, conjugated=False)
        return G

    def test_basic_annotations_and_values(self):
        G = self._build_test_graph_cycle_with_halogen()
        annot = GRAPH_ANNOTATOR.GraphAnnotator(G, in_place=True)
        annot.annotate()
        for n in range(6):
            self.assertIn("atom_degree", G.nodes[n])
            self.assertGreaterEqual(G.nodes[n]["atom_degree"], 2)
        counts0 = G.nodes[0]["nbr_elements_counts_r1"]
        self.assertEqual(counts0.get("Cl", 0), 1)
        for n in range(6):
            self.assertEqual(G.nodes[n]["conj_component_size"], 6)
        self.assertEqual(G.nodes[0]["dist_to_halogen"], 1)
        self.assertIsInstance(G.nodes[3]["dist_to_halogen"], int)
        for n in range(6):
            sizes = G.nodes[n].get("ring_sizes", [])
            self.assertIsInstance(sizes, list)
            self.assertIn(6, sizes)

    def test_defaults_when_attrs_missing(self):
        G = nx.Graph()
        G.add_node(0, element="C")
        G.add_node(1, element="O")
        G.add_edge(0, 1, order=1.0)
        GRAPH_ANNOTATOR.GraphAnnotator(G, in_place=True).annotate()
        self.assertIn("atom_degree", G.nodes[0])
        self.assertIn("nbr_elements_counts_r1", G.nodes[0])
        self.assertIn("dist_to_halogen", G.nodes[0])
        self.assertIn("conj_component_size", G.nodes[0])
        self.assertIn("ring_sizes", G.nodes[0])


if __name__ == "__main__":
    unittest.main()
