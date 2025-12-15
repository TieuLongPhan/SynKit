import unittest

import networkx as nx

from synkit.CRN.Hypergraph.hypergraph import CRNHyperGraph
from synkit.CRN.Hypergraph.conversion import (
    rxns_to_hypergraph,
    hypergraph_to_species_graph,
)
from synkit.CRN.Hypergraph.backend import _CRNGraphBackend


class TestCRNGraphBackend(unittest.TestCase):
    def _example_hypergraph(self) -> CRNHyperGraph:
        """Helper to build the example hypergraph."""
        rxns = [
            "2A>>B+3C",
            "2B>>D",
            "D+C>>E",
            "D+3C>>F",
            "E+2C>>F",
            "3B>>G",
            "G+3C>>H",
            "B+C>>I",
            "I+C>>J",
            "E+I>>K",
            "K+C>>H",
        ]
        H = rxns_to_hypergraph(rxns)
        mol_map = {
            "A": "CH4",
            "B": "C2H2",
            "C": "H2",
            "D": "C4H4",
            "E": "C4H6",
            "F": "C4H10",
            "G": "C6H6",
            "H": "C6H12",
            "I": "C2H4",
            "J": "C2H6",
            "K": "C6H10",
        }
        H.set_mol_map(mapping=mol_map, strict=False)
        return H

    # ------------------------------------------------------------------
    # Species graph backend (include_rule = False)
    # ------------------------------------------------------------------

    def test_species_graph_backend_default_type_and_structure(self):
        H = self._example_hypergraph()
        backend = _CRNGraphBackend(H)  # include_rule=False by default

        # Before access, internal cache should be empty
        self.assertIsNone(backend._G)
        self.assertIsNone(backend._graph_type)

        G = backend.G
        self.assertIsInstance(G, nx.DiGraph)
        self.assertEqual(backend.graph_type, "species")

        # After access, cache should be populated
        self.assertIsNotNone(backend._G)
        self.assertIsNotNone(backend._graph_type)

        # Should match hypergraph_to_species_graph
        S_ref = hypergraph_to_species_graph(H)
        self.assertEqual(set(G.nodes), set(S_ref.nodes))
        self.assertEqual(set(G.edges), set(S_ref.edges))

        # Node attributes (kind/label) should exist
        for n, d in G.nodes(data=True):
            self.assertEqual(d.get("kind"), "species")
            self.assertEqual(d.get("label"), n)

    def test_species_graph_backend_caching_same_object(self):
        H = self._example_hypergraph()
        backend = _CRNGraphBackend(H)

        G1 = backend.G
        G2 = backend.G
        self.assertIs(G1, G2)  # cached object reused
        self.assertEqual(backend.graph_type, "species")

    def test_species_graph_backend_on_empty_hypergraph(self):
        H = CRNHyperGraph()
        backend = _CRNGraphBackend(H)

        G = backend.G
        self.assertEqual(G.number_of_nodes(), 0)
        self.assertEqual(G.number_of_edges(), 0)
        self.assertEqual(backend.graph_type, "species")

    # ------------------------------------------------------------------
    # Bipartite backend (include_rule = True)
    # ------------------------------------------------------------------

    def test_bipartite_backend_string_ids_with_stoich(self):
        H = self._example_hypergraph()
        backend = _CRNGraphBackend(
            H,
            include_rule=True,
            integer_ids=False,
            include_stoich=True,
        )

        G = backend.G
        self.assertIsInstance(G, nx.DiGraph)
        self.assertEqual(backend.graph_type, "bipartite")

        kinds = nx.get_node_attributes(G, "kind")
        self.assertIn("species", kinds.values())
        self.assertIn("reaction", kinds.values())

        # Species nodes should use "S:" prefix by default and label = species
        species_nodes = [n for n, k in kinds.items() if k == "species"]
        self.assertTrue(species_nodes)
        for n in species_nodes:
            self.assertIsInstance(n, str)
            label = G.nodes[n].get("label")
            self.assertIn(label, self._example_hypergraph().species)

        # At least one edge should carry stoich attribute
        has_stoich = any("stoich" in attrs for _, _, attrs in G.edges(data=True))
        self.assertTrue(has_stoich)

    def test_bipartite_backend_integer_ids_all_int_nodes(self):
        H = self._example_hypergraph()
        backend = _CRNGraphBackend(
            H,
            include_rule=True,
            integer_ids=True,
            include_stoich=True,
        )

        G = backend.G
        self.assertEqual(backend.graph_type, "bipartite")

        # All nodes should be integers
        self.assertTrue(all(isinstance(n, int) for n in G.nodes()))

        # Node kinds still distinguish species vs reaction
        kinds = nx.get_node_attributes(G, "kind")
        self.assertIn("species", kinds.values())
        self.assertIn("reaction", kinds.values())

        # Edges should still have stoich attr
        self.assertTrue(all("stoich" in attrs for _, _, attrs in G.edges(data=True)))

    def test_bipartite_backend_without_stoich_removes_stoich_attr(self):
        H = self._example_hypergraph()
        backend = _CRNGraphBackend(
            H,
            include_rule=True,
            integer_ids=False,
            include_stoich=False,
        )

        G = backend.G
        self.assertEqual(backend.graph_type, "bipartite")

        # Edges should NOT have 'stoich' but should still have 'role'
        for _, _, attrs in G.edges(data=True):
            self.assertNotIn("stoich", attrs)
            self.assertIn("role", attrs)

    def test_bipartite_backend_on_empty_hypergraph(self):
        H = CRNHyperGraph()
        backend = _CRNGraphBackend(
            H,
            include_rule=True,
            integer_ids=False,
            include_stoich=True,
        )

        G = backend.G
        self.assertEqual(G.number_of_nodes(), 0)
        self.assertEqual(G.number_of_edges(), 0)
        self.assertEqual(backend.graph_type, "bipartite")


if __name__ == "__main__":
    unittest.main()
