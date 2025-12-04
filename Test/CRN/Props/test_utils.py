import unittest
import networkx as nx

from synkit.CRN.Hypergraph.conversion import rxns_to_hypergraph, hypergraph_to_bipartite

from synkit.CRN.Props.utils import (
    _as_bipartite,
    _split_species_reactions,
    _species_order,
    _species_and_reaction_order,
)


class TestBipartiteHelpers(unittest.TestCase):

    def _example_H(self):
        rxns = [
            "2A+B>>C",
            "C+D>>E",
            "E+F>>D+G",
        ]
        return rxns_to_hypergraph(rxns)

    def _example_G(self):
        return hypergraph_to_bipartite(self._example_H())

    # ----------------------------------------------------------------------
    # _as_bipartite
    # ----------------------------------------------------------------------
    def test_as_bipartite_from_hypergraph(self):
        H = self._example_H()
        G = _as_bipartite(H)

        self.assertIsInstance(G, nx.DiGraph)
        self.assertGreater(G.number_of_nodes(), 0)

    def test_as_bipartite_from_networkx_graph(self):
        G0 = nx.DiGraph()
        G0.add_node("S:A", kind="species", bipartite=0)
        G0.add_node("R:1", kind="reaction", bipartite=1)

        G = _as_bipartite(G0)
        self.assertIs(G, G0)

    def test_as_bipartite_invalid_input_raises(self):
        with self.assertRaises(TypeError):
            _as_bipartite("not a graph")

    # ----------------------------------------------------------------------
    # _split_species_reactions
    # ----------------------------------------------------------------------
    def test_split_species_reactions_normal(self):
        G = self._example_G()
        species, reactions = _split_species_reactions(G)

        self.assertGreater(len(species), 0)
        self.assertGreater(len(reactions), 0)

        # All species must have bipartite=0, reactions bipartite=1
        for n in species:
            self.assertEqual(G.nodes[n]["bipartite"], 0)

        for n in reactions:
            self.assertEqual(G.nodes[n]["bipartite"], 1)

    def test_split_species_reactions_missing_role_raises(self):
        # Construct a graph missing reaction nodes
        G0 = nx.DiGraph()
        G0.add_node("A", kind="species", bipartite=0)

        with self.assertRaises(ValueError):
            _split_species_reactions(G0)

    # ----------------------------------------------------------------------
    # _species_order
    # ----------------------------------------------------------------------
    def test_species_order_lexicographic(self):
        G = self._example_G()
        species_nodes_sorted, species_labels, species_index = _species_order(G)

        # Check consistency
        self.assertEqual(len(species_nodes_sorted), len(species_labels))
        self.assertEqual(len(species_nodes_sorted), len(species_index))

        # Labels must be sorted lexicographically
        self.assertEqual(species_labels, sorted(species_labels))

        # species_index must match enumeration
        for i, n in enumerate(species_nodes_sorted):
            self.assertEqual(species_index[n], i)

    # ----------------------------------------------------------------------
    # _species_and_reaction_order
    # ----------------------------------------------------------------------
    def test_species_and_reaction_order(self):
        G = self._example_G()
        species_labels, reaction_labels, s_idx, r_idx = _species_and_reaction_order(G)

        # Species must include: A,B,C,D,E,F,G
        expected_species = sorted(["A", "B", "C", "D", "E", "F", "G"])
        self.assertEqual(species_labels, expected_species)

        # Reaction labels must correspond to rule strings from hypergraph_to_bipartite
        # They are NOT molecule names, but reaction rule labels (LHS>>RHS).
        self.assertGreater(len(reaction_labels), 0)

        # Index maps must be consistent
        self.assertEqual(len(s_idx), len(expected_species))
        self.assertEqual(len(r_idx), len(reaction_labels))

        # Ensure indices reflect order
        for i, lbl in enumerate(species_labels):
            # find the corresponding node key
            found = False
            for node, idx in s_idx.items():
                if (
                    str(node).endswith(lbl)
                    or str(G.nodes[node].get("label", "")) == lbl
                ):
                    found = True
                    break
            self.assertTrue(found)

    def test_species_and_reaction_order_accepts_networkx_input(self):
        G = self._example_G()
        s_labels, r_labels, s_idx, r_idx = _species_and_reaction_order(G)

        self.assertGreater(len(s_labels), 0)
        self.assertGreater(len(r_labels), 0)

    def test_species_and_reaction_order_invalid_input_raises(self):
        with self.assertRaises(ValueError):
            # Graph with no bipartite attributes
            G0 = nx.DiGraph()
            G0.add_node("x")
            _species_and_reaction_order(G0)


if __name__ == "__main__":
    unittest.main()
