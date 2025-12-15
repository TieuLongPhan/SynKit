import unittest

import networkx as nx

# Adjust this import to your actual package path, e.g.:
from synkit.Graph.Matcher.sing import SING

# from sing import SING


class TestSINGBasic(unittest.TestCase):
    def test_unlabeled_path_match(self) -> None:
        """Simple path pattern in an unlabeled path graph."""
        G = nx.path_graph(4)  # 0-1-2-3
        Q = nx.path_graph(3)  # 0-1-2 (query nodes)

        index = SING(G, max_path_length=2, node_att=[], edge_att=None)
        mappings = index.search(Q, prune=False)

        # For a path of length 3 in path_graph(4), there are four embeddings:
        # 0-1-2 and 1-2-3, each in two directions.
        self.assertEqual(len(mappings), 4)

        # Check that each mapping is injective and preserves edges
        for m in mappings:
            self.assertEqual(len(set(m.values())), len(m))  # injective
            for u, v in Q.edges:
                self.assertTrue(G.has_edge(m[u], m[v]))

    def test_prune_boolean_result(self) -> None:
        """`prune=True` should return boolean existence."""
        G = nx.cycle_graph(4)
        Q = nx.path_graph(3)

        index = SING(G, max_path_length=2, node_att=[], edge_att=None)
        self.assertTrue(index.search(Q, prune=True))

        # Pattern that does not exist: a path of length 4
        Q2 = nx.path_graph(5)
        self.assertFalse(index.search(Q2, prune=True))


class TestSINGAttributes(unittest.TestCase):
    def test_node_attribute_filtering(self) -> None:
        """Node attributes should restrict matches."""
        # Triangle with one special node
        G = nx.cycle_graph(3)
        for n in G.nodes:
            G.nodes[n]["element"] = "C"
        G.nodes[0]["element"] = "O"  # make node 0 distinct

        # Query edge O--C
        Q = nx.Graph()
        Q.add_edge("x", "y")
        Q.nodes["x"]["element"] = "O"
        Q.nodes["y"]["element"] = "C"

        index = SING(G, max_path_length=1, node_att="element", edge_att=None)
        mappings = index.search(Q)

        # We expect two embeddings: x->0, y->1 and x->0, y->2
        self.assertEqual(len(mappings), 2)
        for m in mappings:
            self.assertEqual(G.nodes[m["x"]]["element"], "O")
            self.assertEqual(G.nodes[m["y"]]["element"], "C")

    def test_edge_attribute_filtering(self) -> None:
        """Edge attributes should be respected in refinement."""
        G = nx.Graph()
        G.add_edge(0, 1, order=1)
        G.add_edge(1, 2, order=2)

        # Query edge with order=2 should match only the edge (1, 2),
        # but with both orientations (a->1,b->2 and a->2,b->1).
        Q = nx.Graph()
        Q.add_edge("a", "b", order=2)

        index = SING(G, max_path_length=1, node_att=[], edge_att="order")
        mappings = index.search(Q)

        self.assertEqual(len(mappings), 2)
        for m in mappings:
            edge = tuple(sorted((m["a"], m["b"])))
            self.assertEqual(edge, (1, 2))


class TestSINGReindexAndDunder(unittest.TestCase):
    def test_reindex_with_new_graph(self) -> None:
        """Reindexing with a new graph should update matches."""
        G1 = nx.path_graph(3)
        G2 = nx.cycle_graph(4)
        Q = nx.path_graph(3)

        index = SING(G1, max_path_length=2, node_att=[], edge_att=None)
        mappings1 = index.search(Q)
        # In a P3 inside P3 there are two embeddings (two orientations).
        self.assertEqual(len(mappings1), 2)

        index.reindex(G2)
        mappings2 = index.search(Q)
        # In a 4-cycle, there are 4 distinct triples along the cycle,
        # each in two orientations -> 8 embeddings.
        self.assertEqual(len(mappings2), 8)

    def test_len_and_repr(self) -> None:
        """__len__ and __repr__ sanity checks."""
        G = nx.path_graph(4)
        index = SING(G, max_path_length=2, node_att=[], edge_att=None)

        self.assertEqual(len(index), 4)
        rep = repr(index)
        self.assertIn("SING", rep)
        self.assertIn("|V|=4", rep)
        self.assertIn("max_path_length=2", rep)


if __name__ == "__main__":
    unittest.main()
