import unittest

import networkx as nx

from synkit.Graph.Matcher.sing import SING


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
    def test_attribute_signatures_do_not_have_string_delimiter_collisions(self):
        host = nx.Graph()
        host.add_node(0, first="x|y", second="z")
        query = nx.Graph()
        query.add_node(10, first="x", second="y|z")

        index = SING(host, max_path_length=0, node_att=["first", "second"])

        self.assertEqual(index.search(query), [])

    def test_attribute_signatures_preserve_value_types(self):
        host = nx.Graph()
        host.add_node(0, label=1)
        query = nx.Graph()
        query.add_node(10, label="1")

        index = SING(host, max_path_length=0, node_att="label")

        self.assertEqual(index.search(query), [])

    def test_directed_incoming_edges_are_checked(self):
        host = nx.DiGraph()
        host.add_edge(0, 1)
        host.nodes[0]["element"] = "C"
        host.nodes[1]["element"] = "O"
        query = nx.DiGraph()
        query.add_edge(10, 11)
        query.nodes[10]["element"] = "O"
        query.nodes[11]["element"] = "C"

        index = SING(host, max_path_length=0, node_att="element", edge_att=None)

        self.assertEqual(index.search(query), [])

    def test_query_self_loop_must_exist_in_host(self):
        host = nx.path_graph(3)
        query = nx.Graph()
        query.add_edge(10, 10)

        index = SING(host, max_path_length=0, node_att=[], edge_att=None)

        self.assertEqual(index.search(query), [])

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
