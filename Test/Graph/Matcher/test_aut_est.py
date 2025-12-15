from __future__ import annotations

import unittest
from typing import Set, FrozenSet

import networkx as nx

from synkit.Graph.Matcher.auto_est import AutoEst, estimate_automorphism_groups


class TestAutoEst(unittest.TestCase):
    """Unit tests for AutoEst WL-1 automorphism orbit estimator."""

    def test_cycle_graph_all_symmetric(self) -> None:
        """A 4-cycle should have a single orbit containing all nodes."""
        G = nx.cycle_graph(4)  # nodes 0..3, symmetric
        est = AutoEst(G, node_attrs=[], edge_attrs=[], max_iter=10)
        # before fit, __len__ should be 0 and accessing node_colors should raise
        self.assertEqual(len(est), 0)
        with self.assertRaises(RuntimeError):
            _ = est.node_colors

        est.fit()
        # after fit, one orbit containing all 4 nodes
        self.assertEqual(est.n_orbits, 1)
        expected_orbit = frozenset({0, 1, 2, 3})
        self.assertIn(expected_orbit, est.orbits)
        # groups should be a list-of-lists representation
        self.assertEqual(est.groups, [[0, 1, 2, 3]])
        # orbit_index maps each node to orbit 0
        for n in G.nodes():
            self.assertEqual(est.orbit_index[n], 0)
        # node_colors length matches number of nodes
        self.assertEqual(len(est.node_colors), G.number_of_nodes())
        # pairwise check: all nodes share same color id
        colors = set(est.node_colors.values())
        self.assertEqual(len(colors), 1)

    def test_path_graph_orbits_and_groups(self) -> None:
        """Path on 4 nodes has two orbits: endpoints and middle nodes."""
        P = nx.path_graph(4)  # nodes 0-3
        est = AutoEst(P, node_attrs=[], edge_attrs=[], max_iter=10).fit()
        # Expect two orbits: {0,3} and {1,2}
        expected_orbits: Set[FrozenSet[int]] = {frozenset({0, 3}), frozenset({1, 2})}
        self.assertEqual(set(est.orbits), expected_orbits)
        # groups returns sorted lists and is deterministically ordered
        self.assertEqual(est.groups, [[0, 3], [1, 2]])
        # len(est) equals number of orbits after fit
        self.assertEqual(len(est), est.n_orbits)
        self.assertEqual(est.n_orbits, 2)
        # orbit_index should map nodes properly
        idx_map = est.orbit_index
        self.assertEqual(set([idx_map[0], idx_map[3]]), {0})
        self.assertEqual(set([idx_map[1], idx_map[2]]), {1})

    def test_node_colors_before_fit_raises(self) -> None:
        """Accessing node_colors before fit should raise a RuntimeError."""
        G = nx.path_graph(3)
        est = AutoEst(G, node_attrs=[], edge_attrs=[], max_iter=5)
        with self.assertRaises(RuntimeError):
            _ = est.node_colors
        with self.assertRaises(RuntimeError):
            _ = est.orbits
        with self.assertRaises(RuntimeError):
            _ = est.orbit_index
        # After fit no exceptions
        est.fit()
        _ = est.node_colors
        _ = est.orbits
        _ = est.orbit_index

    def test_estimate_automorphism_groups_with_node_attrs(self) -> None:
        """
        The convenience function should respect node_attrs: if nodes carry
        distinguishing attributes they should be placed in different orbits.
        """
        G = nx.Graph()
        G.add_nodes_from([0, 1])
        G.add_edge(0, 1)
        # give different 'element' attributes so WL must distinguish them
        G.nodes[0]["element"] = "C"
        G.nodes[1]["element"] = "O"

        # use the convenience function with node_attrs=['element']
        est = estimate_automorphism_groups(
            G, node_attrs=["element"], edge_attrs=[], max_iter=5
        )
        # nodes should be in different orbits because attributes differ
        self.assertEqual(est.n_orbits, 2)
        # check that each orbit is a singleton
        self.assertTrue(all(len(o) == 1 for o in est.orbits))
        # orbit_index maps nodes to distinct indices
        self.assertNotEqual(est.orbit_index[0], est.orbit_index[1])

    def test_repr_and_len_behavior(self) -> None:
        """repr should include node count and max_iter; len is 0 pre-fit and >0 post-fit."""
        G = nx.path_graph(5)
        est = AutoEst(G, node_attrs=[], edge_attrs=[], max_iter=3)
        r = repr(est)
        # contains nodes count and max_iter
        self.assertIn("nodes=5", r)
        self.assertIn("max_iter=3", r)
        self.assertEqual(len(est), 0)
        est.fit()
        self.assertGreater(len(est), 0)
        # repr now shows orbits number (should be an integer in the repr string)
        r2 = repr(est)
        self.assertIn("orbits=", r2)

    def test_orbit_index_consistency(self) -> None:
        """orbit_index should be consistent with the orbits list."""
        G = nx.cycle_graph(6)  # all nodes symmetric -> single orbit of size 6
        est = AutoEst(G, node_attrs=[], edge_attrs=[], max_iter=6).fit()
        orbit_list = est.orbits
        idx = est.orbit_index
        # Every node should map to the index of the unique orbit
        self.assertEqual(len(orbit_list), 1)
        orb0 = orbit_list[0]
        for n in G.nodes():
            self.assertIn(n, orb0)
            self.assertEqual(idx[n], 0)


if __name__ == "__main__":
    unittest.main()
