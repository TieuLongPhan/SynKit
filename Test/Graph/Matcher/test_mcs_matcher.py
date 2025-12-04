from __future__ import annotations

import unittest

import networkx as nx

from synkit.Graph.Matcher.mcs_matcher import MCSMatcher


class TestMCSMatcher(unittest.TestCase):
    """Unit tests for :class:`MCSMatcher`."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _cycle_with_labels(n: int) -> nx.Graph:
        G = nx.cycle_graph(n)
        nx.set_node_attributes(G, "C", "element")
        nx.set_edge_attributes(G, 1, "order")
        return G

    @staticmethod
    def _path_with_labels(n: int) -> nx.Graph:
        G = nx.path_graph(n)
        nx.set_node_attributes(G, "C", "element")
        nx.set_edge_attributes(G, 1, "order")
        return G

    # ------------------------------------------------------------------
    # Basic behaviour
    # ------------------------------------------------------------------
    def test_exact_isomorphism_mcs(self) -> None:
        """Two identical cycles should have MCS size equal to n."""
        G1 = self._cycle_with_labels(4)
        G2 = self._cycle_with_labels(4)

        matcher = MCSMatcher()
        matcher.find_common_subgraph(G1, G2, mcs=True)

        self.assertGreaterEqual(matcher.last_size, 4)
        self.assertGreaterEqual(matcher.num_mappings, 1)
        for mapping in matcher.mappings:
            self.assertEqual(len(mapping), 4)

    def test_subgraph_case_mcs(self) -> None:
        """Path of length 3 inside path of length 5 should give MCS size 3."""
        G1 = self._path_with_labels(3)
        G2 = self._path_with_labels(5)

        matcher = MCSMatcher()
        matcher.find_common_subgraph(G1, G2, mcs=True)

        self.assertEqual(matcher.last_size, 3)
        self.assertGreaterEqual(matcher.num_mappings, 1)
        for mapping in matcher.mappings:
            self.assertEqual(len(mapping), 3)
            # All keys must come from G1
            self.assertTrue(all(node in G1.nodes for node in mapping.keys()))
            # All values must come from G2
            self.assertTrue(all(node in G2.nodes for node in mapping.values()))

    def test_mcs_false_collects_smaller_subgraphs(self) -> None:
        """With mcs=False, smaller common subgraphs may also be present."""
        G1 = self._path_with_labels(3)
        G2 = self._path_with_labels(4)

        matcher = MCSMatcher()
        matcher.find_common_subgraph(G1, G2, mcs=False)

        self.assertGreaterEqual(matcher.num_mappings, 1)
        # At least one mapping should have size equal to 3 (full path match)
        has_full = any(len(m) == 3 for m in matcher.mappings)
        self.assertTrue(has_full)

    def test_label_mismatch_gives_no_mappings(self) -> None:
        """Different node labels should prevent matching."""
        G1 = self._path_with_labels(3)
        G2 = self._path_with_labels(3)
        nx.set_node_attributes(G2, "O", "element")  # different label

        matcher = MCSMatcher()
        matcher.find_common_subgraph(G1, G2, mcs=True)
        self.assertEqual(matcher.num_mappings, 0)
        self.assertEqual(matcher.last_size, 0)

    # ------------------------------------------------------------------
    # Orientation / swapping behaviour
    # ------------------------------------------------------------------
    def test_internal_swapping_preserves_orientation(self) -> None:
        """
        When G1 is larger than G2, internal swapping should still produce
        mappings from G1 → G2.
        """
        G_big = self._path_with_labels(5)
        G_small = self._path_with_labels(3)

        matcher = MCSMatcher()
        matcher.find_common_subgraph(G_big, G_small, mcs=True)

        self.assertEqual(matcher.last_size, 3)
        for mapping in matcher.mappings:
            # Keys must be G_big nodes, values G_small nodes
            self.assertTrue(all(node in G_big.nodes for node in mapping.keys()))
            self.assertTrue(all(node in G_small.nodes for node in mapping.values()))

    # ------------------------------------------------------------------
    # Molecule-level matching
    # ------------------------------------------------------------------
    def test_mcs_mol_matches_components(self) -> None:
        """Connected-component matching should combine isomorphic components."""
        # G1: two components: path_3 + path_2
        G1 = nx.disjoint_union(self._path_with_labels(3), self._path_with_labels(2))
        # G2: the same composition but maybe different node ids
        G2 = nx.disjoint_union(self._path_with_labels(3), self._path_with_labels(2))

        matcher = MCSMatcher()
        matcher.find_common_subgraph(G1, G2, mcs_mol=True)

        self.assertEqual(matcher.num_mappings, 1)
        mapping = matcher.mappings[0]
        self.assertEqual(len(mapping), G1.number_of_nodes())
        self.assertEqual(matcher.last_size, G1.number_of_nodes())

    # ------------------------------------------------------------------
    # Niceties
    # ------------------------------------------------------------------
    def test_repr_and_properties(self) -> None:
        """Smoke test for __repr__ and properties."""
        G1 = self._path_with_labels(3)
        G2 = self._path_with_labels(4)

        matcher = MCSMatcher()
        matcher.find_common_subgraph(G1, G2, mcs=True)

        rep = repr(matcher)
        self.assertIn("MCSMatcher", rep)
        self.assertIsInstance(matcher.last_size, int)
        self.assertIsInstance(matcher.num_mappings, int)
        self.assertIsInstance(matcher.help, str)


if __name__ == "__main__":
    unittest.main()
