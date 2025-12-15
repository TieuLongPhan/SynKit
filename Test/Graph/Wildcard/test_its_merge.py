from __future__ import annotations

import unittest
import networkx as nx

from synkit.Graph.Wildcard.its_merge import ITSMerge, fuse_its_graphs


class TestITSMerge(unittest.TestCase):
    """Unit tests for :class:`ITSMerge` and :func:`fuse_its_graphs`."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _simple_its_node(hcount_left: int, hcount_right: int) -> tuple:
        """Build a minimal typesGH tuple with given hydrogen counts."""
        left = ("C", False, hcount_left, 0, ["O"])
        right = ("C", False, hcount_right, 0, ["O"])
        return (left, right)

    def _build_host_and_pattern(self) -> tuple[nx.Graph, nx.Graph, dict]:
        """Construct small ITS graphs for anchor merging tests."""
        G1 = nx.Graph()
        G2 = nx.Graph()

        # pattern: node 1
        G1.add_node(1, element="C", typesGH=self._simple_its_node(2, 3))
        # host: node 10 with lower H counts
        G2.add_node(10, element="C", typesGH=self._simple_its_node(1, 1))

        mapping = {1: 10}
        return G1, G2, mapping

    # ------------------------------------------------------------------
    # Orientation
    # ------------------------------------------------------------------
    def test_orientation_pattern_to_host_direct(self) -> None:
        """Mapping keys in G1 and values in G2 should orient pattern=G1, host=G2."""
        G1, G2, mapping = self._build_host_and_pattern()
        merger = ITSMerge(G1, G2, mapping)

        self.assertIs(merger.pattern_graph, G1)
        self.assertIs(merger.host_graph, G2)
        self.assertEqual(merger.pattern_to_host, mapping)

    def test_orientation_inverted_mapping(self) -> None:
        """If mapping is host→pattern, ITSMerge must invert it."""
        G1, G2, mapping = self._build_host_and_pattern()
        inv_mapping = {v: k for k, v in mapping.items()}

        merger = ITSMerge(G1, G2, inv_mapping)
        self.assertIs(merger.pattern_graph, G1)
        self.assertIs(merger.host_graph, G2)
        self.assertEqual(merger.pattern_to_host, mapping)

    # ------------------------------------------------------------------
    # Anchor merging
    # ------------------------------------------------------------------
    def test_types_gh_hydrogen_counts_merged_max(self) -> None:
        """Hydrogen counts in typesGH should be merged with max semantics."""
        G1, G2, mapping = self._build_host_and_pattern()
        merger = ITSMerge(G1, G2, mapping).merge()
        F = merger.fused_graph

        self.assertIn(10, F)
        t = F.nodes[10]["typesGH"]
        self.assertEqual(t[0][2], 2)  # left side max(1,2)
        self.assertEqual(t[1][2], 3)  # right side max(1,3)

    # ------------------------------------------------------------------
    # Leftover pattern nodes & edges
    # ------------------------------------------------------------------
    def test_leftover_pattern_nodes_added(self) -> None:
        """Non-wildcard leftover pattern nodes should be added with new IDs."""
        G1, G2, mapping = self._build_host_and_pattern()

        # Add an extra pattern node 2 connected to 1
        G1.add_node(2, element="O", typesGH=self._simple_its_node(0, 0))
        G1.add_edge(1, 2)

        merger = ITSMerge(G1, G2, mapping).merge()
        F = merger.fused_graph

        # Host node 10 must exist
        self.assertIn(10, F)

        # There should be exactly one new node from pattern (non-wildcard)
        extra_nodes = [n for n in F.nodes if n not in G2.nodes]
        self.assertEqual(len(extra_nodes), 1)

        extra = extra_nodes[0]
        self.assertEqual(F.nodes[extra]["element"], "O")
        self.assertTrue(F.has_edge(10, extra))

    def test_wildcard_pattern_nodes_not_added(self) -> None:
        """Wildcard leftover pattern nodes should not be added to the fused graph."""
        G1, G2, mapping = self._build_host_and_pattern()

        # Pattern node 2 is wildcard
        G1.add_node(2, element="*")
        G1.add_edge(1, 2)

        merger = ITSMerge(G1, G2, mapping).merge()
        F = merger.fused_graph

        # Extra nodes should not be present (only host nodes)
        extra_nodes = [n for n in F.nodes if n not in G2.nodes]
        self.assertEqual(len(extra_nodes), 0)

    # ------------------------------------------------------------------
    # Wildcard removal at end
    # ------------------------------------------------------------------
    def test_wildcard_nodes_removed_after_merge(self) -> None:
        """Wildcard nodes must be removed from final fused graph."""
        G1, G2, mapping = self._build_host_and_pattern()

        # Add a wildcard node inside host to test removal
        G2.add_node(99, element="*")
        G2.add_edge(10, 99)

        merger = ITSMerge(G1, G2, mapping).merge()
        F = merger.fused_graph

        elements = [d.get("element") for _, d in F.nodes(data=True)]
        self.assertNotIn("*", elements)

    # ------------------------------------------------------------------
    # Functional wrapper
    # ------------------------------------------------------------------
    def test_fuse_its_graphs_wrapper(self) -> None:
        """Functional wrapper should behave identically to ITSMerge.merge()."""
        G1, G2, mapping = self._build_host_and_pattern()

        F1 = ITSMerge(G1, G2, mapping).merge().fused_graph
        F2 = fuse_its_graphs(G1, G2, mapping)

        self.assertEqual(set(F1.nodes), set(F2.nodes))
        self.assertEqual(set(F1.edges), set(F2.edges))

    # ------------------------------------------------------------------
    # Errors & repr
    # ------------------------------------------------------------------
    def test_merge_must_be_called_before_fused_graph(self) -> None:
        """Accessing fused_graph before merge must raise RuntimeError."""
        G1, G2, mapping = self._build_host_and_pattern()
        merger = ITSMerge(G1, G2, mapping)
        with self.assertRaises(RuntimeError):
            _ = merger.fused_graph

    def test_repr_contains_counts(self) -> None:
        """__repr__ should contain basic count info."""
        G1, G2, mapping = self._build_host_and_pattern()
        merger = ITSMerge(G1, G2, mapping).merge()
        rep = repr(merger)
        self.assertIn("ITSMerge", rep)
        self.assertIn("fused_nodes", rep)


if __name__ == "__main__":
    unittest.main()
