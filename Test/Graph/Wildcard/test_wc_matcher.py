from __future__ import annotations

import unittest
import networkx as nx

from synkit.Graph.Wildcard.wc_matcher import WCMatcher, WildcardPatternMatcher


class TestWCMatcher(unittest.TestCase):
    """Unit tests for :class:`WCMatcher` / :class:`WildcardPatternMatcher`."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _path_with_elements(n: int, wildcard_at: int | None = None) -> nx.Graph:
        """Create a simple path graph with element labels and optional wildcard."""
        G = nx.path_graph(n)
        nx.set_node_attributes(G, "C", "element")
        if wildcard_at is not None:
            G.nodes[wildcard_at]["element"] = "*"
        return G

    # ------------------------------------------------------------------
    # Pattern/host selection
    # ------------------------------------------------------------------
    def test_pattern_is_graph_with_wildcard(self) -> None:
        """Graph containing wildcard must be selected as pattern."""
        G1 = self._path_with_elements(3, wildcard_at=1)
        G2 = self._path_with_elements(3, wildcard_at=None)

        matcher = WCMatcher(G1, G2)
        self.assertIs(matcher.pattern_graph, G1)
        self.assertIs(matcher.host_graph, G2)

    def test_pattern_is_smaller_when_both_without_wildcard(self) -> None:
        """When both graphs lack wildcards, smaller graph becomes pattern."""
        G1 = self._path_with_elements(3)
        G2 = self._path_with_elements(5)

        matcher = WCMatcher(G1, G2)
        self.assertIs(matcher.pattern_graph, G1)
        self.assertIs(matcher.host_graph, G2)

    # ------------------------------------------------------------------
    # Core matching / wildcard behaviour
    # ------------------------------------------------------------------
    def test_simple_core_match_without_wildcard(self) -> None:
        """Exact core match when both graphs are identical."""
        G1 = self._path_with_elements(3)
        G2 = self._path_with_elements(3)

        matcher = WCMatcher(
            G1,
            G2,
            node_attrs=["element"],
            edge_attrs=[],
        ).fit()
        self.assertTrue(matcher.is_match)
        self.assertEqual(len(matcher.core_mapping_without_wildcard_regions), 3)

    def test_core_match_with_wildcard_node(self) -> None:
        """
        Pattern C-C-* vs host C-C-C.

        Core mapping should involve only the C-C part (2 nodes).
        """
        pattern = self._path_with_elements(3, wildcard_at=2)
        host = self._path_with_elements(3)

        matcher = WCMatcher(
            pattern,
            host,
            node_attrs=["element"],
            edge_attrs=[],
        ).fit()
        self.assertTrue(matcher.is_match)
        core_map = matcher.core_mapping_without_wildcard_regions
        self.assertEqual(len(core_map), 2)  # only non-wildcard nodes
        self.assertIn(0, core_map)
        self.assertIn(1, core_map)

    def test_no_match_when_core_too_large(self) -> None:
        """
        If the *chosen* pattern core has more nodes than the host, no match is possible.
        """
        # G1 has a wildcard → forced to be the pattern
        pattern = self._path_with_elements(5, wildcard_at=4)  # core size = 4
        host = self._path_with_elements(3)  # host size = 3

        matcher = WCMatcher(
            pattern,
            host,
            node_attrs=["element"],
            edge_attrs=[],
        ).fit()

        # Now pattern_core has 4 nodes, host has 3 → early size check triggers
        self.assertFalse(matcher.is_match)
        self.assertEqual(matcher.core_mapping_without_wildcard_regions, {})

    # ------------------------------------------------------------------
    # Neighbour semantics
    # ------------------------------------------------------------------
    def test_neighbors_lower_bound_semantics(self) -> None:
        """
        Neighbour lists: pattern counts are lower bounds on host counts.

        Pattern neighbours: [C, C]; host neighbours: [C, C, C] → match.
        """
        host = nx.Graph()
        host.add_nodes_from([0, 1, 2])
        host.add_edges_from([(0, 1), (0, 2)])
        nx.set_node_attributes(host, "C", "element")
        host.nodes[0]["neighbors"] = ["C", "C"]

        pattern = nx.Graph()
        pattern.add_node(0, element="C", neighbors=["C"])
        # pattern has only one C neighbour requirement, host has >=1

        matcher = WCMatcher(
            pattern,
            host,
            node_attrs=["element", "neighbors"],
            edge_attrs=[],
        ).fit()
        self.assertTrue(matcher.is_match)

    # ------------------------------------------------------------------
    # Wildcard region extraction
    # ------------------------------------------------------------------
    def test_wildcard_subgraph_regions(self) -> None:
        """
        Wildcard region: nodes adjacent to core anchors but not in core mapping.
        """
        # host: path 0-1-2-3, all C
        host = self._path_with_elements(4)

        # pattern: 0-1-2 with 2 as wildcard (C-C-*)
        pattern = self._path_with_elements(3, wildcard_at=2)

        matcher = WCMatcher(
            pattern,
            host,
            node_attrs=["element"],
            edge_attrs=[],
        ).fit()
        self.assertTrue(matcher.is_match)

        regions = matcher.wildcard_subgraph_mapping
        self.assertIn(2, regions)
        # wildcard at node 2 anchored at pattern node 1 → mapped to host node,
        # whose neighbours should include at least one extra host node.
        self.assertTrue(len(regions[2]) >= 1)

    def test_wildcard_regions_empty_without_match(self) -> None:
        """If no core match, wildcard regions must be empty."""
        pattern = self._path_with_elements(3, wildcard_at=1)
        host = self._path_with_elements(2)  # too small

        matcher = WCMatcher(
            pattern,
            host,
            node_attrs=["element"],
            edge_attrs=[],
        ).fit()
        self.assertFalse(matcher.is_match)
        self.assertEqual(matcher.wildcard_subgraph_mapping, {})

    # ------------------------------------------------------------------
    # Alias / niceties
    # ------------------------------------------------------------------
    def test_wildcard_pattern_matcher_alias(self) -> None:
        """WildcardPatternMatcher should be an alias of WCMatcher."""
        pattern = self._path_with_elements(3, wildcard_at=2)
        host = self._path_with_elements(3)

        matcher = WildcardPatternMatcher(
            pattern,
            host,
            node_attrs=["element"],
            edge_attrs=[],
        ).fit()
        self.assertIsInstance(matcher, WCMatcher)
        self.assertTrue(matcher.is_match)

    def test_repr_and_help(self) -> None:
        """Smoke test for __repr__ and help property."""
        pattern = self._path_with_elements(3)
        host = self._path_with_elements(3)

        matcher = WCMatcher(pattern, host).fit()
        rep = repr(matcher)
        self.assertIn("WCMatcher", rep)
        self.assertIsInstance(matcher.help, str)


if __name__ == "__main__":
    unittest.main()
