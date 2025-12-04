from __future__ import annotations

import unittest

import networkx as nx

from synkit.Graph.Matcher.partial_matcher import PartialMatcher
from synkit.Synthesis.Reactor.strategy import Strategy


class TestPartialMatcher(unittest.TestCase):
    """Unit tests for :class:`PartialMatcher`."""

    @staticmethod
    def _make_simple_pattern() -> nx.Graph:
        """Pattern with two disconnected edges."""
        pattern = nx.Graph()
        pattern.add_edges_from([(0, 1), (2, 3)])
        return pattern

    @staticmethod
    def _make_host_path(n: int = 5) -> nx.Graph:
        """Host graph: simple path on n nodes."""
        return nx.path_graph(n)

    # ------------------------------------------------------------------
    # Behaviour for partial=False
    # ------------------------------------------------------------------
    def test_full_mode_allows_full_pattern_when_possible(self) -> None:
        """
        partial=False, k=None → full-mode only.

        If full pattern is embeddable, we should see at least one
        mapping that uses all pattern nodes.
        """
        host = self._make_host_path(5)
        pattern = self._make_simple_pattern()

        matcher = PartialMatcher(
            host=host,
            pattern=pattern,
            node_attrs=[],
            edge_attrs=[],
            strategy=Strategy.COMPONENT,
            max_results=None,
            partial=False,
        )
        mappings = matcher.get_mappings()
        # If any mapping exists, at least one should cover all pattern nodes
        if mappings:
            has_full = any(len(m) == pattern.number_of_nodes() for m in mappings)
            self.assertTrue(has_full)

    def test_full_mode_can_return_empty_if_full_not_embeddable(self) -> None:
        """
        partial=False, k=None but host too small for full pattern.

        In this case no full-pattern embedding exists, so we expect
        zero mappings (no automatic fallback to partials).
        """
        host = self._make_host_path(3)  # too small for two disjoint edges
        pattern = self._make_simple_pattern()

        matcher = PartialMatcher(
            host=host,
            pattern=pattern,
            node_attrs=[],
            edge_attrs=[],
            strategy=Strategy.COMPONENT,
            max_results=None,
            partial=False,
        )
        self.assertEqual(matcher.num_mappings, 0)

    # ------------------------------------------------------------------
    # Behaviour for partial=True
    # ------------------------------------------------------------------
    def test_partial_mode_excludes_full_pattern_in_auto(self) -> None:
        """
        partial=True, k=None → strict partials only.

        Any mapping must match strictly fewer nodes than the full pattern.
        """
        host = self._make_host_path(5)
        pattern = self._make_simple_pattern()

        matcher = PartialMatcher(
            host=host,
            pattern=pattern,
            node_attrs=[],
            edge_attrs=[],
            strategy=Strategy.COMPONENT,
            max_results=None,
            partial=True,
        )
        mappings = matcher.get_mappings()
        for mapping in mappings:
            self.assertLess(len(mapping), pattern.number_of_nodes())

    def test_partial_mode_still_allows_single_component_matches(self) -> None:
        """
        partial=True, k=None on a too-small host should still give
        single-component matches if possible.
        """
        host = self._make_host_path(3)  # only enough for one edge
        pattern = self._make_simple_pattern()

        matcher = PartialMatcher(
            host=host,
            pattern=pattern,
            node_attrs=[],
            edge_attrs=[],
            strategy=Strategy.COMPONENT,
            max_results=None,
            partial=True,
        )
        mappings = matcher.get_mappings()
        # Should get at least one mapping of a single component (2 nodes)
        self.assertGreaterEqual(len(mappings), 1)
        for mapping in mappings:
            self.assertLess(len(mapping), pattern.number_of_nodes())

    # ------------------------------------------------------------------
    # Explicit k semantics
    # ------------------------------------------------------------------
    def test_explicit_k_uses_exact_component_count(self) -> None:
        """
        Explicit k should restrict embeddings to exactly k components.

        partial flag does not change explicit k behaviour.
        """
        host = self._make_host_path(5)
        pattern = self._make_simple_pattern()

        # Pattern has 2 components, each of size 2 nodes.
        # k=1: any mapping must not cover all pattern nodes.
        mappings_k1 = PartialMatcher.find_partial_mappings(
            host=host,
            pattern=pattern,
            node_attrs=[],
            edge_attrs=[],
            k=1,
            strategy=Strategy.COMPONENT,
            max_results=None,
            partial=True,
        )
        for mapping in mappings_k1:
            self.assertLess(len(mapping), pattern.number_of_nodes())

        # k=2: full-pattern embeddings allowed even if partial=True, because
        # the caller is explicitly requesting it.
        mappings_k2 = PartialMatcher.find_partial_mappings(
            host=host,
            pattern=pattern,
            node_attrs=[],
            edge_attrs=[],
            k=2,
            strategy=Strategy.COMPONENT,
            max_results=None,
            partial=True,
        )
        for mapping in mappings_k2:
            self.assertEqual(len(mapping), pattern.number_of_nodes())

    def test_invalid_k_raises_value_error(self) -> None:
        """Requesting invalid k must raise ValueError."""
        host = self._make_host_path(5)
        pattern = self._make_simple_pattern()

        matcher = PartialMatcher(
            host=host,
            pattern=pattern,
            node_attrs=[],
            edge_attrs=[],
            strategy=Strategy.COMPONENT,
            max_results=None,
            partial=True,
        )
        with self.assertRaises(ValueError):
            matcher._match_components(k=0)
        with self.assertRaises(ValueError):
            matcher._match_components(k=10)

    # ------------------------------------------------------------------
    # Edge cases and niceties
    # ------------------------------------------------------------------
    def test_pattern_without_components_raises(self) -> None:
        """Empty pattern graph should raise ValueError on construction."""
        host = self._make_host_path(5)
        pattern = nx.Graph()  # no nodes, no edges

        with self.assertRaises(ValueError):
            PartialMatcher(
                host=host,
                pattern=pattern,
                node_attrs=[],
                edge_attrs=[],
                strategy=Strategy.COMPONENT,
                max_results=None,
                partial=True,
            )

    def test_repr_and_properties(self) -> None:
        """Basic smoke test for __repr__ and properties."""
        host = self._make_host_path(5)
        pattern = self._make_simple_pattern()

        matcher = PartialMatcher(
            host=host,
            pattern=pattern,
            node_attrs=[],
            edge_attrs=[],
            strategy=Strategy.COMPONENT,
            max_results=10,
            partial=True,
        )

        rep = repr(matcher)
        self.assertIn("PartialMatcher", rep)
        self.assertIsInstance(matcher.num_mappings, int)
        self.assertEqual(matcher.num_pattern_components, 2)
        self.assertIsInstance(matcher.help, str)


if __name__ == "__main__":
    unittest.main()
