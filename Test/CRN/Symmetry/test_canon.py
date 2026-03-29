from __future__ import annotations

import unittest
from typing import Any, Iterable, Set

import networkx as nx

from synkit.CRN.Structure import SynCRN
from synkit.CRN.Symmetry.canon import CRNCanonicalizer, canonical
from synkit.CRN.Symmetry._ir import IRCanonicalEngine
from synkit.CRN.Symmetry._common import SymmetryConfig


class TestCRNCanonicalizerHelpers(unittest.TestCase):
    """Helper assertions for canonicalization tests."""

    @staticmethod
    def assert_partition(
        testcase: unittest.TestCase,
        universe: Set[Any],
        orbits: Iterable[Set[Any]],
    ) -> None:
        seen: Set[Any] = set()
        for cell in orbits:
            testcase.assertIsInstance(cell, set)
            testcase.assertTrue(cell, "Orbit cells should be non-empty.")
            testcase.assertTrue(
                seen.isdisjoint(cell),
                f"Orbit cells overlap: already saw {seen & cell}",
            )
            seen.update(cell)

        testcase.assertEqual(
            seen,
            universe,
            "Orbit partition should cover exactly the graph nodes.",
        )


class TestCRNCanonicalizer(unittest.TestCase):
    """
    Unit tests for exact canonicalization on SynCRN-format graphs.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.medium = [
            "AJ>>N",
            "AK>>AF",
            "A+AK>>AE",
            "AE+AJ>>W",
            "A+AJ>>M",
            "AB+AI>>O",
            "AC+AJ>>U",
            "AB+AK>>Z",
            "AF+AI>>S",
            "AB+AJ>>H",
            "AD+AI>>Q",
            "AE+AI>>E",
            "AE+AJ>>K",
            "A+AJ>>Y",
            "AI>>AD",
            "AE>>AJ",
            "AE+AI>>R",
            "AF+AJ>>L",
            "AD>>AI",
            "AC+AI>>P",
            "AC>>AH",
            "A+AI>>X",
            "A+AI>>G",
            "AE+AK>>AC",
            "AI>>M",
            "AD+AJ>>J",
            "AF+AJ>>X",
            "AD+AI>>D",
            "AJ>>AE",
            "AC+AJ>>I",
            "AF+AK>>AD",
            "AB+AJ>>T",
            "AB+AI>>B",
            "AC+AI>>C",
            "AF+AI>>F",
            "AB>>AG",
            "AC+AK>>AA",
            "AD+AK>>AB",
            "AD+AJ>>V",
            "AF>>AK",
        ]

    def test_build_from_syncrn_and_digraph(self) -> None:
        """
        Canonicalizer should be constructible from both SynCRN and DiGraph.
        """
        syn = SynCRN.from_reaction_strings(["A>>B", "B>>C"])
        g = syn.to_digraph()

        canon_from_syn = CRNCanonicalizer(syn)
        canon_from_g = CRNCanonicalizer(g)

        self.assertIsInstance(canon_from_syn.G, nx.DiGraph)
        self.assertIsInstance(canon_from_g.G, nx.DiGraph)
        self.assertEqual(set(canon_from_syn.G.nodes()), set(canon_from_g.G.nodes()))
        self.assertEqual(set(canon_from_syn.G.edges()), set(canon_from_g.G.edges()))

    def test_graph_property_returns_internal_digraph(self) -> None:
        """
        The public ``G`` property should expose the internal exact-engine graph.
        """
        syn = SynCRN.from_reaction_strings(["A>>B"])
        canon_obj = CRNCanonicalizer(syn)

        self.assertIs(canon_obj.G, canon_obj.engine.G)
        self.assertIsInstance(canon_obj.G, nx.DiGraph)

    def test_engine_property_returns_ir_engine(self) -> None:
        """
        ``engine`` should expose the shared exact IR engine.
        """
        syn = SynCRN.from_reaction_strings(["A>>B", "B>>C"])
        canon_obj = CRNCanonicalizer(syn)

        self.assertIsInstance(canon_obj.engine, IRCanonicalEngine)
        self.assertIs(canon_obj.engine, canon_obj._engine)

    def test_explicit_config_is_respected(self) -> None:
        """
        Passing an explicit config should be preserved.
        """
        syn = SynCRN.from_reaction_strings(["A>>B", "B>>C"])
        cfg = SymmetryConfig.semantic()

        canon_obj = CRNCanonicalizer(syn, config=cfg)

        self.assertIs(canon_obj.config, cfg)

    def test_canonical_result_has_core_fields(self) -> None:
        """
        ``canonical_result`` should return an object with canonical order/key.
        """
        syn = SynCRN.from_reaction_strings(["A>>B", "B>>C"])
        canon_obj = CRNCanonicalizer(syn)

        res = canon_obj.canonical_result(timeout_sec=5.0)

        self.assertTrue(hasattr(res, "canonical_order"))
        self.assertTrue(hasattr(res, "canonical_key"))
        self.assertTrue(hasattr(res, "elapsed_seconds"))
        self.assertIsInstance(res.canonical_order, list)
        self.assertEqual(len(res.canonical_order), canon_obj.G.number_of_nodes())

    def test_canonical_order_matches_graph_nodes(self) -> None:
        """
        Canonical order should be a permutation of graph nodes.
        """
        syn = SynCRN.from_reaction_strings(["A>>B", "B>>C"])
        canon_obj = CRNCanonicalizer(syn)

        order = canon_obj.canonical_order(timeout_sec=5.0)

        self.assertEqual(len(order), canon_obj.G.number_of_nodes())
        self.assertEqual(set(order), set(canon_obj.G.nodes()))

    def test_canonical_graph_uses_consecutive_integer_labels(self) -> None:
        """
        Canonical graph should relabel nodes to 1..n.
        """
        syn = SynCRN.from_reaction_strings(["A>>B", "B>>C"])
        canon_obj = CRNCanonicalizer(syn)

        g_canon = canon_obj.canonical_graph(timeout_sec=5.0)
        n = canon_obj.G.number_of_nodes()

        self.assertIsInstance(g_canon, nx.DiGraph)
        self.assertEqual(set(g_canon.nodes()), set(range(1, n + 1)))

    def test_canonical_wrapper_matches_method(self) -> None:
        """
        The convenience wrapper should match ``canonical_graph``.
        """
        syn = SynCRN.from_reaction_strings(["A>>B", "B>>C"])

        g1 = canonical(syn)
        g2 = CRNCanonicalizer(syn).canonical_graph()

        self.assertEqual(set(g1.nodes()), set(g2.nodes()))
        self.assertEqual(set(g1.edges()), set(g2.edges()))

    def test_semantic_mode_medium_is_rigid(self) -> None:
        """
        In semantic mode, the medium example is currently rigid.
        """
        syn = SynCRN.from_reaction_strings(self.medium)
        canon_obj = CRNCanonicalizer(syn, config=SymmetryConfig.semantic())

        self.assertFalse(canon_obj.has_nontrivial_automorphism(timeout_sec=10.0))

        ares = canon_obj.automorphism_result(max_count=512, timeout_sec=20.0)
        self.assertEqual(ares.automorphism_count, 1)

    def test_topological_mode_medium_has_nontrivial_automorphism(self) -> None:
        """
        In topological mode, the medium example should expose symmetry.
        """
        syn = SynCRN.from_reaction_strings(self.medium)
        canon_obj = CRNCanonicalizer(syn, config=SymmetryConfig.topological())

        self.assertTrue(canon_obj.has_nontrivial_automorphism(timeout_sec=10.0))

    def test_topological_mode_medium_count_is_512(self) -> None:
        """
        In topological mode, the current expected automorphism count is 512.
        """
        syn = SynCRN.from_reaction_strings(self.medium)
        canon_obj = CRNCanonicalizer(syn, config=SymmetryConfig.topological())

        ares = canon_obj.automorphism_result(max_count=512, timeout_sec=20.0)
        self.assertEqual(ares.automorphism_count, 512)

    def test_exact_orbits_form_partition(self) -> None:
        """
        Exact orbit output should form a partition of graph nodes.
        """
        syn = SynCRN.from_reaction_strings(self.medium)
        canon_obj = CRNCanonicalizer(syn, config=SymmetryConfig.topological())

        orbits = canon_obj.orbits(max_count=512, timeout_sec=20.0)
        universe = set(canon_obj.G.nodes())

        TestCRNCanonicalizerHelpers.assert_partition(self, universe, orbits)

    def test_wl_orbits_form_partition(self) -> None:
        """
        WL orbit output should also form a partition of graph nodes.
        """
        syn = SynCRN.from_reaction_strings(self.medium)
        canon_obj = CRNCanonicalizer(syn, config=SymmetryConfig.topological())

        wl_orbits = canon_obj.wl_orbits()
        universe = set(canon_obj.G.nodes())

        TestCRNCanonicalizerHelpers.assert_partition(self, universe, wl_orbits)

    def test_summary_without_automorphisms(self) -> None:
        """
        ``summary(include_automorphisms=False)`` should return canonical data only.
        """
        syn = SynCRN.from_reaction_strings(["A>>B", "B>>C"])
        canon_obj = CRNCanonicalizer(syn)

        info = canon_obj.summary(include_automorphisms=False, timeout_sec=5.0)

        self.assertIn("graph_type", info)
        self.assertIn("canonical_perm", info)
        self.assertIn("canonical_key", info)
        self.assertIn("canon_graph", info)
        self.assertIn("elapsed_seconds", info)

        self.assertNotIn("automorphism_count", info)
        self.assertNotIn("orbits", info)
        self.assertIsInstance(info["canon_graph"], nx.DiGraph)

    def test_summary_with_automorphisms(self) -> None:
        """
        ``summary(include_automorphisms=True)`` should include symmetry fields.
        """
        syn = SynCRN.from_reaction_strings(self.medium)
        canon_obj = CRNCanonicalizer(syn, config=SymmetryConfig.topological())

        info = canon_obj.summary(
            include_automorphisms=True,
            max_count=128,
            timeout_sec=10.0,
        )

        self.assertIn("graph_type", info)
        self.assertIn("canonical_perm", info)
        self.assertIn("canonical_key", info)
        self.assertIn("canon_graph", info)
        self.assertIn("automorphism_count", info)
        self.assertIn("sample_permutations", info)
        self.assertIn("mappings", info)
        self.assertIn("orbits", info)
        self.assertIn("early_stop", info)
        self.assertIn("elapsed_seconds", info)

        self.assertIsInstance(info["canon_graph"], nx.DiGraph)
        self.assertGreaterEqual(info["automorphism_count"], 2)

    def test_construct_from_prebuilt_digraph_semantic(self) -> None:
        """
        Canonicalizer should work from ``syn.to_digraph()`` in semantic mode.
        """
        syn = SynCRN.from_reaction_strings(self.medium)
        crn = syn.to_digraph()

        canon_obj = CRNCanonicalizer(crn, config=SymmetryConfig.semantic())
        ares = canon_obj.automorphism_result(max_count=64, timeout_sec=10.0)

        self.assertIsInstance(canon_obj.G, nx.DiGraph)
        self.assertEqual(ares.automorphism_count, 1)

    def test_construct_from_prebuilt_digraph_topological(self) -> None:
        """
        Canonicalizer should also detect symmetry from a prebuilt digraph in
        topological mode.
        """
        syn = SynCRN.from_reaction_strings(self.medium)
        crn = syn.to_digraph()

        canon_obj = CRNCanonicalizer(crn, config=SymmetryConfig.topological())
        ares = canon_obj.automorphism_result(max_count=64, timeout_sec=10.0)

        self.assertIsInstance(canon_obj.G, nx.DiGraph)
        self.assertGreaterEqual(ares.automorphism_count, 2)

    def test_isomorphic_inputs_have_same_canonical_key(self) -> None:
        """
        Two isomorphic inputs with different reaction insertion orders should
        yield the same canonical key under the same config.
        """
        rxns1 = [
            "A>>B",
            "B>>C",
            "C>>D",
        ]
        rxns2 = [
            "C>>D",
            "A>>B",
            "B>>C",
        ]

        syn1 = SynCRN.from_reaction_strings(rxns1)
        syn2 = SynCRN.from_reaction_strings(rxns2)

        c1 = CRNCanonicalizer(syn1, config=SymmetryConfig.semantic())
        c2 = CRNCanonicalizer(syn2, config=SymmetryConfig.semantic())

        self.assertEqual(
            c1.canonical_key(timeout_sec=5.0),
            c2.canonical_key(timeout_sec=5.0),
        )

    def test_canonical_graph_preserves_edge_count(self) -> None:
        """
        Canonical relabeling should preserve graph size.
        """
        syn = SynCRN.from_reaction_strings(self.medium)
        canon_obj = CRNCanonicalizer(syn, config=SymmetryConfig.topological())

        g_canon = canon_obj.canonical_graph(timeout_sec=10.0)

        self.assertEqual(g_canon.number_of_nodes(), canon_obj.G.number_of_nodes())
        self.assertEqual(g_canon.number_of_edges(), canon_obj.G.number_of_edges())


if __name__ == "__main__":
    unittest.main()
