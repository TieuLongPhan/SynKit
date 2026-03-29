from __future__ import annotations

import unittest

import networkx as nx

from synkit.CRN.Structure import SynCRN
from synkit.CRN.Symmetry._common import SymmetryConfig
from synkit.CRN.Symmetry._ir import IRCanonicalEngine, IRInternalResult


class TestIRCanonicalEngine(unittest.TestCase):
    """
    Unit tests for the exact individualize-refine engine on SynCRN-format graphs.
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
        cls.medium_permuted = list(reversed(cls.medium))

        cls.small_chain = [
            "A>>B",
            "B>>C",
            "C>>D",
        ]
        cls.small_chain_permuted = [
            "C>>D",
            "A>>B",
            "B>>C",
        ]

    def test_build_from_syncrn_and_digraph(self) -> None:
        """
        Engine should be constructible from both SynCRN and its digraph.
        """
        syn = SynCRN.from_reaction_strings(self.small_chain)
        g = syn.to_digraph()

        eng_syn = IRCanonicalEngine(syn)
        eng_g = IRCanonicalEngine(g)

        self.assertIsInstance(eng_syn.G, nx.DiGraph)
        self.assertIsInstance(eng_g.G, nx.DiGraph)
        self.assertEqual(set(eng_syn.G.nodes()), set(eng_g.G.nodes()))
        self.assertEqual(set(eng_syn.G.edges()), set(eng_g.G.edges()))

    def test_graph_property_returns_internal_digraph(self) -> None:
        """
        ``G`` should expose the internal directed graph.
        """
        syn = SynCRN.from_reaction_strings(["A>>B"])
        eng = IRCanonicalEngine(syn)

        self.assertIsInstance(eng.G, nx.DiGraph)

    def test_explicit_config_is_respected(self) -> None:
        """
        Passing an explicit config should be preserved.
        """
        syn = SynCRN.from_reaction_strings(self.small_chain)
        cfg = SymmetryConfig.semantic()

        eng = IRCanonicalEngine(syn, config=cfg)

        self.assertIs(eng.config, cfg)

    def test_run_returns_internal_result(self) -> None:
        """
        ``run`` should return an ``IRInternalResult`` with core fields populated.
        """
        syn = SynCRN.from_reaction_strings(self.small_chain)
        eng = IRCanonicalEngine(syn)

        res = eng.run(timeout_sec=5.0)

        self.assertIsInstance(res, IRInternalResult)
        self.assertIsInstance(res.canonical_order, list)
        self.assertEqual(len(res.canonical_order), eng.G.number_of_nodes())
        self.assertEqual(set(res.canonical_order), set(eng.G.nodes()))
        self.assertIsInstance(res.canonical_key, tuple)
        self.assertIsInstance(res.automorphism_count, int)
        self.assertIsInstance(res.sample_permutations, list)
        self.assertIsInstance(res.sample_mappings, list)
        self.assertIsInstance(res.orbits, list)
        self.assertIsInstance(res.elapsed_seconds, float)
        self.assertIsInstance(res.stopped_early, bool)

    def test_canonical_result_matches_run(self) -> None:
        """
        ``canonical_result`` should agree with ``run`` on order and key.
        """
        syn = SynCRN.from_reaction_strings(self.small_chain)
        eng = IRCanonicalEngine(syn)

        run_res = eng.run(timeout_sec=5.0)
        can_res = eng.canonical_result(timeout_sec=5.0)

        self.assertEqual(can_res.canonical_order, run_res.canonical_order)
        self.assertEqual(can_res.canonical_key, run_res.canonical_key)
        self.assertTrue(can_res.exact)

    def test_automorphism_result_matches_run(self) -> None:
        """
        ``automorphism_result`` should agree with ``run`` on automorphism data.
        """
        syn = SynCRN.from_reaction_strings(self.small_chain)
        eng = IRCanonicalEngine(syn)

        run_res = eng.run(max_count=20, timeout_sec=5.0)
        auto_res = eng.automorphism_result(max_count=20, timeout_sec=5.0)

        self.assertEqual(auto_res.automorphism_count, run_res.automorphism_count)
        self.assertEqual(
            {frozenset(cell) for cell in auto_res.orbits},
            {frozenset(cell) for cell in run_res.orbits},
        )
        self.assertEqual(auto_res.sample_mappings, run_res.sample_mappings)

    def test_small_chain_semantic_is_rigid(self) -> None:
        """
        A small directed chain should be rigid in semantic mode.
        """
        syn = SynCRN.from_reaction_strings(self.small_chain)
        eng = IRCanonicalEngine(syn, config=SymmetryConfig.semantic())

        res = eng.run(max_count=20, timeout_sec=5.0)

        self.assertEqual(res.automorphism_count, 1)
        for cell in res.orbits:
            self.assertEqual(len(cell), 1)

    def test_medium_semantic_is_rigid(self) -> None:
        """
        In semantic mode, the medium example is currently rigid.
        """
        syn = SynCRN.from_reaction_strings(self.medium)
        eng = IRCanonicalEngine(syn, config=SymmetryConfig.semantic())

        res = eng.run(max_count=512, timeout_sec=20.0)

        self.assertEqual(res.automorphism_count, 1)

    def test_medium_topological_has_nontrivial_symmetry(self) -> None:
        """
        In topological mode, the medium example should expose symmetry.
        """
        syn = SynCRN.from_reaction_strings(self.medium)
        eng = IRCanonicalEngine(syn, config=SymmetryConfig.topological())

        res = eng.run(max_count=512, timeout_sec=20.0)

        self.assertGreaterEqual(res.automorphism_count, 2)

    def test_medium_topological_count_is_512(self) -> None:
        """
        Current expected topological automorphism count for medium is 512.
        """
        syn = SynCRN.from_reaction_strings(self.medium)
        eng = IRCanonicalEngine(syn, config=SymmetryConfig.topological())

        res = eng.run(max_count=512, timeout_sec=20.0)

        self.assertEqual(res.automorphism_count, 512)

    def test_stop_after_two_caps_search(self) -> None:
        """
        ``stop_after_two=True`` should stop once nontrivial symmetry is detected.
        """
        syn = SynCRN.from_reaction_strings(self.medium)
        eng = IRCanonicalEngine(syn, config=SymmetryConfig.topological())

        res = eng.run(stop_after_two=True, timeout_sec=20.0)

        self.assertLessEqual(res.automorphism_count, 2)
        self.assertTrue(res.stopped_early or res.automorphism_count <= 2)

    def test_max_count_slices_sample_storage(self) -> None:
        """
        ``max_count`` should limit stored sample permutations and mappings.
        """
        syn = SynCRN.from_reaction_strings(self.medium)
        eng = IRCanonicalEngine(syn, config=SymmetryConfig.topological())

        res = eng.run(max_count=5, timeout_sec=20.0)

        self.assertLessEqual(len(res.sample_permutations), 5)
        self.assertLessEqual(len(res.sample_mappings), 5)
        self.assertLessEqual(res.automorphism_count, 5)

    def test_permuted_reaction_list_same_canonical_key(self) -> None:
        """
        Same CRN with different reaction insertion order should have same
        canonical key under the same config.
        """
        syn1 = SynCRN.from_reaction_strings(self.small_chain)
        syn2 = SynCRN.from_reaction_strings(self.small_chain_permuted)

        eng1 = IRCanonicalEngine(syn1, config=SymmetryConfig.semantic())
        eng2 = IRCanonicalEngine(syn2, config=SymmetryConfig.semantic())

        self.assertEqual(
            eng1.canonical_result(timeout_sec=5.0).canonical_key,
            eng2.canonical_result(timeout_sec=5.0).canonical_key,
        )

    def test_medium_permuted_same_canonical_key_topological(self) -> None:
        """
        Medium CRN and its permuted insertion-order version should have the same
        canonical key in topological mode.
        """
        syn1 = SynCRN.from_reaction_strings(self.medium)
        syn2 = SynCRN.from_reaction_strings(self.medium_permuted)

        eng1 = IRCanonicalEngine(syn1, config=SymmetryConfig.topological())
        eng2 = IRCanonicalEngine(syn2, config=SymmetryConfig.topological())

        self.assertEqual(
            eng1.canonical_result(timeout_sec=20.0).canonical_key,
            eng2.canonical_result(timeout_sec=20.0).canonical_key,
        )

    def test_completed_run_is_cached(self) -> None:
        """
        A full completed run without timeout should populate the exact cache.
        """
        syn = SynCRN.from_reaction_strings(self.small_chain)
        eng = IRCanonicalEngine(syn, config=SymmetryConfig.semantic())

        self.assertIsNone(eng._exact_full_cache)
        res1 = eng.run()
        self.assertIsNotNone(eng._exact_full_cache)

        res2 = eng.run()
        self.assertEqual(res1.canonical_key, res2.canonical_key)
        self.assertEqual(res1.canonical_order, res2.canonical_order)
        self.assertEqual(res1.automorphism_count, res2.automorphism_count)

    def test_timeout_run_is_not_reused_as_full_cache(self) -> None:
        """
        Timed runs should not populate the reusable full exact cache.
        """
        syn = SynCRN.from_reaction_strings(self.small_chain)
        eng = IRCanonicalEngine(syn, config=SymmetryConfig.semantic())

        eng.run(timeout_sec=5.0)
        self.assertIsNone(eng._exact_full_cache)

    def test_run_from_prebuilt_digraph(self) -> None:
        """
        The engine should work directly from ``syn.to_digraph()``.
        """
        syn = SynCRN.from_reaction_strings(self.medium)
        g = syn.to_digraph()

        eng = IRCanonicalEngine(g, config=SymmetryConfig.topological())
        res = eng.run(max_count=64, timeout_sec=10.0)

        self.assertIsInstance(eng.G, nx.DiGraph)
        self.assertGreaterEqual(res.automorphism_count, 2)

    def test_orbits_cover_all_nodes_when_not_truncated(self) -> None:
        """
        For a non-truncated exact run, orbit sets should cover all graph nodes.
        """
        syn = SynCRN.from_reaction_strings(self.small_chain)
        eng = IRCanonicalEngine(syn, config=SymmetryConfig.semantic())

        res = eng.run(timeout_sec=5.0)
        seen = set().union(*res.orbits) if res.orbits else set()

        self.assertEqual(seen, set(eng.G.nodes()))


if __name__ == "__main__":
    unittest.main()
