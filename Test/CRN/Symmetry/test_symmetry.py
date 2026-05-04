from __future__ import annotations

import unittest
from typing import Any, Iterable, Set

import networkx as nx

from synkit.CRN.Structure import SynCRN
from synkit.CRN.Symmetry._common import SymmetryConfig
from synkit.CRN.Symmetry.automorphism import CRNAutomorphism
from synkit.CRN.Symmetry.canon import CRNCanonicalizer
from synkit.CRN.Symmetry.symmetry import CRNSymmetry
from synkit.CRN.Symmetry.isomorphism import CRNIsomorphism
from synkit.CRN.Symmetry.wl_canon import WLCanonicalizer


class TestCRNSymmetryHelpers(unittest.TestCase):
    """Helper assertions for symmetry façade tests."""

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


class TestCRNSymmetry(unittest.TestCase):
    """Unit tests for the unified CRNSymmetry façade."""

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

    def test_default_config_is_topological(self) -> None:
        syn = SynCRN.from_reaction_strings(["A>>B", "B>>C"])
        sym = CRNSymmetry(syn)

        self.assertEqual(sym.config, SymmetryConfig.topological())

    def test_explicit_config_is_respected(self) -> None:
        syn = SynCRN.from_reaction_strings(["A>>B", "B>>C"])
        cfg = SymmetryConfig.semantic()

        sym = CRNSymmetry(syn, config=cfg)

        self.assertIs(sym.config, cfg)
        self.assertIs(sym.kwargs["config"], cfg)

    def test_build_from_syncrn_and_digraph(self) -> None:
        syn = SynCRN.from_reaction_strings(["A>>B", "B>>C"])
        g = syn.to_digraph()

        sym_from_syn = CRNSymmetry(syn)
        sym_from_g = CRNSymmetry(g)

        self.assertEqual(set(sym_from_syn.wl.G.nodes()), set(sym_from_g.wl.G.nodes()))
        self.assertEqual(set(sym_from_syn.wl.G.edges()), set(sym_from_g.wl.G.edges()))

    def test_component_types(self) -> None:
        syn = SynCRN.from_reaction_strings(["A>>B", "B>>C"])
        sym = CRNSymmetry(syn)

        self.assertIsInstance(sym.wl, WLCanonicalizer)
        self.assertIsInstance(sym.automorphism, CRNAutomorphism)
        self.assertIsInstance(sym.canonicalizer, CRNCanonicalizer)
        self.assertIsInstance(sym.isomorphism, CRNIsomorphism)

    def test_kwargs_are_propagated(self) -> None:
        syn = SynCRN.from_reaction_strings(["2A>>B", "B>>C"])
        cfg = SymmetryConfig.semantic()

        sym = CRNSymmetry(
            syn,
            include_rule=False,
            include_stoich=False,
            wl_iters=7,
            wl_digest_size=12,
            config=cfg,
        )

        self.assertEqual(
            sym.kwargs,
            {
                "include_rule": False,
                "include_stoich": False,
                "wl_iters": 7,
                "wl_digest_size": 12,
                "config": cfg,
            },
        )

    def test_wl_orbits_delegate(self) -> None:
        syn = SynCRN.from_reaction_strings(self.medium)
        sym = CRNSymmetry(syn)

        direct = sym.wl.orbits()
        via_facade = sym.wl_orbits()

        self.assertEqual(
            {frozenset(cell) for cell in direct},
            {frozenset(cell) for cell in via_facade},
        )

    def test_orbits_delegate(self) -> None:
        syn = SynCRN.from_reaction_strings(self.medium)
        sym = CRNSymmetry(syn)

        direct = sym.automorphism.orbits(max_count=128, timeout_sec=10.0)
        via_facade = sym.orbits(max_count=128, timeout_sec=10.0)

        self.assertEqual(
            {frozenset(cell) for cell in direct},
            {frozenset(cell) for cell in via_facade},
        )

    def test_has_nontrivial_automorphism_delegate(self) -> None:
        syn = SynCRN.from_reaction_strings(self.medium)
        sym = CRNSymmetry(syn)

        self.assertEqual(
            sym.automorphism.has_nontrivial_automorphism(timeout_sec=10.0),
            sym.has_nontrivial_automorphism(timeout_sec=10.0),
        )

    def test_automorphism_summary_delegate(self) -> None:
        syn = SynCRN.from_reaction_strings(self.medium)
        sym = CRNSymmetry(syn)

        direct = sym.automorphism.summary(max_count=128, timeout_sec=10.0)
        via_facade = sym.automorphism_summary(max_count=128, timeout_sec=10.0)

        self.assertEqual(direct.automorphism_count, via_facade.automorphism_count)
        self.assertEqual(
            {frozenset(cell) for cell in direct.orbits},
            {frozenset(cell) for cell in via_facade.orbits},
        )

    def test_canonical_result_delegate(self) -> None:
        syn = SynCRN.from_reaction_strings(["A>>B", "B>>C"])
        sym = CRNSymmetry(syn)

        direct = sym.canonicalizer.canonical_result(timeout_sec=5.0)
        via_facade = sym.canonical_result(timeout_sec=5.0)

        self.assertEqual(direct.canonical_order, via_facade.canonical_order)
        self.assertEqual(direct.canonical_key, via_facade.canonical_key)

    def test_canonical_graph_delegate(self) -> None:
        syn = SynCRN.from_reaction_strings(["A>>B", "B>>C"])
        sym = CRNSymmetry(syn)

        direct = sym.canonicalizer.canonical_graph(timeout_sec=5.0)
        via_facade = sym.canonical_graph(timeout_sec=5.0)

        self.assertEqual(set(direct.nodes()), set(via_facade.nodes()))
        self.assertEqual(set(direct.edges()), set(via_facade.edges()))

    def test_medium_has_nontrivial_automorphism_by_default(self) -> None:
        """
        CRNSymmetry defaults to topological mode, so the medium network
        should expose nontrivial symmetry.
        """
        syn = SynCRN.from_reaction_strings(self.medium)
        sym = CRNSymmetry(syn)

        self.assertTrue(sym.has_nontrivial_automorphism(timeout_sec=10.0))

    def test_medium_summary_count_is_512_by_default(self) -> None:
        """
        Under the current topological default, the medium network is expected
        to have 512 automorphisms.
        """
        syn = SynCRN.from_reaction_strings(self.medium)
        sym = CRNSymmetry(syn)

        result = sym.automorphism_summary(max_count=512, timeout_sec=20.0)
        self.assertEqual(result.automorphism_count, 512)

    def test_orbits_form_partition(self) -> None:
        syn = SynCRN.from_reaction_strings(self.medium)
        sym = CRNSymmetry(syn)

        orbits = sym.orbits(max_count=512, timeout_sec=20.0)
        universe = set(sym.wl.G.nodes())

        TestCRNSymmetryHelpers.assert_partition(self, universe, orbits)

    def test_wl_orbits_form_partition(self) -> None:
        syn = SynCRN.from_reaction_strings(self.medium)
        sym = CRNSymmetry(syn)

        orbits = sym.wl_orbits()
        universe = set(sym.wl.G.nodes())

        TestCRNSymmetryHelpers.assert_partition(self, universe, orbits)

    def test_canonical_graph_has_consecutive_integer_labels(self) -> None:
        syn = SynCRN.from_reaction_strings(self.medium)
        sym = CRNSymmetry(syn)

        g_canon = sym.canonical_graph(timeout_sec=10.0)
        n = sym.wl.G.number_of_nodes()

        self.assertIsInstance(g_canon, nx.DiGraph)
        self.assertEqual(set(g_canon.nodes()), set(range(1, n + 1)))

    def test_isomorphism_component_on_permuted_input(self) -> None:
        syn1 = SynCRN.from_reaction_strings(self.medium)
        syn2 = SynCRN.from_reaction_strings(self.medium_permuted)

        sym1 = CRNSymmetry(syn1)
        sym2 = CRNSymmetry(syn2)

        result = sym1.isomorphism.isomorphic_to(sym2.isomorphism)

        self.assertTrue(result.isomorphic)
        self.assertIsInstance(result.mapping, dict)

    def test_semantic_mode_can_be_requested(self) -> None:
        syn = SynCRN.from_reaction_strings(self.medium)
        sym = CRNSymmetry(syn, config=SymmetryConfig.semantic())

        self.assertFalse(sym.has_nontrivial_automorphism(timeout_sec=10.0))
        result = sym.automorphism_summary(max_count=64, timeout_sec=10.0)
        self.assertEqual(result.automorphism_count, 1)

    def test_construct_from_prebuilt_digraph(self) -> None:
        syn = SynCRN.from_reaction_strings(self.medium)
        g = syn.to_digraph()

        sym = CRNSymmetry(g)

        self.assertTrue(sym.has_nontrivial_automorphism(timeout_sec=10.0))
        result = sym.automorphism_summary(max_count=64, timeout_sec=10.0)
        self.assertGreaterEqual(result.automorphism_count, 2)

    def test_isomorphic_inputs_have_same_canonical_key(self) -> None:
        syn1 = SynCRN.from_reaction_strings(self.medium)
        syn2 = SynCRN.from_reaction_strings(self.medium_permuted)

        sym1 = CRNSymmetry(syn1)
        sym2 = CRNSymmetry(syn2)

        self.assertEqual(
            sym1.canonical_result(timeout_sec=10.0).canonical_key,
            sym2.canonical_result(timeout_sec=10.0).canonical_key,
        )


if __name__ == "__main__":
    unittest.main()
