from __future__ import annotations

import unittest
from typing import Any, Iterable, List, Set

import networkx as nx

from synkit.CRN.Structure import SynCRN
from synkit.CRN.Symmetry.automorphism import CRNAutomorphism, detect_automorphisms
from synkit.CRN.Symmetry.canon import CRNCanonicalizer
from synkit.CRN.Symmetry._ir import IRCanonicalEngine
from synkit.CRN.Symmetry._common import SymmetryConfig


class TestCRNAutomorphismHelpers(unittest.TestCase):
    """Small helper checks for orbit partitions."""

    @staticmethod
    def flatten_orbits(orbits: Iterable[Set[Any]]) -> Set[Any]:
        out: Set[Any] = set()
        for cell in orbits:
            out.update(cell)
        return out

    @staticmethod
    def assert_partition(
        testcase: unittest.TestCase,
        universe: Set[Any],
        orbits: List[Set[Any]],
    ) -> None:
        """Assert that ``orbits`` forms a valid partition of ``universe``."""
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


class TestCRNAutomorphism(unittest.TestCase):
    """
    Unit tests for exact automorphism analysis on SynCRN-format graphs.

    Assumed SynCRN format
    ---------------------
    - directed bipartite graph
    - species nodes and rule nodes
    - edges species -> rule for reactants
    - edges rule -> species for products
    - reaction ids preserved in the graph representation
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
        """Fresh analyzer should be constructible from SynCRN and digraph."""
        syn = SynCRN.from_reaction_strings(["A>>B", "B>>C"])
        g = syn.to_digraph()

        auto_from_syn = CRNAutomorphism(syn)
        auto_from_g = CRNAutomorphism(g)

        self.assertIsInstance(auto_from_syn.G, nx.DiGraph)
        self.assertIsInstance(auto_from_g.G, nx.DiGraph)
        self.assertEqual(set(auto_from_syn.G.nodes()), set(auto_from_g.G.nodes()))
        self.assertEqual(set(auto_from_syn.G.edges()), set(auto_from_g.G.edges()))

    def test_graph_property_returns_internal_digraph(self) -> None:
        """The public ``G`` property should expose the internal graph."""
        syn = SynCRN.from_reaction_strings(["A>>B"])
        auto = CRNAutomorphism(syn)

        self.assertIs(auto.G, auto._engine.G)
        self.assertIsInstance(auto.G, nx.DiGraph)

    def test_reuse_from_canonicalizer(self) -> None:
        """Passing a canonicalizer should reuse config, WL, and engine."""
        syn = SynCRN.from_reaction_strings(["A>>B", "B>>C"])
        canon = CRNCanonicalizer(syn)

        auto = CRNAutomorphism(canon)

        self.assertIs(auto.config, canon.config)
        self.assertIs(auto.wl, canon.wl)
        self.assertIs(auto._engine, canon.engine)

    def test_reuse_from_engine(self) -> None:
        """Passing an IR engine should reuse that engine directly."""
        syn = SynCRN.from_reaction_strings(["A>>B", "B>>C"])
        engine = IRCanonicalEngine(syn)

        auto = CRNAutomorphism(engine)

        self.assertIs(auto.config, engine.config)
        self.assertIs(auto.wl, engine.wl)
        self.assertIs(auto._engine, engine)

    def test_asymmetric_network_has_no_nontrivial_automorphism(self) -> None:
        """A small chain should not have nontrivial automorphisms."""
        syn = SynCRN.from_reaction_strings(
            [
                "A>>B",
                "B>>C",
                "C>>D",
            ]
        )
        auto = CRNAutomorphism(syn)

        self.assertFalse(auto.has_nontrivial_automorphism(timeout_sec=2.0))

        result = auto.summary(max_count=20, timeout_sec=2.0)
        self.assertGreaterEqual(result.automorphism_count, 1)

        for cell in result.orbits:
            self.assertEqual(len(cell), 1)

    def test_medium_network_is_rigid_in_semantic_mode(self) -> None:
        """
        In semantic mode, the provided medium network is expected to be rigid.

        This matches the current observed behavior: automorphism_count == 1.
        """
        syn = SynCRN.from_reaction_strings(self.medium)
        auto = CRNAutomorphism(syn, config=SymmetryConfig.semantic())

        self.assertFalse(auto.has_nontrivial_automorphism(timeout_sec=10.0))

        result = auto.summary(max_count=512, timeout_sec=20.0)
        self.assertEqual(result.automorphism_count, 1)

    def test_medium_network_has_nontrivial_automorphism_in_topological_mode(
        self,
    ) -> None:
        """
        In topological mode, labels are ignored more aggressively, so the
        medium example should expose nontrivial symmetry.
        """
        syn = SynCRN.from_reaction_strings(self.medium)
        auto = CRNAutomorphism(syn, config=SymmetryConfig.topological())

        self.assertTrue(auto.has_nontrivial_automorphism(timeout_sec=10.0))

    def test_medium_summary_count_is_512_in_topological_mode(self) -> None:
        """
        The provided medium network is expected to have 512 automorphisms
        in topological mode.
        """
        syn = SynCRN.from_reaction_strings(self.medium)
        auto = CRNAutomorphism(syn, config=SymmetryConfig.topological())

        result = auto.summary(max_count=512, timeout_sec=20.0)
        self.assertEqual(result.automorphism_count, 512)

    def test_orbits_form_partition_of_graph_nodes(self) -> None:
        """Exact orbit output should form a valid partition."""
        syn = SynCRN.from_reaction_strings(self.medium)
        auto = CRNAutomorphism(syn, config=SymmetryConfig.topological())

        orbits = auto.orbits(max_count=512, timeout_sec=20.0)
        universe = set(auto.G.nodes())

        TestCRNAutomorphismHelpers.assert_partition(self, universe, orbits)

    def test_wl_orbits_form_partition_of_graph_nodes(self) -> None:
        """WL color classes should also form a valid partition."""
        syn = SynCRN.from_reaction_strings(self.medium)
        auto = CRNAutomorphism(syn, config=SymmetryConfig.topological())

        wl_orbits = auto.wl_orbits()
        universe = set(auto.G.nodes())

        TestCRNAutomorphismHelpers.assert_partition(self, universe, wl_orbits)

    def test_automorphisms_iter_yields_dict_mappings(self) -> None:
        """``automorphisms_iter`` should yield node->node dictionaries."""
        syn = SynCRN.from_reaction_strings(self.medium)
        auto = CRNAutomorphism(syn, config=SymmetryConfig.topological())

        mappings = list(auto.automorphisms_iter(max_count=5, timeout_sec=10.0))

        self.assertGreaterEqual(len(mappings), 1)
        for mapping in mappings:
            self.assertIsInstance(mapping, dict)
            self.assertTrue(set(mapping.keys()).issubset(set(auto.G.nodes())))
            self.assertTrue(set(mapping.values()).issubset(set(auto.G.nodes())))

    def test_detect_automorphisms_wrapper_matches_summary(self) -> None:
        """The convenience wrapper should agree with the class-based API."""
        syn = SynCRN.from_reaction_strings(self.medium)
        cfg = SymmetryConfig.topological()

        result_fn = detect_automorphisms(
            syn,
            max_count=512,
            timeout_sec=20.0,
            config=cfg,
        )
        result_cls = CRNAutomorphism(syn, config=cfg).summary(
            max_count=512,
            timeout_sec=20.0,
        )

        self.assertEqual(result_fn.automorphism_count, result_cls.automorphism_count)
        self.assertEqual(
            {frozenset(cell) for cell in result_fn.orbits},
            {frozenset(cell) for cell in result_cls.orbits},
        )

    def test_explicit_config_is_respected(self) -> None:
        """Passing an explicit config should be preserved."""
        syn = SynCRN.from_reaction_strings(["A>>B", "B>>C"])
        cfg = SymmetryConfig.semantic()

        auto = CRNAutomorphism(syn, config=cfg)

        self.assertIs(auto.config, cfg)

    def test_construct_from_prebuilt_digraph_in_semantic_mode(self) -> None:
        """
        The class should work directly from ``syn.to_digraph()``.

        In semantic mode, the medium graph is currently rigid.
        """
        syn = SynCRN.from_reaction_strings(self.medium)
        crn = syn.to_digraph()

        auto = CRNAutomorphism(crn, config=SymmetryConfig.semantic())
        result = auto.summary(max_count=64, timeout_sec=10.0)

        self.assertIsInstance(auto.G, nx.DiGraph)
        self.assertEqual(result.automorphism_count, 1)

    def test_construct_from_prebuilt_digraph_in_topological_mode(self) -> None:
        """
        The class should also detect symmetry from a prebuilt digraph when
        using topological matching.
        """
        syn = SynCRN.from_reaction_strings(self.medium)
        crn = syn.to_digraph()

        auto = CRNAutomorphism(crn, config=SymmetryConfig.topological())
        result = auto.summary(max_count=64, timeout_sec=10.0)

        self.assertIsInstance(auto.G, nx.DiGraph)
        self.assertGreaterEqual(result.automorphism_count, 2)


if __name__ == "__main__":
    unittest.main()
