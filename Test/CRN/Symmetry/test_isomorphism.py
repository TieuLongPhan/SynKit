from __future__ import annotations

import unittest

import networkx as nx

from synkit.CRN.Structure import SynCRN
from synkit.CRN.Symmetry.isomorphism import (
    CRNIsomorphism,
    are_isomorphic,
    are_subhypergraph_isomorphic,
)
from synkit.CRN.Symmetry._common import SymmetryConfig


class TestCRNIsomorphism(unittest.TestCase):
    """
    Unit tests for pairwise graph isomorphism and subgraph isomorphism
    on SynCRN-format directed bipartite graphs.
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

        cls.medium_permuted = [
            "AF>>AK",
            "AD+AJ>>V",
            "AD+AK>>AB",
            "AC+AK>>AA",
            "AB>>AG",
            "AF+AI>>F",
            "AC+AI>>C",
            "AB+AI>>B",
            "AB+AJ>>T",
            "AF+AK>>AD",
            "AC+AJ>>I",
            "AJ>>AE",
            "AD+AI>>D",
            "AF+AJ>>X",
            "AD+AJ>>J",
            "AI>>M",
            "AE+AK>>AC",
            "A+AI>>G",
            "A+AI>>X",
            "AC>>AH",
            "AC+AI>>P",
            "AD>>AI",
            "AF+AJ>>L",
            "AE+AI>>R",
            "AE>>AJ",
            "AI>>AD",
            "A+AJ>>Y",
            "AE+AJ>>K",
            "AE+AI>>E",
            "AD+AI>>Q",
            "AB+AJ>>H",
            "AF+AI>>S",
            "AB+AK>>Z",
            "AC+AJ>>U",
            "AB+AI>>O",
            "A+AJ>>M",
            "AE+AJ>>W",
            "A+AK>>AE",
            "AK>>AF",
            "AJ>>N",
        ]

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

        cls.small_chain_missing = [
            "A>>B",
            "B>>C",
        ]

        cls.pattern = [
            "A>>B",
            "B>>C",
        ]

        cls.host = [
            "A>>B",
            "B>>C",
            "C>>D",
            "D>>E",
        ]

        cls.nonisomorphic_same_size_a = [
            "A>>B",
            "B>>C",
        ]

        cls.nonisomorphic_same_size_b = [
            "A>>B",
            "A>>C",
        ]

    def test_build_from_syncrn_and_digraph(self) -> None:
        """
        Wrapper should be constructible from both SynCRN and its DiGraph.
        """
        syn = SynCRN.from_reaction_strings(self.small_chain)
        g = syn.to_digraph()

        iso_from_syn = CRNIsomorphism(syn)
        iso_from_g = CRNIsomorphism(g)

        self.assertIsInstance(iso_from_syn.G, nx.DiGraph)
        self.assertIsInstance(iso_from_g.G, nx.DiGraph)
        self.assertEqual(set(iso_from_syn.G.nodes()), set(iso_from_g.G.nodes()))
        self.assertEqual(set(iso_from_syn.G.edges()), set(iso_from_g.G.edges()))

    def test_graph_property_returns_internal_digraph(self) -> None:
        """
        ``G`` should expose the internal directed graph.
        """
        syn = SynCRN.from_reaction_strings(self.small_chain)
        iso = CRNIsomorphism(syn)

        self.assertIsInstance(iso.G, nx.DiGraph)

    def test_explicit_config_is_respected(self) -> None:
        """
        Passing an explicit symmetry config should be preserved.
        """
        syn = SynCRN.from_reaction_strings(self.small_chain)
        cfg = SymmetryConfig.semantic()

        iso = CRNIsomorphism(syn, config=cfg)

        self.assertIs(iso.config, cfg)

    def test_same_reactions_different_insertion_order_are_isomorphic_semantic(
        self,
    ) -> None:
        """
        Same reaction set with different insertion order should be isomorphic.
        """
        syn1 = SynCRN.from_reaction_strings(self.small_chain)
        syn2 = SynCRN.from_reaction_strings(self.small_chain_permuted)

        iso1 = CRNIsomorphism(syn1, config=SymmetryConfig.semantic())
        iso2 = CRNIsomorphism(syn2, config=SymmetryConfig.semantic())

        result = iso1.isomorphic_to(iso2)

        self.assertTrue(result.isomorphic)
        self.assertFalse(result.rejected_by_invariants)
        self.assertEqual(result.mode, "vf2")
        self.assertIsInstance(result.mapping, dict)

    def test_same_medium_different_order_are_isomorphic_semantic(self) -> None:
        """
        Same medium network in different order should be isomorphic.
        """
        syn1 = SynCRN.from_reaction_strings(self.medium)
        syn2 = SynCRN.from_reaction_strings(self.medium_permuted)

        iso1 = CRNIsomorphism(syn1, config=SymmetryConfig.semantic())
        iso2 = CRNIsomorphism(syn2, config=SymmetryConfig.semantic())

        result = iso1.isomorphic_to(iso2)

        self.assertTrue(result.isomorphic)
        self.assertFalse(result.rejected_by_invariants)
        self.assertIsInstance(result.mapping, dict)

    def test_same_medium_different_order_are_isomorphic_topological(self) -> None:
        """
        Same medium network in different order should also be isomorphic
        in topological mode.
        """
        syn1 = SynCRN.from_reaction_strings(self.medium)
        syn2 = SynCRN.from_reaction_strings(self.medium_permuted)

        iso1 = CRNIsomorphism(syn1, config=SymmetryConfig.topological())
        iso2 = CRNIsomorphism(syn2, config=SymmetryConfig.topological())

        result = iso1.isomorphic_to(iso2)

        self.assertTrue(result.isomorphic)
        self.assertFalse(result.rejected_by_invariants)
        self.assertIsInstance(result.mapping, dict)

    def test_syncrn_and_prebuilt_digraph_are_isomorphic(self) -> None:
        """
        SynCRN object and its own ``to_digraph()`` should be isomorphic.
        """
        syn = SynCRN.from_reaction_strings(self.medium)
        g = syn.to_digraph()

        iso1 = CRNIsomorphism(syn, config=SymmetryConfig.semantic())
        iso2 = CRNIsomorphism(g, config=SymmetryConfig.semantic())

        result = iso1.isomorphic_to(iso2)

        self.assertTrue(result.isomorphic)
        self.assertIsInstance(result.mapping, dict)

    def test_remove_one_reaction_breaks_full_isomorphism(self) -> None:
        """
        Removing one reaction should break full graph isomorphism.
        """
        syn1 = SynCRN.from_reaction_strings(self.small_chain)
        syn2 = SynCRN.from_reaction_strings(self.small_chain_missing)

        iso1 = CRNIsomorphism(syn1, config=SymmetryConfig.semantic())
        iso2 = CRNIsomorphism(syn2, config=SymmetryConfig.semantic())

        result = iso1.isomorphic_to(iso2)

        self.assertFalse(result.isomorphic)

    def test_real_structural_difference_is_not_isomorphic(self) -> None:
        """
        Same-size but structurally different CRNs should not be isomorphic.
        """
        syn1 = SynCRN.from_reaction_strings(self.nonisomorphic_same_size_a)
        syn2 = SynCRN.from_reaction_strings(self.nonisomorphic_same_size_b)

        iso1 = CRNIsomorphism(syn1, config=SymmetryConfig.semantic())
        iso2 = CRNIsomorphism(syn2, config=SymmetryConfig.semantic())

        result = iso1.isomorphic_to(iso2)

        self.assertFalse(result.isomorphic)

    def test_invariant_rejection_for_obvious_mismatch(self) -> None:
        """
        Obvious graph mismatch should usually be rejected by fast invariants.
        """
        syn1 = SynCRN.from_reaction_strings(self.small_chain)
        syn2 = SynCRN.from_reaction_strings(self.small_chain_missing)

        iso1 = CRNIsomorphism(syn1, config=SymmetryConfig.semantic())
        iso2 = CRNIsomorphism(syn2, config=SymmetryConfig.semantic())

        result = iso1.isomorphic_to(iso2)

        self.assertFalse(result.isomorphic)
        self.assertTrue(hasattr(result, "rejected_by_invariants"))

    def test_subgraph_isomorphic_true_for_pattern_inside_host(self) -> None:
        """
        Smaller pattern should embed into a larger host.
        """
        pattern = SynCRN.from_reaction_strings(self.pattern)
        host = SynCRN.from_reaction_strings(self.host)

        iso_pattern = CRNIsomorphism(pattern, config=SymmetryConfig.semantic())
        iso_host = CRNIsomorphism(host, config=SymmetryConfig.semantic())

        result = iso_pattern.subgraph_isomorphic_to(iso_host)

        self.assertTrue(result.isomorphic)
        self.assertFalse(result.rejected_by_invariants)
        self.assertEqual(result.mode, "vf2-subgraph")
        self.assertIsInstance(result.mapping, dict)

    def test_subgraph_isomorphic_false_when_pattern_larger_than_host(self) -> None:
        """
        Larger graph cannot be embedded into a smaller host.
        """
        pattern = SynCRN.from_reaction_strings(self.host)
        host = SynCRN.from_reaction_strings(self.pattern)

        iso_pattern = CRNIsomorphism(pattern, config=SymmetryConfig.semantic())
        iso_host = CRNIsomorphism(host, config=SymmetryConfig.semantic())

        result = iso_pattern.subgraph_isomorphic_to(iso_host)

        self.assertFalse(result.isomorphic)

    def test_subgraph_isomorphic_false_for_incompatible_pattern(self) -> None:
        """
        Structurally incompatible pattern should not match as subgraph.
        """
        pattern = SynCRN.from_reaction_strings(["A>>B", "A>>C"])
        host = SynCRN.from_reaction_strings(["A>>B", "B>>C", "C>>D"])

        iso_pattern = CRNIsomorphism(pattern, config=SymmetryConfig.semantic())
        iso_host = CRNIsomorphism(host, config=SymmetryConfig.semantic())

        result = iso_pattern.subgraph_isomorphic_to(iso_host)

        self.assertFalse(result.isomorphic)

    def test_wrapper_are_isomorphic_true(self) -> None:
        """
        Convenience wrapper should return True for isomorphic inputs.
        """
        syn1 = SynCRN.from_reaction_strings(self.medium)
        syn2 = SynCRN.from_reaction_strings(self.medium_permuted)

        ok = are_isomorphic(
            syn1,
            syn2,
            config=SymmetryConfig.semantic(),
        )
        self.assertTrue(ok)

    def test_wrapper_are_isomorphic_false(self) -> None:
        """
        Convenience wrapper should return False for non-isomorphic inputs.
        """
        syn1 = SynCRN.from_reaction_strings(self.small_chain)
        syn2 = SynCRN.from_reaction_strings(self.small_chain_missing)

        ok = are_isomorphic(
            syn1,
            syn2,
            config=SymmetryConfig.semantic(),
        )
        self.assertFalse(ok)

    def test_wrapper_are_subhypergraph_isomorphic_true(self) -> None:
        """
        Convenience wrapper should detect subgraph embedding.
        """
        pattern = SynCRN.from_reaction_strings(self.pattern)
        host = SynCRN.from_reaction_strings(self.host)

        ok = are_subhypergraph_isomorphic(
            pattern,
            host,
            config=SymmetryConfig.semantic(),
        )
        self.assertTrue(ok)

    def test_wrapper_are_subhypergraph_isomorphic_false(self) -> None:
        """
        Convenience wrapper should return False when no embedding exists.
        """
        pattern = SynCRN.from_reaction_strings(["A>>B", "A>>C"])
        host = SynCRN.from_reaction_strings(["A>>B", "B>>C", "C>>D"])

        ok = are_subhypergraph_isomorphic(
            pattern,
            host,
            config=SymmetryConfig.semantic(),
        )
        self.assertFalse(ok)

    def test_isomorphic_result_mapping_nodes_belong_to_graphs(self) -> None:
        """
        Returned mapping should map nodes from one graph to the other graph.
        """
        syn1 = SynCRN.from_reaction_strings(self.small_chain)
        syn2 = SynCRN.from_reaction_strings(self.small_chain_permuted)

        iso1 = CRNIsomorphism(syn1, config=SymmetryConfig.semantic())
        iso2 = CRNIsomorphism(syn2, config=SymmetryConfig.semantic())

        result = iso1.isomorphic_to(iso2)

        self.assertTrue(result.isomorphic)
        self.assertIsNotNone(result.mapping)

        mapping = result.mapping or {}
        self.assertTrue(set(mapping.keys()).issubset(set(iso1.G.nodes())))
        self.assertTrue(set(mapping.values()).issubset(set(iso2.G.nodes())))

    def test_subgraph_result_mapping_targets_host_nodes(self) -> None:
        """
        Subgraph mapping should target host-graph nodes.
        """
        pattern = SynCRN.from_reaction_strings(self.pattern)
        host = SynCRN.from_reaction_strings(self.host)

        iso_pattern = CRNIsomorphism(pattern, config=SymmetryConfig.semantic())
        iso_host = CRNIsomorphism(host, config=SymmetryConfig.semantic())

        result = iso_pattern.subgraph_isomorphic_to(iso_host)

        self.assertTrue(result.isomorphic)
        self.assertIsNotNone(result.mapping)

        mapping = result.mapping or {}
        self.assertTrue(set(mapping.keys()).issubset(set(iso_host.G.nodes())))
        self.assertTrue(set(mapping.values()).issubset(set(iso_pattern.G.nodes())))

    def test_topological_mode_can_ignore_label_identity(self) -> None:
        """
        Two structurally identical CRNs with renamed species should fail in
        semantic mode but pass in topological mode.

        This assumes topological mode does not enforce label identity.
        """
        syn1 = SynCRN.from_reaction_strings(
            [
                "A>>B",
                "B>>C",
            ]
        )
        syn2 = SynCRN.from_reaction_strings(
            [
                "X>>Y",
                "Y>>Z",
            ]
        )

        sem1 = CRNIsomorphism(syn1, config=SymmetryConfig.semantic())
        sem2 = CRNIsomorphism(syn2, config=SymmetryConfig.semantic())
        top1 = CRNIsomorphism(syn1, config=SymmetryConfig.topological())
        top2 = CRNIsomorphism(syn2, config=SymmetryConfig.topological())

        sem_res = sem1.isomorphic_to(sem2)
        top_res = top1.isomorphic_to(top2)

        self.assertFalse(sem_res.isomorphic)
        self.assertTrue(top_res.isomorphic)


if __name__ == "__main__":
    unittest.main()
