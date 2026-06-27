"""
Sprint 2 tests — bipartite component assignment (P3, P4).

Covers both MCSMatcher and ApproxMCSMatcher.

Core invariant being tested
---------------------------
Given two multi-component graphs where greedy size-sorting picks the WRONG
pairs (same-size components with different element types), the bipartite
assignment must find the CORRECT pairing and produce a LARGER total mapping.

Scenario used throughout
------------------------
G1 has components A (C-chain, 6 atoms) and B (N-chain, 4 atoms).
G2 has components X (N-chain, 4 atoms) and Y (C-chain, 6 atoms).

Correct pairing: A ↔ Y (6 C-atoms match), B ↔ X (4 N-atoms match) → 10 mapped.
Greedy (descending size): A ↔ Y first (both size 6, correct here by luck),
then B ↔ X (both size 4, correct). Actually this case is correct for greedy too
UNLESS we make sizes equal to expose the element-type problem.

Harder case (all same size, different elements):
G1: A (all-C, 4 atoms), B (all-N, 4 atoms)
G2: X (all-N, 4 atoms), Y (all-C, 4 atoms)
Greedy: A ↔ X (first pair, no match — C vs N), then B ↔ Y (N vs C, no match) → 0.
Bipartite: A ↔ Y (C vs C, 4 atoms) + B ↔ X (N vs N, 4 atoms) → 8.
"""

import time
import unittest

import networkx as nx

from synkit.Graph.Matcher.mcs_matcher import MCSMatcher
from synkit.Graph.Matcher.approx_mcs import ApproxMCSMatcher

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chain(elements: list[str], offset: int = 0) -> nx.Graph:
    """Linear chain graph with given element sequence. Nodes start at `offset`."""
    G = nx.Graph()
    for i, e in enumerate(elements):
        G.add_node(i + offset, element=e)
        if i:
            G.add_edge(i - 1 + offset, i + offset, order=1)
    return G


def _disjoint(g1: nx.Graph, g2: nx.Graph) -> nx.Graph:
    """Disjoint union that keeps node IDs from each graph distinct."""
    return nx.disjoint_union(g1, g2)


# ---------------------------------------------------------------------------
# Greedy-failure fixture
# ---------------------------------------------------------------------------
#
# All components have the same size (4 atoms) so size-sorting cannot
# differentiate pairs → greedy always picks positional order.
#
# G1 components: A=[C,C,C,C]  B=[N,N,N,N]
# G2 components: X=[N,N,N,N]  Y=[C,C,C,C]
#
# nx.disjoint_union renumbers nodes, so A occupies 0..3, B occupies 4..7
# in G1; X occupies 0..3, Y occupies 4..7 in G2.
#
# Greedy pairs A(pos 0) ↔ X(pos 0): C vs N → zero MCS.
#              B(pos 1) ↔ Y(pos 1): N vs C → zero MCS.  Total = 0.
#
# Bipartite builds score matrix:
#   score[A,X]=0, score[A,Y]=4, score[B,X]=4, score[B,Y]=0
# Optimal assignment: A↔Y + B↔X → total = 8.


def _same_size_mismatched_fixture() -> tuple[nx.Graph, nx.Graph]:
    A = _chain(["C", "C", "C", "C"])
    B = _chain(["N", "N", "N", "N"])
    X = _chain(["N", "N", "N", "N"])
    Y = _chain(["C", "C", "C", "C"])
    G1 = _disjoint(A, B)
    G2 = _disjoint(X, Y)
    return G1, G2


def _asymmetric_fixture() -> tuple[nx.Graph, nx.Graph]:
    """
    Asymmetric sizes so greedy (desc) pairs large first.

    G1: A=[C]*6, B=[N]*4     G2: X=[N]*4, Y=[C]*6
    Greedy descending: A(6)↔Y(6) ✓, B(4)↔X(4) ✓ — greedy works here.
    But score matrix still gives the same result via Hungarian.
    Total correct = 10.
    """
    A = _chain(["C"] * 6)
    B = _chain(["N"] * 4)
    X = _chain(["N"] * 4)
    Y = _chain(["C"] * 6)
    G1 = _disjoint(A, B)
    G2 = _disjoint(X, Y)
    return G1, G2


# ---------------------------------------------------------------------------
# P3 — MCSMatcher bipartite assignment
# ---------------------------------------------------------------------------


class TestMCSMatcherBipartite(unittest.TestCase):

    def test_same_size_wrong_element_greedy_fails_bipartite_wins(self):
        """Greedy maps 0 atoms; bipartite maps 8 atoms."""
        G1, G2 = _same_size_mismatched_fixture()

        greedy = MCSMatcher(node_attrs=["element"])
        greedy_map = greedy._componentwise_mcs(G1, G2, mcs=True)

        bipartite = MCSMatcher(node_attrs=["element"])
        bipartite_map = bipartite._bipartite_assign_components(G1, G2, mcs=True)

        self.assertEqual(len(greedy_map), 0, "greedy should fail for this fixture")
        self.assertEqual(
            len(bipartite_map), 8, f"bipartite mapped {len(bipartite_map)}, expected 8"
        )

    def test_bipartite_ge_greedy_on_asymmetric(self):
        """Bipartite must match at least as many atoms as greedy."""
        G1, G2 = _asymmetric_fixture()
        m = MCSMatcher(node_attrs=["element"])
        greedy_map = m._componentwise_mcs(G1, G2, mcs=True)
        bipartite_map = m._bipartite_assign_components(G1, G2, mcs=True)
        self.assertGreaterEqual(len(bipartite_map), len(greedy_map))
        self.assertEqual(len(bipartite_map), 10)

    def test_find_rc_mapping_component_uses_bipartite(self):
        """
        find_rc_mapping(component=True) must return the bipartite-optimal
        mapping (8 atoms) not the greedy one (0 atoms) for the failing fixture.
        """
        G1, G2 = _same_size_mismatched_fixture()
        m = MCSMatcher(node_attrs=["element"])
        m.find_rc_mapping(G1, G2, side="its", mcs=True, component=True)
        result = m.get_mappings(direction="G1_to_G2")
        self.assertTrue(result, "no mappings returned")
        self.assertEqual(len(result[0]), 8)

    def test_mapping_values_are_valid_node_ids(self):
        """All mapped node ids must actually exist in the respective graphs."""
        G1, G2 = _same_size_mismatched_fixture()
        m = MCSMatcher(node_attrs=["element"])
        combined = m._bipartite_assign_components(G1, G2, mcs=True)
        for k, v in combined.items():
            self.assertIn(k, G1, f"key {k} not in G1")
            self.assertIn(v, G2, f"value {v} not in G2")

    def test_elements_matched_correctly(self):
        """Each mapped pair must have the same element."""
        G1, G2 = _same_size_mismatched_fixture()
        m = MCSMatcher(node_attrs=["element"])
        combined = m._bipartite_assign_components(G1, G2, mcs=True)
        for g1_node, g2_node in combined.items():
            e1 = G1.nodes[g1_node]["element"]
            e2 = G2.nodes[g2_node]["element"]
            self.assertEqual(
                e1, e2, f"element mismatch: {g1_node}({e1}) ↔ {g2_node}({e2})"
            )

    def test_single_component_bipartite_equals_mcs(self):
        """With one component each, bipartite == standard MCS."""
        G1 = _chain(["C", "O", "C"])
        G2 = _chain(["C", "O", "C"])
        m = MCSMatcher(node_attrs=["element"])
        bipartite_map = m._bipartite_assign_components(G1, G2, mcs=True)
        self.assertEqual(len(bipartite_map), 3)

    def test_no_shared_elements_returns_empty(self):
        """No element overlap → empty mapping."""
        G1 = _chain(["C", "C"])
        G2 = _chain(["N", "N"])
        m = MCSMatcher(node_attrs=["element"])
        result = m._bipartite_assign_components(G1, G2, mcs=True)
        self.assertEqual(len(result), 0)

    def test_mcs_false_finds_any_common_subgraph(self):
        """mcs=False must still find mappings (not require max-size)."""
        G1, G2 = _asymmetric_fixture()
        m = MCSMatcher(node_attrs=["element"])
        result = m._bipartite_assign_components(G1, G2, mcs=False)
        self.assertGreater(len(result), 0)


# ---------------------------------------------------------------------------
# P3 — ApproxMCSMatcher bipartite assignment
# ---------------------------------------------------------------------------


class TestApproxMCSMatcherBipartite(unittest.TestCase):

    def test_same_size_wrong_element_greedy_fails_bipartite_wins(self):
        G1, G2 = _same_size_mismatched_fixture()
        m = ApproxMCSMatcher(node_attrs=["element"])
        greedy_map = m._componentwise_approx(G1, G2, max_seeds=8, max_steps=64)
        bipartite_map = m._bipartite_assign_components(
            G1, G2, max_seeds=8, max_steps=64
        )
        self.assertEqual(len(greedy_map), 0, "greedy should fail for this fixture")
        self.assertGreaterEqual(
            len(bipartite_map), 6, "bipartite must map at least 6 atoms"
        )

    def test_bipartite_ge_greedy_on_asymmetric(self):
        G1, G2 = _asymmetric_fixture()
        m = ApproxMCSMatcher(node_attrs=["element"])
        greedy_map = m._componentwise_approx(G1, G2, max_seeds=8, max_steps=64)
        bipartite_map = m._bipartite_assign_components(
            G1, G2, max_seeds=8, max_steps=64
        )
        self.assertGreaterEqual(len(bipartite_map), len(greedy_map))

    def test_find_rc_mapping_component_uses_bipartite(self):
        """find_rc_mapping(component=True) uses _bipartite_assign_components."""
        G1, G2 = _same_size_mismatched_fixture()
        m = ApproxMCSMatcher(node_attrs=["element"])
        m.find_rc_mapping(
            G1, G2, side="its", component=True, max_seeds=16, max_steps=128
        )
        result = m.get_mappings(direction="G1_to_G2")
        self.assertTrue(result)
        self.assertGreater(len(result[0]), 0)

    def test_mapping_values_are_valid_node_ids(self):
        G1, G2 = _same_size_mismatched_fixture()
        m = ApproxMCSMatcher(node_attrs=["element"])
        combined = m._bipartite_assign_components(G1, G2, max_seeds=8, max_steps=64)
        for k, v in combined.items():
            self.assertIn(k, G1, f"key {k} not in G1")
            self.assertIn(v, G2, f"value {v} not in G2")

    def test_elements_matched_correctly(self):
        G1, G2 = _same_size_mismatched_fixture()
        m = ApproxMCSMatcher(node_attrs=["element"])
        combined = m._bipartite_assign_components(G1, G2, max_seeds=8, max_steps=64)
        for g1_node, g2_node in combined.items():
            e1 = G1.nodes[g1_node]["element"]
            e2 = G2.nodes[g2_node]["element"]
            self.assertEqual(
                e1, e2, f"element mismatch: {g1_node}({e1}) ↔ {g2_node}({e2})"
            )

    def test_single_component_bipartite_finds_mapping(self):
        G1 = _chain(["C", "O", "C"])
        G2 = _chain(["C", "O", "C"])
        m = ApproxMCSMatcher(node_attrs=["element"])
        result = m._bipartite_assign_components(G1, G2, max_seeds=8, max_steps=64)
        self.assertEqual(len(result), 3)

    def test_with_wl_augmentation(self):
        """use_wl=True must still produce correct bipartite mapping."""
        G1, G2 = _same_size_mismatched_fixture()
        m = ApproxMCSMatcher(node_attrs=["element"], use_wl=True)
        bipartite_map = m._bipartite_assign_components(
            G1, G2, max_seeds=8, max_steps=64
        )
        self.assertGreaterEqual(len(bipartite_map), 6)


# ---------------------------------------------------------------------------
# P4 — Performance / timing
# ---------------------------------------------------------------------------


class TestBipartitePerformance(unittest.TestCase):

    def test_mcs_bipartite_2x2_timing(self):
        """2×2 bipartite score matrix + Hungarian must finish in < 5 s."""
        G1, G2 = _same_size_mismatched_fixture()
        m = MCSMatcher(node_attrs=["element"])
        t0 = time.perf_counter()
        m._bipartite_assign_components(G1, G2, mcs=True)
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 5.0, f"bipartite took {elapsed:.2f}s, expected < 5s")

    def test_approx_bipartite_2x2_timing(self):
        """Approx 2×2 bipartite must finish in < 0.5 s."""
        G1, G2 = _same_size_mismatched_fixture()
        m = ApproxMCSMatcher(node_attrs=["element"])
        t0 = time.perf_counter()
        m._bipartite_assign_components(G1, G2, max_seeds=16, max_steps=256)
        elapsed = time.perf_counter() - t0
        self.assertLess(
            elapsed, 0.5, f"approx bipartite took {elapsed:.3f}s, expected < 0.5s"
        )

    def test_approx_bipartite_3x3_timing(self):
        """3×3 matrix (6-component total) finishes in < 2 s."""
        C4 = _chain(["C", "C", "C", "C"])
        N4 = _chain(["N", "N", "N", "N"])
        O4 = _chain(["O", "O", "O", "O"])
        G1 = nx.disjoint_union(nx.disjoint_union(C4, N4), O4)
        # Shuffle order in G2
        G2 = nx.disjoint_union(nx.disjoint_union(O4, C4), N4)
        m = ApproxMCSMatcher(node_attrs=["element"])
        t0 = time.perf_counter()
        result = m._bipartite_assign_components(G1, G2, max_seeds=16, max_steps=256)
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 2.0)
        # All 12 atoms should be matched
        self.assertEqual(len(result), 12)

    def test_mcs_bipartite_3x3_correct(self):
        """3×3 bipartite MCS maps all 12 atoms correctly by element."""
        C4 = _chain(["C", "C", "C", "C"])
        N4 = _chain(["N", "N", "N", "N"])
        O4 = _chain(["O", "O", "O", "O"])
        G1 = nx.disjoint_union(nx.disjoint_union(C4, N4), O4)
        G2 = nx.disjoint_union(nx.disjoint_union(O4, C4), N4)
        m = MCSMatcher(node_attrs=["element"])
        result = m._bipartite_assign_components(G1, G2, mcs=True)
        self.assertEqual(len(result), 12)
        for g1n, g2n in result.items():
            self.assertEqual(
                G1.nodes[g1n]["element"],
                G2.nodes[g2n]["element"],
            )


# ---------------------------------------------------------------------------
# Integration — full RBL pipeline uses bipartite under the hood
# ---------------------------------------------------------------------------


class TestBipartiteViaRBLEngine(unittest.TestCase):
    """
    Smoke-test that the RBL engine (which calls find_rc_mapping with
    component=True) still produces valid RSMIs after the bipartite switch.
    """

    def test_transesterification_still_works(self):
        from synkit.Synthesis.Reactor.rbl_engine import RBLEngine

        engine = RBLEngine(mode="full")
        engine.process(
            "CCC(=O)OC>>CCC(=O)OCC",
            "[C:1][O:2].[O:3][H:4]>>[C:1][O:3].[O:2][H:4]",
        )
        self.assertTrue(engine.fused_rsmis, "no fused RSMIs produced")
        for rsmi in engine.fused_rsmis:
            self.assertIn(">>", rsmi)

    def test_ester_formation_still_works(self):
        from synkit.Synthesis.Reactor.rbl_engine import RBLEngine
        from synkit.IO import its_to_rsmi, rsmi_to_its

        raw_template = (
            "[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][O:6][H:7]>>"
            "[CH3:1][C:2](=[O:3])[O:6][CH3:5].[H:7][OH:4]"
        )
        its_template = rsmi_to_its(raw_template, core=True)
        template = its_to_rsmi(its_template)
        engine = RBLEngine(mode="full")
        engine.process("CCC(=O)(O)>>CCC(=O)OC", template, replace_wc=True)
        self.assertTrue(engine.fused_rsmis)


if __name__ == "__main__":
    unittest.main()
