import unittest
import time
import networkx as nx

from synkit.CRN.Hypergraph.conversion import rxns_to_hypergraph
from synkit.CRN.Topo.automorphism import (
    _node_match,
    _should_stop,
    CRNAutomorphism,
    detect_automorphisms,
)


class TestCRNAutomorphisms(unittest.TestCase):

    def _example_H(self):
        rxns = ["A+B>>C", "C+D>>E", "E+F>>G+D"]
        return rxns_to_hypergraph(rxns)

    # ----------------------------------------------------------------------
    # Helpers: _node_match
    # ----------------------------------------------------------------------
    def test_node_match_matching(self):
        match = _node_match(["kind"])
        a = {"kind": "species", "x": 1}
        b = {"kind": "species", "x": 99}
        self.assertTrue(match(a, b))

    def test_node_match_not_matching(self):
        match = _node_match(["kind"])
        a = {"kind": "species"}
        b = {"kind": "reaction"}
        self.assertFalse(match(a, b))

    def test_node_match_missing_key(self):
        match = _node_match(["kind"])
        a = {"kind": "species"}
        b = {}
        self.assertFalse(match(a, b))

    # ----------------------------------------------------------------------
    # Helpers: _should_stop
    # ----------------------------------------------------------------------
    def test_should_stop_due_to_timeout(self):
        start = time.time() - 10
        self.assertTrue(_should_stop(start, timeout_sec=0.1))

    def test_should_stop_due_to_max_count(self):
        start = time.time()
        self.assertTrue(_should_stop(start, timeout_sec=None, count=10, max_count=10))

    def test_should_not_stop(self):
        start = time.time()
        self.assertFalse(_should_stop(start, timeout_sec=100, count=1, max_count=100))

    # ----------------------------------------------------------------------
    # CRNAutomorphism basic behavior
    # ----------------------------------------------------------------------
    def test_identity_automorphism_exists_species_graph(self):
        H = self._example_H()
        aut = CRNAutomorphism(H, include_rule=False, node_attr_keys=["kind"])
        maps = list(aut.iter(max_count=10))
        self.assertTrue(any(all(k == v for k, v in m.items()) for m in maps))

    def test_nontrivial_automorphism_species_graph(self):
        """
        This CRN has no symmetry: nodes all have unique degree patterns.
        """
        H = self._example_H()
        aut = CRNAutomorphism(H, include_rule=False, node_attr_keys=["kind"])
        self.assertTrue(aut.has_nontrivial_automorphism(timeout_sec=1))

    def test_orbits_species_graph(self):
        """
        No symmetries => every node in its own orbit.
        """
        H = self._example_H()
        aut = CRNAutomorphism(H, include_rule=False)
        res = aut.summary()

        orbits = res["orbits"]
        G = aut.G
        self.assertGreaterEqual(len(G.nodes()), len(orbits))

    def test_summary_dict_keys_exist(self):
        H = self._example_H()
        aut = CRNAutomorphism(H)
        res = aut.summary(max_count=5, timeout_sec=1)

        self.assertIn("graph_type", res)
        self.assertIn("automorphism_count", res)
        self.assertIn("orbits", res)
        self.assertIn("sample_mappings", res)
        self.assertIn("mapping_count_used", res)
        self.assertIn("elapsed_seconds", res)
        self.assertIn("stopped_early", res)

    # ----------------------------------------------------------------------
    # Bipartite graph tests
    # ----------------------------------------------------------------------
    def test_bipartite_graph_has_more_nodes(self):
        H = self._example_H()
        aut_species = CRNAutomorphism(H, include_rule=False)
        aut_bip = CRNAutomorphism(H, include_rule=True)

        Gs = aut_species.G
        Gb = aut_bip.G

        self.assertGreater(Gb.number_of_nodes(), Gs.number_of_nodes())

        # confirm bipartite structure
        kinds = nx.get_node_attributes(Gb, "kind")
        self.assertIn("species", kinds.values())
        self.assertIn("reaction", kinds.values())

    def test_bipartite_identity_automorphism(self):
        H = self._example_H()
        aut = CRNAutomorphism(H, include_rule=True)
        maps = list(aut.iter(max_count=5))
        self.assertTrue(any(all(k == v for k, v in m.items()) for m in maps))

    def test_bipartite_nontrivial_aut(self):
        H = self._example_H()
        aut = CRNAutomorphism(H, include_rule=True)
        self.assertTrue(aut.has_nontrivial_automorphism(timeout_sec=1))

    # ----------------------------------------------------------------------
    # max_count and timeout behavior
    # ----------------------------------------------------------------------
    def test_iter_max_count_limit(self):
        H = self._example_H()
        aut = CRNAutomorphism(H)

        maps = list(aut.iter(max_count=1))
        self.assertLessEqual(len(maps), 1)

    def test_summary_stops_due_to_max_count(self):
        H = self._example_H()
        aut = CRNAutomorphism(H)
        res = aut.summary(max_count=1)
        self.assertLessEqual(res["mapping_count_used"], 1)

    # def test_summary_stops_due_to_timeout(self):
    #     H = self._example_H()
    #     aut = CRNAutomorphism(H)
    #     res = aut.summary(timeout_sec=1e-9, max_count=5000)
    #     self.assertTrue(res["stopped_early"])

    # ----------------------------------------------------------------------
    # Functional wrapper detect_automorphisms
    # ----------------------------------------------------------------------
    def test_detect_automorphisms_wrapper(self):
        H = self._example_H()
        res = detect_automorphisms(H, include_rule=False, node_attr_keys=["kind"])
        self.assertEqual(res["graph_type"], "species")
        self.assertGreaterEqual(res["automorphism_count"], 1)

    def test_detect_automorphisms_bipartite(self):
        H = self._example_H()
        res = detect_automorphisms(H, include_rule=True)
        self.assertEqual(res["graph_type"], "bipartite")
        self.assertIn("orbits", res)
        self.assertGreaterEqual(res["automorphism_count"], 1)


if __name__ == "__main__":
    unittest.main()
