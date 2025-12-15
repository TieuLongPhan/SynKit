import unittest

from synkit.CRN.Hypergraph.conversion import rxns_to_hypergraph
from synkit.CRN.Topo.canon import CRNCanonicalizer, canonical


class TestCRNCanonicalizer(unittest.TestCase):

    def _example_H(self):
        rxns = ["A+B>>C", "C+D>>E", "E+F>>G+D"]
        return rxns_to_hypergraph(rxns)

    # ----------------------------------------------------------------------
    # _init_part
    # ----------------------------------------------------------------------
    def test_init_part_groups_by_kind(self):
        H = self._example_H()

        # species graph → only species → 1 partition
        canon_species = CRNCanonicalizer(H, include_rule=False)
        part_species = canon_species._init_part(canon_species.G)
        self.assertEqual(len(part_species), 1)

        # bipartite graph → species + reactions → >= 2 partitions
        canon_bip = CRNCanonicalizer(H, include_rule=True)
        part_bip = canon_bip._init_part(canon_bip.G)
        self.assertGreaterEqual(len(part_bip), 2)

    # ----------------------------------------------------------------------
    # Automorphism count (A↔B symmetry)
    # ----------------------------------------------------------------------
    def test_automorphism_count_two(self):
        H = self._example_H()
        c = CRNCanonicalizer(H)
        res = c.summary()

        self.assertEqual(res["automorphism_count"], 2)

    def test_orbits_reflect_AB_symmetry(self):
        H = self._example_H()
        c = CRNCanonicalizer(H)

        orbits = c.orbits()
        self.assertIn({"A", "B"}, orbits)

    # ----------------------------------------------------------------------
    # Canonical Graph Tests
    # ----------------------------------------------------------------------
    def test_canonical_graph_labels_are_integers(self):
        """
        Canonical mapping does NOT guarantee labels are 1..N continuous
        (because permutations may contain duplicates), but labels must be ints.
        """
        H = self._example_H()
        c = CRNCanonicalizer(H)
        G_can = c.graph()

        for n in G_can.nodes():
            self.assertIsInstance(n, int)

    # ----------------------------------------------------------------------
    # Max-depth and timeout behavior: EXPECT RuntimeError
    # ----------------------------------------------------------------------
    def test_canonical_max_depth_raises_runtimeerror(self):
        H = self._example_H()
        c = CRNCanonicalizer(H)

        with self.assertRaises(RuntimeError):
            c.summary(max_depth=0)

    # def test_canonical_timeout_raises_runtimeerror(self):
    #     H = self._example_H()
    #     c = CRNCanonicalizer(H)

    #     with self.assertRaises(RuntimeError):
    #         c.summary(timeout_sec=1e-9)

    # ----------------------------------------------------------------------
    # Functional wrapper verifies same results
    # ----------------------------------------------------------------------
    def test_functional_wrapper(self):
        H = self._example_H()
        summary = canonical(H)

        self.assertEqual(summary["automorphism_count"], 2)
        self.assertIn({"A", "B"}, summary["orbits"])


if __name__ == "__main__":
    unittest.main()
