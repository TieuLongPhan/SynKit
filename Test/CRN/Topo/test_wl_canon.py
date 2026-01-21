import unittest

from synkit.CRN.Hypergraph.conversion import rxns_to_hypergraph
from synkit.CRN.Topo.canon import (
    CRNCanonicalizer,
)
from synkit.CRN.Topo.wl_canon import (
    WLCanonicalizer,
    wl_canonical,
)


# ----------------------------------------------------------------------
# Local orbit comparison helpers (test-only)
# ----------------------------------------------------------------------


def is_refinement(orbits_fine, orbits_coarse):
    """
    True iff orbits_fine is a refinement of orbits_coarse.
    """
    coarse_index = {}
    for i, o in enumerate(orbits_coarse):
        for v in o:
            coarse_index[v] = i

    for fine in orbits_fine:
        idxs = {coarse_index.get(v) for v in fine}
        if len(idxs) != 1:
            return False
    return True


class TestWLCanonicalizer(unittest.TestCase):

    def _example_H(self):
        # Simple A <-> B symmetry
        rxns = ["A+B>>C", "C+D>>E", "E+F>>G+D"]
        return rxns_to_hypergraph(rxns)

    # ------------------------------------------------------------------
    # Basic construction / refinement
    # ------------------------------------------------------------------
    def test_wl_runs_and_returns_orbits(self):
        H = self._example_H()
        wl = WLCanonicalizer(H)

        orbits = wl.orbits()
        self.assertIsInstance(orbits, list)
        self.assertTrue(all(isinstance(o, set) for o in orbits))

    def test_wl_runs_within_iteration_budget(self):
        H = self._example_H()
        n_iter = 10
        wl = WLCanonicalizer(H, n_iter=n_iter)

        summary = wl.summary()

        self.assertIn("iters_run", summary)
        self.assertLessEqual(summary["iters_run"], n_iter)

    def test_wl_stabilized_flag_is_boolean(self):
        H = self._example_H()
        wl = WLCanonicalizer(H, n_iter=10)

        summary = wl.summary()

        self.assertIn("stabilized", summary)
        self.assertIsInstance(summary["stabilized"], bool)

    # ------------------------------------------------------------------
    # Approximate symmetry detection
    # ------------------------------------------------------------------
    def test_wl_detects_AB_symmetry(self):
        """
        WL should put A and B in the same color class
        for this simple symmetric example.
        """
        H = self._example_H()
        wl = WLCanonicalizer(H)

        orbits = wl.orbits()
        self.assertIn({"A", "B"}, orbits)

    # ------------------------------------------------------------------
    # Canonical graph properties
    # ------------------------------------------------------------------
    def test_wl_canonical_graph_labels_are_integers(self):
        H = self._example_H()
        wl = WLCanonicalizer(H)

        G_can = wl.graph()
        for n in G_can.nodes():
            self.assertIsInstance(n, int)

    def test_wl_canonical_graph_deterministic(self):
        """
        WL canonical labeling must be deterministic for fixed parameters.
        """
        H = self._example_H()
        wl1 = WLCanonicalizer(H)
        wl2 = WLCanonicalizer(H)

        G1 = wl1.graph()
        G2 = wl2.graph()

        self.assertEqual(
            sorted(G1.edges()),
            sorted(G2.edges()),
        )

    # ------------------------------------------------------------------
    # Comparison with exact canonicalizer
    # ------------------------------------------------------------------
    def test_wl_orbits_are_coarser_than_exact(self):
        """
        WL orbits must be a coarsening of exact orbits
        (never split a true orbit).
        """
        H = self._example_H()

        exact = CRNCanonicalizer(H).orbits()
        wl = WLCanonicalizer(H).orbits()

        self.assertTrue(is_refinement(exact, wl))

    # ------------------------------------------------------------------
    # automorphism_count (approximate)
    # ------------------------------------------------------------------
    def test_wl_automorphism_count_present(self):
        H = self._example_H()
        wl = WLCanonicalizer(H, estimate_automorphisms=True)

        summary = wl.summary()
        self.assertIn("automorphism_count", summary)
        self.assertIsInstance(summary["automorphism_count"], int)
        self.assertGreaterEqual(summary["automorphism_count"], 1)

    # ------------------------------------------------------------------
    # Functional wrapper
    # ------------------------------------------------------------------
    def test_wl_functional_wrapper(self):
        H = self._example_H()
        wl = wl_canonical(H)

        summary = wl.summary()
        self.assertIn("orbits", summary)
        self.assertIn({"A", "B"}, summary["orbits"])


if __name__ == "__main__":
    unittest.main()
