from __future__ import annotations

import unittest
from typing import FrozenSet, Hashable


from synkit.Graph.Matcher.orbit import OrbitAccuracy


def froz(*items) -> FrozenSet[Hashable]:
    """Helper to create frozensets succinctly in tests."""
    return frozenset(items)


class TestOrbitAccuracy(unittest.TestCase):
    def assertFloatEqual(self, a: float, b: float, places: int = 9):
        """Small wrapper around assertAlmostEqual for readability."""
        self.assertAlmostEqual(a, b, places=places)

    def test_init_mismatched_nodes_raises(self):
        approx = [froz(1), froz(2)]
        exact = [froz(1), froz(3)]
        with self.assertRaises(ValueError) as ctx:
            OrbitAccuracy(approx, exact)
        msg = str(ctx.exception)
        self.assertIn("nodes missing", msg)

    def test_full_match_metrics_and_confusion(self):
        approx = [froz(1), froz(2, 3)]
        exact = [froz(1), froz(2, 3)]
        oa = OrbitAccuracy(approx, exact).compute()

        metrics = oa.metrics
        self.assertFloatEqual(metrics["node_exact_match_fraction"], 1.0)
        self.assertFloatEqual(metrics["purity"], 1.0)
        self.assertFloatEqual(metrics["pairwise_accuracy"], 1.0)

        cm = oa.confusion_map
        self.assertEqual(cm, {0: {0: 1}, 1: {1: 2}})

    def test_partial_overlap_metrics_pairwise(self):
        approx = [froz(1, 2), froz(3)]
        exact = [froz(1), froz(2, 3)]
        oa = OrbitAccuracy(approx, exact).compute()
        m = oa.metrics

        # No node belongs to exactly the same orbit
        self.assertFloatEqual(m["node_exact_match_fraction"], 0.0)

        # purity: best overlaps are 1 and 1 -> (1+1)/3
        self.assertFloatEqual(m["purity"], 2.0 / 3.0)

        # pairwise accuracy: 1 match out of 3 pairs => 1/3
        self.assertFloatEqual(m["pairwise_accuracy"], 1.0 / 3.0)

    def test_pairwise_non_bruteforce_delegates(self):
        approx = [froz(1, 2), froz(3)]
        exact = [froz(1), froz(2, 3)]
        oa_bf = OrbitAccuracy(approx, exact).compute(brute_force_pairs=True)
        oa_non = OrbitAccuracy(approx, exact).compute(brute_force_pairs=False)
        self.assertFloatEqual(
            oa_bf.metrics["pairwise_accuracy"], oa_non.metrics["pairwise_accuracy"]
        )

    def test_empty_partitions_behavior(self):
        # both partitions empty: compute should succeed
        oa = OrbitAccuracy([], []).compute()
        m = oa.metrics
        self.assertIn("pairwise_accuracy", m)
        self.assertFloatEqual(m["pairwise_accuracy"], 1.0)
        self.assertFloatEqual(m["node_exact_match_fraction"], 0.0)
        self.assertFloatEqual(m["purity"], 0.0)

        r = repr(oa)
        self.assertIn("OrbitAccuracy", r)
        self.assertIn("nodes=", r)

    def test_properties_return_copies_and_confusion_immutable_from_view(self):
        approx = [froz(1), froz(2)]
        exact = [froz(1), froz(2)]
        oa = OrbitAccuracy(approx, exact).compute()

        # approx_orbits returns a list copy; mutating it should not affect the object
        out_list = oa.approx_orbits
        out_list.append(froz(999))
        # original internal approx_orbits should remain unchanged
        all_nodes = set().union(*oa.approx_orbits)
        self.assertNotIn(999, all_nodes)

        # confusion_map returns copies of inner dicts; mutating returned should not change internal
        cm = oa.confusion_map
        cm[0][999] = 12345
        cm2 = oa.confusion_map
        self.assertNotIn(999, cm2[0])

        # metrics returns a copy
        metrics = oa.metrics
        metrics["new_metric"] = 0.5
        self.assertNotIn("new_metric", oa.metrics)

    def test_report_contains_expected_sections_and_confusion_rows(self):
        approx = [froz(1), froz(2, 3)]
        exact = [froz(1), froz(2, 3)]
        oa = OrbitAccuracy(approx, exact).compute()
        rpt = oa.report(max_rows=5)
        self.assertIn("Metrics:", rpt)
        self.assertIn("Top confusion rows", rpt)
        self.assertIn("0 ->", rpt)

    def test_help_method_short_string(self):
        approx = [froz(1)]
        exact = [froz(1)]
        oa = OrbitAccuracy(approx, exact)
        help_str = oa.help()
        # check it contains brief usage hints
        self.assertTrue(
            "Instantiate" in help_str
            or "Call .compute()" in help_str
            or "help" in help_str
        )


if __name__ == "__main__":
    unittest.main()
