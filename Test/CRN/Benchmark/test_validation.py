"""The validation table, run as a test.

Every benchmark network must reproduce its expected structural verdicts, and
every quantity must agree with an independent recomputation. A failure here is a
correctness regression in :mod:`synkit.CRN`, not a flaky test.
"""

import unittest

import numpy as np

from synkit.CRN.Benchmark import (
    BENCHMARK_NETWORKS,
    brute_force_siphons,
    check_conservation_laws,
    check_semiflows,
    deficiency_by_rank_identity,
    get_network,
    network_names,
    persistence_by_brute_force,
    rank_by_svd,
    run_validation,
    validate_network,
    validation_table,
)
from synkit.CRN.Petrinet.structure import find_siphons
from synkit.CRN.Props.stoich import integer_conservation_laws, stoichiometric_matrix
from synkit.CRN.Structure.syncrn import SynCRN


class TestBenchmarkRegistry(unittest.TestCase):
    def test_names_are_unique(self):
        names = network_names()
        self.assertEqual(len(names), len(set(names)))

    def test_every_entry_builds(self):
        for entry in BENCHMARK_NETWORKS:
            with self.subTest(network=entry.name):
                self.assertIsInstance(entry.build(), SynCRN)

    def test_every_entry_cites_a_source(self):
        for entry in BENCHMARK_NETWORKS:
            with self.subTest(network=entry.name):
                self.assertTrue(entry.source)

    def test_expected_deficiency_is_self_consistent(self):
        # delta = n - l - s must hold in the asserted numbers themselves.
        for entry in BENCHMARK_NETWORKS:
            with self.subTest(network=entry.name):
                self.assertEqual(
                    entry.deficiency,
                    entry.n_complexes - entry.n_linkage_classes - entry.rank,
                )

    def test_get_network(self):
        self.assertEqual(get_network("michaelis_menten").deficiency, 0)

    def test_get_unknown_network_raises(self):
        with self.assertRaises(KeyError):
            get_network("not_a_network")


class TestValidationTable(unittest.TestCase):
    """The paper's validation table, asserted network by network."""

    @classmethod
    def setUpClass(cls):
        cls.report = run_validation()

    def test_all_networks_pass(self):
        for result in self.report.results:
            with self.subTest(network=result.name):
                self.assertEqual(result.mismatches, {})
                self.assertEqual(result.crosscheck_failures, [])

    def test_report_is_complete(self):
        self.assertEqual(self.report.n_networks, len(BENCHMARK_NETWORKS))
        self.assertTrue(self.report.all_passed)
        self.assertEqual(self.report.failures, [])

    def test_report_serializes(self):
        payload = self.report.to_dict()
        self.assertTrue(payload["all_passed"])
        self.assertEqual(len(payload["results"]), len(BENCHMARK_NETWORKS))

    def test_markdown_table_lists_every_network(self):
        table = validation_table(self.report)
        for entry in BENCHMARK_NETWORKS:
            self.assertIn(entry.name, table)
        self.assertIn("all networks passed", table)

    def test_rst_table(self):
        table = validation_table(self.report, fmt="rst")
        self.assertIn("+=", table)

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            validation_table(self.report, fmt="latex")

    def test_validate_network_without_crosschecks(self):
        result = validate_network(BENCHMARK_NETWORKS[0], crosscheck=False)
        self.assertEqual(result.crosschecks, {})
        self.assertTrue(result.passed)


class TestCrosschecksAreIndependent(unittest.TestCase):
    """The cross-checks must be able to disagree, or they prove nothing."""

    def test_rank_identity_matches_production_on_every_benchmark(self):
        for entry in BENCHMARK_NETWORKS:
            with self.subTest(network=entry.name):
                crn = entry.build()
                self.assertEqual(deficiency_by_rank_identity(crn), entry.deficiency)

    def test_svd_rank_matches_exact_rank(self):
        for entry in BENCHMARK_NETWORKS:
            with self.subTest(network=entry.name):
                self.assertEqual(rank_by_svd(entry.build()), entry.rank)

    def test_brute_force_siphons_match_the_fast_search(self):
        for entry in BENCHMARK_NETWORKS:
            with self.subTest(network=entry.name):
                crn = entry.build()
                fast = {frozenset(s) for s in find_siphons(crn, names="id")}
                brute = set(brute_force_siphons(crn))
                self.assertEqual(fast, brute)

    def test_brute_force_refuses_large_networks(self):
        big = SynCRN.from_reaction_strings([f"S{i}>>S{i + 1}" for i in range(30)])
        with self.assertRaises(ValueError):
            brute_force_siphons(big, max_species=14)

    def test_persistence_by_brute_force_matches_expectations(self):
        for entry in BENCHMARK_NETWORKS:
            if entry.persistent is None:
                continue
            with self.subTest(network=entry.name):
                self.assertEqual(
                    persistence_by_brute_force(entry.build()), entry.persistent
                )

    def test_conservation_law_check_rejects_a_wrong_law(self):
        crn = SynCRN.from_reaction_strings(["A>>B", "B>>A"])
        checks = check_conservation_laws(crn, [[1, -1]])
        self.assertFalse(checks["in_left_kernel"])
        self.assertFalse(checks["ok"])

    def test_conservation_law_check_accepts_the_real_law(self):
        crn = SynCRN.from_reaction_strings(["A>>B", "B>>A"])
        checks = check_conservation_laws(crn, integer_conservation_laws(crn))
        self.assertTrue(checks["ok"])

    def test_semiflow_check_reports_the_defining_properties(self):
        crn = SynCRN.from_reaction_strings(["A>>B", "B>>A"])
        for kind in ("p", "t"):
            with self.subTest(kind=kind):
                checks = check_semiflows(crn, kind=kind)
                self.assertTrue(checks["nonnegative"])
                self.assertTrue(checks["in_kernel"])
                self.assertTrue(checks["supports_minimal"])

    def test_semiflow_check_rejects_a_bad_kind(self):
        with self.assertRaises(ValueError):
            check_semiflows(SynCRN.from_reaction_strings(["A>>B"]), kind="q")


class TestConservationLawsAreExact(unittest.TestCase):
    """Regression: the float-kernel path returned vectors that were not laws."""

    def test_every_benchmark_law_is_in_the_left_kernel(self):
        for entry in BENCHMARK_NETWORKS:
            with self.subTest(network=entry.name):
                crn = entry.build()
                matrix = stoichiometric_matrix(crn)
                if matrix.size == 0:
                    continue
                for law in integer_conservation_laws(crn):
                    np.testing.assert_allclose(
                        np.array(law, dtype=float) @ matrix, 0.0, atol=1e-12
                    )


if __name__ == "__main__":
    unittest.main()
