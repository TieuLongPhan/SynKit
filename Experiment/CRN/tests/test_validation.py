"""Reference networks and independent CRNT checks."""

import unittest

from Experiment.CRN.studies import (
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
from synkit.CRN.Props.stoich import integer_conservation_laws
from synkit.CRN.Structure.syncrn import SynCRN


class TestBenchmarkRegistry(unittest.TestCase):
    def test_registry_entries_are_named_sourced_and_consistent(self):
        names = network_names()
        self.assertEqual(len(names), len(set(names)))
        for entry in BENCHMARK_NETWORKS:
            with self.subTest(network=entry.name):
                self.assertIs(get_network(entry.name), entry)
                self.assertIsInstance(entry.build(), SynCRN)
                self.assertTrue(entry.source)
                self.assertEqual(entry.deficiency, entry.n_complexes - entry.n_linkage_classes - entry.rank)
        self.assertEqual(get_network("michaelis_menten").deficiency, 0)

    def test_get_unknown_network_raises(self):
        with self.assertRaises(KeyError):
            get_network("not_a_network")


class TestValidationTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_validation()

    def test_complete_report_preserves_each_network_verdict(self):
        self.assertEqual(self.report.n_networks, len(BENCHMARK_NETWORKS))
        self.assertTrue(self.report.all_passed)
        self.assertEqual(self.report.failures, [])
        payload = self.report.to_dict()
        self.assertTrue(payload["all_passed"])
        self.assertEqual(len(payload["results"]), len(BENCHMARK_NETWORKS))
        for result in self.report.results:
            with self.subTest(network=result.name):
                self.assertEqual(result.mismatches, {})
                self.assertEqual(result.crosscheck_failures, [])

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
    """Independent recomputation and invalid-witness checks."""

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
                self.assertEqual(persistence_by_brute_force(entry.build()), entry.persistent)

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


if __name__ == "__main__":
    unittest.main()
