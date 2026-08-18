import unittest

from synkit.CRN.Benchmark.scaling import (
    NETWORK_FAMILIES,
    ScalingRecord,
    TASKS,
    generate_network,
    run_scaling_benchmark,
    scaling_table,
)
from synkit.CRN.Structure.syncrn import SynCRN


class TestNetworkFamilies(unittest.TestCase):
    def test_every_family_generates_a_network(self):
        for family in NETWORK_FAMILIES:
            with self.subTest(family=family):
                crn = generate_network(family, 8)
                self.assertIsInstance(crn, SynCRN)
                self.assertGreater(crn.n_reactions, 0)
                self.assertGreater(crn.n_species, 0)

    def test_chain_sizes(self):
        crn = generate_network("chain", 10)
        self.assertEqual(crn.n_reactions, 10)
        self.assertEqual(crn.n_species, 11)

    def test_reversible_chain_doubles_the_reactions(self):
        self.assertEqual(generate_network("reversible_chain", 10).n_reactions, 20)

    def test_cycle_is_closed(self):
        crn = generate_network("cycle", 10)
        self.assertEqual(crn.n_species, 10)
        self.assertEqual(crn.n_reactions, 10)

    def test_random_sparse_is_seed_reproducible(self):
        first = generate_network("random_sparse", 12, seed=3)
        second = generate_network("random_sparse", 12, seed=3)
        self.assertEqual(first.to_equations(), second.to_equations())

    def test_random_sparse_differs_across_seeds(self):
        first = generate_network("random_sparse", 20, seed=1)
        second = generate_network("random_sparse", 20, seed=2)
        self.assertNotEqual(first.to_equations(), second.to_equations())

    def test_unknown_family_raises(self):
        with self.assertRaises(ValueError):
            generate_network("hypercube", 4)


class TestRunScalingBenchmark(unittest.TestCase):
    def test_produces_one_record_per_combination(self):
        records = run_scaling_benchmark(
            sizes=(4, 6), families=("chain",), tasks=("rank", "crnt_summary")
        )
        self.assertEqual(len(records), 4)
        self.assertTrue(all(isinstance(r, ScalingRecord) for r in records))

    def test_records_carry_timings_and_results(self):
        records = run_scaling_benchmark(
            sizes=(4,), families=("chain",), tasks=("rank",)
        )
        record = records[0]
        self.assertIsNotNone(record.seconds)
        self.assertGreaterEqual(record.seconds, 0.0)
        self.assertIsNone(record.error)
        self.assertEqual(record.result, 4)

    def test_every_task_runs(self):
        records = run_scaling_benchmark(
            sizes=(5,), families=("chain",), tasks=tuple(TASKS)
        )
        for record in records:
            with self.subTest(task=record.task):
                self.assertIsNone(record.error)
                self.assertIsNotNone(record.seconds)

    def test_unknown_family_raises(self):
        with self.assertRaises(ValueError):
            run_scaling_benchmark(sizes=(4,), families=("nope",))

    def test_unknown_task_raises(self):
        with self.assertRaises(ValueError):
            run_scaling_benchmark(sizes=(4,), tasks=("nope",))

    def test_time_budget_skips_larger_sizes(self):
        # A zero budget must stop a task after its first measurement.
        records = run_scaling_benchmark(
            sizes=(4, 6, 8),
            families=("chain",),
            tasks=("rank",),
            time_budget=0.0,
        )
        self.assertEqual(len(records), 1)

    def test_records_serialize(self):
        record = run_scaling_benchmark(
            sizes=(4,), families=("chain",), tasks=("rank",)
        )[0]
        payload = record.to_dict()
        self.assertEqual(payload["family"], "chain")
        self.assertEqual(payload["task"], "rank")


class TestScalingTable(unittest.TestCase):
    def setUp(self):
        self.records = run_scaling_benchmark(
            sizes=(4,), families=("chain",), tasks=("rank", "crnt_summary")
        )

    def test_markdown_lists_every_task(self):
        table = scaling_table(self.records)
        self.assertIn("rank", table)
        self.assertIn("crnt_summary", table)

    def test_csv_has_a_header_and_one_row_per_record(self):
        csv = scaling_table(self.records, fmt="csv").splitlines()
        self.assertTrue(csv[0].startswith("family,size"))
        self.assertEqual(len(csv), len(self.records) + 1)

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            scaling_table(self.records, fmt="latex")

    def test_empty_records(self):
        self.assertIn("family", scaling_table([]))


if __name__ == "__main__":
    unittest.main()
