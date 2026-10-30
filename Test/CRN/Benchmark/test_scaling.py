import unittest
import time
from unittest.mock import patch

from synkit.CRN.Benchmark.scaling import (
    NETWORK_FAMILIES,
    ScalingRecord,
    TASKS,
    generate_network,
    run_scaling_benchmark,
    scaling_table,
)
from synkit.CRN.Structure.syncrn import SynCRN


def _slow_rank(crn):
    """Use an importable task so a spawned Windows process can deserialize it."""
    time.sleep(1.0)
    return crn.n_species


def _node_limited_rank(crn):
    from synkit.CRN.Petrinet.siphon_search import SiphonSearchLimit

    raise SiphonSearchLimit("test node limit")


def _broken_rank(crn):
    raise ValueError("test algorithm error")


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
        self.assertTrue(records[0].timed_out)

    def test_time_budget_terminates_current_measurement(self):
        original = TASKS["rank"]

        TASKS["rank"] = _slow_rank
        started = time.perf_counter()
        try:
            with patch(
                "synkit.CRN.Benchmark.scaling.get_all_start_methods",
                return_value=["spawn"],
            ):
                records = run_scaling_benchmark(
                    sizes=(4,),
                    families=("chain",),
                    tasks=("rank",),
                    time_budget=0.05,
                )
        finally:
            TASKS["rank"] = original

        self.assertLess(time.perf_counter() - started, 0.5)
        self.assertTrue(records[0].timed_out)
        self.assertIsNone(records[0].seconds)
        self.assertIsNone(records[0].error)

    def test_spawned_task_returns_the_registered_result(self):
        with patch(
            "synkit.CRN.Benchmark.scaling.get_all_start_methods", return_value=["spawn"]
        ):
            records = run_scaling_benchmark(
                sizes=(4,),
                families=("chain",),
                tasks=("rank",),
                time_budget=30.0,
            )
        self.assertEqual(records[0].result, 4)
        self.assertFalse(records[0].timed_out)
        self.assertIsNone(records[0].error)

    def test_records_serialize(self):
        record = run_scaling_benchmark(
            sizes=(4,), families=("chain",), tasks=("rank",)
        )[0]
        payload = record.to_dict()
        self.assertEqual(payload["family"], "chain")
        self.assertEqual(payload["task"], "rank")
        self.assertFalse(payload["timed_out"])

    def test_node_limit_is_incomplete_and_has_no_completed_timing(self):
        for budget in (None, 30.0):
            with (
                self.subTest(budget=budget),
                patch.dict(TASKS, rank=_node_limited_rank),
            ):
                records = run_scaling_benchmark(
                    sizes=(4, 6),
                    families=("chain",),
                    tasks=("rank",),
                    time_budget=budget,
                )
                self.assertEqual(len(records), 1)
                row = records[0].to_dict()
                self.assertEqual(row["termination_reason"], "node_limit")
                self.assertFalse(row["complete"])
                self.assertFalse(row["timed_out"])
                self.assertIsNone(row["seconds"])
                self.assertIsNone(row["result"])
                self.assertIsNone(row["error"])
                self.assertGreaterEqual(row["elapsed_seconds"], 0)

    def test_algorithm_errors_remain_errors(self):
        for budget in (None, 30.0):
            with self.subTest(budget=budget), patch.dict(TASKS, rank=_broken_rank):
                record = run_scaling_benchmark(
                    sizes=(4,),
                    families=("chain",),
                    tasks=("rank",),
                    time_budget=budget,
                )[0]
                self.assertEqual(record.termination_reason, "error")
                self.assertIn("ValueError", record.error)
                self.assertFalse(record.timed_out)


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


def test_paper_scaling_rejects_incomplete_measurements():
    from Experiment.CRN.studies.scaling import scaling_report

    record = ScalingRecord(
        family="chain",
        size=4,
        n_species=5,
        n_reactions=4,
        task="rank",
        termination_reason="node_limit",
    )
    with (
        patch(
            "Experiment.CRN.studies.scaling.run_scaling_benchmark",
            return_value=[record],
        ),
        patch("Experiment.CRN.studies.scaling.environment", return_value={}),
    ):
        report = scaling_report(
            sizes=[4],
            families=["chain"],
            tasks=["rank"],
            repeats=1,
            seed=0,
            time_budget=30,
        )
    assert report["status"] == "FAIL"
    assert report["checks"]["all_measurements_complete"] is False


def test_paper_biomodels_rejects_missing_enumerations():
    from pathlib import Path
    from Experiment.CRN.studies.biomodels import biomodels_report

    record = dict(
        model_id="test",
        n_species=2,
        n_reactions=1,
        n_complexes=2,
        n_linkage_classes=1,
        rank=1,
        deficiency=0,
        roundtrip_ok=True,
        n_conserved_moieties=1,
        n_minimal_siphons=None,
        persistent=None,
    )
    with (
        patch(
            "Experiment.CRN.studies.biomodels.fetch", return_value=Path("/tmp/test.xml")
        ),
        patch("Experiment.CRN.studies.biomodels.analyse", return_value=record),
        patch("Experiment.CRN.studies.biomodels.environment", return_value={}),
    ):
        report = biomodels_report(["test"], offline=True, budget=30)
    assert report["status"] == "FAIL"
    assert report["checks"]["every_structural_analysis_complete"] is False
