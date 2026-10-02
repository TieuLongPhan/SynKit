import json
from types import SimpleNamespace

from Experiment.Synister.enumeration_benchmark import execute, run, select, size_bin
from Experiment.Synister.audit_enumeration_benchmark import audit, compare_outputs


def test_bins_and_input_only_shortfall_rule(tmp_path):
    assert [size_bin(n) for n in (1, 20, 21, 40, 41, 60, 61, 80, 81, 512)] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    paths = []
    for source in ("flower", "rhea"):
        path = tmp_path / f"{source}.json"
        path.write_text(json.dumps([{"case_id": str(i), "reaction_id": str(i),
                                     "reaction": r, "status": "timeout"}
                                    for i, r in enumerate(("CO>>CO", "CCO>>CCO", "CCCO>>CCCO"))]))
        paths.append(path)
    rows, accounting = select(paths, per_bin=1)
    assert len(rows) == 6
    assert all(row["status"] == "timeout" for row in rows)
    assert [a["selected"] for a in accounting] == [3, 3]
    assert len({r["benchmark_id"] for r in rows}) == 6


def test_isolated_worker_minimum_and_explicit_empty(tmp_path):
    (tmp_path / "cases").mkdir()
    (tmp_path / "maps").mkdir()
    base = {"reaction": "CO>>CO", "seconds": 5, "memory_gib": 2, "max_maps": 100,
            "benchmark_id": "example", "query": "minimum"}
    results = []
    for method in ("synister", "milp"):
        results.append(execute({**base, "task_id": f"minimum.{method}", "method": method,
                                "target_doubled_cd": "minimal"}, tmp_path))
        empty = execute({**base, "task_id": f"empty.{method}", "method": method,
                         "target_doubled_cd": 1}, tmp_path)
        assert empty["complete"] and empty["mapping_count"] == 0
    assert all(r["complete"] and r["minimum_proved"] and r["minimum_doubled_cd"] == 0 for r in results)
    assert results[0]["mapping_sha256"] == results[1]["mapping_sha256"]
    assert all(r["peak_rss_kib"] > 0 and r["parent_seconds"] > 0 for r in results)


def test_end_to_end_query_derivation_and_preserved_attempts(tmp_path):
    paths = []
    for source, reaction in (("flower", "CO>>CO"), ("rhea", "CC>>CC")):
        path = tmp_path / f"{source}.json"
        path.write_text(json.dumps([{"case_id": "a", "reaction_id": "a", "reaction": reaction}]))
        paths.append(path)
    args = SimpleNamespace(output=tmp_path/"study", flower=paths[0], rhea=paths[1],
                           per_bin=1, seconds=5, memory_gib=2, max_maps=100,
                           workers=2, references=None)
    summary = run(args)
    assert summary["selected"] == 2 and summary["attempts"] == 20
    assert summary["unexplained_disagreements"] == 0
    assert all(r["both_complete"] and r["mapping_sets_equal"] for r in summary["comparisons"])
    derivation = json.loads((args.output/"target_derivation.json").read_text())
    assert all(r["reference_status"] == "not_available" and r["relative_targets_available"] for r in derivation)
    checked = audit(args.output)
    assert checked["attempts"] == 20 and checked["all_output_comparisons_consistent"]


def test_partial_output_can_falsify_a_claimed_complete_set():
    complete, partial = {"complete": True}, {"complete": False}
    mismatch = compare_outputs(complete, partial, [(0, 1)], [(1, 0)])
    assert not mismatch["both_complete"] and not mismatch["consistent"]
    assert mismatch["missing_from_first_complete_set"] == 1
    assert compare_outputs(complete, partial, [(0, 1), (1, 0)], [(1, 0)])["consistent"]
