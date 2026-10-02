import json
from types import SimpleNamespace

import pytest

from Experiment.Synister.repeat_enumeration import make_tasks, run


def test_repeated_minimum_queries_never_receive_a_privileged_target():
    rows = [{"benchmark_id": "example", "reaction": "CO>>CO"}]
    tasks = make_tasks(rows, repeats=3, seconds=60, memory_gib=6, max_maps=100000)
    assert len(tasks) == 6 and len({t["task_id"] for t in tasks}) == 6
    assert {t["target_doubled_cd"] for t in tasks} == {"minimal"}
    assert [t["method"] for t in tasks] == ["synister", "milp", "milp", "synister", "synister", "milp"]


def test_serial_repeats_preserve_every_attempt_and_audit_outputs(tmp_path):
    matched = tmp_path/"matched"
    matched.mkdir()
    rows = [{"benchmark_id": "reaction_000", "reaction": "CC>>CC", "source": "FlowER",
             "size_bin": 0, "selection_sha256": "00", "status": "timeout"}]
    (matched/"inputs.json").write_text(json.dumps(rows))
    (matched/"audit.json").write_text(json.dumps({"all_output_comparisons_consistent": True}))
    args = SimpleNamespace(matched=matched, output=tmp_path/"repeats", after=None, per_bin=1,
                           repeats=2, seconds=5, memory_gib=2, max_maps=100)
    report = run(args)
    assert report["attempts"] == 4 and report["matched_queries"] == 2
    assert report["validated_saved_maps"] == 8 and report["repeat_comparison_count"] == 6
    assert report["all_output_comparisons_consistent"] and report["all_repeat_comparisons_consistent"]
    assert json.loads((args.output/"inputs.json").read_text()) == rows
    with pytest.raises(FileExistsError):
        run(args)
