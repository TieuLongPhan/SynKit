"""Interruption recovery preserves scientific outcomes and frozen protocols."""

from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys

import pytest

from Experiment.Synister import seed_output_recovery as recovery
from Experiment.Synister.audit_seed_output import audit
from Experiment.Synister.audit_seed_output_recovery import verify_continuation


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def interrupted(tmp_path):
    """A bounded two-input version of the four-condition frozen E2 protocol."""
    source, parent, main = (
        tmp_path / name for name in ("interrupted", "pilot", "main")
    )
    for folder in (
        "cases",
        "maps",
        "preparation",
        "classification",
        "details",
        "frozen_source",
    ):
        (source / folder).mkdir(parents=True)
    rows = [
        {"benchmark_id": "reaction_000", "reaction": "CO>>CO"},
        {"benchmark_id": "reaction_001", "reaction": "OC>>OC"},
    ]
    worker_name = "Experiment/Synister/seed_output_worker.py"
    sources = {worker_name: Path(worker_name).read_text()}
    for name, content in sources.items():
        path = source / "frozen_source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    write(parent / "sources.json", sources)
    write(parent / "summary.json", {"synthetic_gate": True})
    write(
        parent / "audit.json",
        {
            "attempts": 200,
            "all_output_comparisons_consistent": True,
            "summary_sha256": recovery.digest(parent / "summary.json"),
        },
    )
    write(parent / "difficult_inputs.json", rows)
    write(main / "summary.json", {"synthetic_gate": True})
    write(
        main / "audit.json",
        {
            "attempts": 320,
            "all_output_comparisons_consistent": True,
            "summary_sha256": recovery.digest(main / "summary.json"),
        },
    )
    inputs = tmp_path / "selected.json"
    write(inputs, rows)
    write(source / "inputs.json", rows)
    # Preserve the same serialization as the pilot, rather than merely equality.
    (source / "sources.json").write_bytes((parent / "sources.json").read_bytes())
    write(
        source / "protocol.json",
        {
            "phase": "followup",
            "search_seconds": 300,
            "parent_seconds": 315,
            "matched_attempts": 8,
        },
    )
    write(source / "orchestrator_source.json", {"synthetic": "test-only orchestration"})
    preparations, tasks = [], []
    for index, row in enumerate(rows):
        bid = row["benchmark_id"]
        preparation = {
            "task_id": bid + ".seed",
            "benchmark_id": bid,
            "reaction": row["reaction"],
            "stage": "seed",
            "memory_gib": 6,
        }
        preparations.append(preparation)
        write(
            source / "preparation" / (preparation["task_id"] + ".json"),
            {
                "task": preparation,
                "complete": True,
                "termination": "valid_prediction",
                "parent_seconds": 0.2,
                "prediction": {"mapping": [0, 1]},
                "seed_doubled_cd": 0,
            },
        )
        for condition in (("none", "slap") if index % 2 == 0 else ("slap", "none")):
            for method in (
                ("synister", "milp") if index % 2 == 0 else ("milp", "synister")
            ):
                tasks.append(
                    {
                        "task_id": f"{bid}.{condition}.{method}.indexed",
                        "benchmark_id": bid,
                        "reaction": row["reaction"],
                        "stage": "matched",
                        "method": method,
                        "output": "indexed",
                        "seed_condition": condition,
                        "initial_mapping": [0, 1] if condition == "slap" else None,
                        "seed_preparation_parent_seconds": (
                            0.2 if condition == "slap" else 0
                        ),
                        "seconds": 300,
                        "max_maps": 100000,
                        "memory_gib": 6,
                    }
                )
    write(source / "preparation_tasks.json", preparations)
    write(source / "tasks.json", tasks)
    task = tasks[0]
    # A recorded timeout is terminal and must not be rerun as a missing task.
    write(
        source / "cases" / (task["task_id"] + ".json"),
        {
            "task": task,
            "complete": False,
            "minimum_proved": False,
            "termination": "time_limit",
            "parent_seconds": 300.1,
            "prediction_already_available_parent_seconds": 300.1,
            "seed_inclusive_parent_seconds": 300.1,
        },
    )
    write(source / "maps" / (tasks[1]["task_id"] + ".json"), [[0, 1]])
    names = ("protocol.json", "inputs.json", "sources.json", "orchestrator_source.json")
    write(
        source / "manifest.json",
        {
            "phase": "followup",
            "python": sys.version,
            "dependencies": {
                name: version(name) for name in ("numpy", "scipy", "rdkit", "networkx")
            },
            "inputs_source": str(inputs),
            "inputs_source_sha256": recovery.digest(inputs),
            "parent": str(parent),
            "parent_sha256": {
                name: recovery.digest(parent / name)
                for name in (
                    "summary.json",
                    "audit.json",
                    "difficult_inputs.json",
                    "sources.json",
                )
            },
            "main_extension": str(main),
            "main_extension_audit_sha256": recovery.digest(main / "audit.json"),
            "file_sha256": {name: recovery.digest(source / name) for name in names},
        },
    )
    controls = tmp_path / "controls.json"
    write(
        controls,
        {
            "all_passed": True,
            "source_sha256": {
                worker_name: sha256(sources[worker_name].encode()).hexdigest()
            },
        },
    )
    return source, controls


def test_only_missing_tasks_run_and_original_timeout_is_preserved(
    interrupted, tmp_path, monkeypatch
):
    source, controls = interrupted
    before = recovery.inventory(source)
    output = recovery.prepare(source, tmp_path / "continuation", controls)
    assert list((output / "orphans" / "maps").glob("*.json"))
    launched = []

    def fake_isolated(output, module, task, seconds):
        assert seconds == 315
        launched.append(task["task_id"])
        return {
            "complete": False,
            "minimum_proved": False,
            "termination": "time_limit",
            "parent_seconds": 300.1,
        }

    monkeypatch.setattr(recovery.benchmark, "isolated", fake_isolated)
    result = recovery.run(output, controls)
    assert result["attempts"] == 8 and len(launched) == 7
    assert "reaction_000.none.synister.indexed" not in launched
    assert recovery.inventory(source) == before
    checked = audit(output, controls)
    assert checked["all_output_comparisons_consistent"]
    assert checked["continuation"]["preserved_attempts"] == 1
    assert recovery.run(output, controls) == result
    assert len(launched) == 7


def test_resume_after_interrupted_parent_does_not_replace_committed_attempts(
    interrupted, tmp_path, monkeypatch
):
    source, controls = interrupted
    output = recovery.prepare(source, tmp_path / "continuation", controls)
    task = recovery.read(output / "tasks.json")[1]
    record = {
        "task": task,
        "complete": False,
        "minimum_proved": False,
        "termination": "mapping_limit",
        "parent_seconds": 1,
        "prediction_already_available_parent_seconds": 1,
        "seed_inclusive_parent_seconds": 1,
    }
    recovery.atomic_save(output / "cases" / (task["task_id"] + ".json"), record)
    before = (output / "cases" / (task["task_id"] + ".json")).read_bytes()
    # Simulate output written before a parent crashed, without a case commit.
    missing = recovery.read(output / "tasks.json")[2]
    write(output / "maps" / (missing["task_id"] + ".json"), [[0, 1]])
    launched = []

    def fake(output, module, task, seconds):
        launched.append(task["task_id"])
        return {
            "complete": False,
            "minimum_proved": False,
            "termination": "time_limit",
            "parent_seconds": 300.1,
        }

    monkeypatch.setattr(recovery.benchmark, "isolated", fake)
    recovery.run(output, controls)
    assert task["task_id"] not in launched and len(launched) == 6
    assert (output / "cases" / (task["task_id"] + ".json")).read_bytes() == before
    assert list((output / "orphans" / "maps").glob(missing["task_id"] + ".*.json"))


@pytest.mark.parametrize(
    "corruption", ["task", "duplicate", "unknown", "worker", "environment"]
)
def test_reject_corrupted_history_before_creating_continuation(
    interrupted, tmp_path, corruption
):
    source, controls = interrupted
    if corruption in ("task", "duplicate"):
        tasks = recovery.read(source / "tasks.json")
        if corruption == "task":
            tasks[0]["seconds"] = 301
        else:
            tasks.append(tasks[0])
        write(source / "tasks.json", tasks)
    elif corruption == "unknown":
        write(source / "cases" / "unknown.json", {"complete": False, "task": {}})
    elif corruption == "worker":
        (
            source / "frozen_source" / "Experiment/Synister/seed_output_worker.py"
        ).write_text("changed")
    else:
        manifest = recovery.read(source / "manifest.json")
        manifest["dependencies"]["numpy"] = "0.0.0"
        write(source / "manifest.json", manifest)
    with pytest.raises(ValueError):
        recovery.prepare(source, tmp_path / "rejected", controls)
    assert not (tmp_path / "rejected").exists()


def test_atomic_publication_survives_interruption_and_refuses_replacement(
    tmp_path, monkeypatch
):
    target = tmp_path / "record.json"
    real_link = recovery.os.link

    def interrupted_link(source, destination):
        raise InterruptedError("Injected crash before publication")

    monkeypatch.setattr(recovery.os, "link", interrupted_link)
    with pytest.raises(InterruptedError):
        recovery.atomic_save(target, {"complete": True})
    assert not target.exists() and not list(tmp_path.iterdir())
    monkeypatch.setattr(recovery.os, "link", real_link)
    recovery.atomic_save(target, {"complete": False})
    with pytest.raises(FileExistsError):
        recovery.atomic_save(target, {"complete": True})
    assert recovery.read(target) == {"complete": False}


def test_auditor_rejects_replaced_original_outcomes(interrupted, tmp_path):
    source, controls = interrupted
    output = recovery.prepare(source, tmp_path / "continuation", controls)
    path = next((output / "cases").glob("*.json"))
    record = recovery.read(path)
    record["complete"] = True
    write(path, record)
    with pytest.raises(ValueError, match="replaced an original terminal outcome"):
        verify_continuation(output)


def test_exclusive_lock_prevents_duplicate_scheduling(tmp_path):
    with recovery.exclusive_run(tmp_path):
        with pytest.raises(ValueError, match="already running"):
            with recovery.exclusive_run(tmp_path):
                pytest.fail("Duplicate orchestrator obtained lock")
