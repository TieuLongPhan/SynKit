"""Independently check preserved E2 history and continuation task coverage."""

from hashlib import sha256
import json
from pathlib import Path


def verify_continuation(directory):
    directory = Path(directory)
    path = directory / "continuation.json"
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    if saved["schema"] != "synister.seed-output-continuation.v1":
        raise ValueError("Unsupported continuation provenance")
    source = Path(saved["source"])
    expected_files = saved["source_files_sha256"]
    actual_files = {
        str(p.relative_to(source)): sha256(p.read_bytes()).hexdigest()
        for p in source.rglob("*.json")
        if "frozen_source" not in p.relative_to(source).parts
        and "orphans" not in p.relative_to(source).parts
    }
    if actual_files != expected_files:
        raise ValueError("Original interrupted history changed")
    original_tasks = json.loads((source / "tasks.json").read_text())
    task_ids = [task["task_id"] for task in original_tasks]
    original_ids = {p.stem for p in (source / "cases").glob("*.json")}
    if (
        len(task_ids) != len(set(task_ids))
        or not original_ids <= set(task_ids)
        or saved["preserved_task_ids"] != sorted(original_ids)
        or saved["scheduled_task_ids"]
        != [key for key in task_ids if key not in original_ids]
    ):
        raise ValueError("Continuation omitted or duplicated original task IDs")
    for name in (
        "manifest.json",
        "protocol.json",
        "inputs.json",
        "sources.json",
        "preparation_tasks.json",
        "tasks.json",
        "orchestrator_source.json",
    ):
        if (
            name in expected_files
            and (directory / name).read_bytes() != (source / name).read_bytes()
        ):
            raise ValueError("Continuation changed original metadata: " + name)
    for key in original_ids:
        name = "cases/" + key + ".json"
        if (directory / name).read_bytes() != (source / name).read_bytes():
            raise ValueError(
                "Continuation replaced an original terminal outcome: " + key
            )
        record = json.loads((source / name).read_text())
        if "mapping_sha256" in record:
            name = "maps/" + key + ".json"
            if (
                sha256((directory / name).read_bytes()).hexdigest()
                != record["mapping_sha256"]
            ):
                raise ValueError("Continuation changed an original map output: " + key)
    for name in expected_files:
        if (
            name.startswith("preparation/")
            and (directory / name).read_bytes() != (source / name).read_bytes()
        ):
            raise ValueError("Continuation changed a shared seed or preparation cost")
    return {
        "source": str(source),
        "source_records_preserved": True,
        "preserved_attempts": len(original_ids),
        "missing_at_continuation": len(task_ids) - len(original_ids),
        "continuation_sha256": sha256(path.read_bytes()).hexdigest(),
    }
