"""Append-only E2 continuation with frozen workers and preserved terminal attempts.

Create a new continuation directory from an interrupted extension/follow-up.
The original directory is read-only. Only absent task records are scheduled;
timeouts and output caps are terminal outcomes, not retry candidates.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from unittest.mock import patch

from Experiment.Synister import seed_output_benchmark as benchmark
from Experiment.Synister.classify_enumeration import digest, encode


def atomic_save(path, value):
    """Publish a durable JSON record atomically, refusing any replacement."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(encode(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def exclusive_run(directory):
    """An OS lock prevents two orchestrators from scheduling the same tasks."""
    with (directory / ".continuation.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Continuation is already running") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def read(path):
    return json.loads(Path(path).read_text())


def inventory(directory):
    """Bind scientific records and metadata; interpreter caches are not evidence."""
    return {
        str(path.relative_to(directory)): digest(path)
        for path in sorted(directory.rglob("*.json"))
        if "frozen_source" not in path.relative_to(directory).parts
        and "orphans" not in path.relative_to(directory).parts
    }


def verify_snapshot(directory, controls):
    """Reject altered snapshots, task plans, seeds, environments or saved output."""
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    protocol = read(directory / "protocol.json")
    phase = manifest.get("phase")
    if phase not in ("extension", "followup"):
        raise ValueError("Recovery supports extension/follow-up snapshots only")
    for name, expected in manifest["file_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError("Snapshot artifact changed: " + name)
    if sys.version != manifest["python"] or any(
        version(name) != expected for name, expected in manifest["dependencies"].items()
    ):
        raise ValueError("Continuation environment differs from the original run")
    sources = read(directory / "sources.json")
    for name, content in sources.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid frozen source path")
        if (directory / "frozen_source" / relative).read_text() != content:
            raise ValueError("Frozen worker source changed: " + name)
    control = read(controls)
    if not control["all_passed"]:
        raise ValueError("E2 literal controls failed")
    for name, expected in control["source_sha256"].items():
        if "/tests/" not in name and (
            name not in sources
            or sha256(sources[name].encode()).hexdigest() != expected
        ):
            raise ValueError("Frozen source differs from literal controls: " + name)
    if digest(Path(manifest["inputs_source"])) != manifest["inputs_source_sha256"]:
        raise ValueError("Original input selection changed")
    parent = Path(manifest["parent"])
    for name, expected in manifest["parent_sha256"].items():
        if digest(parent / name) != expected:
            raise ValueError("Pilot provenance changed: " + name)
    audit = read(parent / "audit.json")
    if (
        not audit["all_output_comparisons_consistent"]
        or audit["attempts"] != 200
        or audit["summary_sha256"] != digest(parent / "summary.json")
    ):
        raise ValueError("Pilot gate is not satisfied")
    if (directory / "sources.json").read_bytes() != (
        parent / "sources.json"
    ).read_bytes():
        raise ValueError("Continuation does not use the original pilot sources")
    if phase == "followup":
        main = Path(manifest["main_extension"])
        audit = read(main / "audit.json")
        if (
            digest(main / "audit.json") != manifest["main_extension_audit_sha256"]
            or not audit["all_output_comparisons_consistent"]
            or audit["attempts"] != 320
            or audit["summary_sha256"] != digest(main / "summary.json")
        ):
            raise ValueError("Completed extension provenance changed")
    rows = read(directory / "inputs.json")
    selection_name = (
        "difficult_inputs.json" if phase == "followup" else "extension_inputs.json"
    )
    if rows != read(parent / selection_name):
        raise ValueError("Prespecified input selection changed")
    seconds = 300 if phase == "followup" else 60
    if (
        protocol["search_seconds"] != seconds
        or protocol["parent_seconds"] != seconds + 15
        or protocol["matched_attempts"] != len(rows) * 4
    ):
        raise ValueError("Original resource protocol changed")
    preparation_tasks = read(directory / "preparation_tasks.json")
    expected_preparation = [
        {
            "task_id": row["benchmark_id"] + ".seed",
            "benchmark_id": row["benchmark_id"],
            "reaction": row["reaction"],
            "stage": "seed",
            "memory_gib": 6,
        }
        for row in rows
    ]
    if preparation_tasks != expected_preparation:
        raise ValueError("Seed preparation plan changed")
    seeds = {}
    if {p.stem for p in (directory / "preparation").glob("*.json")} != {
        task["task_id"] for task in preparation_tasks
    }:
        raise ValueError("Missing, unknown or duplicate seed preparation records")
    for task in preparation_tasks:
        record = read(directory / "preparation" / (task["task_id"] + ".json"))
        if record["task"] != task:
            raise ValueError("Seed preparation record changed")
        if record.get("complete"):
            from Experiment.Synister.mapping_check import independent_map
            from synkit.Chem.Mapper.identifiability import parse_reaction

            reactant, product = parse_reaction(task["reaction"])
            if (
                independent_map(reactant, product, record["prediction"]["mapping"])[0]
                != record["seed_doubled_cd"]
            ):
                raise ValueError(
                    "Saved feasible seed does not match independent rescoring"
                )
        seeds[task["benchmark_id"]] = record
    expected_tasks = []
    for index, row in enumerate(rows):
        bid = row["benchmark_id"]
        seed = seeds[bid]
        for condition in (("none", "slap") if index % 2 == 0 else ("slap", "none")):
            for method in (
                ("synister", "milp") if index % 2 == 0 else ("milp", "synister")
            ):
                expected_tasks.append(
                    {
                        "task_id": f"{bid}.{condition}.{method}.indexed",
                        "benchmark_id": bid,
                        "reaction": row["reaction"],
                        "stage": "matched",
                        "method": method,
                        "output": "indexed",
                        "seed_condition": condition,
                        "initial_mapping": (
                            seed.get("prediction", {}).get("mapping")
                            if condition == "slap"
                            else None
                        ),
                        "seed_preparation_parent_seconds": (
                            seed["parent_seconds"] if condition == "slap" else 0
                        ),
                        "seconds": seconds,
                        "max_maps": 100000,
                        "memory_gib": 6,
                    }
                )
    tasks = read(directory / "tasks.json")
    if tasks != expected_tasks or len({task["task_id"] for task in tasks}) != len(
        tasks
    ):
        raise ValueError("Saved search task plan changed or contains duplicate IDs")
    by_id = {task["task_id"]: task for task in tasks}
    records = {}
    for path in (directory / "cases").glob("*.json"):
        record = read(path)
        if path.stem not in by_id or record["task"] != by_id[path.stem]:
            raise ValueError("Unknown or changed terminal task record: " + path.stem)
        if type(record.get("complete")) is not bool or not record.get("termination"):
            raise ValueError("Case is not a terminal outcome: " + path.stem)
        if (
            "worker_sha256" in record
            and record["worker_sha256"]
            != sha256(
                sources["Experiment/Synister/seed_output_worker.py"].encode()
            ).hexdigest()
        ):
            raise ValueError("Saved attempt used a different worker")
        if "mapping_sha256" in record:
            path = directory / "maps" / (path.stem + ".json")
            data = path.read_bytes()
            maps = json.loads(data)
            if (
                sha256(data).hexdigest() != record["mapping_sha256"]
                or len(data) != record["output_bytes"]
                or len(maps) != record["mapping_count"]
                or len({tuple(m) for m in maps}) != len(maps)
            ):
                raise ValueError("Saved map output changed")
        elif record["complete"]:
            raise ValueError("Complete saved attempt lacks output")
        records[record["task"]["task_id"]] = record
    return rows, tasks, list(seeds.values()), records


def prepare(source, output, controls):
    """Copy committed history into a new directory; preserve orphan files separately."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError("Continuation must be outside the original directory")
    if (source / "summary.json").exists():
        raise ValueError("Source run already has a terminal summary")
    _, tasks, _, records = verify_snapshot(source, controls)
    history = inventory(source)
    output.mkdir(parents=True, exist_ok=False)
    for folder in (
        "cases",
        "maps",
        "preparation",
        "classification",
        "details",
        "frozen_source",
        "orphans",
    ):
        (output / folder).mkdir()
    for name in history:
        relative = Path(name)
        # Map/detail files without an authoritative record remain historical
        # artifacts, but cannot obstruct a fresh execution of a missing task.
        if relative.parts[0] == "maps":
            if (
                relative.stem not in records
                or "mapping_sha256" not in records[relative.stem]
            ):
                destination = output / "orphans" / relative
            else:
                destination = output / relative
        else:
            destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, destination)
    sources = read(source / "sources.json")
    for name, content in sources.items():
        path = output / "frozen_source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    continuation = {
        "schema": "synister.seed-output-continuation.v1",
        "source": str(source),
        "source_files_sha256": history,
        "preserved_task_ids": sorted(records),
        "scheduled_task_ids": [
            task["task_id"] for task in tasks if task["task_id"] not in records
        ],
        "policy": "Preserve all terminal outcomes; execute only absent task IDs at original budgets.",
        "orchestrator_sha256": digest(Path(__file__)),
        "controls_sha256": digest(controls),
    }
    atomic_save(output / "continuation.json", continuation)
    return output


def quarantine_orphans(output, records):
    """Preserve outputs of interrupted uncommitted attempts before starting afresh."""
    for path in (output / "maps").glob("*.json"):
        if path.stem in records and "mapping_sha256" in records[path.stem]:
            continue
        destination = (
            output / "orphans" / "maps" / (path.stem + "." + digest(path) + ".json")
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != path.read_bytes():
                raise ValueError("Orphan identity collision")
            path.unlink()
        else:
            path.rename(destination)


def run(output, controls):
    """Finish missing tasks and classify complete sets; may itself be resumed."""
    output = Path(output).resolve()
    with exclusive_run(output):
        from Experiment.Synister.audit_seed_output_recovery import verify_continuation

        verify_continuation(output)
        if (output / "summary.json").exists():
            return read(output / "summary.json")
        rows, tasks, prepared, records = verify_snapshot(output, controls)
        quarantine_orphans(output, records)
        missing = [task for task in tasks if task["task_id"] not in records]
        started = time.perf_counter()
        # Only the active parent's record publication changes. Every scientific
        # worker runs from the original, byte-identical source snapshot.
        with patch.object(benchmark, "save", atomic_save):
            with ThreadPoolExecutor(max_workers=4) as pool:
                for record in pool.map(
                    lambda task: benchmark.execute(output, task), missing
                ):
                    records[record["task"]["task_id"]] = record
            ordered = [records[task["task_id"]] for task in tasks]
            grouped = defaultdict(list)
            for record in ordered:
                if record["complete"]:
                    grouped[
                        record["task"]["benchmark_id"], record["mapping_sha256"]
                    ].append(record)
            classifications_plan = []
            for (bid, identity), aliases in sorted(grouped.items()):
                chosen = min(aliases, key=lambda record: record["task"]["task_id"])
                classifications_plan.append(
                    {
                        "classification_id": bid + "." + identity[:16],
                        "reaction": chosen["task"]["reaction"],
                        "benchmark_id": bid,
                        "mapping_count": chosen["mapping_count"],
                        "mapping_sha256": identity,
                        "target_doubled_cd": chosen["minimum_doubled_cd"],
                        "map_path": str(
                            output / "maps" / (chosen["task"]["task_id"] + ".json")
                        ),
                        "source_task_id": chosen["task"]["task_id"],
                        "aliases": [record["task"]["task_id"] for record in aliases],
                    }
                )
            plan_path = output / "classification_tasks.json"
            if plan_path.exists():
                if read(plan_path) != classifications_plan:
                    raise ValueError("Existing classification plan differs")
            else:
                atomic_save(plan_path, classifications_plan)
            classifications = []
            for task in classifications_plan:
                path = output / "classification" / (task["classification_id"] + ".json")
                if path.exists():
                    record = read(path)
                    if record["task"] != task:
                        raise ValueError("Saved classification task differs")
                else:
                    detail = output / "details" / (task["classification_id"] + ".json")
                    if detail.exists():
                        destination = (
                            output
                            / "orphans"
                            / "details"
                            / (detail.stem + "." + digest(detail) + ".json")
                        )
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        detail.rename(destination)
                    record = benchmark.classify(output, task)
                classifications.append(record)
        summary = {
            "phase": read(output / "manifest.json")["phase"],
            "selected": len(rows),
            "preparation_attempts": len(prepared),
            "attempts": len(ordered),
            "matched_attempts": len(ordered),
            "diagnostic_attempts": 0,
            "seed_preparation": dict(
                Counter(record["termination"] for record in prepared)
            ),
            "classification_terminations": dict(
                Counter(record["termination"] for record in classifications)
            ),
            "structural_contradictions": sum(
                record.get("structural_audit", {}).get("consistent") is False
                for record in classifications
            ),
            "continuation": {
                "preserved_attempts": len(ordered) - len(missing),
                "new_attempts": len(missing),
                "orchestration_seconds": time.perf_counter() - started,
            },
            "files_sha256": {
                str(path.relative_to(output)): digest(path)
                for folder in (
                    "cases",
                    "maps",
                    "preparation",
                    "classification",
                    "details",
                )
                for path in sorted((output / folder).glob("*.json"))
            },
            "plans_sha256": {
                name: digest(output / name)
                for name in (
                    "preparation_tasks.json",
                    "tasks.json",
                    "classification_tasks.json",
                    "continuation.json",
                )
            },
        }
        atomic_save(output / "summary.json", summary)
        print(
            encode(
                {
                    k: v
                    for k, v in summary.items()
                    if k not in ("files_sha256", "plans_sha256")
                }
            ),
            flush=True,
        )
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        help="Interrupted original run; omit to resume a continuation",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--controls",
        type=Path,
        default=Path("paper/synister/evidence/seed_output_controls_v1.json"),
    )
    args = parser.parse_args()
    output = (
        prepare(args.source, args.output, args.controls) if args.source else args.output
    )
    run(output, args.controls)


if __name__ == "__main__":
    main()
