#!/usr/bin/env python
"""Benchmark fresh formose DPO expansions with optional independent engines.

Run ``python Experiment/CRN/studies/formose_dpo.py --help`` for the standalone
source-distribution interface. Imports, native build/startup, fixed GML format
translation and independent identity audits are outside timed API calls.
Every timed call parses rules and seeds and constructs a fresh network.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import statistics
import subprocess
import sys
import threading
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.CRN.studies.formose_case_study import RULES, SEEDS  # noqa: E402
from Experiment.CRN.tools.common import _default_rdkit_sdk, environment  # noqa: E402

SCHEMA = "synkit.formose-dpo-comparison/5"
EXPECTED_COUNTS = {1: (3, 1), 2: (5, 4), 3: (9, 10), 4: (37, 44), 5: (302, 371), 6: (10572, 11239)}


def save(path, value):
    from Experiment.CRN.tools.reports import save_result

    save_result(path, value)


def rule_gml(smarts, name):
    """Translate the four element/bond-only formose rules to explicit-H GML.

    This restricted translation does not support general SMARTS query logic.
    Every mapped atom is retained and only bond additions/deletions are used.
    """
    from rdkit import Chem

    sides, nodes = [], {}
    for side in smarts.split(">>"):
        molecule = Chem.MolFromSmarts(side)
        if molecule is None:
            raise ValueError("Invalid mapped formose rule")
        labels = {a.GetAtomMapNum(): a.GetSymbol() for a in molecule.GetAtoms()}
        if 0 in labels or len(labels) != molecule.GetNumAtoms():
            raise ValueError("Formose GML translation requires unique atom maps")
        if nodes and labels != nodes:
            raise ValueError("Formose GML translation requires retained atom labels")
        nodes = labels
        edges = {}
        for bond in molecule.GetBonds():
            ends = tuple(sorted((bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum())))
            edges[ends] = {1: "-", 2: "=", 3: "#"}[int(bond.GetBondTypeAsDouble())]
        sides.append(edges)
    if len(sides) != 2:
        raise ValueError("Expected one directed reaction rule")
    shared = {edge: label for edge, label in sides[0].items() if sides[1].get(edge) == label}

    def edges_text(edges):
        return "\n".join(f'edge [ source {a} target {b} label "{label}" ]' for (a, b), label in sorted(edges.items()))

    return (
        f'rule [ ruleID "{name}" left [\n'
        + edges_text({e: v for e, v in sides[0].items() if e not in shared})
        + "\n] context [\n"
        + "\n".join(f'node [ id {i} label "{label}" ]' for i, label in sorted(nodes.items()))
        + "\n"
        + edges_text(shared)
        + "\n] right [\n"
        + edges_text({e: v for e, v in sides[1].items() if e not in shared})
        + "\n] ]"
    )


def normalize(species, reactions):
    """Compare directed multiset incidences without cancelling participants."""
    from rdkit import Chem

    cache = {}

    def canonical(value):
        if value not in cache:
            molecule = Chem.MolFromSmiles(value)
            if molecule is None:
                raise ValueError(f"Invalid exported molecule: {value}")
            cache[value] = Chem.MolToSmiles(Chem.RemoveHs(molecule), canonical=True, isomericSmiles=False)
        return cache[value]

    normalized_species = sorted({canonical(s) for s in species})

    def side(values):
        return sorted(Counter(canonical(value) for value in values).items())

    normalized_reactions = sorted({json.dumps([side(a), side(b)], separators=(",", ":")) for a, b in reactions})
    payload = {"species": normalized_species, "reactions": [json.loads(value) for value in normalized_reactions]}
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def source_hashes():
    """Cover the CRN controller, typed DPO kernel, inputs and build definitions."""
    files = {ROOT / "pyproject.toml", ROOT / "CMakeLists.txt", ROOT / "hatch_build.py"}
    for directory in (ROOT / "synkit", ROOT / "Experiment/CRN"):
        for candidate in directory.rglob("*"):
            if (
                candidate.is_file()
                and "results" not in candidate.relative_to(directory).parts
                and (
                    candidate.suffix in {".py", ".cpp", ".c", ".h", ".hpp", ".cmake"}
                    or candidate.name == "CMakeLists.txt"
                )
            ):
                files.add(candidate)
    files.update((ROOT / "Experiment/CRN/data").glob("*.json"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files) if p.is_file()}


class Worker:
    """One engine process with a timeout and a separate diagnostic stream."""

    def __init__(self, command, log, env, cpu, memory_gib):
        self.log = log.open("w")
        self.messages = queue.Queue()
        self.resource_limits_enforced = os.name == "posix"

        def limits():
            import resource

            if cpu is not None and hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(0, {cpu})
            if memory_gib is not None:
                cap = int(memory_gib * 1024**3)
                resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                env=env,
                text=True,
                bufsize=1,
                start_new_session=True,
                preexec_fn=limits if os.name == "posix" else None,
            )
        except Exception:
            self.log.close()
            raise

        def read():
            try:
                for line in self.process.stdout:
                    try:
                        self.messages.put(json.loads(line))
                    except ValueError:
                        self.messages.put({"phase": "error", "error": "Invalid worker JSON: " + line[:500]})
            finally:
                self.messages.put({"phase": "error", "error": "Worker exited before response"})

        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def receive(self, timeout, expected):
        try:
            result = self.messages.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"Worker exceeded {timeout:g} s while waiting for {expected}") from None
        if result.get("phase") != expected:
            raise RuntimeError(result.get("error", f"Unexpected worker response: {result.get('phase')}"))
        return result

    def send(self, command):
        self.process.stdin.write(json.dumps(command) + "\n")
        self.process.stdin.flush()

    def close(self, *, force=False):
        try:
            if self.process.poll() is None and not force:
                self.send({"phase": "stop"})
                self.process.wait(timeout=15)
            if self.process.poll() is None:
                if os.name == "posix":
                    os.killpg(self.process.pid, signal.SIGKILL)
                else:
                    self.process.kill()
                self.process.wait(timeout=15)
            if not force and self.process.returncode:
                raise RuntimeError(f"Worker exited with code {self.process.returncode}")
        finally:
            self.log.close()


def summarize(row):
    samples = row["samples"]
    if samples:
        q1, _, q3 = statistics.quantiles(samples, n=4, method="inclusive") if len(samples) > 1 else [samples[0]] * 3
        row.update(median_seconds=statistics.median(samples), q1_seconds=q1, q3_seconds=q3)


def finish_report(report, args, unchanged):
    """Require every requested repetition before reporting network agreement."""
    rows = report["runs"]
    names = ["synreactor-native"] + (["mod", "molfoundry"] if args.external else [])
    expected = {(backend, depth) for backend in names for depth in args.depths}
    actual = [(row["backend"], row["depth"]) for row in rows]
    completed = all(
        row["status"] == "complete"
        and len(row["samples"]) == args.repeats
        and len(row["warmup_samples"]) == args.warmups
        for row in rows
    )
    equality = all(
        len({row["snapshot_sha256"] for row in rows if row["depth"] == depth and row["status"] == "complete"}) == 1
        for depth in args.depths
    )
    present = len(actual) == len(set(actual)) and set(actual) == expected
    report["source_unchanged"] = unchanged
    report["checks"] = {
        "all_requested_runs_complete": present and completed,
        "all_requested_depths_present": present,
        "identical_completed_networks": present and completed and equality,
        "sources_unchanged": unchanged,
    }
    report["complete"] = present and completed
    report["status"] = "PASS" if all(report["checks"].values()) else "FAIL"
    return report


def _start_worker(args, backend, depth, executable, output, env, row):
    interpreter = getattr(args, backend + "_python") if backend != "synreactor-native" else sys.executable
    command = [
        interpreter,
        str(ROOT / "Experiment/CRN/tools/formose_dpo_worker.py"),
        backend,
        str(depth),
        str(output / "inputs.json"),
        str(executable),
    ]
    worker = None
    try:
        worker = Worker(command, output / row["log"], env, args.cpu, args.memory_gib)
        ready = worker.receive(args.startup_timeout, "ready")
        row.update(status="running", version=ready["version"], python=ready["python"])
        return worker
    except Exception as error:
        row["failure"] = f"{type(error).__name__}: {error}"
        if worker is not None:
            worker.close(force=True)
        return None


def _measure(worker, row, event, args, snapshots):
    backend, depth = event["backend"], event["depth"]
    phase, index = event["phase"], event["index"]
    started = time.monotonic()
    try:
        worker.send({"phase": phase, "index": index})
        generated = worker.receive(args.timeout, "generation_complete")
        event["seconds"] = generated["seconds"]
        exported = worker.receive(args.audit_timeout, "done")
        snapshot = normalize(exported["species"], exported["reactions"])
        digest = snapshot["sha256"]
        if row.get("snapshot_sha256", digest) != digest:
            raise ValueError("Network changed between repeated expansions")
        snapshots.setdefault((backend, depth), snapshot)
        row.update(snapshot_sha256=digest, n_species=len(snapshot["species"]), n_reactions=len(snapshot["reactions"]))
        if depth in EXPECTED_COUNTS and (row["n_species"], row["n_reactions"]) != EXPECTED_COUNTS[depth]:
            raise ValueError(f"Unexpected formose counts at depth {depth}")
        if "dpo_report" in exported:
            row["dpo_report"] = exported["dpo_report"]
        event["snapshot_sha256"] = digest
        row["warmup_samples" if phase == "warmup" else "samples"].append(generated["seconds"])
        row["audits"].append({"phase": phase, "index": index, "snapshot_sha256": digest})
        summarize(row)
        return True
    except Exception as error:
        row.update(
            status="timeout" if isinstance(error, TimeoutError) else "error",
            failed_phase=phase,
            failed_index=index,
            failure_stage="audit" if "seconds" in event else "expansion",
            failure=f"{type(error).__name__}: {error}",
        )
        worker.close(force=True)
        return False
    finally:
        event["wall_seconds_with_audit"] = time.monotonic() - started


def _finish_workers(workers, rows, output, depth, snapshots):
    for backend in list(workers):
        worker = workers.pop(backend)
        try:
            worker.close()
            rows[backend]["status"] = "complete"
        except Exception as error:
            rows[backend].update(status="error", failure=f"{type(error).__name__}: {error}")
        save(output / rows[backend]["result"], {**rows[backend], "snapshot": snapshots.get((backend, depth))})


def benchmark(args):
    from synkit.CRN.Construct.native_dpo import build_native_dpo_expander, dpo_expander_manifest

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    before = source_hashes()
    inputs = {"rules": RULES, "seeds": SEEDS, "gml_rules": [rule_gml(rule, f"r{i}") for i, rule in enumerate(RULES)]}
    save(output / "inputs.json", inputs)
    report = {
        "schema": SCHEMA,
        "status": "FAIL",
        "complete": False,
        "environment": environment(),
        "source_sha256": before,
        "inputs_sha256": hashlib.sha256((output / "inputs.json").read_bytes()).hexdigest(),
        "threads": 1,
        "cpu_affinity": [args.cpu] if args.cpu is not None else None,
        "timeout_seconds_per_expansion": args.timeout,
        "audit_timeout_seconds": args.audit_timeout,
        "memory_gib_per_process": args.memory_gib,
        "resource_limits_enforced": os.name == "posix",
        "repeats_requested": args.repeats,
        "warmups": args.warmups,
        "cross_tool_equality_checked": args.external,
        "timing_scope": (
            "Fresh rule/seed preparation, matching, DPO rewriting, molecular identity and network "
            "construction through each API; imports, controller startup, fixed GML translation and "
            "independent identity audits excluded."
        ),
        "order_policy": (
            "One active expansion; cyclic rotation of tools within each warm-up and measured round "
            "and across depths."
        ),
        "radical_policy": (
            "ignore: generic mapped formose templates encode element/bond queries without radical "
            "guards; host radical deltas remain zero"
        ),
        "component_policy": "union",
        "allow_self_mixtures": False,
        "scope": (
            "Exact species and directed stoichiometric reaction sets after canonicalization without "
            "explicit hydrogen vertices or stereochemistry; no kinetic or thermodynamic comparison."
        ),
        "schedule": [],
        "runs": [],
    }
    save(output / "comparison.json", report)
    try:
        executable = args.dpo_expander
        if executable is None:
            sdk = args.rdkit_dir or _default_rdkit_sdk()
            if sdk is None:
                raise ValueError("An RDKit C++ SDK is required; pass --rdkit-dir or SYNKIT_RDKIT_CPP_DIR")
            executable = build_native_dpo_expander(
                args.native_build_dir or Path(tempfile.gettempdir()) / "synkit-crn-native/dpo-expander", rdkit_dir=sdk
            )
        executable = Path(executable).resolve()
        report.update(native_dpo_expander=str(executable), native_dpo_manifest=dpo_expander_manifest(executable))
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        save(output / "comparison.json", report)
        return report

    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONHASHSEED="0",
        PYTHONPATH=str(ROOT),
    )
    names = ["synreactor-native"] + (["mod", "molfoundry"] if args.external else [])
    snapshots = {}
    for depth in args.depths:
        workers, rows = {}, {}
        try:
            for backend in names:
                row = {
                    "backend": backend,
                    "depth": depth,
                    "status": "error",
                    "samples": [],
                    "warmup_samples": [],
                    "audits": [],
                    "result": f"{backend}-{depth}.json",
                    "log": f"{backend}-{depth}.log",
                }
                report["runs"].append(row)
                rows[backend] = row
                worker = _start_worker(args, backend, depth, executable, output, env, row)
                if worker is not None:
                    workers[backend] = worker
                save(output / row["result"], row)
            for round_index in range(args.warmups + args.repeats):
                phase = "warmup" if round_index < args.warmups else "measured"
                index = round_index if phase == "warmup" else round_index - args.warmups
                shift = (round_index + depth - 1) % len(names)
                for backend in names[shift:] + names[:shift]:
                    if backend not in workers:
                        continue
                    worker, row = workers[backend], rows[backend]
                    event = {"depth": depth, "backend": backend, "phase": phase, "index": index}
                    report["schedule"].append(event)
                    if not _measure(worker, row, event, args, snapshots):
                        del workers[backend]
                    save(output / row["result"], {**row, "snapshot": snapshots.get((backend, depth))})
                    save(output / "comparison.json", report)
                print(f"Depth {depth}: {phase} {index + 1} complete", flush=True)
            _finish_workers(workers, rows, output, depth, snapshots)
        finally:
            for worker in workers.values():
                worker.close(force=True)
        save(output / "comparison.json", report)
    finish_report(report, args, before == source_hashes())
    save(output / "comparison.json", report)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--rdkit-dir", type=Path)
    result.add_argument("--native-build-dir", type=Path)
    result.add_argument("--dpo-expander", type=Path)
    result.add_argument("--external", action="store_true", help="Require MØD and MolFoundry comparisons")
    result.add_argument("--mod-python", default="python")
    result.add_argument("--molfoundry-python", default="python")
    result.add_argument("--depths", type=int, nargs="+", default=list(range(1, 7)))
    result.add_argument("--repeats", type=int, default=20)
    result.add_argument("--warmups", type=int, default=3)
    result.add_argument("--timeout", type=float, default=180)
    result.add_argument("--audit-timeout", type=float, default=180)
    result.add_argument("--startup-timeout", type=float, default=180)
    result.add_argument("--memory-gib", type=float, default=8)
    result.add_argument(
        "--cpu", type=int, default=min(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    return result


def main(argv=None):
    options = parser()
    args = options.parse_args(argv)
    if args.repeats < 1 or args.warmups < 0 or not args.depths or any(d < 1 for d in args.depths):
        options.error("Require positive depths and repeats and non-negative warmups")
    if len(set(args.depths)) != len(args.depths):
        options.error("Each depth may be requested only once")
    if min(args.timeout, args.audit_timeout, args.startup_timeout, args.memory_gib) <= 0:
        options.error("Time and memory budgets must be positive")
    report = benchmark(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "complete": report["complete"],
                "output": str(args.output_dir / "comparison.json"),
            }
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
