"""Controlled, isolated configured-graph costs with explicit censoring.

Each case/repetition/output mode runs in a fresh worker. Full code equality
uses a bounded disk store; hashes remain indices. Native process memory includes
the interpreter and imported libraries. Search, witness output and packaging
are measured separately, following the frozen study design.
"""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

import networkx as nx
from rdkit import Chem, RDLogger

from Experiment.Stereo.Canonicalization.exact_code_store import ExactCodeStore
from Experiment.Stereo.Canonicalization.independent_model import to_descriptor
from Experiment.Stereo.Canonicalization.independent_orbits import _selector, color_signature
from Experiment.Stereo.Canonicalization.performance_cases import cases, selection_metadata, state_from_json
from Experiment.Stereo.Canonicalization.runtime import (
    load_average, peak_rss_mib, process_cpu_seconds, process_rss_mib,
)
from Experiment.Stereo.provenance import ROOT, capture_source, finish_source, sha256
from synkit.Graph.Canon.exact import ExactColoredGraphCanonicalizer
from synkit.Graph.Stereo import (
    StereographMirrorStatus, canonicalize_configured_registry, classify_stereograph_mirror,
    descriptor_id, expand_configured_stereograph, mirror_stereo_registry, stereo_isomorphic,
)
from synkit.Graph.Stereo.canonical import _rdkit_graph_and_registry
from synkit.Graph.Stereo.rdkit_resonance import _resonance_graphs

SHARED_FRAMES = frozenset({"tetrahedral", "square_planar", "trigonal_bipyramidal", "octahedral",
                           "planar_bond", "atrop_bond"})
MAX_WORKER_OUTPUT_BYTES = 64 * 1024 * 1024


class _TimedCanonicalizer(ExactColoredGraphCanonicalizer):
    """Observe witness-output time without changing the search or its result."""

    witness_seconds = 0.0

    def _generated_witnesses(self, *args):
        started = time.perf_counter()
        try:
            return super()._generated_witnesses(*args)
        finally:
            self.witness_seconds += time.perf_counter() - started

    def _generator_witnesses(self, *args):
        started = time.perf_counter()
        try:
            return super()._generator_witnesses(*args)
        finally:
            self.witness_seconds += time.perf_counter() - started


def _checkpoint(result):
    print(json.dumps({"event": "checkpoint", "metrics": result}), flush=True)


def _prepare(specification):
    started = time.perf_counter()
    if specification["kind"] in {"public", "resonance"}:
        molecule = Chem.MolFromSmiles(specification["smiles"])
        if molecule is None:
            raise ValueError("Controlled input cannot be parsed.")
        graph, registry = _rdkit_graph_and_registry(molecule)
        graphs = (graph,)
    else:
        graph = nx.Graph()
        graph.add_nodes_from(specification["nodes"])
        graph.add_edges_from(specification["edges"])
        descriptors = tuple(to_descriptor(state_from_json(value)) for value in specification["states"])
        registry = {descriptor_id(value): value for value in descriptors}
        graphs = (graph,)
    preparation = time.perf_counter() - started
    started = time.perf_counter()
    if specification["kind"] == "resonance":
        graphs = _resonance_graphs(molecule)
    normalization = time.perf_counter() - started if specification["kind"] == "resonance" else 0.0
    return graphs, registry, {"preparation_seconds": preparation, "resonance_seconds": normalization,
                              "resonance_drawings": len(graphs)}


def _one_certificate(graph, registry, selectors, mode, seconds, max_nodes=None):
    started = time.perf_counter()
    expansion = expand_configured_stereograph(graph, registry.values(), **selectors)
    expansion_seconds = time.perf_counter() - started
    started = time.perf_counter()
    engine = _TimedCanonicalizer(expansion, node_color="color", edge_color=None, prune_automorphisms=True,
                                 enumerate_automorphism_group=mode == "complete_group")
    snapshot_seconds = time.perf_counter() - started
    started = time.perf_counter()
    result = engine.search(timeout_seconds=seconds, max_search_nodes=max_nodes)
    total = time.perf_counter() - started
    metrics = {"expansion_seconds": expansion_seconds, "snapshot_seconds": snapshot_seconds,
               "search_seconds": result.statistics.elapsed_seconds, "witness_seconds": engine.witness_seconds,
               "packaging_seconds": max(0.0, total - result.statistics.elapsed_seconds - engine.witness_seconds),
               "search_nodes": result.statistics.visited_nodes, "leaves": result.statistics.leaves,
               "refinement_rounds": result.statistics.refinement_rounds,
               "auxiliary_vertices": len(expansion), "auxiliary_edges": expansion.number_of_edges(),
               "returned_witnesses": len(result.automorphisms), "complete": result.complete,
               "automorphisms_complete": getattr(result, "automorphisms_complete", False)}
    if not result.complete:
        return None, {**metrics, "refusal_reason": result.reason}
    return result.canonical_code, {**metrics, "certificate_bytes": len(result.canonical_code.encode("utf-8"))}


def _pairwise(graphs, registry, selectors):
    if not all(value.descriptor_class in SHARED_FRAMES for value in registry.values()):
        return {"pairwise_seconds": None, "pairwise_scope": "not_measured_outside_selected_six_frame_families"}
    node_color, edge_color = _selector(selectors["atom_color"]), _selector(selectors["bond_color"])
    graph = graphs[0].copy()
    graph.graph["stereo_descriptors"] = registry
    reflected = graph.copy()
    reflected.graph["stereo_descriptors"] = mirror_stereo_registry(registry)
    started = time.perf_counter()
    equal = stereo_isomorphic(
        graph, reflected,
        node_match=lambda a, b: color_signature(node_color(a)) == color_signature(node_color(b)),
        edge_match=lambda a, b: color_signature(edge_color(a)) == color_signature(edge_color(b)),
    )
    return {"pairwise_seconds": time.perf_counter() - started, "pairwise_mirror_equal": equal,
            "pairwise_scope": "first Lewis drawing, selected six families, identical full attribute selectors"}


def _boundary(graph, registry, specification, selectors):
    if specification["expected"] == "incomplete_input":
        result = classify_stereograph_mirror(graph, registry, **selectors)
        if result.status is not StereographMirrorStatus.INCOMPLETE:
            raise ValueError("Unknown-input boundary failed to refuse.")
        return "incomplete_input"
    if specification["expected"] == "unsupported_input":
        class Unsupported:
            descriptor_class = "unsupported_performance_probe"
        try:
            canonicalize_configured_registry(graph, {"unsupported": Unsupported()}, **selectors)
        except TypeError:
            return "unsupported_input"
        raise ValueError("Unsupported registry did not refuse.")
    return None


def measure(specification, mode, seconds):
    """Measure one workload; limits cover the whole worker as well as search."""
    graphs, registry, metrics = _prepare(specification)
    selectors = {key: specification[key] for key in ("atom_color", "bond_color")}
    metrics.update({"base_vertices": len(graphs[0]), "base_edges": graphs[0].number_of_edges(),
                    "components": nx.number_connected_components(graphs[0]), "carriers": len(registry),
                    "families": dict(Counter(value.descriptor_class for value in registry.values()))})
    _checkpoint(metrics)
    boundary = _boundary(graphs[0], registry, specification, selectors)
    if boundary:
        return {**metrics, "status": boundary}
    codes, details = [], []
    for graph in graphs:
        code, detail = _one_certificate(graph, registry, selectors, mode, seconds,
                                        specification.get("max_search_nodes"))
        details.append(detail)
        _checkpoint({**metrics, "primary_members": details})
        if code is None:
            return {**metrics, "status": "bounded_search", "primary_members": details}
        codes.append(code)
    full_code = json.dumps(sorted(set(codes)), separators=(",", ":"))
    started = time.perf_counter()
    mirror_codes = [_one_certificate(graph, mirror_stereo_registry(registry), selectors, "generators", seconds)[0]
                    for graph in graphs]
    if any(value is None for value in mirror_codes):
        return {**metrics, "status": "bounded_mirror", "primary_members": details}
    mirror_equal = full_code == json.dumps(sorted(set(mirror_codes)), separators=(",", ":"))
    metrics.update({"mirror_seconds": time.perf_counter() - started, "canonical_mirror_equal": mirror_equal})
    _checkpoint({**metrics, "primary_members": details})
    metrics.update(_pairwise(graphs, registry, selectors))
    # A single-drawing pairwise predicate is not a resonance-family predicate.
    if specification["kind"] != "resonance" and metrics.get("pairwise_seconds") is not None:
        if metrics["pairwise_mirror_equal"] != mirror_equal:
            raise ValueError("Matched pairwise and canonical mirror decisions disagree.")
    result = {**metrics, "status": "complete", "primary_members": details, "canonical_code": full_code,
              "certificate_bytes": len(full_code.encode("utf-8"))}
    started = time.perf_counter()
    json.dumps(result)
    result["report_serialization_seconds"] = time.perf_counter() - started
    return result


def worker(args):
    RDLogger.DisableLog("rdApp.*")
    specification = json.loads(args.worker.read_text())
    started, started_cpu = time.perf_counter(), time.process_time()
    try:
        result = measure(specification, args.mode, args.worker_seconds)
    except Exception as error:
        result = {"status": "error", "error": f"{type(error).__name__}: {error}"}
    usage = os.times()
    result.update({"worker_operation_seconds": time.perf_counter() - started,
                   "worker_operation_cpu_seconds": time.process_time() - started_cpu,
                   "worker_cpu_user_seconds": usage.user,
                   "worker_cpu_system_seconds": usage.system,
                   "worker_cpu_total_seconds": usage.user + usage.system,
                   "worker_peak_rss_mib": peak_rss_mib()})
    print(json.dumps({"event": "result", "metrics": result}), flush=True)
    return 0


def _rss(pid):
    return process_rss_mib(pid)


def _process_cpu_seconds(pid):
    """Observe process CPU, including startup, before a censored worker exits."""
    return process_cpu_seconds(pid)


def isolated(specification, mode, seconds, max_rss_mib, environment, work):
    """Retain phase checkpoints when a time or native-memory limit kills a worker."""
    input_path, output_path, error_path = work / "input.json", work / "stdout.jsonl", work / "stderr.log"
    input_path.write_text(json.dumps(specification))
    argv = [sys.executable, "-m", "Experiment.Stereo.Canonicalization.performance", "--worker", str(input_path),
            "--mode", mode, "--worker-seconds", str(seconds)]
    started, observed, observed_cpu, interruption = time.monotonic(), 0.0, 0.0, None
    load_before = load_average()
    with output_path.open("w") as output, error_path.open("w") as errors:
        with subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=output, stderr=errors) as process:
            while process.poll() is None:
                observed = max(observed, _rss(process.pid))
                observed_cpu = max(observed_cpu, _process_cpu_seconds(process.pid))
                if observed > max_rss_mib:
                    interruption = "rss_limit"
                elif output_path.stat().st_size > MAX_WORKER_OUTPUT_BYTES:
                    interruption = "worker_output_limit"
                elif time.monotonic() - started > seconds:
                    interruption = "timeout"
                if interruption:
                    process.kill()
                    process.wait()
                    break
                time.sleep(0.05)
            returncode = process.returncode
    result = {}
    with output_path.open() as output:
        while line := output.readline(MAX_WORKER_OUTPUT_BYTES):
            if not line.endswith("\n"):
                interruption = interruption or "worker_output_limit"
                break
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            result.update(event["metrics"])
    if result.get("worker_peak_rss_mib", 0) > max_rss_mib:
        interruption = "rss_limit"
    if interruption or returncode != 0 or "status" not in result:
        result.pop("canonical_code", None)
        result["status"] = interruption or "worker_error"
        result["stderr_tail"] = error_path.read_text()[-2000:]
    return {**result, "parent_observed_peak_rss_mib": observed, "worker_returncode": returncode,
            "parent_observed_cpu_seconds": observed_cpu,
            "host_load_average_before": load_before, "host_load_average_after": load_average(),
            "worker_wall_seconds": time.monotonic() - started, "budget_seconds": seconds,
            "budget_peak_rss_mib": max_rss_mib, "rss_monitor_interval_seconds": 0.05}


def distribution(values):
    """Nearest-rank p95 and completed-only summaries, with explicit count."""
    if not values:
        return {"count": 0, "median": None, "p95": None, "maximum": None}
    values = sorted(values)
    return {"count": len(values), "median": statistics.median(values),
            "p95": values[(95 * len(values) + 99) // 100 - 1], "maximum": values[-1]}


def summarize(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["mode"]].append(record)
    result = {}
    for mode, values in grouped.items():
        complete = [value for value in values if value["status"] == "complete"]
        phases = ("worker_operation_seconds", "worker_operation_cpu_seconds", "worker_cpu_total_seconds",
                  "worker_wall_seconds", "preparation_seconds", "resonance_seconds",
                  "mirror_seconds", "pairwise_seconds", "report_serialization_seconds", "worker_peak_rss_mib",
                  "certificate_bytes")
        result[mode] = {"planned": len(values), "outcomes": dict(Counter(value["status"] for value in values)),
                        "completed_only": {key: distribution([value[key] for value in complete
                                                              if value.get(key) is not None]) for key in phases}}
        for field in ("expansion_seconds", "snapshot_seconds", "search_seconds", "witness_seconds",
                      "packaging_seconds", "search_nodes", "leaves", "refinement_rounds", "returned_witnesses"):
            totals = [sum(member[field] for member in value["primary_members"]) for value in complete]
            result[mode]["completed_only"][field] = distribution(totals)
    return result


def hardware():
    cpu_info = Path("/proc/cpuinfo")
    cpu_lines = cpu_info.read_text().splitlines() if cpu_info.exists() else ()
    model = next((line.split(":", 1)[1].strip() for line in cpu_lines if line.startswith("model name")),
                 platform.processor() or "unknown")
    return {"platform": platform.platform(), "cpu_model": model, "logical_cpu_count": os.cpu_count(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "load_average_at_report": load_average(),
            "native_memory_scope": "process RSS/peak RSS, including native allocations and imported libraries"}


def run(args):
    source = capture_source()
    selected = tuple(value for value in cases() if not args.case or value["id"] in args.case)
    if not selected or set(args.case or ()) - {value["id"] for value in selected}:
        raise ValueError("Unknown or empty controlled-study selection.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inputs_path = args.output.with_name(args.output.stem + "_inputs.json")
    inputs_path.write_text(json.dumps({"schema": "synkit.controlled-cost-inputs/1", "cases": selected,
                                      **selection_metadata(selected)}, indent=2) + "\n")
    environment = {**os.environ, **{key: "1" for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
    started, records, codes, disagreements = time.monotonic(), [], {}, []
    with tempfile.TemporaryDirectory(prefix="synkit-controlled-cost-") as temporary, ExactCodeStore() as store:
        for specification in selected:
            for mode in ("generators", "complete_group"):
                for repetition in range(args.repetitions):
                    seconds = args.stress_seconds if specification["stress"] else args.ordinary_seconds
                    if time.monotonic() - started >= args.total_seconds:
                        result = {"status": "study_total_cap"}
                    else:
                        seconds = min(seconds, args.total_seconds - (time.monotonic() - started))
                        result = isolated(specification, mode, seconds, args.max_rss_mib, environment, Path(temporary))
                    code = result.pop("canonical_code", None)
                    if code is not None:
                        index = store.intern(code)
                        previous = codes.setdefault(specification["id"], index)
                        if index != previous:
                            disagreements.append({"case": specification["id"], "mode": mode,
                                                  "repetition": repetition})
                        result.update({"full_code_intern_id": index, "digest_index": store.index_digest(index)})
                    result.update({"id": specification["id"], "mode": mode, "repetition": repetition,
                                   "kind": specification["kind"], "stress": specification["stress"],
                                   "expected": specification["expected"]})
                    records.append(result)
            print(json.dumps({"case": specification["id"], "outcomes": dict(Counter(
                value["status"] for value in records if value["id"] == specification["id"]))}), flush=True)
    unexpected = [value for value in records if value["status"] in {"error", "worker_error"} or
                  (value["expected"] != "complete" and value["status"] != value["expected"])]
    completed_modes = defaultdict(set)
    for value in records:
        if value["status"] == "complete":
            completed_modes[value["id"]].add(value["mode"])
    report = {"schema": "synkit.controlled-stereo-costs/1", "provenance": finish_source(source),
              "hardware": hardware(), "inputs": {"path": str(inputs_path), "sha256": sha256(inputs_path)},
              "protocol": {"repetitions": args.repetitions, "worker_processes": 1, "native_threads": 1,
                           "ordinary_seconds": args.ordinary_seconds, "stress_seconds": args.stress_seconds,
                           "total_seconds": args.total_seconds, "max_rss_mib": args.max_rss_mib,
                           "max_worker_output_bytes": MAX_WORKER_OUTPUT_BYTES,
                           "host_condition": "background contention permitted and disclosed by per-trial load",
                           "cpu_time_scope": "exact completed worker CPU; parent-sampled CPU for censored runs",
                           "search_statistics_exclude_group_and_serialization": True,
                           "mirror_output_mode": "generators", "full_codes_checked_in_disk_store": True},
              "summary": summarize(records), "records": records, "full_code_disagreements": disagreements,
              "cases_with_complete_code": len(completed_modes),
              "cases_with_both_output_modes_complete": sum(len(modes) == 2 for modes in completed_modes.values()),
              "unexpected_outcomes": [{key: value[key] for key in ("id", "mode", "repetition", "status")}
                                      for value in unexpected],
              "study_valid": not disagreements and not unexpected,
              "planned_runs_executed": all(value["status"] != "study_total_cap" for value in records),
              "wall_seconds": time.monotonic() - started,
              "claim_boundary": ("Censored runs contribute coverage, never completed identity "
                                 "or completed-only medians.")}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["study_valid"] and report["planned_runs_executed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", action="append")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--ordinary-seconds", type=float, default=30)
    parser.add_argument("--stress-seconds", type=float, default=120)
    parser.add_argument("--total-seconds", type=float, default=14400)
    parser.add_argument("--max-rss-mib", type=float, default=1024)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--mode", choices=("generators", "complete_group"), default="generators")
    parser.add_argument("--worker-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.worker is not None:
        return worker(args)
    if args.output is None or args.repetitions < 1 or min(
        args.ordinary_seconds, args.stress_seconds, args.total_seconds, args.max_rss_mib,
    ) <= 0:
        parser.error("Output, positive repetition count and positive resource budgets are required.")
    RDLogger.DisableLog("rdApp.*")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
