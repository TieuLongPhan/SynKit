"""Schedule disjoint resumable native subtrees and merge exact records online."""

import math
import multiprocessing as mp
import os
import resource
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

from .native_candidates import NativeEnumerationStop, enumerate_native_candidates, prepare_native_candidates
from .orbit_aggregation import OrbitAccumulator

_STATE = None


def _init(lgp, target, config, seed, library, counter, deadline, barrier, cpu_index, cpus):
    global _STATE
    from ..analysis import _BlindShellObserver, _property_vectors
    from ..graph.automorphism import bounded_automorphism_permutations
    from ..slap.lap import _adjacency_and_elements
    from .symmetry import permutation_group_order

    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    with cpu_index.get_lock():
        index = cpu_index.value
        cpu_index.value += 1
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpus[index % len(cpus)]})
    a, labels = _adjacency_and_elements(lgp[0], False)
    b, _ = _adjacency_and_elements(lgp[1], False)
    observer = _BlindShellObserver(a, b, labels, _property_vectors(lgp, config.reaction_center_properties), config)
    rg, rc = bounded_automorphism_permutations(lgp[0], binary=False, node_properties=config.symmetry_node_properties)
    pg, pc = bounded_automorphism_permutations(lgp[1], binary=False, node_properties=config.symmetry_node_properties)
    if not rc or not pc:
        raise ValueError("frontier search requires complete side-group proofs")
    accumulator = OrbitAccumulator(observer, rg[1:], permutation_group_order(rg[1:]),
                                   permutation_group_order(pg[1:]), library_path=library, record_only=True)
    prepared = prepare_native_candidates(lgp, target, library_path=library,
                                         initial_mapping=seed, node_properties=config.symmetry_node_properties)
    _STATE = (lgp, target, config, seed, library, counter, deadline, barrier, accumulator, prepared)


def _ready():
    barrier = _STATE[7]
    barrier.wait(timeout=30)
    barrier.wait(timeout=30)


def _slice(prefixes, slice_nodes):
    from ..analysis import _mapping_sha256

    lgp, target, config, seed, library, counter, deadline, _, accumulator, prepared = _STATE
    records, hashes = {}, set()

    def receive(mapping, cost):
        if time.perf_counter() >= deadline.value:
            raise NativeEnumerationStop("time_limit")
        hashes.add(_mapping_sha256(mapping))
        before = len(accumulator.seen)
        accumulator.observe(mapping, cost)
        if len(accumulator.seen) == before:
            return
        key = next(reversed(accumulator.seen))
        with counter.get_lock():
            if config.max_mappings is not None and counter.value >= config.max_mappings:
                accumulator.seen.pop(key)
                raise NativeEnumerationStop("mapping_limit")
            counter.value += 1
        records[key] = accumulator.seen[key]
        # Membership alone is sufficient to deduplicate subsequent slices.
        accumulator.seen[key] = None

    remaining = deque(prefixes)
    frontier = []
    totals = {"candidate_count": 0, "visited_nodes": 0, "visited_leaves": 0, "pruned": 0}
    batch_started = time.perf_counter()
    reason = None
    while remaining:
        prefix = remaining.popleft()
        result = enumerate_native_candidates(
            lgp, target, library_path=library,
            time_limit_seconds=max(0, deadline.value - time.perf_counter()),
            max_mappings=None, callback=receive, prefix=prefix,
            slice_nodes=slice_nodes, _prepared=prepared,
        )
        for name in totals:
            totals[name] += result[name]
        frontier.extend(result["frontier"])
        if result["reason"] not in (None, "work_slice"):
            reason = result["reason"]
            break
        if time.perf_counter() - batch_started >= .25:
            break
    frontier.extend(remaining)
    if reason is None and frontier:
        reason = "work_slice"
    totals.update(complete=reason is None, reason=reason, frontier=frontier,
                  retained_classes=len(records), elapsed_seconds=time.perf_counter()-batch_started)
    return totals, records, hashes


def frontier_orbit_search(lgp, target, config, seed, observer, *,
                          library_path, workers=8, slice_nodes=8192):
    """Consume every emitted subtree before claiming complete coverage.

    The shared cap counts unique records retained within each worker, including
    duplicate classes held by different workers. Prefix replay is deterministic
    and preserves the original two-sided pruning state.
    """
    from ..graph.automorphism import bounded_automorphism_permutations
    from .symmetry import permutation_group_order

    if (isinstance(workers, bool) or not isinstance(workers, int) or workers < 1
            or isinstance(slice_nodes, bool) or not isinstance(slice_nodes, int)
            or not 0 < slice_nodes < (1 << 63)):
        raise ValueError("workers and slice_nodes must be positive integers")
    if (config.binary or set(config.symmetry_node_properties) != set(config.reaction_center_properties)
            or config.time_limit_seconds is None or not math.isfinite(config.time_limit_seconds)):
        raise ValueError("frontier search requires weighted aligned properties and a finite deadline")
    rg, rc = bounded_automorphism_permutations(lgp[0], binary=False, node_properties=config.symmetry_node_properties)
    pg, pc = bounded_automorphism_permutations(lgp[1], binary=False, node_properties=config.symmetry_node_properties)
    if not rc or not pc:
        raise ValueError("frontier search requires complete side-group proofs")
    merged = OrbitAccumulator(observer, rg[1:], permutation_group_order(rg[1:]),
                              permutation_group_order(pg[1:]), library_path=library_path)
    context = mp.get_context("spawn")
    counter, deadline = context.Value("q", 0), context.Value("d", 0)
    cpu_index = context.Value("i", 0)
    barrier = context.Barrier(workers + 1)
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(workers))
    cpus = cpus[:workers]
    started = time.perf_counter()
    pending, running = deque([()]), set()
    records, results, reasons = {}, [], set()
    peak_pending = 1
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=context, initializer=_init,
        initargs=(lgp, target, config, seed, str(library_path), counter, deadline,
                  barrier, cpu_index, cpus),
    ) as pool:
        ready = [pool.submit(_ready) for _ in range(workers)]
        barrier.wait(timeout=30)
        search_started = time.perf_counter()
        deadline.value = search_started + config.time_limit_seconds
        barrier.wait(timeout=30)
        for future in ready:
            future.result()
        while pending or running:
            if time.perf_counter() >= deadline.value:
                reasons.add("time_limit")
            while pending and len(running) < workers and not reasons:
                prefixes = [pending.popleft() for _ in range(min(16, len(pending)))]
                running.add(pool.submit(_slice, prefixes, slice_nodes))
            if not running:
                break
            finished, running = wait(running, timeout=.25, return_when=FIRST_COMPLETED)
            for future in finished:
                result, found, hashes = future.result()
                observer.mapping_hashes.update(hashes)
                new_tasks = result.pop("frontier")
                if result["reason"] not in (None, "work_slice"):
                    reasons.add(result["reason"])
                else:
                    pending.extend(new_tasks)
                    peak_pending = max(peak_pending, len(pending))
                results.append(result)
                for key, value in found.items():
                    old = records.get(key)
                    if old is not None and (old[:2] != value[:2] or old[3:] != value[3:]):
                        raise RuntimeError("inconsistent exact records across frontier workers")
                    if old is None:
                        records[key] = value
                        merged.merge_record(key, value)
        merged.finish()
    if time.perf_counter() > deadline.value:
        reasons.add("time_limit")
    return {
        "complete": not reasons and not pending,
        "reasons": sorted(reasons),
        "timed_phase_seconds": time.perf_counter() - search_started,
        "retained_worker_records": counter.value,
        "candidate_count": sum(item["candidate_count"] for item in results),
        "mapping_limit_scope": "retained_worker_double_orbit_representatives",
        "workers": workers, "scheduler": "resumable_subtree_queue",
        "slice_nodes": slice_nodes, "completed_slices": len(results),
        "pending_subtrees": len(pending), "peak_pending_subtrees": peak_pending,
        "shards": results, "unique_classes": len(records),
        "weighted_product_representatives": observer.count,
        "labeled_mappings": observer.count * merged.product_order,
        "elapsed_seconds": time.perf_counter() - started,
    }, merged
