"""Bounded parallel enumeration with an exact double-orbit merge."""

import math
import multiprocessing as mp
import os
import resource
import time
from concurrent.futures import ProcessPoolExecutor

from .native_candidates import NativeEnumerationStop, enumerate_native_candidates
from .orbit_aggregation import OrbitAccumulator

_STATE = None
_READY = False


class NativeSharedBudget(NativeEnumerationStop):
    pass


def _init_worker(counter, deadline, barrier, cap):
    global _STATE, _READY
    _READY = False
    _STATE = counter, deadline, barrier, cap
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))


def _worker(lgp, target, config, seed, library_path, index, workers, cpu):
    global _READY
    from ..analysis import _BlindShellObserver, _mapping_key, _property_vectors
    from ..graph.automorphism import bounded_automorphism_permutations
    from ..slap.lap import _adjacency_and_elements
    from .symmetry import permutation_group_order

    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu})
    a, labels = _adjacency_and_elements(lgp[0], config.binary)
    b, _ = _adjacency_and_elements(lgp[1], config.binary)
    properties = _property_vectors(lgp, config.reaction_center_properties)
    observer = _BlindShellObserver(a, b, labels, properties, config)
    rg, rc = bounded_automorphism_permutations(
        lgp[0], binary=False, node_properties=config.symmetry_node_properties
    )
    pg, pc = bounded_automorphism_permutations(
        lgp[1], binary=False, node_properties=config.symmetry_node_properties
    )
    if not rc or not pc:
        raise ValueError("parallel orbit search requires complete side groups")
    accumulator = OrbitAccumulator(
        observer,
        rg[1:],
        permutation_group_order(rg[1:]),
        permutation_group_order(pg[1:]),
        library_path=library_path,
        record_only=True,
    )
    counter, deadline, barrier, cap = _STATE
    if not _READY:
        barrier.wait(timeout=30)
        barrier.wait(timeout=30)
        _READY = True

    def receive(mapping, cost):
        if time.perf_counter() >= deadline.value:
            raise NativeSharedBudget("time_limit")
        observer.mapping_hashes.add(_mapping_key(mapping))
        before = len(accumulator.seen)
        accumulator.observe(mapping, cost)
        if len(accumulator.seen) > before:
            with counter.get_lock():
                if cap is not None and counter.value >= cap:
                    accumulator.seen.popitem()
                    raise NativeSharedBudget("mapping_limit")
                counter.value += 1

    result = enumerate_native_candidates(
        lgp,
        target,
        library_path=library_path,
        time_limit_seconds=max(0, deadline.value - time.perf_counter()),
        max_mappings=None,
        initial_mapping=seed,
        node_properties=config.symmetry_node_properties,
        callback=receive,
        shard_index=index,
        shards=workers,
    )
    result = {
        key: value
        for key, value in result.items()
        if key not in {"mappings", "reactant_generators"}
    }
    result["retained_classes"] = len(accumulator.seen)
    return result, accumulator.seen, observer.mapping_hashes


def parallel_orbit_search(
    lgp, target, config, seed, observer, *, library_path, workers=8, shard_count=None
):
    from ..graph.automorphism import bounded_automorphism_permutations
    from .symmetry import permutation_group_order

    if config.binary or set(config.symmetry_node_properties) != set(
        config.reaction_center_properties
    ):
        raise ValueError(
            "parallel orbit analysis requires weighted matrices and aligned property sets"
        )
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    shard_count = workers if shard_count is None else shard_count
    if not isinstance(shard_count, int) or shard_count < workers:
        raise ValueError("shard_count must be at least workers")
    if (
        config.time_limit_seconds is None
        or not math.isfinite(config.time_limit_seconds)
        or config.time_limit_seconds < 0
    ):
        raise ValueError("parallel orbit analysis requires a finite time budget")
    rg, rc = bounded_automorphism_permutations(
        lgp[0], binary=False, node_properties=config.symmetry_node_properties
    )
    pg, pc = bounded_automorphism_permutations(
        lgp[1], binary=False, node_properties=config.symmetry_node_properties
    )
    if not rc or not pc:
        raise ValueError("parallel orbit search requires complete side groups")
    merged = OrbitAccumulator(
        observer,
        rg[1:],
        permutation_group_order(rg[1:]),
        permutation_group_order(pg[1:]),
        library_path=library_path,
    )
    context = mp.get_context("spawn")
    counter = context.Value("q", 0)
    deadline = context.Value("d", 0)
    barrier = context.Barrier(workers + 1)
    cpus = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(workers))
    )
    started = time.perf_counter()
    results = []
    # One complete representative per class survives the exact-key union.
    records = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_worker,
        initargs=(counter, deadline, barrier, config.max_mappings),
    ) as pool:
        futures = [
            pool.submit(
                _worker,
                lgp,
                target,
                config,
                seed,
                str(library_path),
                i,
                shard_count,
                cpus[i % min(workers, len(cpus))],
            )
            for i in range(shard_count)
        ]
        barrier.wait(timeout=30)
        deadline.value = time.perf_counter() + config.time_limit_seconds
        search_started = time.perf_counter()
        barrier.wait(timeout=30)
        for future in futures:
            result, found, hashes = future.result()
            observer.mapping_hashes.update(hashes)
            results.append(result)
            for key, value in found.items():
                old = records.get(key)
                if old is not None and (old[:2] != value[:2] or old[3:] != value[3:]):
                    raise RuntimeError(
                        "inconsistent double-orbit records across shards"
                    )
                records.setdefault(key, value)
    search_and_transfer_seconds = time.perf_counter() - search_started
    for key, record in records.items():
        merged.merge_record(key, record)
    merged.finish()
    within_deadline = time.perf_counter() <= deadline.value
    reasons = {result["reason"] for result in results if result.get("reason")}
    if not within_deadline:
        reasons.add("time_limit")
    return {
        "complete": all(result["complete"] for result in results) and within_deadline,
        "reasons": sorted(reasons),
        "timed_phase_seconds": time.perf_counter() - search_started,
        "retained_worker_records": counter.value,
        "candidate_count": sum(result["candidate_count"] for result in results),
        "mapping_limit_scope": "retained_worker_double_orbit_representatives",
        "workers": workers,
        "shard_count": shard_count,
        "shards": results,
        "unique_classes": len(records),
        "weighted_product_representatives": observer.count,
        "labeled_mappings": observer.count * merged.product_order,
        "elapsed_seconds": time.perf_counter() - started,
        "search_and_transfer_seconds": search_and_transfer_seconds,
    }, merged
