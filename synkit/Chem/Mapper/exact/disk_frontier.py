"""Explicit disk-backed exact shell protocol for outputs exceeding RAM budgets.

This is separate from the capped in-memory protocol. Full certificates, never
hashes alone, decide equality. Worker history is bounded by a single slice.
"""

import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import shutil
import sqlite3
import time
import zlib
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, replace
from pathlib import Path

from . import native_frontier
from .native_candidates import prepare_native_candidates
from .native_canonical import compact_code_identifier, transport_canonical_key
from .orbit_aggregation import OrbitAccumulator


def _key(key):
    # Transport contains only bytes or a pair of byte strings. Pickling this
    # fixed grammar, followed by lossless compression, preserves exact equality.
    return zlib.compress(pickle.dumps(transport_canonical_key(key), protocol=5), 1)


def _batch(prefixes, slice_nodes):
    # Every earlier batch has transferred its complete records to the parent.
    # Reobservations are safe: the disk store deduplicates globally. Clearing
    # history does not change native subtree coverage or discard class counts.
    native_frontier._STATE[8].seen.clear()
    return native_frontier._slice(prefixes, slice_nodes)


class DiskOrbitStore:
    """Exact global records and spectra, with bounded SQLite page cache."""

    def __init__(self, path):
        path = Path(path)
        if path.exists():
            raise ValueError("refusing to overwrite an existing exact store")
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA page_size=16384")
        self.db.execute("PRAGMA cache_size=-131072")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.execute("PRAGMA mmap_size=0")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("CREATE TABLE its (k BLOB PRIMARY KEY, identifier TEXT, w TEXT, record BLOB) WITHOUT ROWID")
        self.db.execute("CREATE TABLE templates (k BLOB PRIMARY KEY, identifier TEXT, w TEXT) WITHOUT ROWID")
        self.db.execute("CREATE TABLE witnesses (k BLOB PRIMARY KEY) WITHOUT ROWID")
        self.classes = self.templates = 0

    def add(self, key, value, accumulator):
        weight, stabilizer, digest, template, atoms, bonds = value
        if (not stabilizer or accumulator.reactant_order % stabilizer
                or accumulator.product_order % stabilizer
                or weight != accumulator.reactant_order // stabilizer):
            raise RuntimeError("unproved disk merge multiplicity")
        encoded = _key(key)
        template_encoded = _key(template)
        # Mapping witness is deliberately excluded from the invariant record:
        # different witnesses of the same exact orbit must agree on all else.
        invariant = pickle.dumps((weight, stabilizer, template_encoded, atoms, bonds), 5)
        row = self.db.execute("SELECT record FROM its WHERE k=?", (encoded,)).fetchone()
        if row is not None:
            if row[0] != invariant:
                raise RuntimeError("inconsistent exact records across disk batches")
            return False
        self.db.execute("INSERT INTO its VALUES (?,?,?,?)",
                        (encoded, compact_code_identifier(key), str(weight), invariant))
        row = self.db.execute("SELECT w FROM templates WHERE k=?", (template_encoded,)).fetchone()
        if row is None:
            self.db.execute("INSERT INTO templates VALUES (?,?,?)",
                            (template_encoded, compact_code_identifier(template), str(weight)))
            self.templates += 1
        else:
            # Counts remain arbitrary precision Python integers, not SQLite
            # int64 or floating point arithmetic.
            self.db.execute("UPDATE templates SET w=? WHERE k=?",
                            (str(int(row[0]) + weight), template_encoded))
        self.classes += 1
        observer = accumulator.observer
        observer.count += weight
        accumulator.atom_frequencies.update(dict(atoms))
        accumulator.bond_frequencies.update(dict(bonds))
        payload = str(weight).encode() + b":" + digest
        observer.stream_digest.update(len(payload).to_bytes(8, "little"))
        observer.stream_digest.update(payload)
        return True

    def add_witnesses(self, witnesses):
        self.db.executemany("INSERT OR IGNORE INTO witnesses VALUES (?)", ((w,) for w in witnesses))

    def contains(self, table, key):
        assert table in {"its", "templates", "witnesses"}
        encoded = key if table == "witnesses" else _key(key)
        return self.db.execute(f"SELECT 1 FROM {table} WHERE k=?", (encoded,)).fetchone() is not None

    def export(self, table, path):
        assert table in {"its", "templates"}
        count = total = 0
        digest = hashlib.sha256()
        with gzip.open(path, "wb") as stream:
            for identifier, weight in self.db.execute(f"SELECT identifier,w FROM {table} ORDER BY identifier,k"):
                payload = (json.dumps([identifier, int(weight)], separators=(",", ":")) + "\n").encode()
                stream.write(payload)
                digest.update(payload)
                count += 1
                total += int(weight)
        return {"path": str(path), "rows": count, "total_weight": str(total),
                "uncompressed_sha256": digest.hexdigest(), "format": "gzip_jsonl_identifier_count"}

    def close(self):
        self.db.commit()
        self.db.close()


def analyze_disk_reference_shell(lgp, reference_mapping, *, library_path, output,
                                 seconds=3600, workers=16, slice_nodes=8192):
    """Enumerate a reference-CD shell with exact external class tables.

    There is no cumulative record cap in this explicit protocol. Each worker
    retains only one native batch; its 4 GiB address-space limit is unchanged.
    Class-count arrays are streamed separately and bound to the small result
    manifest by hashes. This is not the original 60-second/capped API.
    """
    from ..analysis import (
        GlobalShellConfig,
        _BlindShellObserver,
        _mapping_key,
        _property_vectors,
        _reference_free_slap_seed,
    )
    from ..slap.lap import _adjacency_and_elements, chemical_distance
    from ..spectrum import exact_its_and_template_codes

    if (isinstance(workers, bool) or not isinstance(workers, int) or workers < 1
            or isinstance(slice_nodes, bool) or not isinstance(slice_nodes, int) or not 0 < slice_nodes < 2**63
            or not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("positive finite budget, integer workers and slice size required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    deadline_at = started + seconds
    library_path = Path(library_path).resolve()
    a, labels = _adjacency_and_elements(lgp[0], False)
    b, right_labels = _adjacency_and_elements(lgp[1], False)
    reference = tuple(int(i) for i in reference_mapping)
    if sorted(reference) != list(range(len(labels))) or any(labels[i] != right_labels[j] for i, j in enumerate(reference)):
        raise ValueError("invalid atom-compatible reference")
    config = GlobalShellConfig(time_limit_seconds=seconds, max_mappings=None)
    properties = _property_vectors(lgp, config.reaction_center_properties)
    symmetry_properties = tuple(n for n in config.symmetry_node_properties if n in _property_vectors(lgp, (n,)))
    if set(properties) != set(symmetry_properties):
        raise ValueError("symmetry must preserve all reported properties")
    config = replace(config, symmetry_node_properties=symmetry_properties, reaction_center_properties=tuple(properties))
    target = chemical_distance(lgp, reference, binary=False)
    seed, seed_stats = _reference_free_slap_seed(lgp, False, repair=True)
    observer = _BlindShellObserver(a, b, labels, properties, config)
    prepared = prepare_native_candidates(lgp, target, library_path=library_path,
                                         initial_mapping=seed, node_properties=symmetry_properties)
    rg, _, ro, po = prepared[4:8]
    merged = OrbitAccumulator(observer, rg[1:], ro, po, library_path=library_path, transport_keys=True)
    store = DiskOrbitStore(output / "exact.sqlite")
    context = mp.get_context("spawn")
    counter, deadline = context.Value("q", 0), context.Value("d", deadline_at)
    index, barrier = context.Value("i", 0), context.Barrier(workers + 1)
    cpus = sorted(os.sched_getaffinity(0))[:workers]
    pending, running = deque([()]), set()
    reasons, totals = set(), Counter()
    batches, last_progress = 0, 0.0
    protocol = {"name": "disk_exact_reference_shell_v1", "seconds": seconds,
                "workers": workers, "worker_address_space_bytes": 4 * 1024**3,
                "record_cap": None, "worker_history": "one_batch", "class_equality": "full_exact_certificate",
                "cpu_affinity": cpus, "library_sha256": hashlib.sha256(library_path.read_bytes()).hexdigest()}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                 initializer=native_frontier._init,
                                 initargs=(lgp, target, config, seed, str(library_path), counter,
                                           deadline, barrier, index, cpus, prepared[:-1])) as pool:
            ready = [pool.submit(native_frontier._ready) for _ in range(workers)]
            barrier.wait(timeout=30)
            barrier.wait(timeout=30)
            for future in ready:
                future.result()
            with (output / "batches.jsonl").open("w") as log:
                while pending or running:
                    if time.perf_counter() >= deadline_at:
                        reasons.add("time_limit")
                    while pending and len(running) < workers and not reasons:
                        size = min(64, max(1, len(pending) // (workers - len(running))))
                        prefixes = [pending.popleft() for _ in range(size)]
                        running.add(pool.submit(_batch, prefixes, slice_nodes))
                    if not running:
                        break
                    finished, running = wait(running, timeout=0.25, return_when=FIRST_COMPLETED)
                    for future in finished:
                        result, records, witnesses = future.result()
                        frontier = result.pop("frontier")
                        if result["reason"] in (None, "work_slice"):
                            pending.extend(frontier)
                        else:
                            reasons.add(result["reason"])
                        store.add_witnesses(witnesses)
                        for key, value in records.items():
                            store.add(key, value, merged)
                        store.db.commit()
                        log.write(json.dumps(result) + "\n")
                        batches += 1
                        for name in ("candidate_count", "visited_nodes", "visited_leaves", "new_search_nodes"):
                            totals[name] += result[name] or 0
                    now = time.perf_counter()
                    if now - last_progress >= 10:
                        if shutil.disk_usage(output).free < 5 * 1024**3:
                            raise OSError("disk store stopped before exhausting the filesystem")
                        progress = dict(totals, elapsed_seconds=now-started, batches=batches,
                                        unique_classes=store.classes, template_classes=store.templates,
                                        weighted_representatives=str(observer.count), pending=len(pending),
                                        running=len(running), reasons=sorted(reasons))
                        temporary = output / "progress.tmp"
                        temporary.write_text(json.dumps(progress, indent=2) + "\n")
                        temporary.replace(output / "progress.json")
                        print(json.dumps(progress), flush=True)
                        log.flush()
                        last_progress = now
                    # Release the last returned batch before waiting again.
                    records = witnesses = finished = None
        if time.perf_counter() > deadline_at:
            reasons.add("time_limit")
        complete = not reasons and not pending
        merged.finish()
        # Only now query the held-out mapping and structure identities.
        its_key, template_key, reference_reason = exact_its_and_template_codes(
            a, b, labels, properties, reference, template_radius=config.template_radius,
            tolerance=config.tolerance, timeout_seconds=config.structure_timeout_seconds,
            max_search_nodes=config.structure_max_search_nodes, _code_cache=observer.structure._code_cache)
        reference_its = None if its_key is None else store.contains("its", its_key)
        reference_template = None if template_key is None else store.contains("templates", template_key)
        exports = {table: store.export(table, output / f"{table}_class_counts.jsonl.gz")
                   for table in ("its", "templates")}
        assert exports["its"]["rows"] == store.classes
        assert exports["templates"]["rows"] == store.templates
        assert int(exports["its"]["total_weight"]) == int(exports["templates"]["total_weight"]) == observer.count
        result = {"protocol": protocol, "target_mode": "reference_cd", "target": target,
                  "reference_cd": target, "complete": complete, "status": "complete" if complete else "timeout",
                  "truncation_reason": None if complete else ",".join(sorted(reasons)),
                  "representative_solution_count": observer.count,
                  "labeled_solution_count": str(observer.count * po), "symmetry_group_order": str(po),
                  "mapping_hartley_entropy_nats": math.log(observer.count) if complete and observer.count else None,
                  "symmetry_quotient_complete": True,
                  "reference_mapping_observed": store.contains("witnesses", _mapping_key(reference)),
                  "reference_class_observed": reference_its,
                  "reaction_center": asdict(observer.spectrum("verified_product_subgroup_orbit_representatives")),
                  "structure": {"complete": complete and reference_reason is None,
                                "incomplete_reason": reference_reason or (None if complete else "shell_incomplete"),
                                "observed_its_class_count": store.classes,
                                "observed_template_class_count": store.templates,
                                "its_hartley_entropy_nats": math.log(store.classes) if complete and store.classes else None,
                                "reference_its_class_observed": reference_its,
                                "reference_template_class_observed": reference_template,
                                "class_count_exports": exports},
                  "seed": seed_stats, "counters": dict(totals), "batches": batches,
                  "end_to_end_wall_seconds": time.perf_counter()-started}
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        store.close()
