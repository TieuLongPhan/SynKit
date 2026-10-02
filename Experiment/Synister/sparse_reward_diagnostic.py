"""Offline checked sparse-dual diagnostics on the two unresolved root gaps."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from Experiment.Synister.global_milp import doubled_distance
from synkit.Chem.Mapper.identifiability import parse_reaction
from synkit.Chem.Mapper.slap.lap import _adjacency_and_elements
from synkit.Chem.Mapper.exact.distance_bounds import atom_profile_costs
from synkit.Chem.Mapper.exact.incremental_assignment import solve_assignment
from synkit.Chem.Mapper.exact.propagation_limits import PropagationDeadline
from synkit.Chem.Mapper.exact.sparse_reward_bound import (
    checked_sparse_reward_bound,
    diffuse_factor_messages,
)


def run(output, seconds):
    root = Path(__file__).resolve().parents[2]
    rows = json.loads(
        (root / "paper/synister/evidence/enumeration_main_v1/inputs.json").read_text()
    )
    seed_path = (
        root
        / "Experiment/Synister/runs/separator_spectrum_100_minimal_v1/prepared_seeds.json"
    )
    seeds = json.loads(seed_path.read_text())
    records = []
    for row in rows:
        if row["benchmark_id"] not in {"reaction_082", "reaction_093"}:
            continue
        r, p = parse_reaction(row["reaction"])
        a_raw, labels = _adjacency_and_elements(r.graph(), False)
        b_raw, product_labels = _adjacency_and_elements(p.graph(), False)
        a, b = np.rint(a_raw * 4).astype(np.int64), np.rint(b_raw * 4).astype(np.int64)
        allowed = np.asarray([[x == y for y in product_labels] for x in labels])
        profile = atom_profile_costs(a_raw, b_raw, labels, product_labels)
        costs = np.rint(np.where(np.isfinite(profile), profile * 4, 0)).astype(np.int64)
        baseline = solve_assignment(
            costs, allowed, tuple(range(len(a))), tuple(range(len(a)))
        )
        unary = np.zeros_like(a)
        record = {
            "reaction": row["benchmark_id"],
            "reaction_sha256": hashlib.sha256(row["reaction"].encode()).hexdigest(),
            "atoms": len(a),
            "profile_lower_bound_quarters": baseline.lower_bound,
            "seed_upper_bound_quarters": 2
            * doubled_distance(r, p, seeds[row["benchmark_id"]]["mapping"]),
            "stages": [],
        }
        started = time.perf_counter()
        stop = started + seconds
        messages = None
        for sweep in range(3):
            stage_started = time.perf_counter()
            try:
                if sweep:
                    messages = diffuse_factor_messages(
                        a, b, allowed, messages, deadline=stop
                    )
                bound = checked_sparse_reward_bound(
                    a, b, unary, allowed, messages, deadline=stop
                )
            except PropagationDeadline:
                record["interrupted_sweep"] = sweep
                break
            if bound is None:
                raise ValueError(
                    "Known feasible seed unexpectedly has an infeasible residual"
                )
            record["stages"].append(
                {
                    "sweeps": sweep,
                    "lower_bound_quarters": bound.lower_bound,
                    "seconds": time.perf_counter() - stage_started,
                    "adjacency_entries_scanned": bound.adjacency_entries_scanned,
                    "assignment_certificate_checked": True,
                }
            )
        record["total_seconds"] = time.perf_counter() - started
        record["best_sparse_lower_bound_quarters"] = max(
            (stage["lower_bound_quarters"] for stage in record["stages"]), default=None
        )
        records.append(record)
    output.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__),
        root / "synkit/Chem/Mapper/exact/sparse_reward_bound.py",
        root / "synkit/Chem/Mapper/exact/incremental_assignment.py",
    ]
    for path in sources:
        target = output / "frozen_source" / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    report = {
        "scope": "Static root-domain bound diagnostic, not a new production proof result",
        "seconds_per_case": seconds,
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "seeds_sha256": hashlib.sha256(seed_path.read_bytes()).hexdigest(),
        "records": records,
    }
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=0.25)
    args = parser.parse_args()
    if not 0 < args.seconds <= 5:
        parser.error("seconds must be in (0, 5]")
    run(args.output, args.seconds)
