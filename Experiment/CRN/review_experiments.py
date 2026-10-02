#!/usr/bin/env python
"""Audit representation sensitivity and repeat a paired siphon benchmark.

The cancellation arm keeps all species and reaction identities fixed. Thus a
change in enabling cannot be attributed to deleting the catalyst's matrix row.
Timings compare fresh network objects in alternating algorithm order; all raw
samples are retained and summarized by median and interquartile range.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.CRN.common import environment, write_report
from synkit.CRN import PathwayRealizability, crnt_summary, find_siphons
from synkit.CRN.Benchmark import build_kegg_crn, GLYCOLYSIS_FLUX
from synkit.CRN.Benchmark.crosschecks import brute_force_siphons
from synkit.CRN.Benchmark.scaling import generate_network


def execute(crn, flow, marking):
    """Run bounded search and independently replay any returned certificate."""
    search = PathwayRealizability().load_syncrn_and_flow(
        crn, flow=flow, initial_marking=marking, species="label", reaction="label"
    )
    search.build_petri_net_from_flow()
    outcome = search.realizability_result().to_dict()
    replay = {sid: marking.get(s.label, 0) for sid, s in crn.species.items()}
    reactions = {r.label: r for r in crn.reactions.values()}
    counts = {}
    valid = True
    for label in outcome.get("certificate") or []:
        reaction = reactions[label]
        valid &= all(replay[s] >= n for s, n in reaction.lhs.items())
        for s, n in reaction.lhs.items():
            replay[s] -= n
        for s, n in reaction.rhs.items():
            replay[s] += n
        valid &= all(n >= 0 for n in replay.values())
        counts[label] = counts.get(label, 0) + 1
    outcome["independent_replay_ok"] = (
        valid and counts == flow if outcome["status"] == "realizable" else None
    )
    outcome["initial_marking"] = marking
    outcome["requested_flux"] = flow
    return outcome


def sensitivity():
    """Compare intact/cancelled incidence and retained/removed currencies."""
    intact = build_kegg_crn("M00001", drop_currency=True)
    cancelled = deepcopy(intact)
    changes = []
    for reaction in cancelled.reactions.values():
        for sid in set(reaction.lhs.counts) & set(reaction.rhs.counts):
            count = min(reaction.lhs.get(sid), reaction.rhs.get(sid))
            changes.append(
                {
                    "reaction": reaction.label,
                    "species": cancelled.species[sid].label,
                    "coefficient": count,
                }
            )
            for side in (reaction.lhs, reaction.rhs):
                side.counts[sid] -= count
                if not side.counts[sid]:
                    del side.counts[sid]
    matrices = [c.to_stoichiometric_matrices() for c in (intact, cancelled)]
    arms = []
    for name, crn in (("intact", intact), ("cancelled", cancelled)):
        summary = crnt_summary(crn)
        for catalyst in (0, 1):
            marking = {"alpha-D-Glucose": 1, "Polyphosphate": catalyst}
            arms.append(
                {
                    "arm": name,
                    "catalyst_tokens": catalyst,
                    "deficiency": summary.deficiency,
                    "n_complexes": summary.n_complexes,
                    "n_linkage_classes": summary.n_linkage_classes,
                    "outcome": execute(crn, {"R02189": 1}, marking),
                }
            )
    currency = []
    for drop in (False, True):
        crn = build_kegg_crn("M00001", drop_currency=drop)
        summary = crnt_summary(crn)
        currency.append(
            {
                "drop_currency": drop,
                "n_species": crn.n_species,
                "n_reactions": crn.n_reactions,
                "deficiency": summary.deficiency,
                "outcome": execute(crn, dict(GLYCOLYSIS_FLUX), {"alpha-D-Glucose": 1}),
            }
        )
    return {
        "cancelled_incidences": changes,
        "species_order_unchanged": matrices[0]["species_order"]
        == matrices[1]["species_order"],
        "reaction_order_unchanged": matrices[0]["reaction_order"]
        == matrices[1]["reaction_order"],
        "net_matrix_unchanged": bool(
            np.array_equal(matrices[0]["S"], matrices[1]["S"])
        ),
        "catalyst_arms": arms,
        "currency_arms": currency,
    }


def paired_timings(sizes, repeats):
    """Time two algorithms on identical networks and verify every output."""
    rows = []
    for size in sizes:
        samples = {"closure": [], "exhaustive": []}
        for repeat in range(repeats):
            outputs = {}
            order = (
                ("closure", "exhaustive")
                if repeat % 2 == 0
                else ("exhaustive", "closure")
            )
            for algorithm in order:
                crn = generate_network("reversible_chain", size)
                start = time.perf_counter()
                result = (
                    find_siphons(crn)
                    if algorithm == "closure"
                    else brute_force_siphons(crn, max_species=22)
                )
                samples[algorithm].append(time.perf_counter() - start)
                labels = {sid: s.label for sid, s in crn.species.items()}
                outputs[algorithm] = {
                    frozenset(labels.get(p, p) for p in s) for s in result
                }
            if outputs["closure"] != outputs["exhaustive"]:
                raise AssertionError(f"Siphon disagreement at size {size}")
        for algorithm, values in samples.items():
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            rows.append(
                {
                    "n_species": size + 1,
                    "n_reactions": 2 * size,
                    "algorithm": algorithm,
                    "samples_seconds": values,
                    "median_seconds": float(median),
                    "q1_seconds": float(q1),
                    "q3_seconds": float(q3),
                    "outputs_agree": True,
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 14, 18, 20])
    args = parser.parse_args()
    if args.repeats < 3 or any(size < 1 or size > 21 for size in args.sizes):
        parser.error("use at least 3 repeats and sizes from 1 to 21")
    representation = sensitivity()
    rows = paired_timings(args.sizes, args.repeats)
    statuses = [r["outcome"]["status"] for r in representation["catalyst_arms"]]
    checks = {
        "net_matrix_preserved": representation["net_matrix_unchanged"],
        "orders_preserved": representation["species_order_unchanged"]
        and representation["reaction_order_unchanged"],
        "catalyst_control": statuses
        == ["unrealizable", "realizable", "realizable", "realizable"],
        "currency_control": [
            r["outcome"]["status"] for r in representation["currency_arms"]
        ]
        == ["unrealizable", "realizable"],
        "all_certificates_replayed": all(
            r["outcome"]["independent_replay_ok"] is not False
            for r in representation["catalyst_arms"] + representation["currency_arms"]
        ),
        "paired_outputs_agree": all(r["outputs_agree"] for r in rows),
    }
    sources = list((ROOT / "synkit/CRN").rglob("*.py")) + [
        Path(__file__),
        ROOT / "Experiment/CRN/common.py",
        ROOT / "synkit/CRN/Benchmark/data/kegg_modules.json",
    ]
    report = {
        "schema": "synkit.crn-review-experiments/1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "environment": environment(),
        "checks": checks,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(sources)
        },
        "parameters": {
            "repeats": args.repeats,
            "sizes": args.sizes,
            "summary": "median and linear-interpolated quartiles",
            "order": "alternating; fresh objects; construction excluded",
        },
        "claim_boundary": "Controlled model edits and small reversible chains; no biological or asymptotic inference.",
        "representation": representation,
        "paired_siphon_timings": rows,
    }
    return write_report(report, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
