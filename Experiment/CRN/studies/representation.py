#!/usr/bin/env python
"""Measure representation sensitivity and repeat a paired siphon benchmark.

The cancellation arm keeps all species and reaction identities fixed. A change
in enabling cannot be attributed to deleting the catalyst's matrix row.
Timings compare fresh network objects in alternating algorithm order. All raw
samples are retained and summarized by median and interquartile range.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.CRN.tools.common import environment, write_report, source_hashes, native_experiment  # noqa: E402
from synkit.CRN import PathwayRealizability, SynCRN, crnt_summary, find_siphons  # noqa: E402
from Experiment.CRN.studies import build_kegg_crn, GLYCOLYSIS_FLUX  # noqa: E402
from Experiment.CRN.studies.crosschecks import brute_force_siphons  # noqa: E402
from Experiment.CRN.studies.scaling import generate_network  # noqa: E402


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
    outcome["independent_replay_ok"] = valid and counts == flow if outcome["status"] == "realizable" else None
    outcome["initial_marking"] = marking
    outcome["requested_flux"] = flow
    outcome["final_marking"] = (
        {crn.species[sid].label: int(n) for sid, n in replay.items() if n}
        if outcome["status"] == "realizable" and valid and counts == flow
        else None
    )
    outcome["decision_complete"] = (outcome["status"] == "realizable" and outcome["independent_replay_ok"] is True) or (
        outcome["status"] == "unrealizable" and outcome["exhaustive"] is True
    )
    return outcome


def sensitivity():
    """Compare intact/cancelled incidence and retained/removed currencies."""
    # KEGG's generic polyphosphate identifier conflates two polymer lengths.
    # An explicit catalyst avoids interpreting a phosphoryl donor as a catalyst.
    intact = SynCRN.from_reaction_strings(["A+C>>B+C"])
    for reaction in intact.reactions.values():
        reaction.label = "catalysed_conversion"
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
            marking = {"A": 1, "C": catalyst}
            arms.append(
                {
                    "arm": name,
                    "catalyst_tokens": catalyst,
                    "deficiency": summary.deficiency,
                    "n_complexes": summary.n_complexes,
                    "n_linkage_classes": summary.n_linkage_classes,
                    "outcome": execute(crn, {"catalysed_conversion": 1}, marking),
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
        "participant_control": "A+C -> B+C; both isolated networks have deficiency zero",
        "cancelled_incidences": changes,
        "species_order_unchanged": matrices[0]["species_order"] == matrices[1]["species_order"],
        "reaction_order_unchanged": matrices[0]["reaction_order"] == matrices[1]["reaction_order"],
        "net_matrix_unchanged": bool(np.array_equal(matrices[0]["S"], matrices[1]["S"])),
        "catalyst_arms": arms,
        "currency_arms": currency,
    }


def inventory_controls():
    """Separate ATP initiation, redox supply and phosphate for the fixed flux.

    The full KEGG model is closed to replenishment during firing. A successful
    initial inventory therefore supplies two NAD+, two orthophosphates and two
    ADP as well as two ATP. No ferredoxin or polyphosphate bypass is requested.
    """
    crn = build_kegg_crn("M00001", drop_currency=False)
    labels = {sid: species.label for sid, species in crn.species.items()}
    # These species form a siphon on the prescribed transitions: ATP production
    # consumes ADP, and ADP production consumes ATP. Other module reactions are
    # deliberately outside this fixed-route claim.
    adenylates = {"ATP", "ADP"}
    transitions = [r for r in crn.reactions.values() if r.label in GLYCOLYSIS_FLUX]
    siphon_valid = all(
        not (adenylates & {labels[s] for s, _ in r.rhs.items()})
        or bool(adenylates & {labels[s] for s, _ in r.lhs.items()})
        for r in transitions
    )
    cases = [
        ("glucose_only", {}),
        ("one_atp_only", {"ATP": 1}),
        ("two_atp_only", {"ATP": 2}),
        ("two_atp_and_redox", {"ATP": 2, "NAD+": 2}),
        ("two_atp_with_redox_and_phosphate", {"ATP": 2, "NAD+": 2, "Orthophosphate": 2}),
        ("one_atp_with_adp_redox_and_phosphate", {"ATP": 1, "ADP": 2, "NAD+": 2, "Orthophosphate": 2}),
        ("two_atp_with_adp_redox_and_phosphate", {"ATP": 2, "ADP": 2, "NAD+": 2, "Orthophosphate": 2}),
    ]
    rows = []
    for name, supply in cases:
        marking = {"alpha-D-Glucose": 1, **supply}
        outcome = execute(crn, dict(GLYCOLYSIS_FLUX), marking)
        net_final = {sid: marking.get(label, 0) for sid, label in labels.items()}
        for reaction in transitions:
            count = GLYCOLYSIS_FLUX[reaction.label]
            for sid, coefficient in reaction.lhs.items():
                net_final[sid] -= count * coefficient
            for sid, coefficient in reaction.rhs.items():
                net_final[sid] += count * coefficient
        rows.append(
            {
                "name": name,
                "outcome": outcome,
                "net_inventory_feasible": all(n >= 0 for n in net_final.values()),
                "net_inventory_deficits": {labels[sid]: -int(n) for sid, n in net_final.items() if n < 0},
            }
        )
    return {
        "scope": (
            "Prescribed KEGG glycolysis reaction multiset; fixed initial inventory; no replenishment " "during firing"
        ),
        "adenylate_siphon": {
            "species": sorted(adenylates),
            "valid_on_prescribed_transitions": siphon_valid,
            "initially_empty_for_glucose_only": True,
        },
        "cases": rows,
    }


def _exhaustive_siphons(crn, max_species=22):
    from synkit.CRN.backend import resolve_backend

    backend, library = resolve_backend()
    if backend == "cpp":
        from synkit.CRN.native_analysis import exhaustive_siphons

        return exhaustive_siphons(crn, library_path=library, max_species=max_species)
    return brute_force_siphons(crn, max_species=max_species)


def paired_timings(sizes, repeats):
    """Time two algorithms on identical networks and verify every output."""
    rows = []
    for size in sizes:
        samples = {"closure": [], "exhaustive": []}
        for repeat in range(repeats):
            outputs = {}
            order = ("closure", "exhaustive") if repeat % 2 == 0 else ("exhaustive", "closure")
            for algorithm in order:
                crn = generate_network("reversible_chain", size)
                start = time.perf_counter()
                result = (
                    find_siphons(crn, strict_limit=True)
                    if algorithm == "closure"
                    else _exhaustive_siphons(crn, max_species=22)
                )
                samples[algorithm].append(time.perf_counter() - start)
                labels = {sid: s.label for sid, s in crn.species.items()}
                outputs[algorithm] = {frozenset(labels.get(p, p) for p in s) for s in result}
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


def representation_report(sizes=(10, 14, 18, 20), repeats=5):
    """Return representation controls and paired complete siphon timings."""
    if repeats < 3 or not sizes or any(size < 1 or size > 21 for size in sizes):
        raise ValueError("use at least 3 repeats and sizes from 1 to 21")
    representation = sensitivity()
    inventory = inventory_controls()
    rows = paired_timings(sizes, repeats)
    statuses = [r["outcome"]["status"] for r in representation["catalyst_arms"]]
    checks = {
        "all_decisions_complete": all(
            row["outcome"]["decision_complete"]
            for row in representation["catalyst_arms"] + representation["currency_arms"]
        ),
        "net_matrix_preserved": representation["net_matrix_unchanged"],
        "orders_preserved": representation["species_order_unchanged"] and representation["reaction_order_unchanged"],
        "catalyst_control": statuses == ["unrealizable", "realizable", "realizable", "realizable"],
        "currency_control": [r["outcome"]["status"] for r in representation["currency_arms"]]
        == ["unrealizable", "realizable"],
        "all_certificates_replayed": all(
            r["outcome"]["independent_replay_ok"] is not False
            for r in representation["catalyst_arms"] + representation["currency_arms"]
        ),
        "paired_outputs_agree": all(r["outputs_agree"] for r in rows),
        "inventory_controls_complete": all(r["outcome"]["decision_complete"] for r in inventory["cases"]),
        "inventory_supply_threshold": [r["outcome"]["status"] for r in inventory["cases"]]
        == ["unrealizable"] * 6 + ["realizable"],
        "one_atp_is_balanced_but_order_blocked": inventory["cases"][-2]["net_inventory_feasible"]
        and inventory["cases"][-2]["outcome"]["status"] == "unrealizable",
        "empty_adenylate_siphon_verified": inventory["adenylate_siphon"]["valid_on_prescribed_transitions"],
        "supplied_flux_yields_two_pyruvates": (
            inventory["cases"][-1]["outcome"]["final_marking"].get("Pyruvate") == 2
            if inventory["cases"][-1]["outcome"]["final_marking"] is not None
            else False
        ),
    }
    report = {
        "schema": "synkit.crn-representation/1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "environment": environment(),
        "checks": checks,
        "source_sha256": source_hashes(),
        "parameters": {
            "repeats": repeats,
            "sizes": list(sizes),
            "summary": "median and linear-interpolated quartiles",
            "order": "alternating; fresh objects; construction excluded",
        },
        "claim_boundary": "Controlled model edits and small reversible chains; no biological or asymptotic inference.",
        "representation": representation,
        "glycolysis_inventory_controls": inventory,
        "paired_siphon_timings": rows,
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 14, 18, 20])
    args = parser.parse_args(argv)
    return write_report(representation_report(args.sizes, args.repeats), args.output)


main = native_experiment(main)

if __name__ == "__main__":
    raise SystemExit(main())
