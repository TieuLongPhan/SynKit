"""Strong factorized symbolic reference with exact inventory-set comparison."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Experiment.MTG.component_factorization import (  # noqa: E402
    combined_inventory,
    independent_factors,
)
from Experiment.MTG.component_observation_audit import (  # noqa: E402
    fork_product,
    cartesian_cuts,
    oracle,
    witness,
)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def run():
    protocol_path = Path(__file__).with_name("component_factorization_protocol.json")
    protocol = json.loads(protocol_path.read_text())
    input_path = ROOT / "Experiment/MTG/inputs/rcsb_atp_connectivity.json"
    data = json.loads(input_path.read_text())
    cases = []
    for retained in (False, True):
        for copies in (1, 2, 4, 8, 16):
            print(f"factor copies={copies} retention_control={retained}", flush=True)
            history, prepared, cuts, source_seconds, common_seconds = fork_product(
                data, copies, retained=retained
            )
            controls = {}
            if copies == 2:
                linked = replace(
                    prepared,
                    order=replace(
                        prepared.order,
                        pairs=prepared.order.pairs
                        + (("copy0:gamma", "copy1:capture"),),
                    ),
                )
                assert len(independent_factors(prepared)) == 2
                assert len(independent_factors(linked)) == 1
                controls["cross_copy_precedence_merges_factors"] = True
            expected = None
            if copies <= 2:
                expected = {
                    (v["total"], v["labelled"])
                    for _, g in cartesian_cuts(cuts)
                    for v in (oracle(g),)
                }
            rounds = []
            sets = []
            for repetition in range(3):
                for quotient in (
                    (False, True) if repetition % 2 == 0 else (True, False)
                ):
                    started = perf_counter()
                    inventories, details = combined_inventory(
                        prepared, quotient=quotient
                    )
                    elapsed = perf_counter() - started
                    assert details["independent_factors"] == copies
                    if expected is not None:
                        assert set(inventories) == expected
                    sets.append(set(inventories))
                    answers = []
                    for majority in (True, False):
                        feasible = [
                            pair
                            for pair in sorted(inventories)
                            if pair[0] > 0 and (2 * pair[1] > pair[0]) == majority
                        ]
                        answer = {
                            "status": "sat" if feasible else "unsat",
                            "ideal": None,
                            "inventory": None,
                        }
                        if feasible:
                            total, labelled = feasible[0]
                            answer.update(
                                ideal=sorted(inventories[feasible[0]]),
                                inventory={"total": total, "labelled": labelled},
                            )
                        answer["source_witness"] = witness(history, answer)
                        answers.append(answer)
                    rounds.append(
                        {
                            "repetition": repetition,
                            "method": "quotient" if quotient else "full_atom",
                            "complete_query_seconds": elapsed,
                            "answers": answers,
                            **details,
                        }
                    )
            assert all(s == sets[0] for s in sets)
            statuses = [(a["status"] for a in r["answers"]) for r in rounds]
            wanted = ["sat", "sat"] if retained else ["sat", "unsat"]
            assert all(list(s) == wanted for s in statuses)
            cases.append(
                {
                    "retention_control": retained,
                    "copies": copies,
                    "formal_ideals": 5**copies,
                    "ideal_enumeration_used_for_scale_queries": False,
                    "source_construction_seconds": source_seconds,
                    "shared_preparation_seconds": common_seconds,
                    "inventories": [list(p) for p in sorted(sets[0])],
                    "controls": controls,
                    "rounds": rounds,
                    "complete_query_median_seconds": {
                        m: statistics.median(
                            r["complete_query_seconds"]
                            for r in rounds
                            if r["method"] == m
                        )
                        for m in ("full_atom", "quotient")
                    },
                }
            )
    pinned = [
        Path(__file__),
        protocol_path,
        input_path,
        Path(__file__).with_name("component_factorization.py"),
        Path(__file__).with_name("component_observation_audit.py"),
        Path(__file__).with_name("component_observation.py"),
        Path(__file__).with_name("ranked_component_observation.py"),
    ]
    return {
        "schema": "mtg.component-factorization-evidence/1",
        "status": "PASS",
        "protocol": protocol,
        "cases": cases,
        "counts": {
            "timed_rounds": 60,
            "majority_answers": 120,
            "native_palette_checks": sum(
                r["native_palette_checks"] for c in cases for r in c["rounds"]
            ),
            "source_witness_applications": sum(
                a["source_witness"]["primitive_applications"]
                for c in cases
                for r in c["rounds"]
                for a in r["answers"]
            ),
        },
        "all_exact_inventory_sets_agree": True,
        "all_preselected_cases_retained": True,
        "physical_parallelism_established": False,
        "new_generic_factor_algorithm": False,
        "ncs_readiness_established": False,
        "sha256": {str(p.relative_to(ROOT)): digest(p) for p in pinned},
    }


if __name__ == "__main__":
    result = run()
    (ROOT / "Experiment/MTG/evidence/component_factorization.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({"status": "PASS", "counts": result["counts"]}))
