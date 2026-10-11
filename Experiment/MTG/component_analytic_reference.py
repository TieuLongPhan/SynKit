"""Explicit stronger linear summary for the two replicated fork workloads.

Coefficients are checked on every original single-copy labelled cut. This
reference is case-specific; the synthetic connectivity panel is nonadditive.
"""

from hashlib import sha256
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

import z3

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Experiment.MTG.phosphoenzyme_refinement import (  # noqa: E402
    fork_graphs,
    free_phosphate,
)


def run():
    input_path = ROOT / "Experiment/MTG/inputs/rcsb_atp_connectivity.json"
    data = json.loads(input_path.read_text())
    coefficients = []
    for retained in (False, True):
        for labels in ((18, 18), (18, 0), (0, 18), (0, 0)):
            _, _, cuts = fork_graphs(data, labels, retained_beta=retained)
            beta = int(18 in labels) if retained else int(labels[0] == 18)
            gamma = 0 if retained else int(labels[1] == 18)
            for ideal, world in cuts:
                expected = {
                    "total": int("beta" in ideal) + int("gamma" in ideal),
                    "labelled": beta * int("beta" in ideal)
                    + gamma * int("gamma" in ideal),
                }
                assert free_phosphate(world) == expected
            coefficients.append(
                {
                    "retention_control": retained,
                    "labels": list(labels),
                    "beta_label_delta": beta,
                    "gamma_label_delta": gamma,
                    "cuts_checked": 5,
                }
            )
    panel = []
    for retained in (False, True):
        for copies in (1, 2, 4, 8, 16):
            rounds = []
            for repetition in range(3):
                started = perf_counter()
                solver = z3.Solver()
                solver.set(timeout=10000, random_seed=0)
                capture = [z3.Bool(f"capture{i}") for i in range(copies)]
                beta = [z3.Bool(f"beta{i}") for i in range(copies)]
                gamma = [z3.Bool(f"gamma{i}") for i in range(copies)]
                solver.add(
                    *(
                        z3.Implies(branch[i], capture[i])
                        for branch in (beta, gamma)
                        for i in range(copies)
                    )
                )
                total = z3.Sum(*(z3.If(v, 1, 0) for v in beta + gamma))
                labelled = z3.Sum(
                    *(z3.If(v, 1, 0) for v in (beta if retained else beta + gamma))
                )
                answers = []
                for majority in (True, False):
                    solver.push()
                    solver.add(
                        total > 0,
                        2 * labelled > total if majority else 2 * labelled <= total,
                    )
                    status = solver.check()
                    assert status != z3.unknown
                    answers.append(str(status))
                    solver.pop()
                assert answers == (["sat", "sat"] if retained else ["sat", "unsat"])
                rounds.append(
                    {
                        "repetition": repetition,
                        "seconds": perf_counter() - started,
                        "answers": answers,
                    }
                )
            panel.append(
                {
                    "retention_control": retained,
                    "copies": copies,
                    "rounds": rounds,
                    "median_preparation_plus_two_queries_seconds": statistics.median(
                        r["seconds"] for r in rounds
                    ),
                }
            )
    # These completed synthetic cuts rule out fixed additive label increments.
    completed = json.loads(
        (ROOT / "Experiment/MTG/evidence/component_observation.json").read_text()
    )
    actual = {
        frozenset(row["ideal"]): row["inventory"]["labelled"]
        for row in completed["synthetic"]["states"]
    }
    nonadditive = {
        "empty_L": actual[frozenset()],
        "label_edge_only_L": actual[frozenset(("add0",))],
        "port_edge_only_L": actual[frozenset(("add1",))],
        "both_edges_L": actual[frozenset(("add0", "add1"))],
    }
    assert (
        nonadditive["both_edges_L"]
        != nonadditive["label_edge_only_L"]
        + nonadditive["port_edge_only_L"]
        - nonadditive["empty_L"]
    )
    pinned = [
        Path(__file__),
        input_path,
        ROOT / "Experiment/MTG/phosphoenzyme_refinement.py",
        ROOT / "Experiment/MTG/evidence/component_observation.json",
    ]
    return {
        "schema": "mtg.component-analytic-reference/1",
        "status": "PASS",
        "coefficients": coefficients,
        "cases": panel,
        "counts": {
            "single_copy_cut_checks": 40,
            "fresh_rounds": 30,
            "native_answers": 60,
        },
        "nonadditive_countercase": nonadditive,
        "coefficient_derivation_timed": False,
        "general_component_compiler": False,
        "cases_have_simpler_additive_reference": True,
        "ncs_readiness_established": False,
        "sha256": {
            str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in pinned
        },
    }


if __name__ == "__main__":
    result = run()
    (ROOT / "Experiment/MTG/evidence/component_analytic_reference.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({"status": "PASS", "counts": result["counts"]}))
