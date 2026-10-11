"""Check pointed-marking inversion against independently enumerated source cuts."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Experiment.MTG.partial_order_construction import build_history  # noqa: E402
from Experiment.MTG.resource_flux_comparison import (  # noqa: E402
    incidence,
    recover_pointed_configuration,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    process, history, _ = build_history()
    tokens, reactions, producer, consumer, *_ = incidence(history)
    reverse = {number: event for event, number in reactions.items()}
    producer = {m: reverse[number] for m, number in producer.items()}
    consumer = {m: reverse[number] for m, number in consumer.items()}
    # Build the oracle from all five legal schedules, not the inversion equations.
    oracle = {}
    for order in process.linear_extensions():
        active = set(history.initial_material_ids)
        oracle[frozenset(active)] = frozenset()
        for index, event_id in enumerate(order, start=1):
            event = process.event_by_id[event_id]
            active.difference_update(b.material_id for b in event.inputs)
            active.update(b.material_id for b in event.outputs)
            config = frozenset(order[:index])
            oracle[frozenset(active)] = config
            assert history.configuration_cut(config).active_material_ids == active
    accepted = refused = 0
    for mask in range(1 << len(tokens)):
        marking = frozenset(m for i, m in enumerate(tokens) if mask & (1 << i))
        try:
            result = recover_pointed_configuration(
                process.causal_graph, marking, tokens, producer, consumer
            )
        except AssertionError:
            assert marking not in oracle
            refused += 1
        else:
            assert oracle[marking] == result
            accepted += 1
    assert (accepted, refused) == (8, 120)
    pair = (frozenset(("a", "b")), frozenset(("a", "c")))
    counts = []
    for config in pair:
        active = history.configuration_cut(config).active_material_ids
        inventory = Counter()
        for m in active:
            value = history.material_by_id[m].value
            assert len(value.node_ids) == 1 and not value.edge_keys
            inventory[value.node_labels(next(iter(value.node_ids)))["state"]] += 1
        counts.append(dict(inventory))
    assert counts[0] == counts[1] == {0: 2, 1: 1}
    report = {
        "schema": "synkit.pointed-marking-audit/1",
        "status": "PASS",
        "summary": {
            "markings": 128,
            "accepted": accepted,
            "refused": refused,
            "source_configurations": len(oracle),
        },
        "unpointed_collision": {
            "configurations": [sorted(c) for c in pair],
            "shared_state_inventory": counts[0],
        },
        "sha256": {
            name: sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "Experiment/MTG/pointed_marking_audit.py",
                "Experiment/MTG/resource_flux_comparison.py",
                "Experiment/MTG/partial_order_construction.py",
                "synkit/Graph/MTG/history.py",
                "synkit/Graph/MTG/process.py",
            )
        },
        "limits": "Complete pointed ports of the nonempty-input acyclic profile; not anonymous species inversion.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
