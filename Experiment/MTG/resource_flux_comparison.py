"""Execute native fSSA on explicitly bound MTG resource projections.

The upstream binary performs simulation and producer-bin sampling. This adapter
checks its actual outputs against supplied histories; it does not infer rates,
atom origins, elementary events or physical independence.
"""

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Experiment.MTG.mcsa155_case import local_history  # noqa: E402
from Experiment.MTG.mcsa155_network import (  # noqa: E402
    build_models,
    _semantic_equal,
)
from Experiment.MTG.partial_order_construction import build_history  # noqa: E402
from Experiment.MTG.verified_cases import _unpack_certificate  # noqa: E402
from synkit.Graph.Morphism import LewisLabelledGraph  # noqa: E402
from synkit.Graph.MTG import cooccurrence_configuration  # noqa: E402

PROTOCOL = ROOT / "Experiment/MTG/resource_flux_protocol.json"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def histories(case_path):
    _, generic, _ = build_history()
    rows = [("n_poset", generic, ())]
    case = json.loads(case_path.read_text())
    for row in case["candidates"]:
        certificates = [_unpack_certificate(c) for c in row["certificates"]]
        assert all(c.replay().valid for c in certificates)
        states = [certificates[0].host.to_networkx()] + [
            c.result.to_networkx() for c in certificates
        ]
        history, _ = local_history(states, process_id="resource-flux-proposal")
        rows.append((f"mcsa155_proposal_{row['proposal']}", history, ()))
    _, parallel, serial, thioester = build_models(case_path)
    for name, composition in (
        ("mcsa155_two_sites", parallel),
        ("mcsa155_one_site_reused", serial),
    ):
        request = tuple(
            dict(embedding.material_mapping)[thioester]
            for embedding in (composition.left_embedding, composition.right_embedding)
        )
        rows.append((name, composition.history, request))
    return rows


def incidence(history):
    """Pointed resource ports retain the same binding information as MTG."""
    tokens = {m: f"M{i}" for i, m in enumerate(sorted(history.material_by_id))}
    reactions = {e.occurrence_id: i for i, e in enumerate(history.events, start=1)}
    producer = {
        b.material_id: reactions[e.occurrence_id]
        for e in history.events
        for b in e.outputs
    }
    consumer = {
        b.material_id: reactions[e.occurrence_id]
        for e in history.events
        for b in e.inputs
    }
    expected = {
        (producer.get(m, 0), target, tokens[m]): 1 for m, target in consumer.items()
    }
    lines = ["reactions"]
    vectors = {}
    for event in history.events:
        # Nonempty inputs and acyclic single-use ports enforce at most one firing.
        assert event.inputs
        inputs = [tokens[b.material_id] for b in event.inputs]
        outputs = [tokens[b.material_id] for b in event.outputs]
        assert not set(inputs) & set(outputs)
        lines.append(
            f"r{reactions[event.occurrence_id]} : {' + '.join(inputs)} -> "
            f"{' + '.join(outputs)}, 1.0;"
        )
        vector = Counter(outputs)
        vector.subtract(inputs)
        vectors[event.occurrence_id] = {k: v for k, v in vector.items() if v}
    assert len({tuple(sorted(v.items())) for v in vectors.values()}) == len(vectors)
    lines.append("initial state")
    lines.extend(
        f"1 {tokens[m]};" for m in sorted(history.initial_material_ids) if m in consumer
    )
    return tokens, reactions, producer, consumer, expected, vectors, lines


def ideals(graph):
    events = sorted(graph)
    dependencies = {n: nx.ancestors(graph, n) for n in graph}
    return [
        frozenset(events[i] for i in range(len(events)) if mask & (1 << i))
        for mask in range(1 << len(events))
        if all(
            not mask & (1 << i)
            or all(mask & (1 << events.index(p)) for p in dependencies[n])
            for i, n in enumerate(events)
        )
    ]


def active_copies(config, tokens, producer, consumer):
    return frozenset(
        m
        for m in tokens
        if (m not in producer or producer[m] in config)
        and (m not in consumer or consumer[m] not in config)
    )


def decode_world(history, active):
    graph = nx.Graph()
    origins = history.node_lineage_by_carrier
    for material in sorted(active):
        value = history.material_by_id[material].value
        mapping = {n: origins[material, n] for n in value.node_ids}
        part = nx.relabel_nodes(value.to_networkx(), mapping)
        assert not set(graph) & set(part)
        graph = nx.compose(graph, part)
    return LewisLabelledGraph.from_networkx(graph, history.schema)


def native_flux(data):
    rows = data["fluxes"]["complete"]
    result = {}
    for row in rows:
        key = (row["source"], row["target"], row["molecule"])
        assert key not in result and row["weight"] > 0
        result[key] = row["weight"]
    return result


def recover_pointed_configuration(graph, present, materials, producer, consumer):
    """Invert a complete pointed marking without enumerating configurations."""
    assert set(present) <= set(materials)
    inputs = {event: [] for event in graph}
    for material, event in consumer.items():
        inputs[event].append(material)
    executed = {}
    for event in nx.topological_sort(graph):
        assert inputs[event]
        candidates = {
            (
                executed[producer[m]] - int(m in present)
                if m in producer
                else 1 - int(m in present)
            )
            for m in inputs[event]
        }
        assert len(candidates) == 1
        value = candidates.pop()
        assert value in (0, 1)
        executed[event] = value
    result = frozenset(e for e, value in executed.items() if value)
    assert active_copies(result, materials, producer, consumer) == frozenset(present)
    return result


def native_snapshots(csv_text, vectors, tokens, producer, consumer, references, graph):
    records = list(csv.DictReader(csv_text.splitlines()))
    assert records
    # The native binary can batch events even with the prescribed every=1.
    frames = {m for m in tokens if m not in producer and m not in consumer}
    assert set(records[0]) == {"Time"} | {tokens[m] for m in tokens if m not in frames}
    observed, blocks = [], []
    prior = frozenset()
    prior_values = None
    for row in records:
        values = {tokens[m]: 1 if m in frames else int(row[tokens[m]]) for m in tokens}
        assert all(value in (0, 1) for value in values.values())
        present = frozenset(m for m, name in tokens.items() if values[name])
        config = recover_pointed_configuration(
            graph, present, tokens, producer, consumer
        )
        assert config in references
        assert prior <= config
        active = active_copies(config, tokens, producer, consumer)
        assert values == {v: int(m in active) for m, v in tokens.items()}
        if prior_values is None:
            assert not config
        else:
            block = sorted(config - prior)
            assert block
            delta = Counter()
            for event in block:
                delta.update(vectors[event])
            assert all(
                values[name] - prior_values[name] == delta[name] for name in values
            )
            blocks.append(block)
        observed.append(config)
        prior, prior_values = config, values
    assert prior == frozenset(vectors)
    return blocks, observed


def compare(history, request, directory, binary, panel):
    assert history.replay().valid
    tokens, reactions, producer, consumer, expected, vectors, lines = incidence(history)
    reverse = {v: k for k, v in reactions.items()}
    lines.extend([f"until {panel['until']}", f"every {panel['every']}"])
    text = "\n".join(lines) + "\n"
    references = {
        config: history.configuration_cut(config)
        for config in ideals(history.to_process().causal_graph)
    }
    all_configs = set(references)
    checked_native_diagram = False
    rows, snapshot_chains, visited = [], set(), set()
    for ordinal in range(1, panel["runs_per_history"] + 1):
        run = directory / str(ordinal)
        run.mkdir()
        model = run / "model.txt"
        model.write_text(text)
        with (run / "native.log").open("w") as stream:
            subprocess.run(
                [str(binary), model.name],
                cwd=run,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        native_path = run / "model.txt_fluxes.json"
        assert native_path.exists(), (run / "native.log").read_text()
        data = json.loads(native_path.read_text())
        flux = native_flux(data)
        assert flux == expected
        assert set(data["reactions"]) == {str(n) for n in range(len(reactions) + 1)}
        graph = nx.DiGraph()
        graph.add_nodes_from(reverse.values())
        graph.add_edges_from((reverse[a], reverse[b]) for a, b, _ in flux if a)
        assert nx.is_directed_acyclic_graph(graph)
        if not checked_native_diagram:
            configs = ideals(graph)
            assert set(configs) == all_configs
            for config in configs:
                numbers = frozenset(reactions[e] for e in config)
                active = active_copies(numbers, tokens, producer, consumer)
                source = references[config]
                assert active == source.active_material_ids
                assert _semantic_equal(decode_world(history, active), source.world)
            checked_native_diagram = True
        csv_text = (run / "model.txt.csv").read_text()
        # Event names are used for cut keys; resource incidence has numeric ids.
        event_producers = {m: reverse[n] for m, n in producer.items()}
        event_consumers = {m: reverse[n] for m, n in consumer.items()}
        blocks, observed = native_snapshots(
            csv_text,
            vectors,
            tokens,
            event_producers,
            event_consumers,
            references,
            graph,
        )
        witness = [
            event
            for block in blocks
            for event in nx.lexicographical_topological_sort(graph.subgraph(block))
        ]
        assert history.to_process().is_linear_extension(witness)
        snapshot_chains.add(tuple(tuple(sorted(c)) for c in observed))
        visited.update(observed)
        rows.append(
            {
                "ordinal": ordinal,
                "recorded_event_blocks": blocks,
                "block_linear_extension_witness": witness,
                "native_output": data,
                "native_csv": csv_text,
                "prefixes_checked": len(observed),
                "requested_copies_seen_together": (
                    any(
                        set(request) <= references[c].active_material_ids
                        for c in observed
                    )
                    if request
                    else None
                ),
            }
        )
    structural = (
        any(set(request) <= c.active_material_ids for c in references.values())
        if request
        else None
    )
    if request:
        assert structural == (cooccurrence_configuration(history, request) is not None)
    return {
        "model_text": text,
        "model_sha256": sha256(text.encode()).hexdigest(),
        "material_species": tokens,
        "inactive_frame_materials": sorted(
            m for m in tokens if m not in producer and m not in consumer
        ),
        "event_reactions": reactions,
        "expected_flux_cells": [
            {"source": a, "target": b, "molecule": s, "weight": w}
            for (a, b, s), w in sorted(expected.items())
        ],
        "source_replay_valid": True,
        "configuration_worlds_checked": len(references),
        "native_flux_relation": sorted([a, b] for a, b in graph.edges),
        "distinct_recorded_configuration_chains": len(snapshot_chains),
        "sampled_configuration_coverage": len(visited),
        "requested_material_copies": list(request),
        "requested_copies_structurally_coexist": structural,
        "runs": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--downloads", type=Path, default=Path("/tmp"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(PROTOCOL.read_text())
    binary = args.binary.resolve()
    assert digest(binary) == plan["upstream"]["downloads"]["mtg-fSSA_Linux"]["sha256"]
    for name, row in plan["upstream"]["downloads"].items():
        path = binary if name == "mtg-fSSA_Linux" else args.downloads / name
        assert digest(path) == row["sha256"], name
    for name, value in {**plan["inputs"], **plan["constructor_sha256"]}.items():
        assert digest(ROOT / name) == value, name
    cases = histories(ROOT / "Experiment/MTG/evidence/mcsa155_case.json")
    assert [name for name, _, _ in cases] == plan["panel"]["histories"]
    output = []
    with tempfile.TemporaryDirectory(
        prefix="mtg-resource-flux-", dir="/tmp"
    ) as directory:
        for name, history, request in cases:
            case_dir = Path(directory) / name
            case_dir.mkdir()
            row = compare(history, request, case_dir, binary, plan["panel"])
            output.append({"history": name, **row})
            print(
                json.dumps(
                    {
                        "history": name,
                        "runs": len(row["runs"]),
                        "configurations": row["configuration_worlds_checked"],
                        "status": "PASS",
                    }
                ),
                flush=True,
            )
    report = {
        "schema": "synkit.mtg-resource-flux-comparison/1",
        "status": "PASS",
        "script_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(PROTOCOL),
        "adapter_amendments_sha256": digest(
            ROOT / "Experiment/MTG/resource_flux_adapter_amendments.json"
        ),
        "upstream": plan["upstream"],
        "inputs": plan["inputs"],
        "source_sha256": {
            name: digest(ROOT / name)
            for name in (
                *plan["constructor_sha256"],
                "Experiment/MTG/mcsa_drawing.py",
                "Experiment/MTG/verified_cases.py",
                "synkit/Graph/MTG/history.py",
                "synkit/Graph/MTG/process.py",
                "synkit/Graph/MTG/gluing.py",
                "synkit/Graph/MTG/queries.py",
                "synkit/Graph/Morphism/llg.py",
                "synkit/Rule/Apply/dpo.py",
                "synkit/Rule/Compose/laws.py",
            )
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "networkx": nx.__version__,
        },
        "histories": output,
        "summary": {
            "histories": len(output),
            "native_runs": sum(len(r["runs"]) for r in output),
            "configuration_worlds_checked": sum(
                r["configuration_worlds_checked"] for r in output
            ),
            "native_prefix_states_checked": sum(
                run["prefixes_checked"] for r in output for run in r["runs"]
            ),
            "native_rule_applications": sum(
                sum(len(block) for block in run["recorded_event_blocks"])
                for r in output
                for run in r["runs"]
            ),
            "flux_cells_compared": sum(
                len(r["expected_flux_cells"]) * len(r["runs"]) for r in output
            ),
            "structural_thioester_coexistence": [
                r["requested_copies_structurally_coexist"]
                for r in output
                if r["requested_material_copies"]
            ],
            "all_pointed_resource_projections_agree": True,
            "complete_pipeline_advantage": False,
        },
        "limits": plan["limits"]
        + [
            plan["panel"]["randomness"],
            "The native CSV batches events with every=1. Recorded markings and blocks are checked; "
            "within-block order/times are not observed. A legal block extension is a witness, not the native schedule.",
            "Inactive initial frames are not allowed in the native grammar. They remain constant shared metadata "
            "and are restored explicitly in graph cuts; no native inference of those frames is claimed.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, separators=(",", ":")) + "\n")
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
