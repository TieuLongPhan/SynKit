#!/usr/bin/env python
"""Replay both frozen formyl-CoA transferase drawings against PDB connectivity.

The qualitative observation is independent of the curated mechanism drawings.
The depicted protonation and capped residue fragments are retained explicitly;
this does not validate every proposed intermediate or physical event timing.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shlex
import sys

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Experiment.MTG.mcsa_drawing import (  # noqa: E402
    align_drawing,
    observed_atoms,
    push_group,
    read_drawing,
    value,
)
from Experiment.MTG.verified_cases import _pack_certificate  # noqa: E402
from synkit.Graph.MTG import (  # noqa: E402
    MaterialBinding,
    MaterialOccurrence,
    OccurrenceMTG,
    OccurrenceProcess,
    RuleOccurrence,
)
from synkit.Rule.Apply import RuleSpan, apply_dpo  # noqa: E402


def _pdb_connections(path):
    lines = path.read_text().splitlines()
    start = lines.index("_struct_conn.id ")
    headers, rows = [], []
    for line in lines[start:]:
        line = line.strip()
        if line.startswith("_struct_conn."):
            headers.append(line.strip().split(".", 1)[1])
        elif line == "#":
            break
        else:
            fields = shlex.split(line)
            if len(fields) != len(headers):
                raise ValueError(
                    "The selected deposited connection loop has an unsupported row."
                )
            rows.append(dict(zip(headers, fields)))
    observed = []
    for row in rows:
        if row["conn_type_id"] != "covale":
            continue
        if (
            row["ptnr1_label_comp_id"] == "ASP"
            and row["ptnr1_auth_seq_id"] == "169"
            and row["ptnr1_label_atom_id"] == "CG"
            and row["ptnr2_label_comp_id"] == "COA"
            and row["ptnr2_label_atom_id"] == "S1P"
        ):
            observed.append(row)
    if len(observed) != 2:
        raise ValueError(
            "The frozen two-site experimental intermediate observation was not recovered."
        )
    return observed


def _bindings(materials):
    return tuple(
        MaterialBinding(
            m.occurrence_id,
            frozenset(m.value.node_ids),
            {n: n for n in m.value.node_ids},
        )
        for m in materials
    )


def _components(graph, prefix):
    return tuple(
        MaterialOccurrence(f"{prefix}:{i}", value(graph.subgraph(nodes).copy()))
        for i, nodes in enumerate(
            sorted(nx.connected_components(graph), key=lambda c: tuple(sorted(c)))
        )
    )


def local_history(states, *, process_id):
    """Keep unchanged components as frames, with certified local and global steps."""
    active = list(_components(states[0], "input"))
    materials, events, certificates = list(active), [], []
    initial = tuple(m.occurrence_id for m in active)
    for index, (before, after) in enumerate(zip(states, states[1:]), start=1):
        changed = {n for n in before if before.nodes[n] != after.nodes[n]}
        for edge in set(map(frozenset, before.edges)) | set(
            map(frozenset, after.edges)
        ):
            a, b = tuple(edge)
            if before.get_edge_data(a, b) != after.get_edge_data(a, b):
                changed.update(edge)
        inputs = [m for m in active if set(m.value.node_ids) & changed]
        participants = {n for m in inputs for n in m.value.node_ids}
        outputs = _components(after.subgraph(participants).copy(), f"e{index}:output")
        left, right = value(before.subgraph(participants).copy()), value(
            after.subgraph(participants).copy()
        )
        rule = RuleSpan.from_mapping(
            left,
            right,
            {n: n for n in participants},
            boundary="closed",
            name=f"e{index}",
        )
        local = apply_dpo(rule, left, {n: n for n in participants})
        global_app = apply_dpo(rule, value(before), {n: n for n in participants})
        assert (
            local.certificate.replay().valid and global_app.certificate.replay().valid
        )
        assert global_app.result == value(after)
        events.append(
            RuleOccurrence(f"e{index}", rule, _bindings(inputs), _bindings(outputs))
        )
        certificates.append(global_app.certificate)
        input_ids = {m.occurrence_id for m in inputs}
        active = [m for m in active if m.occurrence_id not in input_ids] + list(outputs)
        materials.extend(outputs)
    process = OccurrenceProcess(process_id, tuple(materials), tuple(events))
    history = OccurrenceMTG.from_process(process)
    assert history.replay().valid
    assert set(initial) == set(history.initial_material_ids)
    return history, certificates


def _world_matches(history, cut, graph):
    renaming = {
        lineage.lineage_id: state.carrier
        for lineage, state in cut.node_states
        if state.present
    }
    return cut.world.relabel(renaming) == value(graph)


def proposal(input_dir, rows, number):
    drawings = [
        read_drawing(input_dir / row["local_file"])
        for row in sorted(rows, key=lambda r: r["step"])
    ]
    current = drawings[0][0]
    states, arrows, bindings = [current], [], []
    anchors = observed_atoms(current)
    for index, (drawing, moves) in enumerate(drawings[:-1], start=1):
        maps = align_drawing(current, drawing)
        mapping = maps[0]
        translated = tuple(
            tuple(tuple(mapping[n] for n in p) for p in move) for move in moves
        )
        result = push_group(current, translated)
        following = align_drawing(result, drawings[index][0])
        bindings.append(
            {
                "step": index,
                "before_correspondence_alternatives": [sorted(m.items()) for m in maps],
                "selected_before_correspondence": sorted(mapping.items()),
                "next_drawing_correspondence_count": len(following),
                "omitted_carriers_retained": sorted(
                    set(current) - set(mapping.values())
                ),
                "observed_anchors_fixed_under_all_bindings": True,
            }
        )
        states.append(result)
        arrows.append(translated)
        current = result
    history, certificates = local_history(
        states, process_id=f"MCSA-155-proposal-{number}"
    )
    cuts, bond_presence, thioester = [], [], []
    carbon, sulfur = anchors
    for length, graph in enumerate(states):
        cut = history.configuration_cut(f"e{i}" for i in range(1, length + 1))
        assert _world_matches(history, cut, graph)
        present = graph.has_edge(carbon, sulfur)
        carbonyl = any(
            graph.nodes[n]["element"] == "O" and graph.edges[carbon, n]["pi_order"] == 1
            for n in graph.neighbors(carbon)
        )
        bond_presence.append(present)
        thioester.append(present and carbonyl)
        cuts.append(sorted(cut.active_material_ids))
    return (
        {
            "proposal": number,
            "status": "PASS",
            "native_closed_steps": len(certificates),
            "next_source_drawings_bound": len(bindings),
            "source_correspondences": bindings,
            "observed_origin_atoms": {
                "Asp169_carbonyl_carbon": carbon,
                "CoA_sulfur": sulfur,
            },
            "covalent_Asp_CoA_at_cuts": bond_presence,
            "thioester_motif_at_cuts": thioester,
            "observation_supported": any(bond_presence),
            "state_atoms": len(states[0]),
            "initial_total_charge": sum(
                d["charge"] for _, d in states[0].nodes(data=True)
            ),
            "source_protonation_note": (
                "Drawn oxalate includes one inferred O-H; no silent correction to the named dianion."
            ),
            "source_fragments_note": "Aliased residue caps are depicted fragments, not reconstructed full protein.",
            "material_cuts": cuts,
            "event_participating_atoms": [
                len(e.rule.left.node_ids) for e in history.events
            ],
            "arrows": arrows,
            "certificates": [_pack_certificate(c) for c in certificates],
        },
        history,
        certificates,
        states,
    )


def run(protocol_path, input_dir):
    protocol = json.loads(protocol_path.read_text())
    for row in protocol["source_files"]:
        path = input_dir / row["local_file"]
        if sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Frozen external input checksum differs: {path.name}")
    observations = _pdb_connections(input_dir / "mtg-2vjl.cif")
    rows, histories, certificates, states = [], {}, {}, {}
    for number in protocol["case"]["proposals"]:
        files = [r for r in protocol["source_files"] if r.get("proposal") == number]
        report, history, certs, worlds = proposal(input_dir, files, number)
        rows.append(report)
        histories[number], certificates[number], states[number] = history, certs, worlds
    initial_equal = value(states[1][0]) == value(states[2][0])
    final_equal = value(states[1][-1]) == value(states[2][-1])
    return {
        "schema": "synkit.mtg-mcsa155-case/1",
        "status": "PASS",
        "protocol_sha256": sha256(protocol_path.read_bytes()).hexdigest(),
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "adapter_sha256": sha256(
            (ROOT / "Experiment/MTG/mcsa_drawing.py").read_bytes()
        ).hexdigest(),
        "observation_connections": observations,
        "initial_worlds_equal": initial_equal,
        "final_anchored_worlds_equal": final_equal,
        "candidates": rows,
        "summary": {
            "proposals": len(rows),
            "native_closed_steps": sum(r["native_closed_steps"] for r in rows),
            "configuration_worlds_checked": sum(len(r["material_cuts"]) for r in rows),
            "observation_supported": [r["observation_supported"] for r in rows],
            "pdb_covalent_connections": len(observations),
        },
        "claim_limits": (
            "Exploratory depicted candidate pathways; qualitative intermediate connectivity only. "
            "No full-pathway truth, population performance or event timing."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "Experiment/MTG/mcsa155_protocol.json"
    )
    parser.add_argument("--input-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.protocol, args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
