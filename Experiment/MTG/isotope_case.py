#!/usr/bin/env python
"""Qualitative atom-origin witness from the Doering--Dorfman experiment.

The supplied candidate groups are author reconstructions. The independent
observation concerns the labelled oxygen's product role, not event timing.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.MTG.synepd_construction import _replay, _states  # noqa: E402
from Experiment.MTG.verified_cases import (  # noqa: E402
    _mtg_states,
    _pack_certificate,
    _process,
    _unpack_certificate,
)
from synkit.Graph.Morphism import (  # noqa: E402
    ELECTRON_LLG_SCHEMA,
    ISOTOPE_ELECTRON_LLG_SCHEMA,
    LewisLabelledGraph,
    derive_electron_labeled_graph,
)
from synkit.Graph.MTG import OccurrenceMTG  # noqa: E402
from synkit.Rule.Apply import RuleSpan, apply_dpo  # noqa: E402

DOI = "https://doi.org/10.1021/ja01118a035"
OBSERVATION = {
    "source": DOI,
    "substrate": "carbonyl-oxygen-18 benzophenone",
    "reagent": "perbenzoic acid",
    "solvent": "benzene (primary paper page 5597)",
    "product": "phenyl benzoate",
    "labelled_product_role": "carbonyl",
    "analytical_discrimination": "Reduction gives enriched benzyl alcohol and naturally abundant phenol.",
    "source_access": "Indexed primary abstract and page-5597 passages; full PDF could not be retrieved.",
    "scope": "Qualitative positional isotope observation only; no kinetics or quantitative enrichment fitted.",
}


def _aryl(start):
    return (
        f"[c:{start}]1"
        + "".join(f"[cH:{i}]" for i in range(start + 1, start + 6))
        + "1"
    )


def candidates():
    """Return a frozen, explicit pair with the same isotope-masked boundaries."""
    a, b, acid = _aryl(3), _aryl(9), _aryl(19)
    initial = f"[C:1](=[18O:2])({a}){b}.[C:15](=[O:16])([O:17][O:18][H:25]){acid}"
    carbonyl_product = (
        f"[C:1](=[18O:2])({a})[O:18]{b}.[C:15](=[O:16])([O:17][H:25]){acid}"
    )
    ether_product = f"[C:1](=[O:18])({a})[18O:2]{b}.[C:15](=[O:16])([O:17][H:25]){acid}"
    addition = [
        ["LP-/Sigma+", [18], [1, 18]],
        ["Pi-/LP+", [1, 2], [2]],
        ["LP-/Sigma+", [2], [2, 25]],
        ["Sigma-/LP+", [18, 25], [18]],
    ]
    rearrangement = [
        ["Sigma-/Sigma+", [1, 9], [18, 9]],
        ["Sigma-/LP+", [17, 18], [17]],
        ["LP-/Pi+", [2], [1, 2]],
        ["Sigma-/LP+", [2, 25], [2]],
        ["LP-/Sigma+", [17], [17, 25]],
    ]
    dioxirane = [
        ["LP-/Sigma+", [18], [1, 18]],
        ["Pi-/LP+", [1, 2], [2]],
        ["Sigma-/LP+", [17, 18], [17]],
        ["LP-/Sigma+", [2], [2, 18]],
        ["Sigma-/LP+", [18, 25], [18]],
        ["LP-/Sigma+", [17], [17, 25]],
    ]
    migration = [
        ["Sigma-/Sigma+", [1, 9], [2, 9]],
        ["Sigma-/Pi+", [2, 18], [1, 18]],
    ]
    return [
        {
            "name": "carbonyl-retaining supplied candidate",
            "rsmi": initial + ">>" + carbonyl_product,
            "epd": addition + rearrangement,
            "cuts": (len(addition),),
        },
        {
            "name": "ether-retaining supplied countercandidate",
            "rsmi": initial + ">>" + ether_product,
            "epd": dioxirane + migration,
            "cuts": (len(dioxirane),),
        },
    ]


def isotope_llg(graph):
    value = nx.relabel_nodes(
        graph, {n: a["atom_map"] for n, a in graph.nodes(data=True)}
    )
    schema = ISOTOPE_ELECTRON_LLG_SCHEMA
    for _, attrs in value.nodes(data=True):
        attrs.setdefault("isotope", 0)
        for key in set(attrs) - set(schema.declared_node_keys):
            del attrs[key]
    for _, _, attrs in value.edges(data=True):
        for key in set(attrs) - set(schema.declared_edge_keys):
            del attrs[key]
    return LewisLabelledGraph.from_networkx(
        derive_electron_labeled_graph(value), schema
    )


def oxygen_role(graph, oxygen=2, carbon=1):
    """Inspect an anchored oxygen's bonds without looking at the expected answer."""
    labels = graph.nodes[oxygen]
    if labels["element"] != "O" or labels["isotope"] != 18:
        return "missing labelled origin"
    neighbours = graph[oxygen]
    if (
        carbon in neighbours
        and neighbours[carbon]["sigma_order"] + neighbours[carbon]["pi_order"] == 2
    ):
        return "carbonyl"
    if (
        len(neighbours) == 2
        and all(e["sigma_order"] + e["pi_order"] == 1 for e in neighbours.values())
        and all(graph.nodes[n]["element"] == "C" for n in neighbours)
    ):
        return "ether"
    return "other"


def run():
    rows = []
    masked_boundaries = []
    for candidate in candidates():
        profile, replay = _replay(candidate, candidate["cuts"])
        if profile["status"] != "VALID":
            raise ValueError({"candidate": candidate["name"], "replay": profile})
        states = [isotope_llg(g) for g in _states(replay)]
        masked_boundaries.append(
            [
                LewisLabelledGraph.from_networkx(g.to_networkx(), ELECTRON_LLG_SCHEMA)
                for g in (states[0], states[-1])
            ]
        )
        rules, certificates = [], []
        for before, after in zip(states, states[1:]):
            rule = RuleSpan.from_mapping(
                before, after, {n: n for n in before.node_ids}, boundary="closed"
            )
            app = apply_dpo(rule, before, {n: n for n in before.node_ids})
            assert app.result.nodes == after.nodes and app.result.edges == after.edges
            assert app.certificate.replay().valid
            rules.append(rule)
            certificates.append(_pack_certificate(app.certificate))
        process = _process(states, rules)
        history = OccurrenceMTG.from_process(process)
        order = tuple(e.occurrence_id for e in process.events)
        decoded = _mtg_states(history, order)
        roles = [oxygen_role(g) for g in decoded]
        raw_roles = [oxygen_role(g.to_networkx()) for g in states]
        assert roles == raw_roles
        rows.append(
            {
                "name": candidate["name"],
                "mapped_reaction": candidate["rsmi"],
                "groups": candidate["cuts"],
                "arrow_count": len(candidate["epd"]),
                "execution_status": profile["status"],
                "certificates": certificates,
                "mtg_oxygen_roles": roles,
                "trajectory_oxygen_roles": raw_roles,
                "observation_compatible": roles[-1]
                == OBSERVATION["labelled_product_role"],
                "event_boundaries_physically_validated": False,
            }
        )
    packed = json.loads(json.dumps(rows))
    assert all(a.is_isomorphic(b) for a, b in zip(*masked_boundaries))
    assert all(
        _unpack_certificate(c).replay().valid
        for row in packed
        for c in row["certificates"]
    )
    return {
        "schema": "synkit.mtg-isotope-observation/1",
        "observation": OBSERVATION,
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "candidate_policy": "Frozen supplied two-candidate reconstruction, not blind mechanism prediction.",
        "cases": rows,
        "isotope_masked_boundaries_equal": True,
        "certificates_reloaded": 4,
        "scope": (
            "Positional observation distinguishes the supplied candidates; "
            "equal-information baselines should agree."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "certificates_reloaded": result["certificates_reloaded"],
                "compatible_candidates": sum(
                    row["observation_compatible"] for row in result["cases"]
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
