#!/usr/bin/env python
"""Compare the PDB intermediate query with actual published overlay abstractions."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Experiment.MTG.overlay_history_comparison import (  # noqa: E402
    UPSTREAM_COMMIT,
    _history,
)
from Experiment.MTG.verified_cases import _unpack_certificate  # noqa: E402


def run(upstream, case_path):
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=upstream, text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError("Use the frozen published overlay checkout.")
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--"], cwd=upstream, check=True
    )
    sys.path.insert(0, str(upstream))
    import mod
    import networkx as nx
    from overlay_graphs.canonicalisation import GraphCanonicaliser
    from overlay_graphs.reaction_sequence_tracking import IsomorphismCache
    from overlay_graphs.mechanism import Mechanism, Step
    from overlay_graphs.make_overlay_graphs import compute_overlay_graphs
    from overlay_graphs.overlay_graph import OverlayGraph
    from overlay_graphs.substrate_rules import _make_substrate_rule

    api = GraphCanonicaliser, IsomorphismCache, Mechanism, Step, compute_overlay_graphs
    cases = json.loads(case_path.read_text())
    settings = mod.LabelSettings(mod.LabelType.String, mod.LabelRelation.Isomorphism)
    rows = []
    for case in cases["candidates"]:
        result = _history(
            mod,
            api,
            f"M-CSA155 proposal {case['proposal']}",
            [_unpack_certificate(c) for c in case["certificates"]],
        )
        anchors = case["observed_origin_atoms"]
        roots = result["origin_to_upstream_vertex"]
        carbon = roots[str(anchors["Asp169_carbonyl_carbon"])]
        sulfur = roots[str(anchors["CoA_sulfur"])]
        overlays = []
        for index, item in enumerate(result["overlays"]):
            overlay = OverlayGraph.deserialise(item["overlay"])
            marked = overlay.host_graph.has_edge(carbon, sulfur)
            if marked:
                # Presence initially or a received unit establishes this existential
                # bond query in the admitted supplied chemical projection domain.
                marked = (
                    overlay.host_graph.edges[carbon, sulfur]["label"] != "?"
                    or overlay.electrons_received((carbon, sulfur)) > 0
                )
            assert marked == case["observation_supported"]
            gml, left_catalysts, right_catalysts = _make_substrate_rule(
                overlay, f"MCSA155_P{case['proposal']}_{index}", settings, False
            )
            rule = mod.ruleGMLString(gml)
            retained = not rule.getVertexFromExternalId(carbon).isNull()
            overlays.append(
                {
                    "actual_overlay": item["overlay"],
                    "query_from_overlay": marked,
                    "product_matches_shared_projection": item[
                        "product_matches_shared_projection"
                    ],
                    "substrate_rule_gml": gml,
                    "observed_Asp_carbon_retained_in_substrate_rule": retained,
                    "left_catalyst_original_vertices": [
                        sorted(graph.nodes) for graph in left_catalysts.values()
                    ],
                    "right_catalyst_original_vertices": [
                        sorted(graph.nodes) for graph in right_catalysts.values()
                    ],
                    "substrate_rule_query_status": (
                        "outside_retained_catalyst_domain"
                        if not retained
                        else "requires_history_not_endpoint_rule"
                    ),
                }
            )
        rows.append(
            {
                "proposal": case["proposal"],
                "steps": case["native_closed_steps"],
                "mtg_query": case["observation_supported"],
                "overlays": overlays,
            }
        )
    return {
        "schema": "synkit.mtg-mcsa155-published-comparison/1",
        "status": "PASS",
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "case_sha256": sha256(case_path.read_bytes()).hexdigest(),
        "adapter_sha256": sha256(
            (ROOT / "Experiment/MTG/overlay_history_comparison.py").read_bytes()
        ).hexdigest(),
        "upstream_commit": commit,
        "upstream_module_sha256": {
            str(p.relative_to(upstream)): sha256(p.read_bytes()).hexdigest()
            for p in sorted((upstream / "overlay_graphs").glob("*.py"))
        },
        "environment": {
            "python": platform.python_version(),
            "mod": str(mod.version()),
            "networkx": nx.__version__,
        },
        "candidates": rows,
        "summary": {
            "proposals": len(rows),
            "supplied_steps": sum(row["steps"] for row in rows),
            "actual_overlays": sum(len(row["overlays"]) for row in rows),
            "overlay_query_answers": [
                [og["query_from_overlay"] for og in row["overlays"]] for row in rows
            ],
        },
        "claim_limits": (
            "Equal supplied projected histories and pinned original atoms. "
            "The full overlay answers the observation; substrate-rule catalyst removal "
            "is deliberate abstraction, not pipeline failure or MTG superiority."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument(
        "--case", type=Path, default=ROOT / "Experiment/MTG/evidence/mcsa155_case.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.upstream, args.case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
