"""Isolated full-API DPO engines for the formose timing driver.

This worker imports only the selected engine. Rule format translation and the
cross-tool molecular identity audit run in the parent interpreter, so MØD and
MolFoundry environments need no SynKit installation or RDKit Python package.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import logging
from pathlib import Path
import sys
import time
import traceback


def emit(value):
    print(json.dumps(value), flush=True)


def _native_engine(inputs, depth, executable):
    from synkit import __version__
    from synkit.CRN import CRNExpand
    from synkit.CRN.Construct.native_dpo import native_dpo_expander

    version = __version__
    context = native_dpo_expander(executable)

    def expand():
        builder = CRNExpand(
            rules=inputs["rules"],
            repeats=depth,
            rule_backend="synreactor-native",
            native_dpo_expander=executable,
            dpo_radical_policy="ignore",
            dpo_component_policy="union",
            keep_aam=False,
            explicit_h=False,
            implicit_temp=True,
            hydrogen_model="lewis",
            threads=1,
            max_mixtures_per_rule_step=2**60,
            max_tasks_per_step=2**60,
        )
        return builder, builder.build(inputs["seeds"])

    def export(value):
        builder, graph = value
        if builder.dpo_report.get("status") != "complete":
            raise RuntimeError("Native controller did not complete the requested expansion")
        species = [d["smiles"] for _, d in graph.nodes(data=True) if d.get("kind") == "species"]
        reactions = []
        for node, data in graph.nodes(data=True):
            if data.get("kind") != "rule":
                continue
            left, right = [], []
            for source, _, edge in graph.in_edges(node, data=True):
                left.extend([graph.nodes[source]["smiles"]] * int(edge["stoich"]))
            for _, target, edge in graph.out_edges(node, data=True):
                right.extend([graph.nodes[target]["smiles"]] * int(edge["stoich"]))
            reactions.append((left, right))
        return {"species": species, "reactions": reactions, "dpo_report": builder.dpo_report}

    return version, context, expand, export


def _external_engine(backend, inputs, depth):
    if backend == "mod":
        import mod as api

        version = str(api.version())
    else:
        import molfoundry as api

        version = importlib.metadata.version("molfoundry")

    def expand():
        if backend == "mod":
            seeds = [api.Graph.fromSMILES(s, add=False) for s in inputs["seeds"]]
            rules = [api.Rule.fromGMLString(s, add=False) for s in inputs["gml_rules"]]
        else:
            seeds = [api.smiles(s, add=False) for s in inputs["seeds"]]
            rules = [api.ruleGMLString(s, add=False) for s in inputs["gml_rules"]]
        strategy = api.addSubset(seeds) >> api.repeat[depth](
            api.leftPredicate[lambda derivation: len({g.id for g in derivation.left}) == len(derivation.left)](rules)
        )
        graph = api.DG(graphDatabase=seeds)
        with graph.build() as builder:
            builder.execute(strategy, verbosity=0)
        return graph

    def export(graph):
        return {
            "species": [v.graph.smiles for v in graph.vertices],
            "reactions": [
                ([v.graph.smiles for v in e.sources], [v.graph.smiles for v in e.targets]) for e in graph.edges
            ],
        }

    return version, contextlib.nullcontext(), expand, export


def serve(backend, depth, inputs_path, executable=None):
    logging.disable(logging.INFO)
    depth = int(depth)
    inputs = json.loads(Path(inputs_path).read_text())
    if backend == "synreactor-native":
        version, context, expand, export = _native_engine(inputs, depth, executable)
    else:
        version, context, expand, export = _external_engine(backend, inputs, depth)

    with context:
        emit({"phase": "ready", "version": version, "python": sys.version})
        for line in sys.stdin:
            command = json.loads(line)
            if command["phase"] == "stop":
                return
            if command["phase"] not in {"warmup", "measured"}:
                raise ValueError("Unknown benchmark phase")
            with contextlib.redirect_stdout(sys.stderr):
                started = time.perf_counter()
                value = expand()
                elapsed = time.perf_counter() - started
            emit({"phase": "generation_complete", "seconds": elapsed})
            with contextlib.redirect_stdout(sys.stderr):
                exported = export(value)
            emit({"phase": "done", **exported})
            del value, exported


if __name__ == "__main__":
    # The selected interpreter can be outside the SynKit installation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    try:
        serve(*sys.argv[1:])
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        emit({"phase": "error", "error": f"{type(error).__name__}: {error}"})
        raise SystemExit(1)
