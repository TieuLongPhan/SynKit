"""Native witness retries by target-independent permutations of host order."""

import argparse
import gzip
import json
import logging
from pathlib import Path
import random
from time import perf_counter

import networkx as nx
from rdkit import RDLogger

from synkit.IO import smiles_to_graph
from synkit.Rule import WildcardRuleExtractor
from synkit.Synthesis.Reactor import WildcardEngine, build_native_reactor
from .benchmark import ROOT, ReplayTimeout, canonical_fragments, deadline, digest


def retry(directory, record_id):
    logging.disable(logging.CRITICAL)
    RDLogger.DisableLog("rdApp.*")
    with gzip.open(directory / "records.jsonl.gz", "rt") as stream:
        record = next(json.loads(line) for line in stream if json.loads(line)["r_id"] == record_id)
    rule = WildcardRuleExtractor(endpoint_policy="observed").extract(record["reaction"]).rule
    host = smiles_to_graph(record["runs"]["cpp/full/fw"]["input"],
                           use_index_as_atom_map=False, drop_non_aam=False)
    library = build_native_reactor(ROOT / "build/partial/native")
    target = canonical_fragments(record["reaction"].split(">>")[1])
    attempts = []
    for seed in range(1, 33):
        nodes = list(host)
        random.Random(seed).shuffle(nodes)
        shuffled = nx.Graph(**host.graph)
        shuffled.add_nodes_from((node, dict(host.nodes[node])) for node in nodes)
        shuffled.add_edges_from((u, v, dict(attrs)) for u, v, attrs in host.edges(data=True))
        started = perf_counter()
        attempt = {"seed": seed, "per_attempt_seconds": 10, "candidate_limit": 32768}
        try:
            with deadline(10):
                engine = WildcardEngine(shuffled, rule, partial=False, first_application=True,
                                        backend="cpp", library_path=library)
                outputs = engine.smarts_list
                outcomes = [canonical_fragments(value.split(">>")[1]) for value in outputs]
                attempt.update(status="complete", exact=target in outcomes,
                    search_complete=engine.native_statistics["search_complete"], outcomes=outcomes,
                    witness=next((value for value in outputs
                                  if canonical_fragments(value.split(">>")[1]) == target), None))
        except (ReplayTimeout, Exception) as error:
            attempt.update(status="error", error=str(error))
        attempt["seconds"] = perf_counter() - started
        attempts.append(attempt)
        if attempt.get("exact"):
            break
    result = {"schema": "synkit.native-observed-recovery-retry/1", "record": record_id,
        "primary_summary_sha256": digest(directory / "summary.json"),
        "script_sha256": digest(__file__),
        "method": "Reorder the same unmapped host vertices with fixed seeds; request one native constructible application. No rule edits, GT binding oracle, or Python execution fallback. GT is used only for scoring.",
        "primary_results_unchanged": True, "attempts": attempts}
    (directory / "targeted-retry.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--id", required=True)
    args = parser.parse_args()
    result = retry(args.directory, args.id)
    print(json.dumps({"record": result["record"], "attempts": len(result["attempts"]),
                      "recovered": any(a.get("exact") for a in result["attempts"])}))
