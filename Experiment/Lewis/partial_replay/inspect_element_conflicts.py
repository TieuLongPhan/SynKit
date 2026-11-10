"""Audit raw element/map conflicts and test clearly labelled diagnostic copies.

No dataset or primary benchmark result is modified. Element-compatible label
permutations establish consistency only, not experimentally correct atom fate.
"""

import argparse
from collections import Counter
import csv
import gzip
import hashlib
from itertools import permutations
import json
import logging
import multiprocessing as mp
from pathlib import Path
import random

import networkx as nx
from rdkit import Chem, RDLogger

from synkit.Graph.ITS.its_reverter import ITSReverter
from synkit.IO import rsmi_to_its, smiles_to_graph
from synkit.Rule import WildcardRuleExtractor, WildcardRuleExtractionError
from synkit.Synthesis.Reactor import WildcardEngine, build_native_reactor
from .benchmark import ROOT, ReplayTimeout, canonical_fragments, deadline, digest


def atoms(smiles, remove_hs):
    params = Chem.SmilesParserParams()
    params.removeHs = remove_hs
    molecule = Chem.MolFromSmiles(smiles, params)
    assert molecule is not None
    values = {a.GetAtomMapNum(): a for a in molecule.GetAtoms()}
    assert 0 not in values and len(values) == molecule.GetNumAtoms()
    return molecule, values


def material_signature(atom):
    return atom.GetAtomicNum(), atom.GetIsotope()


def inspect(record):
    reaction = record["reaction"]
    parsed = [atoms(side, False) for side in reaction.split(">>")]
    normalized = [atoms(side, True) for side in reaction.split(">>")]
    left, right = [value[1] for value in parsed]
    conflict_ids = sorted(n for n in left.keys() & right.keys()
                          if left[n].GetAtomicNum() != right[n].GetAtomicNum())
    assert conflict_ids
    normalized_conflicts = sorted(n for n in normalized[0][1].keys() & normalized[1][1].keys()
        if normalized[0][1][n].GetAtomicNum() != normalized[1][1][n].GetAtomicNum())
    assert conflict_ids == normalized_conflicts
    canonical = ">>".join(Chem.MolToSmiles(mol) for mol, _ in normalized)
    its = rsmi_to_its(canonical, format="tuple", drop_non_aam=False, use_index_as_atom_map=True)
    reverter = ITSReverter(its)
    for side, graph in enumerate((reverter.to_reactant_graph(), reverter.to_product_graph())):
        assert {a["atom_map"]: a["element"] for _, a in graph.nodes(data=True)} == {
            n: a.GetSymbol() for n, a in normalized[side][1].items()}
    try:
        WildcardRuleExtractor(endpoint_policy="observed").extract(reaction)
    except WildcardRuleExtractionError as error:
        assert error.code == "CHANGED_ELEMENT"
    else:
        raise AssertionError("Original conflict was not reproduced")
    conflicts = [{"map": n, "left": left[n].GetSymbol(), "right": right[n].GetSymbol(),
        "left_atom": left[n].GetSmarts(), "right_atom": right[n].GetSmarts(),
        "left_degree": left[n].GetDegree(), "right_degree": right[n].GetDegree(),
        "compatible_left_only_maps": [m for m in sorted(left.keys() - right.keys())
                                     if material_signature(left[m]) == material_signature(right[n])]}
        for n in conflict_ids]
    first = Counter(c["left"] for c in conflicts)
    second = Counter(c["right"] for c in conflicts)
    inventories = [Counter(a.GetSymbol() for a in side.values()) for side in (left, right)]
    plans = []
    if first == second:
        for ids in permutations(conflict_ids):
            if all(material_signature(right[old]) == material_signature(left[new])
                   for old, new in zip(conflict_ids, ids)):
                plans.append({"kind": "element_consistent_permutation",
                              "product_map_changes": dict(zip(conflict_ids, ids))})
    else:
        fresh = max(left.keys() | right.keys()) + 1
        plans.append({"kind": "identity_separation_only",
                      "product_map_changes": {n: fresh+i for i,n in enumerate(conflict_ids)}})
    for plan in plans:
        product = Chem.Mol(parsed[1][0])
        for atom in product.GetAtoms():
            atom.SetAtomMapNum(plan["product_map_changes"].get(atom.GetAtomMapNum(), atom.GetAtomMapNum()))
        candidate = reaction.split(">>")[0] + ">>" + Chem.MolToSmiles(product)
        assert [canonical_fragments(s) for s in candidate.split(">>")] == [
            canonical_fragments(s) for s in reaction.split(">>")]
        for atom in product.GetAtoms():
            n = atom.GetAtomMapNum()
            assert n not in left or material_signature(atom) == material_signature(left[n])
        plan["diagnostic_reaction"] = candidate
        plan["unmapped_molecules_unchanged"] = True
    return {"r_id": record["r_id"], "source_row": record["row"], "reaction": reaction,
            "conflicts": conflicts, "raw_conflicts": len(conflict_ids),
            "normalization_introduced_conflicts": 0, "conversion_element_labels_preserved": True,
            "class": "balanced_element_swaps" if first == second else "unbalanced_identity_conflict",
            "product_element_deficit": dict(inventories[1] - inventories[0]),
            "source_element_excess": dict(inventories[0] - inventories[1]),
            "restricted_permutation_count": len(plans) if first == second else 0,
            "plans": plans}


def replay(task):
    index, plan_index, reaction, library = task
    logging.disable(logging.CRITICAL)
    RDLogger.DisableLog("rdApp.*")
    result = {"index": index, "plan": plan_index, "runs": {}}
    try:
        rule = WildcardRuleExtractor(endpoint_policy="observed").extract(reaction).rule
        sides = [canonical_fragments(side) for side in reaction.split(">>")]
        for invert in (False, True):
            run = {"status": "not_recovered", "exact": False}
            try:
                with deadline(15):
                    host = smiles_to_graph(".".join(sides[int(invert)]),
                                           use_index_as_atom_map=False, drop_non_aam=False)
                    for seed in range(17):
                        nodes = list(host)
                        if seed:
                            random.Random(seed).shuffle(nodes)
                        ordered = nx.Graph(**host.graph)
                        ordered.add_nodes_from((n, dict(host.nodes[n])) for n in nodes)
                        ordered.add_edges_from((u,v,dict(a)) for u,v,a in host.edges(data=True))
                        engine = WildcardEngine(ordered, rule, backend="cpp", library_path=library,
                            partial=False, first_application=True, invert=invert)
                        outputs = engine.smarts_list
                        witness = next((s for s in outputs if canonical_fragments(
                            s.split(">>")[0 if invert else 1]) == sides[1-int(invert)]), None)
                        if witness:
                            run.update(status="recovered", exact=True, witness=witness,
                                       host_order_seed=seed, search_complete=False)
                            break
                    if not run["exact"]:
                        engine = WildcardEngine(".".join(sides[int(invert)]), rule, backend="cpp",
                            library_path=library, partial=False, invert=invert, embed_threshold=32768)
                        witness = next((s for s in engine.smarts_list if canonical_fragments(
                            s.split(">>")[0 if invert else 1]) == sides[1-int(invert)]), None)
                        run.update(exact=witness is not None, witness=witness, search_complete=True,
                                   status="recovered" if witness else "not_recovered")
            except (ReplayTimeout, Exception) as error:
                run.update(status="error", error_type=type(error).__name__, error=str(error))
            result["runs"]["bw" if invert else "fw"] = run
    except Exception as error:
        result.update(extraction_error=str(error))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    logging.disable(logging.CRITICAL)
    RDLogger.DisableLog("rdApp.*")
    manifest = json.loads((args.run / "run-manifest.json").read_text())
    assert digest(manifest["dataset"]) == manifest["dataset_sha256"]
    primary_hash = digest(args.run / "records.jsonl.gz")
    with gzip.open(args.run / "records.jsonl.gz", "rt") as stream:
        originals = [json.loads(line) for line in stream if '"code": "CHANGED_ELEMENT"' in line]
    assert len(originals) == 89
    by_id = {r["r_id"]:r for r in originals}
    with gzip.open(manifest["dataset"], "rt") as stream:
        verified = {r["r_id"] for r in csv.DictReader(stream)
                    if r["r_id"] in by_id and r["aam"] == by_id[r["r_id"]]["reaction"]}
    assert verified == set(by_id)
    records = [inspect(record) for record in originals]
    library = str(build_native_reactor(ROOT / "build/partial/native"))
    tasks = [(i,j,plan["diagnostic_reaction"],library) for i,record in enumerate(records)
             for j,plan in enumerate(record["plans"])]
    with mp.Pool(8) as pool:
        for result in pool.imap_unordered(replay, tasks):
            records[result["index"]]["plans"][result["plan"]]["native_replay"] = result
    assert digest(manifest["dataset"]) == manifest["dataset_sha256"]
    assert digest(args.run / "records.jsonl.gz") == primary_hash
    transitions = Counter(c["left"]+"->"+c["right"] for r in records for c in r["conflicts"])
    summary = {"schema": "synkit.element-map-conflict-inspection/1", "rows": len(records),
        "classes": dict(Counter(r["class"] for r in records)), "transitions": dict(transitions),
        "conflicts_per_record": dict(Counter(r["raw_conflicts"] for r in records)),
        "raw_conflicts_present": len(records), "normalization_introduced_cases": 0,
        "conversion_preserves_element_labels": True, "source_and_primary_results_unchanged": True,
        "diagnostic_replay": {direction:sum(any(plan["native_replay"].get("runs",{}).get(direction,{}).get("exact")
                                for plan in record["plans"]) for record in records) for direction in ("fw","bw")},
        "interpretation": "Map-only diagnostic copies preserve both observed molecular structures. Element consistency and replay do not validate chemical atom provenance. Identity separation for the two single conflicts is not a validated remapping or a completion of missing reactants. Original benchmark results remain unchanged.",
        "dataset_sha256": manifest["dataset_sha256"], "primary_records_sha256": primary_hash,
        "script_sha256": digest(__file__), "native_library_sha256": digest(library)}
    (args.output / "analysis.json").write_text(json.dumps({**summary,"records":records}, indent=2)+"\n")
    (args.output / "summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    with (args.output / "cases.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(("r_id","class","conflicting_maps","transitions","product_element_deficit",
                         "restricted_permutations","diagnostic_fw","diagnostic_bw","original_reaction"))
        for record in records:
            writer.writerow((record["r_id"],record["class"],json.dumps([c["map"]for c in record["conflicts"]]),
                json.dumps([c["left"]+"->"+c["right"]for c in record["conflicts"]]),
                json.dumps(record["product_element_deficit"]),record["restricted_permutation_count"],
                any(p["native_replay"].get("runs",{}).get("fw",{}).get("exact")for p in record["plans"]),
                any(p["native_replay"].get("runs",{}).get("bw",{}).get("exact")for p in record["plans"]),
                record["reaction"]))
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
