from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from synkit.CRN.Construct import CRNExpand
from synkit.CRN.Pathway import PathwayRealizability, run_pathfinder_from_syncrn
from synkit.CRN.Petrinet.analyzer import PetriAnalyzer
from synkit.CRN.Props.stoich import build_S, integer_conservation_laws, summary
from synkit.CRN.Props.thermo import compute_thermo_summary
from synkit.CRN.Structure import SynCRN
from synkit.CRN.Symmetry import CRNCanonicalizer
from synkit.Chem.utils import reverse_reaction
from synkit.IO import gml_to_smart

# =============================================================================
# Output / logging
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
OUTDIR = BASE_DIR / "case_formose"
OUTDIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUTDIR / "formose.log"
JSON_DIR = OUTDIR / "json"
TXT_DIR = OUTDIR / "txt"
JSON_DIR.mkdir(exist_ok=True)
TXT_DIR.mkdir(exist_ok=True)


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("formose")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


logger = setup_logger(LOG_PATH)


# =============================================================================
# Generic helpers
# =============================================================================


def log_header(title: str) -> None:
    line = "=" * 88
    logger.info("\n%s\n%s\n%s", line, title, line)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, set):
        return sorted(to_jsonable(x) for x in obj)
    if isinstance(obj, Path):
        return str(obj)

    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:
            pass

    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        try:
            return to_jsonable(obj.tolist())
        except Exception:
            pass

    return obj


def save_text(filename: str, text: str) -> None:
    path = TXT_DIR / filename
    path.write_text(text, encoding="utf-8")
    logger.info("Saved text: %s", path)


def save_json(filename: str, payload: Any) -> None:
    path = JSON_DIR / filename
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved JSON: %s", path)


def canon_smiles(smi: str, *, strict: bool = False) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        if strict:
            raise ValueError(f"Invalid SMILES: {smi}")
        return smi
    return Chem.MolToSmiles(mol)


def canonicalize_marking(
    marking: Dict[str, int],
    *,
    strict: bool = False,
) -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for smi, coeff in marking.items():
        csmi = canon_smiles(smi, strict=strict)
        merged[csmi] = merged.get(csmi, 0) + int(coeff)
    return merged


def atom_stats(smiles: str) -> Dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "formula": None,
            "nC": None,
            "nO": None,
            "heavy_atoms": None,
        }
    return {
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "nC": sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C"),
        "nO": sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "O"),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
    }


def carbon_count_from_smiles(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")


def total_carbon(marking: Dict[str, int]) -> int:
    return sum(carbon_count_from_smiles(smi) * coeff for smi, coeff in marking.items())


def count_semiflow_basis(x: Any) -> int:
    if x is None:
        return 0

    shape = getattr(x, "shape", None)
    if shape is not None:
        if len(shape) == 2:
            return int(shape[1])
        if len(shape) == 1:
            return 0 if int(shape[0]) == 0 else 1

    try:
        return len(x)
    except TypeError:
        return 0


def law_to_label_dict(
    species_labels: Sequence[str], law: Sequence[Any]
) -> Dict[str, Any]:
    return {
        label: to_jsonable(coeff) for label, coeff in zip(species_labels, list(law))
    }


# =============================================================================
# User input
# =============================================================================

KETO_ENOL_GML = """rule [
    ruleID "Keto-enol isomerization"
    left [
        edge [ source 1 target 4 label "-" ]
        edge [ source 1 target 2 label "-" ]
        edge [ source 2 target 3 label "=" ]
    ]
    context [
        node [ id 1 label "C" ]
        node [ id 2 label "C" ]
        node [ id 3 label "O" ]
        node [ id 4 label "H" ]
    ]
    right [
        edge [ source 1 target 2 label "=" ]
        edge [ source 2 target 3 label "-" ]
        edge [ source 3 target 4 label "-" ]
    ]
]"""

ALDOL_ADD_GML = """rule [
    ruleID "Aldol Addition"
    left [
        edge [ source 1 target 2 label "=" ]
        edge [ source 2 target 3 label "-" ]
        edge [ source 3 target 4 label "-" ]
        edge [ source 5 target 6 label "=" ]
    ]
    context [
        node [ id 1 label "C" ]
        node [ id 2 label "C" ]
        node [ id 3 label "O" ]
        node [ id 4 label "H" ]
        node [ id 5 label "O" ]
        node [ id 6 label "C" ]
    ]
    right [
        edge [ source 1 target 2 label "-" ]
        edge [ source 2 target 3 label "=" ]
        edge [ source 5 target 6 label "-" ]
        edge [ source 4 target 5 label "-" ]
        edge [ source 6 target 1 label "-" ]
    ]
]"""

RAW_SEEDS = ["C=O", "OCC=O"]

RAW_TARGETS = {
    "C4_aldose": "O=CC(O)C(O)CO",
    "C4_enediol": "OC=C(O)C(O)CO",
    "C5_aldose": "O=CC(O)C(O)C(O)CO",
    "C6_aldose": "O=CC(O)C(O)C(O)C(O)CO",
}

RAW_INITIAL_POOLS = {
    "minimal": {"C=O": 1, "O=CCO": 1},
    "C4_exact": {"C=O": 2, "O=CCO": 1},
    "C5_exact": {"C=O": 3, "O=CCO": 1},
    "C6_exact": {"C=O": 4, "O=CCO": 1},
    "C6_alt": {"C=O": 2, "O=CCO": 2},
}

REPEATS_SCAN = [1, 2, 3, 4]
ANALYSIS_REPEAT = 4
REPEAT_TAG = f"repeat{ANALYSIS_REPEAT}"

r1 = gml_to_smart(KETO_ENOL_GML)
r2 = gml_to_smart(ALDOL_ADD_GML)
r3 = reverse_reaction(r1)
r4 = reverse_reaction(r2)

RULES = [r1, r2, r3, r4]
RULE_NAMES = {
    0: "keto_enol",
    1: "aldol_add",
    2: "retro_keto_enol",
    3: "retro_aldol",
}

SEEDS = [canon_smiles(s, strict=True) for s in RAW_SEEDS]
TARGETS = {name: canon_smiles(smi, strict=True) for name, smi in RAW_TARGETS.items()}
INITIAL_POOLS = {
    name: canonicalize_marking(pool, strict=True)
    for name, pool in RAW_INITIAL_POOLS.items()
}


# =============================================================================
# CRN helpers
# =============================================================================


def build_formose_crn(repeats: int) -> Tuple[CRNExpand, Any, SynCRN]:
    dg = CRNExpand(
        rules=RULES,
        repeats=repeats,
        explicit_h=False,
        implicit_temp=False,
        keep_aam=False,
        use_frontier=True,
        dedup_delta=True,
    )
    g = dg.build(seeds=SEEDS, parallel=False)
    syn = SynCRN.from_digraph(g)
    return dg, g, syn


def derivation_rows(dg: CRNExpand) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in getattr(dg, "derivation_records", []):
        payload = dict(row)
        payload["rule_name"] = RULE_NAMES.get(
            payload.get("rule_index"), payload.get("rule_index")
        )
        if "reactants" in payload:
            payload["reactants"] = [canon_smiles(x) for x in payload["reactants"]]
        if "products" in payload:
            payload["products"] = [canon_smiles(x) for x in payload["products"]]
        rows.append(payload)
    return rows


def rule_usage_distribution(
    deriv_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    counts = Counter(row.get("rule_name") for row in deriv_rows)
    out = [{"rule_name": k, "count": v} for k, v in counts.items()]
    out.sort(key=lambda x: (-x["count"], str(x["rule_name"])))
    return out


def species_table(syn: SynCRN) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sid in syn.species_ids:
        sp = syn.species[sid]
        label = getattr(sp, "label", None)
        smiles = getattr(sp, "smiles", None) or label
        smiles = canon_smiles(str(smiles)) if smiles is not None else None
        stats = (
            atom_stats(smiles) if isinstance(smiles, str) else atom_stats(str(label))
        )

        rows.append(
            {
                "sid": sid,
                "label": label,
                "smiles": smiles,
                "source_id": getattr(sp, "source_id", None),
                "formula": stats["formula"],
                "nC": stats["nC"],
                "nO": stats["nO"],
                "heavy_atoms": stats["heavy_atoms"],
            }
        )

    rows.sort(
        key=lambda x: (x["nC"] is None, x["nC"], x["formula"] or "", x["smiles"] or "")
    )
    return rows


def formula_distribution(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts: Dict[Tuple[Any, Any], int] = {}
    for row in rows:
        key = (row.get("nC"), row.get("formula"))
        counts[key] = counts.get(key, 0) + 1

    out = [{"nC": k[0], "formula": k[1], "n_species": v} for k, v in counts.items()]
    out.sort(
        key=lambda x: (x["nC"] is None, x["nC"], -(x["n_species"]), x["formula"] or "")
    )
    return out


def carbon_distribution(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts: Dict[Any, int] = {}
    for row in rows:
        key = row.get("nC")
        counts[key] = counts.get(key, 0) + 1

    out = [{"nC": k, "n_species": v} for k, v in counts.items()]
    out.sort(key=lambda x: (x["nC"] is None, x["nC"]))
    return out


def select_targets(
    species_rows: Sequence[Dict[str, Any]],
    *,
    formula: Optional[str] = None,
    nC: Optional[int] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in species_rows:
        if formula is not None and row.get("formula") != formula:
            continue
        if nC is not None and row.get("nC") != nC:
            continue
        out.append(dict(row))

    out.sort(key=lambda x: (x["formula"] or "", x["smiles"] or ""))
    return out


def find_species_row_by_smiles(
    species_rows: Sequence[Dict[str, Any]],
    target_smiles: str,
) -> Optional[Dict[str, Any]]:
    target_smiles = canon_smiles(target_smiles)
    for row in species_rows:
        row_smiles = row.get("smiles")
        if row_smiles is not None and canon_smiles(row_smiles) == target_smiles:
            return dict(row)
    return None


def target_presence_summary(
    species_rows: Sequence[Dict[str, Any]],
    targets: Dict[str, str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, smi in targets.items():
        hit = find_species_row_by_smiles(species_rows, smi)
        out[name] = {
            "target_smiles": smi,
            "present": hit is not None,
            "row": hit,
        }
    return out


def reachability_summary(species_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    c4 = select_targets(species_rows, nC=4)
    c6 = select_targets(species_rows, formula="C6H12O6")
    return {
        "has_C4": bool(c4),
        "n_C4": len(c4),
        "has_C6H12O6": bool(c6),
        "n_C6H12O6": len(c6),
        "C4_candidates": c4,
        "C6H12O6_candidates": c6,
    }


def rid_to_rule_repr_map(syn: SynCRN) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for rid in syn.reaction_ids:
        rxn = syn.reactions[rid]
        out[rid] = getattr(rxn, "rule_repr", None)
    return out


def rid_to_equation_map(
    syn: SynCRN,
    *,
    species: str = "label",
    include_rule: bool = False,
) -> Dict[str, str]:
    return {
        rid: syn.format_reaction(
            rid,
            species=species,
            include_id=False,
            include_rule=include_rule,
        )
        for rid in syn.reaction_ids
    }


def map_species_ids_to_labels(syn: SynCRN, ids: Sequence[Any]) -> List[str]:
    out: List[str] = []
    for sid in ids:
        if sid in syn.species:
            out.append(getattr(syn.species[sid], "label", str(sid)))
            continue

        sid_str = str(sid)
        if sid_str in syn.species:
            out.append(getattr(syn.species[sid_str], "label", sid_str))
            continue

        found = False
        for sp in syn.species.values():
            if getattr(sp, "label", None) == sid:
                out.append(getattr(sp, "label", str(sid)))
                found = True
                break
        if found:
            continue

        out.append(str(sid))

    return out


def species_lookup_table(syn: SynCRN) -> List[Dict[str, Any]]:
    rows = []
    for sid in syn.species_ids:
        sp = syn.species[sid]
        rows.append(
            {
                "sid": sid,
                "label": getattr(sp, "label", None),
                "smiles": canon_smiles(
                    getattr(sp, "smiles", None) or getattr(sp, "label", "")
                ),
                "source_id": getattr(sp, "source_id", None),
            }
        )
    return rows


def candidate_to_dict(cand: Any) -> Dict[str, Any]:
    return {
        "depth": int(cand.depth),
        "reactions": list(cand.reactions),
        "flow": dict(cand.flow),
        "reached_species": list(cand.reached_species),
        "realizable": bool(cand.realizable),
        "certificate": None if cand.certificate is None else list(cand.certificate),
    }


def first_realizable_certificate(cands: Sequence[Any]) -> Optional[List[str]]:
    for cand in cands:
        if getattr(cand, "realizable", False) and getattr(cand, "certificate", None):
            return list(cand.certificate)
    return None


def build_subnetwork_from_certificate(
    syn: SynCRN,
    certificate: Sequence[str],
    *,
    species: str = "label",
) -> SynCRN:
    rid2eq = rid_to_equation_map(syn, species=species, include_rule=False)
    rid2rule = rid_to_rule_repr_map(syn)
    sub_rxns = [rid2eq[rid] for rid in certificate]
    sub_rules = [rid2rule[rid] for rid in certificate]
    return SynCRN.from_reaction_strings(sub_rxns, rules=sub_rules)


def reaction_trace_rows(syn: SynCRN, rid: str) -> Dict[str, Any]:
    rxn = syn.reactions[rid]

    lhs = []
    for sid, coeff in rxn.lhs.items():
        sp = syn.species[sid]
        lhs.append(
            {
                "sid": sid,
                "label": getattr(sp, "label", None),
                "coeff": coeff,
                "smiles": canon_smiles(
                    getattr(sp, "smiles", None) or getattr(sp, "label", "")
                ),
            }
        )

    rhs = []
    for sid, coeff in rxn.rhs.items():
        sp = syn.species[sid]
        rhs.append(
            {
                "sid": sid,
                "label": getattr(sp, "label", None),
                "coeff": coeff,
                "smiles": canon_smiles(
                    getattr(sp, "smiles", None) or getattr(sp, "label", "")
                ),
            }
        )

    return {
        "reaction_id": rid,
        "rule_repr": getattr(syn.reactions[rid], "rule_repr", None),
        "equation_label": syn.format_reaction(
            rid, species="label", include_id=True, include_rule=True
        ),
        "equation_smiles": syn.format_reaction(
            rid, species="smiles", include_id=True, include_rule=True
        ),
        "lhs": lhs,
        "rhs": rhs,
    }


def certificate_trace(syn: SynCRN, certificate: Sequence[str]) -> List[Dict[str, Any]]:
    return [reaction_trace_rows(syn, rid) for rid in certificate]


def dump_certificate_trace_text(
    title: str, trace_rows: Sequence[Dict[str, Any]]
) -> str:
    lines: List[str] = [title, "=" * len(title)]

    for row in trace_rows:
        lines.append("")
        lines.append(f"reaction_id: {row['reaction_id']}")
        lines.append(f"rule_repr  : {row['rule_repr']}")
        lines.append(f"label      : {row['equation_label']}")
        lines.append(f"smiles     : {row['equation_smiles']}")
        lines.append("lhs:")
        for item in row["lhs"]:
            lines.append(
                f"  - coeff={item['coeff']} label={item['label']} smiles={item['smiles']}"
            )
        lines.append("rhs:")
        for item in row["rhs"]:
            lines.append(
                f"  - coeff={item['coeff']} label={item['label']} smiles={item['smiles']}"
            )

    return "\n".join(lines)


def pretty_candidate_block(title: str, cands: Sequence[Any], syn: SynCRN) -> str:
    rid2rule = rid_to_rule_repr_map(syn)
    rid2eq = rid_to_equation_map(syn, species="label", include_rule=False)

    lines: List[str] = [title, "=" * len(title)]
    if not cands:
        lines.append("No candidates found.")
        return "\n".join(lines)

    for i, cand in enumerate(cands, start=1):
        lines.append("")
        lines.append(f"Candidate {i}")
        lines.append(f"depth: {cand.depth}")
        lines.append(f"reactions: {list(cand.reactions)}")
        lines.append(f"rules: {[rid2rule.get(rid) for rid in cand.reactions]}")
        lines.append(f"equations: {[rid2eq.get(rid) for rid in cand.reactions]}")
        lines.append(f"flow: {dict(cand.flow)}")
        lines.append(f"reached_species: {list(cand.reached_species)}")
        lines.append(f"realizable: {bool(cand.realizable)}")
        lines.append(
            f"certificate: {None if cand.certificate is None else list(cand.certificate)}"
        )

    return "\n".join(lines)


def petri_json_payload(an: PetriAnalyzer, syn: SynCRN) -> Dict[str, Any]:
    if hasattr(an, "as_dict"):
        payload = an.as_dict()
    else:
        summary_obj = getattr(an, "summary", None)
        payload = {
            "place_order": list(getattr(summary_obj, "place_order", [])),
            "transition_order": list(getattr(summary_obj, "transition_order", [])),
            "persistence_ok": bool(getattr(summary_obj, "persistence_ok", False)),
            "siphons": [sorted(list(x)) for x in getattr(summary_obj, "siphons", [])],
            "traps": [sorted(list(x)) for x in getattr(summary_obj, "traps", [])],
            "p_semiflows": getattr(summary_obj, "p_semiflows", []),
            "t_semiflows": getattr(summary_obj, "t_semiflows", []),
            "persistence_details": getattr(an, "persistence_details", {}),
        }

    p_semiflows = payload.get("p_semiflows", [])
    t_semiflows = payload.get("t_semiflows", [])
    place_order = list(payload.get("place_order", []))

    return {
        "place_order": place_order,
        "place_order_labels": map_species_ids_to_labels(syn, place_order),
        "transition_order": list(payload.get("transition_order", [])),
        "persistence_ok": bool(payload.get("persistence_ok", False)),
        "siphons": payload.get("siphons", []),
        "traps": payload.get("traps", []),
        "n_p_semiflows": count_semiflow_basis(p_semiflows),
        "n_t_semiflows": count_semiflow_basis(t_semiflows),
        "persistence_details": payload.get("persistence_details", {}),
    }


def pool_vs_target_payload(target_smiles: str, init: Dict[str, int]) -> Dict[str, Any]:
    return {
        "target_smiles": target_smiles,
        "target_formula": atom_stats(target_smiles)["formula"],
        "target_carbon": carbon_count_from_smiles(target_smiles),
        "initial_marking": dict(init),
        "initial_total_carbon": total_carbon(init),
        "carbon_sufficient": total_carbon(init)
        >= carbon_count_from_smiles(target_smiles),
    }


def run_target_search(
    syn: SynCRN,
    *,
    target_name: str,
    target_smiles: str,
    init_name: str,
    init: Dict[str, int],
    max_depth: int,
    max_paths: int,
    validate: bool,
) -> Tuple[Sequence[Any], Dict[str, Any]]:
    payload = pool_vs_target_payload(target_smiles, init)
    logger.info(
        "target=%s smiles=%s init=%s target_carbon=%s pool_carbon=%s carbon_sufficient=%s",
        target_name,
        target_smiles,
        init_name,
        payload["target_carbon"],
        payload["initial_total_carbon"],
        payload["carbon_sufficient"],
    )

    cands = run_pathfinder_from_syncrn(
        syn,
        source_species=sorted(init.keys()),
        target_species=[target_smiles],
        initial_marking=init,
        species="label",
        reaction="id",
        max_depth=max_depth,
        max_paths=max_paths,
        validate=validate,
        verbose=True,
    )
    return cands, payload


def total_stoich(marking: Dict[str, int]) -> int:
    return sum(int(v) for v in marking.values())


def nonzero_marking(marking: Dict[str, int]) -> Dict[str, int]:
    return {k: int(v) for k, v in marking.items() if int(v) > 0}


def candidate_flow_dict(cand: Any) -> Dict[str, int]:
    """
    Preserve the candidate firing counts if available.
    Falls back to certificate counts, then to unit counts on listed reactions.
    """
    flow = dict(getattr(cand, "flow", {}) or {})
    if flow:
        return {str(rid): int(v) for rid, v in flow.items() if int(v) > 0}

    cert = getattr(cand, "certificate", None)
    if cert:
        return {str(rid): int(v) for rid, v in Counter(cert).items() if int(v) > 0}

    rxns = list(getattr(cand, "reactions", []) or [])
    return {str(rid): 1 for rid in rxns}


def default_seed_pool_bounds(
    target_smiles: str,
    seed_order: Sequence[str],
) -> Dict[str, int]:
    """
    Conservative bounded search range for initial pools.
    For each seed, allow a little slack above the carbon-minimal requirement.
    """
    target_c = carbon_count_from_smiles(target_smiles)
    bounds: Dict[str, int] = {}

    for seed in seed_order:
        seed_c = carbon_count_from_smiles(seed)
        if seed_c <= 0:
            bounds[seed] = 0
        elif seed_c == 1:
            bounds[seed] = target_c + 2
        else:
            bounds[seed] = target_c // seed_c + 2

    return bounds


def enumerate_seed_pools(
    seed_order: Sequence[str],
    max_coeff_by_seed: Dict[str, int],
    *,
    min_total_tokens: int = 1,
    min_total_carbon: Optional[int] = None,
) -> List[Dict[str, int]]:
    """
    Enumerate bounded initial pools over the allowed seeds.
    Pools are sorted so smaller stoichiometric support is tested first.
    """
    out: List[Dict[str, int]] = []

    def rec(i: int, cur: Dict[str, int]) -> None:
        if i == len(seed_order):
            m = nonzero_marking(cur)
            if total_stoich(m) < min_total_tokens:
                return
            if min_total_carbon is not None and total_carbon(m) < min_total_carbon:
                return
            out.append(dict(m))
            return

        seed = seed_order[i]
        max_coeff = int(max_coeff_by_seed.get(seed, 0))
        for coeff in range(max_coeff + 1):
            cur[seed] = coeff
            rec(i + 1, cur)
        cur.pop(seed, None)

    rec(0, {})
    out.sort(
        key=lambda m: (
            total_stoich(m),
            total_carbon(m),
            tuple(int(m.get(seed, 0)) for seed in seed_order),
        )
    )
    return out


def candidate_rank_key(
    cand: Any,
    init: Dict[str, int],
    seed_order: Sequence[str],
) -> Tuple[Any, ...]:
    """
    Lexicographic ranking:
    1. shortest realizable route
    2. smallest total stoichiometric initial pool
    3. smallest initial carbon inventory
    4. lexicographically smallest seed counts
    5. smallest total reaction firing count
    """
    flow = candidate_flow_dict(cand)
    return (
        int(cand.depth),
        total_stoich(init),
        total_carbon(init),
        tuple(int(init.get(seed, 0)) for seed in seed_order),
        sum(int(v) for v in flow.values()),
        tuple(str(rid) for rid in list(getattr(cand, "reactions", []) or [])),
    )


def find_shortest_realizable_min_init(
    syn: SynCRN,
    *,
    target_name: str,
    target_smiles: str,
    seed_order: Sequence[str],
    max_coeff_by_seed: Dict[str, int],
    max_depth: int,
    max_paths: int = 100,
    top_k: int = 20,
) -> Dict[str, Any]:
    """
    Search over bounded initial pools and return the best realizable hit,
    ranked by shortest route first and then by smallest initial stoichiometry.
    """
    rid2rule = rid_to_rule_repr_map(syn)
    rid2eq = rid_to_equation_map(syn, species="label", include_rule=False)

    pools = enumerate_seed_pools(
        seed_order,
        max_coeff_by_seed,
        min_total_tokens=1,
        min_total_carbon=carbon_count_from_smiles(target_smiles),
    )

    all_hits: List[Dict[str, Any]] = []

    for i, init in enumerate(pools, start=1):
        init_name = f"{target_name}_poolscan_{i}"

        cands, meta = run_target_search(
            syn,
            target_name=target_name,
            target_smiles=target_smiles,
            init_name=init_name,
            init=init,
            max_depth=max_depth,
            max_paths=max_paths,
            validate=True,
        )

        realizable_cands = [
            cand for cand in cands if bool(getattr(cand, "realizable", False))
        ]

        for cand in realizable_cands:
            reactions = list(getattr(cand, "reactions", []) or [])
            flow = candidate_flow_dict(cand)
            rank_key = candidate_rank_key(cand, init, seed_order)

            all_hits.append(
                {
                    "rank_key": list(rank_key),
                    "target_name": target_name,
                    "target_smiles": target_smiles,
                    "initial_marking": dict(init),
                    "initial_total_stoich": total_stoich(init),
                    "initial_total_carbon": total_carbon(init),
                    "depth": int(cand.depth),
                    "reactions": reactions,
                    "rules": [rid2rule.get(rid) for rid in reactions],
                    "equations": [rid2eq.get(rid) for rid in reactions],
                    "flow": flow,
                    "certificate": (
                        None if cand.certificate is None else list(cand.certificate)
                    ),
                    "meta": meta,
                }
            )

    all_hits.sort(key=lambda row: tuple(row["rank_key"]))

    return {
        "target_name": target_name,
        "target_smiles": target_smiles,
        "seed_order": list(seed_order),
        "max_coeff_by_seed": dict(max_coeff_by_seed),
        "searched_pools": len(pools),
        "n_realizable_hits": len(all_hits),
        "best": None if not all_hits else all_hits[0],
        "top_hits": all_hits[:top_k],
    }


def dump_shortest_realizable_min_init_text(
    title: str,
    result: Dict[str, Any],
) -> str:
    lines: List[str] = [title, "=" * len(title)]
    lines.append(f"target_name: {result['target_name']}")
    lines.append(f"target_smiles: {result['target_smiles']}")
    lines.append(f"seed_order: {result['seed_order']}")
    lines.append(f"max_coeff_by_seed: {result['max_coeff_by_seed']}")
    lines.append(f"searched_pools: {result['searched_pools']}")
    lines.append(f"n_realizable_hits: {result['n_realizable_hits']}")

    best = result.get("best")
    if best is None:
        lines.append("No realizable route found in scanned pool range.")
        return "\n".join(lines)

    lines.append("")
    lines.append("Best hit")
    lines.append("--------")
    lines.append(f"rank_key: {best['rank_key']}")
    lines.append(f"initial_marking: {best['initial_marking']}")
    lines.append(f"initial_total_stoich: {best['initial_total_stoich']}")
    lines.append(f"initial_total_carbon: {best['initial_total_carbon']}")
    lines.append(f"depth: {best['depth']}")
    lines.append(f"reactions: {best['reactions']}")
    lines.append(f"rules: {best['rules']}")
    lines.append(f"flow: {best['flow']}")
    lines.append(f"certificate: {best['certificate']}")
    lines.append("equations:")
    for eq in best["equations"]:
        lines.append(f"  - {eq}")

    lines.append("")
    lines.append("Top hits")
    lines.append("--------")
    for i, row in enumerate(result.get("top_hits", []), start=1):
        lines.append("")
        lines.append(f"Hit {i}")
        lines.append(f"  rank_key: {row['rank_key']}")
        lines.append(f"  initial_marking: {row['initial_marking']}")
        lines.append(f"  initial_total_stoich: {row['initial_total_stoich']}")
        lines.append(f"  initial_total_carbon: {row['initial_total_carbon']}")
        lines.append(f"  depth: {row['depth']}")
        lines.append(f"  rules: {row['rules']}")
        lines.append(f"  flow: {row['flow']}")
        for eq in row["equations"]:
            lines.append(f"    - {eq}")

    return "\n".join(lines)


# =============================================================================
# Main workflow
# =============================================================================


def main() -> None:
    log_header("Experiment 0 — Input normalization and global settings")

    save_json(
        "00_inputs.json",
        {
            "analysis_repeat": ANALYSIS_REPEAT,
            "repeats_scan": REPEATS_SCAN,
            "raw_seeds": RAW_SEEDS,
            "canonical_seeds": SEEDS,
            "raw_targets": RAW_TARGETS,
            "canonical_targets": TARGETS,
            "raw_initial_pools": RAW_INITIAL_POOLS,
            "canonical_initial_pools": INITIAL_POOLS,
            "rule_names": RULE_NAMES,
        },
    )

    logger.info("analysis_repeat: %s", ANALYSIS_REPEAT)
    logger.info("canonical seeds: %s", SEEDS)
    logger.info("canonical targets: %s", TARGETS)
    logger.info("canonical pools: %s", INITIAL_POOLS)

    # =========================================================================
    # Experiment 1 — Growth scan across repeat depth
    # =========================================================================

    log_header("Experiment 1 — Build formose rule set and scan network growth")

    scan_rows: List[Dict[str, Any]] = []

    for rep in REPEATS_SCAN:
        dg_i, _, syn_i = build_formose_crn(repeats=rep)
        rows_i = species_table(syn_i)
        deriv_i = derivation_rows(dg_i)

        valid_c = [x["nC"] for x in rows_i if x["nC"] is not None]
        max_carbon = max(valid_c) if valid_c else None

        row = {
            "repeats": rep,
            "n_species": int(syn_i.n_species),
            "n_reactions": int(syn_i.n_reactions),
            "n_rules": int(syn_i.n_rules),
            "n_derivations": len(deriv_i),
            "max_carbon": max_carbon,
            "n_C1": sum(1 for x in rows_i if x["nC"] == 1),
            "n_C2": sum(1 for x in rows_i if x["nC"] == 2),
            "n_C3": sum(1 for x in rows_i if x["nC"] == 3),
            "n_C4plus": sum(1 for x in rows_i if x["nC"] is not None and x["nC"] >= 4),
        }
        scan_rows.append(row)
        logger.info("repeat=%s -> %s", rep, row)

    save_json("01_growth_scan.json", scan_rows)
    save_text("01_growth_scan.txt", json.dumps(scan_rows, indent=2))

    # =========================================================================
    # Experiment 2 — Build analysis network (repeat = 4) and inspect product space
    # =========================================================================

    log_header(
        f"Experiment 2 — Build analysis SynCRN and inspect product space ({REPEAT_TAG})"
    )

    dg, _, syn = build_formose_crn(repeats=ANALYSIS_REPEAT)

    species_rows = species_table(syn)
    formula_rows = formula_distribution(species_rows)
    carbon_rows = carbon_distribution(species_rows)
    reachability = reachability_summary(species_rows)
    target_presence = target_presence_summary(species_rows, TARGETS)
    deriv_rows = derivation_rows(dg)
    rule_usage_rows = rule_usage_distribution(deriv_rows)

    logger.info("repr: %s", repr(syn))
    logger.info("n_species: %s", syn.n_species)
    logger.info("n_reactions: %s", syn.n_reactions)
    logger.info("n_rules: %s", syn.n_rules)
    logger.info("n_derivations: %s", len(deriv_rows))
    logger.info("reachability summary: %s", reachability)
    logger.info("target presence: %s", target_presence)
    logger.info("rule usage: %s", rule_usage_rows)

    syn_desc = syn.describe(include_species=True, species="label")
    logger.info("\n%s", syn_desc)

    save_text("02_syn_description.txt", syn_desc)
    save_text(
        "03_equations_label.txt",
        "\n".join(
            syn.to_equations(species="label", include_id=True, include_rule=True)
        ),
    )
    save_text(
        "04_equations_smiles.txt",
        "\n".join(
            syn.to_equations(species="smiles", include_id=True, include_rule=True)
        ),
    )

    save_json("02_species_table.json", species_rows)
    save_json("03_formula_distribution.json", formula_rows)
    save_json("04_carbon_distribution.json", carbon_rows)
    save_json("05_reachability_targets.json", reachability)
    save_json("06_target_presence_exact_smiles.json", target_presence)
    save_json(f"07_derivation_records_{REPEAT_TAG}.json", deriv_rows)
    save_json("08_rule_usage_distribution.json", rule_usage_rows)
    save_json("09_species_lookup.json", species_lookup_table(syn))

    # =========================================================================
    # Experiment 3 — Canonicalization / automorphism sanity check (repeat = 4)
    # =========================================================================

    log_header(
        f"Experiment 3 — Canonicalization / automorphism sanity check ({REPEAT_TAG})"
    )

    canon = CRNCanonicalizer(
        syn,
        include_rule=True,
        include_stoich=True,
    )

    has_auto = canon.has_nontrivial_automorphism(timeout_sec=5.0)
    orbits = [
        sorted(list(orb)) for orb in canon.orbits(max_count=2000, timeout_sec=10.0)
    ]
    canon_summary = canon.summary(
        max_count=2000,
        timeout_sec=10.0,
        include_automorphisms=True,
    )
    canon_graph = canon.canonical_graph(timeout_sec=10.0)

    logger.info("analysis repeat: %s", ANALYSIS_REPEAT)
    logger.info("n_species: %s", syn.n_species)
    logger.info("n_reactions: %s", syn.n_reactions)
    logger.info("canonical graph nodes: %s", canon_graph.number_of_nodes())
    logger.info("canonical graph edges: %s", canon_graph.number_of_edges())
    logger.info("Has nontrivial automorphism: %s", has_auto)
    logger.info("Automorphism count: %s", canon_summary.get("automorphism_count"))
    logger.info(
        "Nontrivial orbit sizes: %s", sorted(len(x) for x in orbits if len(x) > 1)
    )

    save_json(
        f"10_canonicalization_{REPEAT_TAG}.json",
        {
            "repeat": ANALYSIS_REPEAT,
            "n_species": int(syn.n_species),
            "n_reactions": int(syn.n_reactions),
            "canonical_graph_nodes": int(canon_graph.number_of_nodes()),
            "canonical_graph_edges": int(canon_graph.number_of_edges()),
            "has_nontrivial_automorphism": has_auto,
            "automorphism_count": canon_summary.get("automorphism_count"),
            "orbits": orbits,
            "canonical_key": str(canon_summary["canonical_key"]),
        },
    )

    # =========================================================================
    # Experiment 4 — Stoichiometric summary (repeat = 4)
    # =========================================================================

    log_header(f"Experiment 4 — Stoichiometric summary ({REPEAT_TAG})")

    species_order, reaction_order, S = build_S(syn)
    sto = summary(syn)
    laws = integer_conservation_laws(syn)

    species_order_labels = map_species_ids_to_labels(syn, species_order)

    logger.info("Stoichiometric matrix shape: %s", S.shape)
    logger.info("Species order (raw): %s", list(species_order))
    logger.info("Species order (labels): %s", species_order_labels)
    logger.info("Reaction order: %s", list(reaction_order))
    logger.info("Stoich summary: %s", sto)
    logger.info("Total integer conservation laws: %s", len(laws))

    save_json(
        f"11_stoich_summary_{REPEAT_TAG}.json",
        {
            "repeat": ANALYSIS_REPEAT,
            "S_shape": list(S.shape),
            "species_order_ids": list(species_order),
            "species_order_labels": species_order_labels,
            "reaction_order": list(reaction_order),
            "summary": {
                "n_species": int(sto.n_species),
                "n_reactions": int(sto.n_reactions),
                "rank": int(sto.rank),
                "dim_left_kernel": int(sto.dim_left_kernel),
                "dim_right_kernel": int(sto.dim_right_kernel),
            },
            "n_integer_conservation_laws": int(len(laws)),
            "first_5_laws": [
                law_to_label_dict(species_order_labels, law) for law in list(laws)[:5]
            ],
        },
    )

    # =========================================================================
    # Experiment 5 — Exact target presence and qualitative path search (repeat = 4)
    # =========================================================================

    log_header(f"Experiment 5A — Exact target presence by SMILES ({REPEAT_TAG})")
    for target_name, target_smiles in TARGETS.items():
        payload = target_presence[target_name]
        logger.info(
            "target=%s smiles=%s present=%s row=%s",
            target_name,
            target_smiles,
            payload["present"],
            payload["row"],
        )

    log_header(
        f"Experiment 5B — Qualitative path search from minimal seed pool ({REPEAT_TAG})"
    )

    qualitative_results: Dict[str, List[Dict[str, Any]]] = {}
    qualitative_meta: Dict[str, Dict[str, Any]] = {}
    qualitative_txt_parts: List[str] = []

    for target_name, target_smiles, max_depth in [
        ("C4_aldose", TARGETS["C4_aldose"], 8),
        ("C6_aldose", TARGETS["C6_aldose"], 12),
    ]:
        cands, meta = run_target_search(
            syn,
            target_name=target_name,
            target_smiles=target_smiles,
            init_name="minimal",
            init=INITIAL_POOLS["minimal"],
            max_depth=max_depth,
            max_paths=20,
            validate=False,
        )
        block = pretty_candidate_block(
            f"Qualitative candidates to {target_name} from minimal pool",
            cands,
            syn,
        )
        logger.info("\n%s", block)
        qualitative_results[target_name] = [candidate_to_dict(c) for c in cands]
        qualitative_meta[target_name] = meta
        qualitative_txt_parts.append(block)
        qualitative_txt_parts.append("")

    save_text(
        "05_path_targets_qualitative.txt", "\n".join(qualitative_txt_parts).strip()
    )
    save_json(
        "12_path_targets_qualitative.json",
        {"meta": qualitative_meta, "results": qualitative_results},
    )

    # =========================================================================
    # Experiment 6 — Exact validation under multiple initial pools (repeat = 4)
    # =========================================================================

    log_header(
        f"Experiment 6 — Exact validation under multiple initial pools ({REPEAT_TAG})"
    )

    validated_results: Dict[str, Dict[str, Any]] = {}
    validated_txt_parts: List[str] = []

    validation_plan = [
        ("C4_minimal", TARGETS["C4_aldose"], INITIAL_POOLS["minimal"], 8),
        ("C4_exact", TARGETS["C4_aldose"], INITIAL_POOLS["C4_exact"], 8),
        ("C6_minimal", TARGETS["C6_aldose"], INITIAL_POOLS["minimal"], 12),
        ("C6_exact", TARGETS["C6_aldose"], INITIAL_POOLS["C6_exact"], 12),
        ("C6_alt", TARGETS["C6_aldose"], INITIAL_POOLS["C6_alt"], 12),
    ]

    for name, target_smiles, init, max_depth in validation_plan:
        cands, meta = run_target_search(
            syn,
            target_name=name,
            target_smiles=target_smiles,
            init_name=name,
            init=init,
            max_depth=max_depth,
            max_paths=20,
            validate=True,
        )
        cert = first_realizable_certificate(cands)
        block = pretty_candidate_block(f"Validated candidates to {name}", cands, syn)
        logger.info("\n%s", block)
        logger.info("realizable_certificate=%s", cert)

        validated_results[name] = {
            "meta": meta,
            "n_candidates": len(cands),
            "results": [candidate_to_dict(c) for c in cands],
            "realizable_certificate": cert,
        }
        validated_txt_parts.append(block)
        validated_txt_parts.append(f"realizable_certificate={cert}")
        validated_txt_parts.append("")

    save_text("06_path_targets_validated.txt", "\n".join(validated_txt_parts).strip())
    save_json("13_path_targets_validated.json", validated_results)

    # =========================================================================
    # Experiment 7 — Exact realizability details for accepted paths (repeat = 4)
    # =========================================================================

    log_header(
        f"Experiment 7 — Exact realizability details for accepted paths ({REPEAT_TAG})"
    )

    realizability_payload: Dict[str, Any] = {}
    trace_payload: Dict[str, Any] = {}
    subnetwork_payload: Dict[str, Any] = {}

    for name, target_smiles, init, _ in validation_plan:
        cert = validated_results[name]["realizable_certificate"]
        if cert is None:
            logger.info("%s: no realizable certificate found.", name)
            realizability_payload[name] = {
                "found": False,
                "target_smiles": target_smiles,
                "initial_marking": dict(init),
            }
            continue

        flow = {rid: 1 for rid in cert}
        pr = PathwayRealizability().load_syncrn_and_flow(
            syn,
            flow=flow,
            initial_marking=init,
            species="label",
            reaction="id",
        )
        pr.build_petri_net_from_flow()

        ok, cert_real = pr.is_realizable()
        scaled_ok, k = pr.is_scaled_realizable(k_max=4)
        borrow_ok, borrow = pr.is_borrow_realizable(max_borrow_each=1)
        summ = pr.summary()

        logger.info("%s exact realizable: %s", name, ok)
        logger.info("%s certificate: %s", name, cert_real)
        logger.info("%s scaled realizable: %s scale=%s", name, scaled_ok, k)
        logger.info("%s borrow realizable: %s borrow=%s", name, borrow_ok, borrow)
        logger.info("%s summary: %s", name, summ)

        realizability_payload[name] = {
            "found": True,
            "target_smiles": target_smiles,
            "flow": flow,
            "exact_realizable": bool(ok),
            "certificate": None if cert_real is None else list(cert_real),
            "scaled_realizable": bool(scaled_ok),
            "scale": k,
            "borrow_realizable": bool(borrow_ok),
            "borrow": borrow,
            "summary": {
                "n_species": int(summ.n_species),
                "n_reactions": int(summ.n_reactions),
                "active_flow": dict(summ.active_flow),
                "initial_marking": dict(summ.initial_marking),
                "goal_exact": dict(summ.goal_exact),
                "goal_atleast": dict(summ.goal_atleast),
            },
        }

        trace_rows = certificate_trace(syn, cert)
        trace_txt = dump_certificate_trace_text(f"{name} certificate trace", trace_rows)
        trace_payload[name] = trace_rows
        save_text(f"07_trace_{name}.txt", trace_txt)
        logger.info("\n%s", trace_txt)

        sub_syn = build_subnetwork_from_certificate(syn, cert, species="label")
        sub_desc = sub_syn.describe(include_species=True, species="label")
        sub_eq_label = "\n".join(
            sub_syn.to_equations(species="label", include_id=True, include_rule=True)
        )
        sub_eq_smiles = "\n".join(
            sub_syn.to_equations(species="smiles", include_id=True, include_rule=True)
        )

        save_text(f"08_subnetwork_{name}_description.txt", sub_desc)
        save_text(f"09_subnetwork_{name}_label.txt", sub_eq_label)
        save_text(f"10_subnetwork_{name}_smiles.txt", sub_eq_smiles)

        subnetwork_payload[name] = {
            "description": sub_desc,
            "equations_label": sub_eq_label.splitlines(),
            "equations_smiles": sub_eq_smiles.splitlines(),
        }

    save_json("14_realizability_details.json", realizability_payload)
    save_json("15_certificate_traces.json", trace_payload)
    save_json("16_minimal_subnetworks.json", subnetwork_payload)

    # =========================================================================
    # Experiment 8 — Thermo-like and Petri structural analysis (repeat = 4)
    # =========================================================================

    log_header(
        f"Experiment 8 — Thermo-like and Petri structural analysis ({REPEAT_TAG})"
    )

    thermo = compute_thermo_summary(syn)
    logger.info("Thermo summary: %s", thermo)

    save_json(
        f"17_thermo_summary_{REPEAT_TAG}.json",
        {
            "repeat": ANALYSIS_REPEAT,
            "conservative": bool(thermo.conservative),
            "consistent": bool(thermo.consistent),
            "irreversible_futile_cycles": bool(thermo.irreversible_futile_cycles),
            "example_conservation_law": (
                None
                if thermo.example_conservation_law is None
                else [float(x) for x in thermo.example_conservation_law]
            ),
        },
    )

    an = PetriAnalyzer(
        syn,
        rtol=1e-12,
        max_siphon_size=4,
    ).compute_all()

    logger.info("%s", an.explain())
    logger.info("Petri summary object: %s", an.summary)

    petri_payload = petri_json_payload(an, syn)
    save_json(f"18_petri_summary_{REPEAT_TAG}.json", petri_payload)
    save_text("11_petri_explain.txt", an.explain())

    # =========================================================================
    # Experiment 9 — Search for shortest realizable route with minimal init pool
    # =========================================================================

    log_header(
        f"Experiment 9 — Shortest realizable route with minimal initial stoichiometry ({REPEAT_TAG})"
    )

    shortest_realizable_min_init_results: Dict[str, Any] = {}
    shortest_realizable_min_init_txt_parts: List[str] = []

    search_plan = [
        ("C4_aldose", TARGETS["C4_aldose"], 8),
        ("C5_aldose", TARGETS["C5_aldose"], 10),
        ("C6_aldose", TARGETS["C6_aldose"], 12),
    ]

    for target_name, target_smiles, max_depth in search_plan:
        max_coeff_by_seed = default_seed_pool_bounds(target_smiles, SEEDS)

        logger.info(
            "Scanning initial pools for %s with bounds=%s and max_depth=%s",
            target_name,
            max_coeff_by_seed,
            max_depth,
        )

        result = find_shortest_realizable_min_init(
            syn,
            target_name=target_name,
            target_smiles=target_smiles,
            seed_order=SEEDS,
            max_coeff_by_seed=max_coeff_by_seed,
            max_depth=max_depth,
            max_paths=100,
            top_k=20,
        )

        text_block = dump_shortest_realizable_min_init_text(
            f"Shortest realizable + minimal init search for {target_name}",
            result,
        )

        logger.info("\n%s", text_block)

        shortest_realizable_min_init_results[target_name] = result
        shortest_realizable_min_init_txt_parts.append(text_block)
        shortest_realizable_min_init_txt_parts.append("")

    save_json(
        "18b_shortest_realizable_min_init.json",
        shortest_realizable_min_init_results,
    )
    save_text(
        "18b_shortest_realizable_min_init.txt",
        "\n".join(shortest_realizable_min_init_txt_parts).strip(),
    )

    # =========================================================================
    # Final compact summary
    # =========================================================================

    log_header("Final compact summary")

    final_summary = {
        "analysis_repeat": ANALYSIS_REPEAT,
        "growth": scan_rows,
        "analysis_network": {
            "n_species": int(syn.n_species),
            "n_reactions": int(syn.n_reactions),
            "n_rules": int(syn.n_rules),
            "target_presence": {
                name: {
                    "present": bool(payload["present"]),
                    "target_smiles": payload["target_smiles"],
                }
                for name, payload in target_presence.items()
            },
        },
        "symmetry": {
            "has_nontrivial_automorphism": has_auto,
            "automorphism_count": canon_summary.get("automorphism_count"),
            "n_nontrivial_orbits": sum(1 for orb in orbits if len(orb) > 1),
        },
        "stoich": {
            "shape": list(S.shape),
            "rank": int(sto.rank),
            "dim_left_kernel": int(sto.dim_left_kernel),
            "dim_right_kernel": int(sto.dim_right_kernel),
            "n_integer_conservation_laws": len(laws),
        },
        "path_tests": {
            name: {
                "target_smiles": validated_results[name]["meta"]["target_smiles"],
                "target_carbon": validated_results[name]["meta"]["target_carbon"],
                "initial_total_carbon": validated_results[name]["meta"][
                    "initial_total_carbon"
                ],
                "carbon_sufficient": validated_results[name]["meta"][
                    "carbon_sufficient"
                ],
                "n_candidates": validated_results[name]["n_candidates"],
                "realizable_certificate": validated_results[name][
                    "realizable_certificate"
                ],
            }
            for name in validated_results
        },
        "thermo": {
            "conservative": bool(thermo.conservative),
            "consistent": bool(thermo.consistent),
            "irreversible_futile_cycles": bool(thermo.irreversible_futile_cycles),
        },
        "petri": {
            "persistence_ok": bool(an.persistence_ok),
            "n_siphons": len(an.siphons),
            "n_traps": len(an.traps),
            "n_p_semiflows": petri_payload["n_p_semiflows"],
            "n_t_semiflows": petri_payload["n_t_semiflows"],
        },
        "shortest_realizable_min_init": {
            name: {
                "best_initial_marking": (
                    None
                    if payload["best"] is None
                    else payload["best"]["initial_marking"]
                ),
                "best_initial_total_stoich": (
                    None
                    if payload["best"] is None
                    else payload["best"]["initial_total_stoich"]
                ),
                "best_depth": (
                    None if payload["best"] is None else payload["best"]["depth"]
                ),
                "best_flow": (
                    None if payload["best"] is None else payload["best"]["flow"]
                ),
                "n_realizable_hits": payload["n_realizable_hits"],
                "searched_pools": payload["searched_pools"],
            }
            for name, payload in shortest_realizable_min_init_results.items()
        },
    }

    logger.info("%s", json.dumps(to_jsonable(final_summary), indent=2))
    save_json("19_final_summary.json", final_summary)

    logger.info("All outputs saved under: %s", OUTDIR.resolve())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Formose analysis pipeline failed.")
        raise
