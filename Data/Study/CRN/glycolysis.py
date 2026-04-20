from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # adjust level if needed
sys.path.insert(0, str(ROOT))
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from synkit.CRN.Structure import SynCRN
from synkit.CRN.Pathway import run_pathfinder_from_syncrn, PathwayRealizability
from synkit.CRN.Props.stoich import build_S, summary, integer_conservation_laws
from synkit.CRN.Petrinet.analyzer import PetriAnalyzer
from synkit.CRN.Symmetry import CRNCanonicalizer
from synkit.CRN.Props.thermo import compute_thermo_summary
from synkit.CRN.Props.dynamics import (
    jacobian_sparsity,
    jacobian_sign_pattern,
    species_influence_graph,
    structural_singularity_summary,
)

# =============================================================================
# User input
# =============================================================================

rxns = [
    "A+B>>C+D",
    "E>>B+F",
    "C+G>>A+H",
    "I>>J",
    "J+K+L>>M+N+O",
    "J+K+P>>M+Q+O",
    "H>>I+J",
    "A+M>>C+R",
    "R>>E",
    "C+S>>A+T",
    "U+S>>V+T",
    "A+G>>W+H",
    "J+F+X+X>>R+O+O+Y+Y",
    "S+A>>T+W",
    "T>>G",
    "C+Z>>A+B+AA",
    "AB+Z>>AC+B+AA",
    "B+F>>E",
    "AD+Z>>AE+B+AA",
    "H+F>>G+K",
    "J>>I",
    "M+N+O>>J+K+L",
    "M+Q+O>>J+K+P",
    "I+J>>H",
    "C+R>>A+M",
    "E>>R",
    "D+AF>>AG+AA",
    "D+AH+L>>AI+AA+N+O",
    "D+AH+P>>AI+AA+Q+O",
    "X+X+D+AH>>Y+Y+AI+AA+O+O",
    "AH+AJ>>AI+AK",
    "AG+AL>>AJ+AF",
    "AK+L>>AL+N+O",
    "D+AH+AM>>AI+AA+AN",
]

rules = [f"r{i}" for i in range(1, len(rxns) + 1)]

species_smiles = {
    "A": "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O",
    "B": "C=C(OP(=O)(O)O)C(=O)O",
    "C": "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O",
    "D": "CC(=O)C(=O)O",
    "E": "O=C(O)[C@@H](CO)OP(=O)(O)O",
    "F": "O",
    "G": "O=P(O)(O)OC[C@H]1OC(O)(CO)[C@@H](O)[C@@H]1O",
    "H": "O=P(O)(O)OC[C@H]1OC(O)(COP(=O)(O)O)[C@@H](O)[C@@H]1O",
    "I": "O=C(CO)COP(=O)(O)O",
    "J": "O=C[C@H](O)COP(=O)(O)O",
    "K": "O=P(O)(O)O",
    "L": "NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1",
    "M": "O=C(OP(=O)(O)O)[C@H](O)COP(=O)(O)O",
    "N": "NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1",
    "O": "[H+]",
    "P": "NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)(O)O)[C@@H]3O)[C@@H](O)[C@H]2O)c1",
    "Q": "NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)(O)O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1",
    "R": "O=C(O)[C@H](O)COP(=O)(O)O",
    "S": "OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
    "T": "O=P(O)(O)OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
    "U": "O=P(O)(O)OP(=O)(O)OP(=O)(O)O",
    "V": "O=P(O)(O)OP(=O)(O)O",
    "W": "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O",
    "X": "S1[Fe+]S[Fe+]1",
    "Y": "S1[Fe]S[Fe+]1",
    "Z": "O=C(O)CC(=O)C(=O)O",
    "AA": "O=C=O",
    "AB": "Nc1nc2c(ncn2[C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)[nH]1",
    "AC": "Nc1nc2c(ncn2[C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)[nH]1",
    "AD": "O=c1[nH]cnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O",
    "AE": "O=c1[nH]cnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O",
    "AF": "Cc1ncc(C[n+]2csc(CCOP(=O)(O)OP(=O)(O)O)c2C)c(N)n1",
    "AG": "Cc1ncc(C[n+]2c(C(C)O)sc(CCOP(=O)(O)OP(=O)(O)O)c2C)c(N)n1",
    "AH": "CC(C)(COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O)[C@@H](O)C(=O)NCCC(=O)NCCS",
    "AI": "CC(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O",
    "AJ": "NC(=O)CCCC[C@@H](S)CCSC(C)=O",
    "AK": "NC(=O)CCCC[C@@H](S)CCS",
    "AL": "NC(=O)CCCC[C@@H]1CCSS1",
    "AM": "CC2(C=C1(N=C3(C(=O)NC(=O)N=C(N(CC(O)C(O)C(O)COP([O-])(=O)[O-])C1=CC(C)=2)3)))",
    "AN": "CC2(C=C1(NC3(C(=O)NC(=O)NC(N(CC(O)C(O)C(O)COP([O-])(=O)[O-])C1=CC(C)=2)=3)))",
}


# =============================================================================
# Output / logging
# =============================================================================

# save next to the current script file
BASE_DIR = Path(__file__).resolve().parent
OUTDIR = BASE_DIR / "case_glycolysis"
OUTDIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUTDIR / "glycolysis.log"
JSON_DIR = OUTDIR / "json"
TXT_DIR = OUTDIR / "txt"
JSON_DIR.mkdir(exist_ok=True)
TXT_DIR.mkdir(exist_ok=True)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("glycolysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


logger = setup_logger()


# =============================================================================
# Helpers
# =============================================================================


def log_header(title: str) -> None:
    line = "=" * 88
    logger.info("\n%s\n%s\n%s", line, title, line)


def save_text(filename: str, text: str) -> None:
    path = TXT_DIR / filename
    path.write_text(text, encoding="utf-8")
    logger.info("Saved text: %s", path)


def save_json(filename: str, payload: Any) -> None:
    path = JSON_DIR / filename
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved JSON: %s", path)


def stringify(obj: Any) -> str:
    return str(obj)


def first_realizable_certificate(cands) -> Optional[List[str]]:
    for cand in cands:
        if getattr(cand, "realizable", False) and getattr(cand, "certificate", None):
            return list(cand.certificate)
    return None


def rid_to_rule_repr_map(syn: SynCRN) -> Dict[str, Optional[str]]:
    return {rid: syn.reactions[rid].rule_repr for rid in syn.reaction_ids}


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


def map_species_ids_to_labels(syn: SynCRN, ids: Sequence[str]) -> List[str]:
    return [syn.species[str(sid)].label for sid in ids]


def safe_sorted_sets(sets_like: Sequence[Sequence[str]]) -> List[List[str]]:
    return [sorted(list(x)) for x in sets_like]


def candidate_to_dict(cand) -> Dict[str, Any]:
    return {
        "depth": cand.depth,
        "reactions": list(cand.reactions),
        "flow": dict(cand.flow),
        "reached_species": list(cand.reached_species),
        "realizable": bool(cand.realizable),
        "certificate": None if cand.certificate is None else list(cand.certificate),
    }


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


def attach_smiles_to_species(syn: SynCRN, smiles_map: Dict[str, str]) -> None:
    for sid, sp in syn.species.items():
        sp.smiles = smiles_map.get(sp.label, sp.smiles)


def species_lookup_table(syn: SynCRN) -> List[Dict[str, Any]]:
    rows = []
    for sid in syn.species_ids:
        sp = syn.species[sid]
        rows.append(
            {
                "sid": sid,
                "label": getattr(sp, "label", None),
                "smiles": getattr(sp, "smiles", None),
                "source_id": getattr(sp, "source_id", None),  # safe fallback
            }
        )
    return rows


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
                "smiles": getattr(sp, "smiles", None),
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
                "smiles": getattr(sp, "smiles", None),
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
    lines: List[str] = []
    lines.append(title)
    lines.append("=" * len(title))

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


def pretty_candidate_block(title: str, cands, syn: SynCRN) -> str:
    rid2rule = rid_to_rule_repr_map(syn)
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
        summary = getattr(an, "summary", None)
        payload = {
            "place_order": list(getattr(summary, "place_order", [])),
            "transition_order": list(getattr(summary, "transition_order", [])),
            "persistence_ok": bool(getattr(summary, "persistence_ok", False)),
            "siphons": [sorted(list(x)) for x in getattr(summary, "siphons", [])],
            "traps": [sorted(list(x)) for x in getattr(summary, "traps", [])],
            "p_semiflows": getattr(summary, "p_semiflows", []),
            "t_semiflows": getattr(summary, "t_semiflows", []),
            "persistence_details": getattr(an, "persistence_details", {}),
        }

    place_order = list(payload.get("place_order", []))

    return {
        "place_order": place_order,
        "place_order_labels": map_species_ids_to_labels(syn, place_order),
        "transition_order": list(payload.get("transition_order", [])),
        "persistence_ok": bool(payload.get("persistence_ok", False)),
        "siphons": payload.get("siphons", []),
        "traps": payload.get("traps", []),
        "n_p_semiflows": len(payload.get("p_semiflows", [])),
        "n_t_semiflows": len(payload.get("t_semiflows", [])),
        "persistence_details": payload.get("persistence_details", {}),
    }


def numpy_matrix_to_list(A) -> List[List[Any]]:
    return [[x.item() if hasattr(x, "item") else x for x in row] for row in A.tolist()]


def edge_list_with_sign(G) -> List[Dict[str, Any]]:
    rows = []
    for u, v, data in G.edges(data=True):
        rows.append(
            {
                "source": u,
                "target": v,
                "sign": data.get("sign"),
                "source_species": data.get("source_species"),
                "target_species": data.get("target_species"),
            }
        )
    return rows


# =============================================================================
# Experiment 0 — Build and annotate SynCRN
# =============================================================================

log_header("Experiment 0 — Build and annotate SynCRN")

syn = SynCRN.from_reaction_strings(rxns, rules=rules)
attach_smiles_to_species(syn, species_smiles)

logger.info("repr: %s", repr(syn))
logger.info("n_species: %s", syn.n_species)
logger.info("n_reactions: %s", syn.n_reactions)
logger.info("n_rules: %s", syn.n_rules)

desc = syn.describe(include_species=True, species="label")
logger.info("\n%s", desc)
save_text("00_syn_description.txt", desc)

eq_label = "\n".join(
    syn.to_equations(species="label", include_id=True, include_rule=True)
)
eq_smiles = "\n".join(
    syn.to_equations(species="smiles", include_id=True, include_rule=True)
)
save_text("01_equations_label.txt", eq_label)
save_text("02_equations_smiles.txt", eq_smiles)

lookup_rows = species_lookup_table(syn)
save_json("00_species_lookup.json", lookup_rows)

rule_table = []
for rid in syn.rule_ids:
    rule_obj = syn.rules[rid]
    rule_table.append(
        {
            "rule_id": rid,
            "rule_repr": rule_obj.rule_repr,
            "rule_index": rule_obj.rule_index,
            "label": rule_obj.label,
        }
    )
save_json("01_rule_table.json", rule_table)


# =============================================================================
# Experiment 1 — Canonicalization / automorphism sanity check
# =============================================================================

log_header("Experiment 1 — Canonicalization sanity check")

canon = CRNCanonicalizer(
    syn,
    include_rule=True,
    include_stoich=True,
)

has_auto = canon.has_nontrivial_automorphism(timeout_sec=5.0)
orbits = [sorted(list(orb)) for orb in canon.orbits(max_count=200, timeout_sec=5.0)]
canon_summary = canon.summary(
    max_count=200,
    timeout_sec=5.0,
    include_automorphisms=True,
)

logger.info("Has nontrivial automorphism: %s", has_auto)
logger.info("Automorphism count: %s", canon_summary.get("automorphism_count"))
logger.info("Orbits: %s", orbits)

save_json(
    "02_canonicalization.json",
    {
        "has_nontrivial_automorphism": has_auto,
        "automorphism_count": canon_summary.get("automorphism_count"),
        "orbits": orbits,
        "canonical_key": stringify(canon_summary["canonical_key"]),
    },
)


# =============================================================================
# Experiment 2 — Stoichiometric summary
# =============================================================================

log_header("Experiment 2 — Stoichiometric summary")

species_order, reaction_order, S = build_S(syn)
sto = summary(syn)
laws = integer_conservation_laws(syn)

species_order_labels = map_species_ids_to_labels(syn, species_order)

logger.info("Stoichiometric matrix shape: %s", S.shape)
logger.info("Species order (internal ids): %s", species_order)
logger.info("Species order (labels): %s", species_order_labels)
logger.info("Reaction order: %s", reaction_order)
logger.info("Stoich summary: %s", sto)
logger.info("Total integer conservation laws: %s", len(laws))

save_json(
    "03_stoich_summary.json",
    {
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
            dict(zip(species_order_labels, list(law))) for law in list(laws)[:5]
        ],
    },
)


# =============================================================================
# Experiment 3 — Qualitative path search
# =============================================================================

log_header("Experiment 3A — Qualitative path search to H")

cands_H_qual = run_pathfinder_from_syncrn(
    syn,
    source_species=["C", "S"],
    target_species=["H"],
    initial_marking={"C": 1, "S": 1},
    species="label",
    reaction="id",
    max_depth=4,
    max_paths=10,
    validate=False,
    verbose=True,
)

block = pretty_candidate_block("Qualitative candidates to H", cands_H_qual, syn)
logger.info("\n%s", block)
save_text("03_path_H_qualitative.txt", block)
save_json("04_path_H_qualitative.json", [candidate_to_dict(c) for c in cands_H_qual])


log_header("Experiment 3B — Qualitative path search to J")

cands_J_qual = run_pathfinder_from_syncrn(
    syn,
    source_species=["C", "S"],
    target_species=["J"],
    initial_marking={"C": 1, "S": 1},
    species="label",
    reaction="id",
    max_depth=5,
    max_paths=10,
    validate=False,
    verbose=True,
)

block = pretty_candidate_block("Qualitative candidates to J", cands_J_qual, syn)
logger.info("\n%s", block)
save_text("04_path_J_qualitative.txt", block)
save_json("05_path_J_qualitative.json", [candidate_to_dict(c) for c in cands_J_qual])


# =============================================================================
# Experiment 4 — Exact validation / realizability
# =============================================================================

log_header("Experiment 4A — Exact validation to H")

cands_H = run_pathfinder_from_syncrn(
    syn,
    source_species=["C", "S"],
    target_species=["H"],
    initial_marking={"C": 1, "S": 1},
    species="label",
    reaction="id",
    max_depth=4,
    max_paths=10,
    validate=True,
    verbose=True,
)

block = pretty_candidate_block("Validated candidates to H", cands_H, syn)
logger.info("\n%s", block)
save_text("05_path_H_validated.txt", block)
save_json("06_path_H_validated.json", [candidate_to_dict(c) for c in cands_H])


log_header("Experiment 4B — Exact validation to J")

cands_J = run_pathfinder_from_syncrn(
    syn,
    source_species=["C", "S"],
    target_species=["J"],
    initial_marking={"C": 1, "S": 1},
    species="label",
    reaction="id",
    max_depth=5,
    max_paths=10,
    validate=True,
    verbose=True,
)

block = pretty_candidate_block("Validated candidates to J", cands_J, syn)
logger.info("\n%s", block)
save_text("06_path_J_validated.txt", block)
save_json("07_path_J_validated.json", [candidate_to_dict(c) for c in cands_J])


# =============================================================================
# Experiment 5 — Exact realizability of accepted path
# =============================================================================

log_header("Experiment 5A — Exact realizability for best H certificate")

cert_H = first_realizable_certificate(cands_H)
if cert_H is None:
    logger.info("No realizable H certificate found.")
    save_json("08_realizability_H.json", {"found": False})
else:
    flow_H = {rid: 1 for rid in cert_H}
    pr_H = PathwayRealizability().load_syncrn_and_flow(
        syn,
        flow=flow_H,
        initial_marking={"C": 1, "S": 1},
        species="label",
        reaction="id",
    )
    pr_H.build_petri_net_from_flow()

    ok_H, cert_real_H = pr_H.is_realizable()
    scaled_ok_H, k_H = pr_H.is_scaled_realizable(k_max=4)
    borrow_ok_H, borrow_H = pr_H.is_borrow_realizable(max_borrow_each=1)
    summ_H = pr_H.summary()

    logger.info("Exact realizable: %s", ok_H)
    logger.info("Certificate: %s", cert_real_H)
    logger.info("Scaled realizable: %s scale=%s", scaled_ok_H, k_H)
    logger.info("Borrow realizable: %s borrow=%s", borrow_ok_H, borrow_H)
    logger.info("Summary: %s", summ_H)

    save_json(
        "08_realizability_H.json",
        {
            "found": True,
            "flow": flow_H,
            "exact_realizable": bool(ok_H),
            "certificate": None if cert_real_H is None else list(cert_real_H),
            "scaled_realizable": bool(scaled_ok_H),
            "scale": k_H,
            "borrow_realizable": bool(borrow_ok_H),
            "borrow": borrow_H,
            "summary": {
                "n_species": int(summ_H.n_species),
                "n_reactions": int(summ_H.n_reactions),
                "active_flow": dict(summ_H.active_flow),
                "initial_marking": dict(summ_H.initial_marking),
                "goal_exact": dict(summ_H.goal_exact),
                "goal_atleast": dict(summ_H.goal_atleast),
            },
        },
    )

log_header("Experiment 5B — Initial-marking sensitivity for false-positive H path")

false_positive_flow = {"50": 1, "55": 1, "43": 1}
initial_markings = [
    {"C": 1, "S": 1},
    {"C": 2, "S": 1},
    {"C": 1, "S": 1, "G": 1},
]

sensitivity_rows = []
for init in initial_markings:
    pr = PathwayRealizability().load_syncrn_and_flow(
        syn,
        flow=false_positive_flow,
        initial_marking=init,
        species="label",
        reaction="id",
    )
    pr.build_petri_net_from_flow()
    ok, cert = pr.is_realizable()
    row = {
        "initial_marking": init,
        "realizable": bool(ok),
        "certificate": None if cert is None else list(cert),
    }
    sensitivity_rows.append(row)
    logger.info("init=%s -> realizable=%s certificate=%s", init, ok, cert)

save_json("09_initial_marking_sensitivity.json", sensitivity_rows)


# =============================================================================
# Experiment 6 — Trace back to real molecules
# =============================================================================

log_header("Experiment 6A — Trace realizable H certificate back to real molecules")

if cert_H is None:
    logger.info("No realizable H certificate to trace.")
else:
    trace_H = certificate_trace(syn, cert_H)
    trace_H_txt = dump_certificate_trace_text("H certificate trace", trace_H)
    logger.info("\n%s", trace_H_txt)
    save_text("07_trace_H.txt", trace_H_txt)
    save_json("10_trace_H.json", trace_H)

log_header("Experiment 6B — Trace realizable J certificate back to real molecules")

cert_J = first_realizable_certificate(cands_J)
if cert_J is None:
    logger.info("No realizable J certificate to trace.")
else:
    trace_J = certificate_trace(syn, cert_J)
    trace_J_txt = dump_certificate_trace_text("J certificate trace", trace_J)
    logger.info("\n%s", trace_J_txt)
    save_text("08_trace_J.txt", trace_J_txt)
    save_json("11_trace_J.json", trace_J)


# =============================================================================
# Experiment 7 — Minimal realizing subnetworks
# =============================================================================

log_header("Experiment 7A — Minimal realizing subnetwork to H")

if cert_H is None:
    logger.info("No realizable H certificate.")
else:
    syn_H = build_subnetwork_from_certificate(syn, cert_H, species="label")
    attach_smiles_to_species(syn_H, species_smiles)

    desc_H = syn_H.describe(include_species=True, species="label")
    eq_H_label = "\n".join(
        syn_H.to_equations(species="label", include_id=True, include_rule=True)
    )
    eq_H_smiles = "\n".join(
        syn_H.to_equations(species="smiles", include_id=True, include_rule=True)
    )

    logger.info("\n%s", desc_H)
    save_text("09_subnetwork_H_description.txt", desc_H)
    save_text("10_subnetwork_H_label.txt", eq_H_label)
    save_text("11_subnetwork_H_smiles.txt", eq_H_smiles)

log_header("Experiment 7B — Minimal realizing subnetwork to J")

if cert_J is None:
    logger.info("No realizable J certificate.")
else:
    syn_J = build_subnetwork_from_certificate(syn, cert_J, species="label")
    attach_smiles_to_species(syn_J, species_smiles)

    desc_J = syn_J.describe(include_species=True, species="label")
    eq_J_label = "\n".join(
        syn_J.to_equations(species="label", include_id=True, include_rule=True)
    )
    eq_J_smiles = "\n".join(
        syn_J.to_equations(species="smiles", include_id=True, include_rule=True)
    )

    logger.info("\n%s", desc_J)
    save_text("12_subnetwork_J_description.txt", desc_J)
    save_text("13_subnetwork_J_label.txt", eq_J_label)
    save_text("14_subnetwork_J_smiles.txt", eq_J_smiles)

# =============================================================================
# Experiment 8 — Petri structural analysis
# =============================================================================

log_header("Experiment 8 — Petri structural analysis")

an = PetriAnalyzer(
    syn,
    rtol=1e-12,
    max_siphon_size=4,
).compute_all()

logger.info("%s", an.explain())
logger.info("Petri summary object: %s", an.summary)

petri_payload = petri_json_payload(an, syn)
save_json("12_petri_summary.json", petri_payload)
save_text("15_petri_explain.txt", an.explain())


# =============================================================================
# Experiment 9 — Thermo-like and dynamics structural analysis
# =============================================================================

log_header("Experiment 9A — Thermo-like structural analysis")

thermo = compute_thermo_summary(syn, rtol=1e-12, eps=1e-8)

logger.info("Thermo summary: %s", thermo)

thermo_payload = {
    "conservative": thermo.conservative,
    "consistent": thermo.consistent,
    "irreversible_futile_cycles": thermo.irreversible_futile_cycles,
    "example_conservation_law": (
        None
        if thermo.example_conservation_law is None
        else [float(x) for x in thermo.example_conservation_law.tolist()]
    ),
}
save_json("14_thermo_summary.json", thermo_payload)
save_text("16_thermo_summary.txt", str(thermo))


log_header("Experiment 9B — Structural dynamics analysis")

species_order_sparse, A_sparse = jacobian_sparsity(syn, tol=1e-12)
species_order_sign, P_sign = jacobian_sign_pattern(syn, tol=1e-12)
G_inf = species_influence_graph(syn, tol=1e-12, use_labels=False)

logger.info("Jacobian sparsity species order: %s", species_order_sparse)
logger.info("Jacobian sparsity shape: %s", A_sparse.shape)
logger.info(
    "Species influence graph: %s nodes, %s edges",
    G_inf.number_of_nodes(),
    G_inf.number_of_edges(),
)

save_json(
    "15_dynamics_jacobian_sparsity.json",
    {
        "species_order": list(species_order_sparse),
        "species_order_labels": map_species_ids_to_labels(syn, species_order_sparse),
        "sparsity": numpy_matrix_to_list(A_sparse.astype(int)),
    },
)

save_json(
    "16_dynamics_jacobian_sign_pattern.json",
    {
        "species_order": list(species_order_sign),
        "species_order_labels": map_species_ids_to_labels(syn, species_order_sign),
        "sign_pattern": numpy_matrix_to_list(P_sign),
    },
)

save_json(
    "17_species_influence_graph.json",
    {
        "n_nodes": G_inf.number_of_nodes(),
        "n_edges": G_inf.number_of_edges(),
        "nodes": list(G_inf.nodes()),
        "edges": edge_list_with_sign(G_inf),
    },
)

# Exact symbolic determinant check is attempted only for sufficiently small systems.
# Your glycolysis network has 40 species, so keep max_exact_size below that unless you
# explicitly want a very large symbolic computation.
try:
    dyn_summary = structural_singularity_summary(
        syn,
        tol=1e-12,
        max_exact_size=7,
        symbol_prefix="rprime",
    )
    logger.info("Structural singularity summary:\n%s", dyn_summary)
    save_json("18_structural_singularity_summary.json", dyn_summary.to_dict())
    save_text("17_structural_singularity_summary.txt", str(dyn_summary))
except ImportError as exc:
    logger.info("Dynamics exact symbolic check skipped: %s", exc)
    save_json(
        "18_structural_singularity_summary.json",
        {
            "skipped": True,
            "reason": str(exc),
        },
    )

# =============================================================================
# Final compact summary
# =============================================================================

log_header("Final compact summary")

final_summary = {
    "network": {
        "n_species": syn.n_species,
        "n_reactions": syn.n_reactions,
        "n_rules": syn.n_rules,
    },
    "symmetry": {
        "has_nontrivial_automorphism": has_auto,
        "automorphism_count": canon_summary.get("automorphism_count"),
    },
    "stoich": {
        "shape": list(S.shape),
        "rank": int(sto.rank),
        "dim_left_kernel": int(sto.dim_left_kernel),
        "dim_right_kernel": int(sto.dim_right_kernel),
    },
    "path_H": {
        "qualitative_n": len(cands_H_qual),
        "validated_n": len(cands_H),
        "realizable_certificate": cert_H,
    },
    "path_J": {
        "qualitative_n": len(cands_J_qual),
        "validated_n": len(cands_J),
        "realizable_certificate": cert_J,
    },
    "petri": {
        "persistence_ok": bool(an.persistence_ok),
        "n_siphons": len(an.siphons),
        "n_traps": len(an.traps),
        "n_p_semiflows": len(an.p_semiflows),
        "n_t_semiflows": len(an.t_semiflows),
    },
    "thermo": {
        "conservative": thermo.conservative,
        "consistent": thermo.consistent,
        "irreversible_futile_cycles": thermo.irreversible_futile_cycles,
    },
    "dynamics": {
        "jacobian_n_species": len(species_order_sparse),
        "jacobian_sparsity_shape": list(A_sparse.shape),
        "species_influence_nodes": G_inf.number_of_nodes(),
        "species_influence_edges": G_inf.number_of_edges(),
        "structural_singularity_classification": (
            None if "dyn_summary" not in locals() else dyn_summary.classification
        ),
    },
}

logger.info("%s", json.dumps(final_summary, indent=2))
save_json("13_final_summary.json", final_summary)

logger.info("All outputs saved under: %s", OUTDIR.resolve())
