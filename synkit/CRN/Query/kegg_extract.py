from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from rxnmapper import RXNMapper

from .kegg_api import KEGGClient
from .kegg_parse import (
    collect_compound_ids_from_equations,
    extract_compound_ids_from_text,
    molblock_to_smiles,
    normalize_module_id,
    parse_kegg_field_blocks,
    reaction_smiles_from_equation,
)


_RID_PATTERN = re.compile(r"R\d{5}")


@dataclass
class KEGGExtractor:
    """
    High-level extractor for KEGG pathway/module reaction data.

    This class downloads KEGG entries, resolves module membership, collects
    reaction equations, builds compound tables, constructs reaction SMILES, and
    optionally computes atom-mapped reaction rules.

    :param client:
        Optional KEGG REST client.
    :type client: Optional[KEGGClient]
    """

    client: Optional[KEGGClient] = None

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = KEGGClient()

    # ---------------------------------------------------------------------
    # Raw KEGG accessors
    # ---------------------------------------------------------------------
    def get_modules_from_pathway(self, pathway_id: str) -> List[str]:
        """
        Extract module IDs from a KEGG pathway entry.

        :param pathway_id:
            KEGG pathway identifier, e.g. ``"hsa00010"``.
        :type pathway_id: str

        :returns:
            Canonical module IDs such as ``["M00001", "M00002"]``.
        :rtype: List[str]
        """
        text = self.client.get_text(f"get/{pathway_id}")
        payloads = parse_kegg_field_blocks(text, "MODULE")

        modules: List[str] = []
        for payload in payloads:
            for token in payload.split():
                normalized = normalize_module_id(token)
                if normalized is not None:
                    modules.append(normalized)

        return modules

    def get_reaction_ids_from_module(self, module_id: str) -> List[str]:
        """
        Collect KEGG reaction IDs from a module entry.

        :param module_id:
            KEGG module ID, e.g. ``"M00001"``.
        :type module_id: str

        :returns:
            Sorted reaction IDs.
        :rtype: List[str]
        """
        text = self.client.get_text(f"get/{module_id}")
        payloads = parse_kegg_field_blocks(text, "REACTION")

        reaction_ids: Set[str] = set()
        for payload in payloads:
            reaction_ids.update(_RID_PATTERN.findall(payload))

        return sorted(reaction_ids)

    def get_equation_for_reaction(self, reaction_id: str) -> Optional[str]:
        """
        Fetch the KEGG equation string for a reaction.

        :param reaction_id:
            KEGG reaction ID, e.g. ``"R00200"``.
        :type reaction_id: str

        :returns:
            Equation string if present, otherwise ``None``.
        :rtype: Optional[str]
        """
        text = self.client.get_text(f"get/rn:{reaction_id}")
        payloads = parse_kegg_field_blocks(text, "EQUATION")
        return payloads[0].strip() if payloads else None

    def get_module_equations(self, module_id: str) -> Dict[str, Optional[str]]:
        """
        Build ``{reaction_id: equation}`` for a module.

        :param module_id:
            KEGG module ID.
        :type module_id: str

        :returns:
            Reaction-equation mapping.
        :rtype: Dict[str, Optional[str]]
        """
        reaction_ids = self.get_reaction_ids_from_module(module_id)
        return {
            reaction_id: self.get_equation_for_reaction(reaction_id)
            for reaction_id in reaction_ids
        }

    def get_pathway_module_equations(
        self,
        pathway_id: str,
    ) -> Dict[str, Dict[str, Optional[str]]]:
        """
        Build nested module/reaction equation mapping for a pathway.

        :param pathway_id:
            KEGG pathway identifier.
        :type pathway_id: str

        :returns:
            ``{module_id: {reaction_id: equation}}``.
        :rtype: Dict[str, Dict[str, Optional[str]]]
        """
        modules = self.get_modules_from_pathway(pathway_id)
        return {module_id: self.get_module_equations(module_id) for module_id in modules}

    # ---------------------------------------------------------------------
    # Compound accessors
    # ---------------------------------------------------------------------
    def get_compound_name(self, compound_id: str) -> Optional[str]:
        """
        Retrieve the primary KEGG compound name.

        :param compound_id:
            KEGG compound ID, e.g. ``"C00001"``.
        :type compound_id: str

        :returns:
            First listed compound name if available, otherwise ``None``.
        :rtype: Optional[str]
        """
        text = self.client.get_text(f"get/cpd:{compound_id}")
        payloads = parse_kegg_field_blocks(text, "NAME")
        if not payloads:
            return None

        first_payload = payloads[0].strip()
        return first_payload.split(";")[0].strip()

    def get_compound_molblock(self, compound_id: str) -> Optional[str]:
        """
        Retrieve the KEGG MOL block for a compound.

        :param compound_id:
            KEGG compound ID.
        :type compound_id: str

        :returns:
            MOL block text or ``None`` if unavailable.
        :rtype: Optional[str]
        """
        return self.client.get_optional_text(f"get/cpd:{compound_id}/mol")

    def build_compound_table(
        self,
        compound_ids: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build a KEGG compound table.

        :param compound_ids:
            KEGG compound IDs.
        :type compound_ids: List[str]

        :returns:
            Mapping
            ``{cid: {"id", "name", "smiles", "molblock"}}``.
        :rtype: Dict[str, Dict[str, Any]]
        """
        compounds: Dict[str, Dict[str, Any]] = {}

        for compound_id in compound_ids:
            name = self.get_compound_name(compound_id)
            molblock = self.get_compound_molblock(compound_id)
            smiles = molblock_to_smiles(molblock)

            compounds[compound_id] = {
                "id": compound_id,
                "name": name,
                "smiles": smiles,
                "molblock": molblock,
            }

        return compounds

    # ---------------------------------------------------------------------
    # Reaction SMILES and atom mapping
    # ---------------------------------------------------------------------
    def build_reaction_smiles_dict(
        self,
        parsed_by_rid: Dict[str, Any],
        compounds_by_cid: Dict[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, str], Dict[str, Dict[str, List[str]]]]:
        """
        Build reaction SMILES for a reaction dictionary.

        :param parsed_by_rid:
            Parsed equation objects keyed by reaction ID.
        :type parsed_by_rid: Dict[str, Any]

        :param compounds_by_cid:
            Compound table keyed by compound ID.
        :type compounds_by_cid: Dict[str, Dict[str, Any]]

        :returns:
            Tuple ``(reaction_smiles_by_id, missing_by_id)``.
        :rtype: Tuple[Dict[str, str], Dict[str, Dict[str, List[str]]]]
        """
        reaction_smiles: Dict[str, str] = {}
        missing_by_rid: Dict[str, Dict[str, List[str]]] = {}

        for reaction_id, parsed_equation in parsed_by_rid.items():
            rsmi, missing = reaction_smiles_from_equation(
                parsed_equation,
                compounds_by_cid,
            )
            reaction_smiles[reaction_id] = rsmi
            missing_by_rid[reaction_id] = missing

        return reaction_smiles, missing_by_rid

    def atom_map_reactions(
        self,
        reaction_smiles_by_id: Dict[str, str],
    ) -> Dict[str, Optional[str]]:
        """
        Atom-map reaction SMILES using RXNMapper.

        :param reaction_smiles_by_id:
            Mapping ``{reaction_id: reaction_smiles}``.
        :type reaction_smiles_by_id: Dict[str, str]

        :returns:
            Mapping ``{reaction_id: mapped_reaction_smiles_or_none}``.
        :rtype: Dict[str, Optional[str]]
        """
        mapper = RXNMapper()
        mapped_by_id: Dict[str, Optional[str]] = {}

        for reaction_id, reaction_smiles in reaction_smiles_by_id.items():
            if (
                ">>" not in reaction_smiles
                or reaction_smiles.startswith(">>")
                or reaction_smiles.endswith(">>")
            ):
                mapped_by_id[reaction_id] = None
                continue

            try:
                result = mapper.get_attention_guided_atom_maps([reaction_smiles])[0]
                mapped_by_id[reaction_id] = result.get("mapped_rxn")
            except Exception:
                mapped_by_id[reaction_id] = None

        return mapped_by_id

    # ---------------------------------------------------------------------
    # Missing data reporting
    # ---------------------------------------------------------------------
    def build_missing_compound_report(
        self,
        equations_by_rid: Dict[str, Optional[str]],
        compounds_by_cid: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Build a report for compounds lacking SMILES.

        :param equations_by_rid:
            Reaction equations keyed by reaction ID.
        :type equations_by_rid: Dict[str, Optional[str]]

        :param compounds_by_cid:
            Compound records keyed by KEGG compound ID.
        :type compounds_by_cid: Dict[str, Dict[str, Any]]

        :returns:
            Missing compound report with per-reaction provenance.
        :rtype: Dict[str, Any]
        """
        cid_to_rids: Dict[str, Set[str]] = {}

        for reaction_id, equation in equations_by_rid.items():
            if not equation:
                continue

            for compound_id in extract_compound_ids_from_text(equation):
                cid_to_rids.setdefault(compound_id, set()).add(reaction_id)

        missing_compounds: List[Dict[str, Any]] = []
        missing_ids: Set[str] = set()
        involving_reactions: Set[str] = set()

        for compound_id, record in compounds_by_cid.items():
            if record.get("smiles") is None:
                reaction_ids = sorted(cid_to_rids.get(compound_id, set()))
                missing_compounds.append(
                    {
                        "id": compound_id,
                        "name": record.get("name"),
                        "reactions": reaction_ids,
                    }
                )
                missing_ids.add(compound_id)
                involving_reactions.update(reaction_ids)

        missing_compounds.sort(key=lambda record: record["id"])

        return {
            "missing_compounds": missing_compounds,
            "missing_compound_ids": sorted(missing_ids),
            "reactions_involving_missing": sorted(involving_reactions),
        }

    # ---------------------------------------------------------------------
    # JSON builders
    # ---------------------------------------------------------------------
    def build_kegg_json(
        self,
        equations_by_rid: Dict[str, Optional[str]],
        *,
        smiles_by_rid: Optional[Dict[str, str]] = None,
        rules_by_rid: Optional[Dict[str, Optional[str]]] = None,
        molecules_by_cid: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Build a compact KEGG JSON block with reactions and molecules.

        :param equations_by_rid:
            Reaction equations keyed by reaction ID.
        :type equations_by_rid: Dict[str, Optional[str]]

        :param smiles_by_rid:
            Optional reaction SMILES keyed by reaction ID.
        :type smiles_by_rid: Optional[Dict[str, str]]

        :param rules_by_rid:
            Optional mapped rule strings keyed by reaction ID.
        :type rules_by_rid: Optional[Dict[str, Optional[str]]]

        :param molecules_by_cid:
            Optional molecule table keyed by compound ID.
        :type molecules_by_cid: Optional[Dict[str, Dict[str, Any]]]

        :returns:
            JSON-like dictionary with ``"reactions"`` and ``"molecules"``.
        :rtype: Dict[str, Any]
        """
        smiles_by_rid = smiles_by_rid or {}
        rules_by_rid = rules_by_rid or {}
        molecules_by_cid = molecules_by_cid or {}

        all_compound_ids: Set[str] = set()
        reactions: List[Dict[str, Any]] = []

        for reaction_id in sorted(equations_by_rid.keys()):
            equation = equations_by_rid[reaction_id]
            if not equation:
                continue

            all_compound_ids.update(extract_compound_ids_from_text(equation))
            reactions.append(
                {
                    "id": reaction_id,
                    "reaction": equation,
                    "rule": rules_by_rid.get(reaction_id),
                    "smiles": smiles_by_rid.get(reaction_id),
                }
            )

        molecules: List[Dict[str, Any]] = []
        for compound_id in sorted(all_compound_ids):
            record = molecules_by_cid.get(
                compound_id,
                {"id": compound_id, "name": None, "smiles": None},
            )
            molecules.append(
                {
                    "id": compound_id,
                    "name": record.get("name"),
                    "smiles": record.get("smiles"),
                }
            )

        return {"reactions": reactions, "molecules": molecules}

    def build_module_json(
        self,
        module_id: str,
        *,
        with_compounds: bool = True,
        with_atom_maps: bool = True,
        save_as: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a JSON block for a KEGG module.

        :param module_id:
            KEGG module ID.
        :type module_id: str

        :param with_compounds:
            Whether to resolve compound names/MOL/SMILES.
        :type with_compounds: bool

        :param with_atom_maps:
            Whether to compute atom-mapped reaction rules.
        :type with_atom_maps: bool

        :param save_as:
            Optional JSON output path.
        :type save_as: Optional[str]

        :returns:
            Module JSON dictionary.
        :rtype: Dict[str, Any]
        """
        equations_by_rid = self.get_module_equations(module_id)
        compound_ids, parsed_by_rid = collect_compound_ids_from_equations(
            equations_by_rid
        )

        compounds_by_cid = (
            self.build_compound_table(compound_ids) if with_compounds else {}
        )

        reaction_smiles_by_rid, _ = (
            self.build_reaction_smiles_dict(parsed_by_rid, compounds_by_cid)
            if with_compounds
            else ({}, {})
        )

        rules_by_rid = (
            self.atom_map_reactions(reaction_smiles_by_rid)
            if (with_compounds and with_atom_maps)
            else {}
        )

        data = {"module_id": module_id}
        data.update(
            self.build_kegg_json(
                equations_by_rid,
                smiles_by_rid=reaction_smiles_by_rid,
                rules_by_rid=rules_by_rid,
                molecules_by_cid={
                    cid: {
                        "id": cid,
                        "name": compounds_by_cid[cid]["name"],
                        "smiles": compounds_by_cid[cid]["smiles"],
                    }
                    for cid in compounds_by_cid
                },
            )
        )

        if with_compounds:
            data["missing"] = self.build_missing_compound_report(
                equations_by_rid,
                compounds_by_cid,
            )
        else:
            data["missing"] = {
                "missing_compounds": [],
                "missing_compound_ids": [],
                "reactions_involving_missing": [],
            }

        if save_as:
            with open(save_as, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)

        return data

    def build_pathway_json(
        self,
        pathway_id: str,
        *,
        with_compounds: bool = True,
        with_atom_maps: bool = True,
        save_as: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a JSON block for a KEGG pathway, organized by module.

        :param pathway_id:
            KEGG pathway ID.
        :type pathway_id: str

        :param with_compounds:
            Whether to resolve compound records.
        :type with_compounds: bool

        :param with_atom_maps:
            Whether to compute atom-mapped rules.
        :type with_atom_maps: bool

        :param save_as:
            Optional JSON output path.
        :type save_as: Optional[str]

        :returns:
            Pathway JSON dictionary.
        :rtype: Dict[str, Any]
        """
        modules = self.get_modules_from_pathway(pathway_id)
        by_module: Dict[str, Any] = {}

        aggregate_missing_ids: Set[str] = set()
        aggregate_reaction_ids: Set[str] = set()

        for module_id in modules:
            module_block = self.build_module_json(
                module_id,
                with_compounds=with_compounds,
                with_atom_maps=with_atom_maps,
            )
            by_module[module_id] = module_block

            if with_compounds and "missing" in module_block:
                aggregate_missing_ids.update(
                    module_block["missing"].get("missing_compound_ids", [])
                )
                aggregate_reaction_ids.update(
                    module_block["missing"].get("reactions_involving_missing", [])
                )

        data = {
            "pathway_id": pathway_id,
            "modules": modules,
            "by_module": by_module,
        }

        if with_compounds:
            data["missing"] = {
                "missing_compound_ids": sorted(aggregate_missing_ids),
                "reactions_involving_missing": sorted(aggregate_reaction_ids),
            }

        if save_as:
            with open(save_as, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)

        return data