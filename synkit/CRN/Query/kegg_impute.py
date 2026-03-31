from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from .kegg_extract import KEGGExtractor
from .kegg_parse import parse_kegg_equation, reaction_smiles_from_equation


_CID_PATTERN = re.compile(r"C\d{5}")


@dataclass
class KEGGImputer:
    """
    Impute missing compound SMILES inside KEGG-style module/pathway JSON blocks,
    then rebuild reaction SMILES and atom-mapped rules.

    :param extractor:
        Optional high-level KEGG extractor used for atom mapping utilities.
    :type extractor: Optional[KEGGExtractor]
    """

    extractor: Optional[KEGGExtractor] = None

    def __post_init__(self) -> None:
        if self.extractor is None:
            self.extractor = KEGGExtractor()

    @staticmethod
    def _build_molecule_map(
        molecules: List[Dict[str, Any]],
        *,
        id_key: str = "id",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Convert molecule list to a dictionary keyed by molecule ID.
        """
        return {record[id_key]: record for record in molecules if id_key in record}

    @staticmethod
    def _restore_molecule_list(
        original_molecules: List[Dict[str, Any]],
        molecules_by_id: Dict[str, Dict[str, Any]],
        *,
        id_key: str = "id",
    ) -> List[Dict[str, Any]]:
        """
        Restore a molecule list while preserving original order and appending new IDs.
        """
        original_ids = [
            record[id_key] for record in original_molecules if id_key in record
        ]
        restored: List[Dict[str, Any]] = []

        for molecule_id in original_ids:
            if molecule_id in molecules_by_id:
                restored.append(molecules_by_id[molecule_id])

        original_id_set = set(original_ids)
        for molecule_id in sorted(molecules_by_id):
            if molecule_id not in original_id_set:
                restored.append(molecules_by_id[molecule_id])

        return restored

    @staticmethod
    def _build_cid_to_reactions(
        reactions: List[Dict[str, Any]],
        *,
        reaction_id_key: str,
        equation_key: str,
    ) -> Dict[str, Set[str]]:
        """
        Build ``compound_id -> reaction_ids`` mapping from equation text.
        """
        cid_to_rids: Dict[str, Set[str]] = {}

        for reaction in reactions:
            reaction_id = reaction.get(reaction_id_key)
            equation = reaction.get(equation_key)
            if not reaction_id or not equation:
                continue

            for compound_id in set(_CID_PATTERN.findall(equation)):
                cid_to_rids.setdefault(compound_id, set()).add(reaction_id)

        return cid_to_rids

    def _recompute_missing_block(
        self,
        reactions: List[Dict[str, Any]],
        molecules: List[Dict[str, Any]],
        *,
        reaction_id_key: str,
        equation_key: str,
        molecule_id_key: str = "id",
    ) -> Dict[str, Any]:
        """
        Rebuild the ``missing`` block from molecule SMILES and equation strings.
        """
        molecules_by_id = self._build_molecule_map(molecules, id_key=molecule_id_key)
        cid_to_rids = self._build_cid_to_reactions(
            reactions,
            reaction_id_key=reaction_id_key,
            equation_key=equation_key,
        )

        missing_compounds: List[Dict[str, Any]] = []
        missing_ids: List[str] = []
        involving_reactions: Set[str] = set()

        for compound_id, molecule in molecules_by_id.items():
            if molecule.get("smiles") is None and compound_id in cid_to_rids:
                reaction_ids = sorted(cid_to_rids[compound_id])
                missing_compounds.append(
                    {
                        "id": compound_id,
                        "name": molecule.get("name"),
                        "reactions": reaction_ids,
                    }
                )
                missing_ids.append(compound_id)
                involving_reactions.update(reaction_ids)

        missing_compounds.sort(key=lambda record: record["id"])
        missing_ids.sort()

        return {
            "missing_compounds": missing_compounds,
            "missing_compound_ids": missing_ids,
            "reactions_involving_missing": sorted(involving_reactions),
        }

    @staticmethod
    def _infer_impacted_reaction_ids(
        reactions: List[Dict[str, Any]],
        missing_block: Dict[str, Any],
        updated_compound_ids: Set[str],
        *,
        reaction_id_key: str,
        equation_key: str,
    ) -> Set[str]:
        """
        Infer which reactions are affected by updated compound IDs.
        """
        impacted: Set[str] = set()

        for record in (missing_block or {}).get("missing_compounds", []):
            if record.get("id") in updated_compound_ids:
                impacted.update(record.get("reactions", []))

        if impacted:
            return impacted

        for reaction in reactions:
            reaction_id = reaction.get(reaction_id_key)
            equation = reaction.get(equation_key, "") or ""
            if reaction_id and any(cid in equation for cid in updated_compound_ids):
                impacted.add(reaction_id)

        return impacted

    def _rebuild_reaction_fields(
        self,
        reactions: List[Dict[str, Any]],
        molecules: List[Dict[str, Any]],
        impacted_reaction_ids: Set[str],
        *,
        reaction_id_key: str,
        equation_key: str,
        reaction_smiles_key: str = "smiles",
        reaction_rule_key: str = "rule",
        molecule_id_key: str = "id",
    ) -> None:
        """
        Recompute reaction SMILES and mapped rules in place for impacted reactions.
        """
        if not impacted_reaction_ids:
            return

        molecules_by_id = self._build_molecule_map(molecules, id_key=molecule_id_key)
        compounds_by_cid = {
            compound_id: {"smiles": molecules_by_id[compound_id].get("smiles")}
            for compound_id in molecules_by_id
        }

        reactions_by_id = {
            reaction.get(reaction_id_key): reaction
            for reaction in reactions
            if reaction.get(reaction_id_key)
        }

        rebuilt_smiles: Dict[str, str] = {}

        for reaction_id in impacted_reaction_ids:
            reaction = reactions_by_id.get(reaction_id)
            if reaction is None:
                continue

            equation = reaction.get(equation_key)
            if not equation:
                continue

            parsed_equation = parse_kegg_equation(equation)
            rsmi, _ = reaction_smiles_from_equation(parsed_equation, compounds_by_cid)
            reaction[reaction_smiles_key] = rsmi
            rebuilt_smiles[reaction_id] = rsmi

        if rebuilt_smiles:
            mapped_rules = self.extractor.atom_map_reactions(rebuilt_smiles)
            for reaction_id, rule in mapped_rules.items():
                reaction = reactions_by_id.get(reaction_id)
                if reaction is not None:
                    reaction[reaction_rule_key] = rule

    def impute_module(
        self,
        module_data: Dict[str, Any],
        fixes: List[Dict[str, str]],
        save_as: Optional[str] = None,
        *,
        molecule_id_key: str = "id",
        reaction_id_key: str = "id",
        equation_key: str = "reaction",
        reaction_smiles_key: str = "smiles",
        reaction_rule_key: str = "rule",
    ) -> Dict[str, Any]:
        """
        Apply compound SMILES fixes to a module JSON block.

        :param module_data:
            Module JSON dictionary.
        :type module_data: Dict[str, Any]

        :param fixes:
            List of update records such as
            ``[{"id": "C00138", "name": "...", "smiles": "..."}, ...]``.
        :type fixes: List[Dict[str, str]]

        :param save_as:
            Optional output path.
        :type save_as: Optional[str]

        :param molecule_id_key:
            Molecule identifier key.
        :type molecule_id_key: str

        :param reaction_id_key:
            Reaction identifier key.
        :type reaction_id_key: str

        :param equation_key:
            Equation text key.
        :type equation_key: str

        :param reaction_smiles_key:
            Reaction SMILES field key.
        :type reaction_smiles_key: str

        :param reaction_rule_key:
            Atom-mapped rule field key.
        :type reaction_rule_key: str

        :returns:
            Updated module JSON dictionary.
        :rtype: Dict[str, Any]
        """
        new_data = copy.deepcopy(module_data)

        molecules = new_data.get("molecules", []) or []
        reactions = new_data.get("reactions", []) or []
        missing_block = new_data.get("missing", {}) or {}

        molecules_by_id = self._build_molecule_map(molecules, id_key=molecule_id_key)
        updated_compound_ids: Set[str] = set()

        for fix in fixes:
            compound_id = fix.get("id")
            if not compound_id:
                continue

            updated_compound_ids.add(compound_id)

            if compound_id not in molecules_by_id:
                molecules_by_id[compound_id] = {
                    molecule_id_key: compound_id,
                    "name": fix.get("name"),
                    "smiles": fix.get("smiles"),
                }
            else:
                if fix.get("name") is not None:
                    molecules_by_id[compound_id]["name"] = fix["name"]
                molecules_by_id[compound_id]["smiles"] = fix.get("smiles")

        new_data["molecules"] = self._restore_molecule_list(
            molecules,
            molecules_by_id,
            id_key=molecule_id_key,
        )

        impacted_reaction_ids = self._infer_impacted_reaction_ids(
            reactions,
            missing_block,
            updated_compound_ids,
            reaction_id_key=reaction_id_key,
            equation_key=equation_key,
        )

        self._rebuild_reaction_fields(
            reactions,
            new_data["molecules"],
            impacted_reaction_ids,
            reaction_id_key=reaction_id_key,
            equation_key=equation_key,
            reaction_smiles_key=reaction_smiles_key,
            reaction_rule_key=reaction_rule_key,
            molecule_id_key=molecule_id_key,
        )

        new_data["reactions"] = reactions
        new_data["missing"] = self._recompute_missing_block(
            reactions,
            new_data["molecules"],
            reaction_id_key=reaction_id_key,
            equation_key=equation_key,
            molecule_id_key=molecule_id_key,
        )

        if save_as:
            with open(save_as, "w", encoding="utf-8") as handle:
                json.dump(new_data, handle, ensure_ascii=False, indent=2)

        return new_data

    def impute_pathway(
        self,
        pathway_data: Dict[str, Any],
        fixes: List[Dict[str, str]],
        save_as: Optional[str] = None,
        *,
        molecule_id_key: str = "id",
        reaction_id_key: str = "id",
        equation_key: str = "reaction",
        reaction_smiles_key: str = "smiles",
        reaction_rule_key: str = "rule",
    ) -> Dict[str, Any]:
        """
        Apply compound SMILES fixes across all modules in a pathway JSON block.

        :param pathway_data:
            Pathway JSON dictionary with ``"by_module"``.
        :type pathway_data: Dict[str, Any]

        :param fixes:
            Compound fix records.
        :type fixes: List[Dict[str, str]]

        :param save_as:
            Optional output path.
        :type save_as: Optional[str]

        :returns:
            Updated pathway JSON dictionary.
        :rtype: Dict[str, Any]
        """
        new_pathway = copy.deepcopy(pathway_data)
        by_module = new_pathway.get("by_module", {}) or {}

        for module_id, block in list(by_module.items()):
            by_module[module_id] = self.impute_module(
                block,
                fixes,
                save_as=None,
                molecule_id_key=molecule_id_key,
                reaction_id_key=reaction_id_key,
                equation_key=equation_key,
                reaction_smiles_key=reaction_smiles_key,
                reaction_rule_key=reaction_rule_key,
            )

        missing_compound_ids: Set[str] = set()
        missing_reaction_ids: Set[str] = set()

        for _, block in by_module.items():
            missing = block.get("missing", {}) or {}
            missing_compound_ids.update(missing.get("missing_compound_ids", []) or [])
            missing_reaction_ids.update(
                missing.get("reactions_involving_missing", []) or []
            )

        new_pathway["by_module"] = by_module
        new_pathway["missing"] = {
            "missing_compound_ids": sorted(missing_compound_ids),
            "reactions_involving_missing": sorted(missing_reaction_ids),
        }

        if save_as:
            with open(save_as, "w", encoding="utf-8") as handle:
                json.dump(new_pathway, handle, ensure_ascii=False, indent=2)

        return new_pathway
