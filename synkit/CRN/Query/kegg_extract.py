from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional, Mapping

try:
    from rxnmapper import RXNMapper as _RXNMapper
except Exception:
    _RXNMapper = None

from .kegg_api import KEGGClient
from .kegg_parse import (
    equation_to_text,
    get_compound_ids_from_equations,
    get_compound_ids_from_text,
    molblock_to_smiles,
    normalize_module_id,
    orient_equation_to_module,
    parse_equation,
    parse_kegg_field_blocks,
    parse_module_reaction_directions,
    reaction_smiles_from_equation,
)

_RID_PATTERN = re.compile(r"R\d{5}")


ReactionEquationMap = dict[str, Optional[str]]
CompoundTable = dict[str, dict[str, Any]]
ReactionSmilesMap = dict[str, str]
MissingByReaction = dict[str, dict[str, list[str]]]
JSONDict = dict[str, Any]


@dataclass(slots=True)
class KEGGExtractor:
    """
    High-level extractor for KEGG pathway and module reaction data.

    This class orchestrates KEGG entry retrieval, module membership parsing,
    reaction equation collection, compound-table construction, reaction SMILES
    assembly, and optional atom mapping.

    :param client:
        Optional KEGG REST client. When omitted, a default :class:`KEGGClient`
        instance is created during :meth:`__post_init__`.
    :type client: Optional[KEGGClient]
    :param mapper_cls:
        Optional atom-mapper class used by :meth:`atom_map_reactions`. The
        class must be instantiable without arguments and must provide
        ``get_attention_guided_atom_maps``.
    :type mapper_cls: Optional[type[Any]]

    Example
    -------
    .. code-block:: python

        extractor = KEGGExtractor()
        data = extractor.build_module_json(
            "M00001",
            with_compounds=True,
            with_atom_maps=False,
        )
    """

    client: Optional[KEGGClient] = None
    mapper_cls: Optional[type[Any]] = _RXNMapper

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = KEGGClient()

    @staticmethod
    def save_json(data: Mapping[str, Any], save_as: Optional[str]) -> None:
        """
        Save JSON data to disk when an output path is provided.

        :param data:
            JSON-serializable data to write.
        :type data: Mapping[str, Any]
        :param save_as:
            Optional output path.
        :type save_as: Optional[str]

        :returns:
            ``None``.
        :rtype: None

        Example
        -------
        .. code-block:: python

            KEGGExtractor._save_json({"x": 1}, "out.json")
        """
        if save_as:
            with open(save_as, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)

    def get_modules_from_pathway(self, pathway_id: str) -> list[str]:
        """
        Extract module IDs from a KEGG pathway entry.

        :param pathway_id:
            KEGG pathway identifier such as ``"hsa00010"``.
        :type pathway_id: str

        :returns:
            Canonical KEGG module identifiers such as ``["M00001",
            "M00002"]``.
        :rtype: list[str]

        Example
        -------
        .. code-block:: python

            modules = extractor.get_modules_from_pathway("hsa00010")
        """
        text = self.client.get_text(f"get/{pathway_id}")
        payloads = parse_kegg_field_blocks(text, "MODULE")

        modules: list[str] = []
        for payload in payloads:
            for token in payload.split():
                normalized = normalize_module_id(token)
                if normalized is not None:
                    modules.append(normalized)

        return modules

    def get_reaction_ids_from_module(self, module_id: str) -> list[str]:
        """
        Collect KEGG reaction IDs from a module entry, preserving module order when
        directional REACTION lines can be parsed.

        :param module_id:
            KEGG module identifier such as ``"M00001"``.
        :type module_id: str

        :returns:
            Sorted unique KEGG reaction identifiers.
        :rtype: list[str]

        Example
        -------
        .. code-block:: python

            reaction_ids = extractor.get_reaction_ids_from_module("M00001")
        """
        text = self.client.get_text(f"get/{module_id}")
        directions = parse_module_reaction_directions(text)
        if directions:
            return list(directions.keys())

        payloads = parse_kegg_field_blocks(text, "REACTION")
        reaction_ids: set[str] = set()
        for payload in payloads:
            reaction_ids.update(_RID_PATTERN.findall(payload))

        return sorted(reaction_ids)

    def get_equation_for_reaction(self, reaction_id: str) -> Optional[str]:
        """
        Fetch the KEGG equation string for a reaction.

        :param reaction_id:
            KEGG reaction identifier such as ``"R00200"``.
        :type reaction_id: str

        :returns:
            Equation string when present, otherwise ``None``.
        :rtype: Optional[str]

        Example
        -------
        .. code-block:: python

            equation = extractor.get_equation_for_reaction("R00200")
        """
        text = self.client.get_text(f"get/rn:{reaction_id}")
        payloads = parse_kegg_field_blocks(text, "EQUATION")
        return payloads[0].strip() if payloads else None

    def get_module_equations(self, module_id: str) -> ReactionEquationMap:
        """
        Build a reaction-to-equation mapping for a KEGG module.

        :param module_id:
            KEGG module identifier.
        :type module_id: str

        :returns:
            Mapping from reaction identifier to equation string.
        :rtype: dict[str, Optional[str]]

        Example
        -------
        .. code-block:: python

            equations = extractor.get_module_equations("M00001")
        """
        module_text = self.client.get_text(f"get/{module_id}")
        directions = parse_module_reaction_directions(module_text)

        reaction_ids = list(directions.keys()) if directions else self.get_reaction_ids_from_module(module_id)

        equations_by_rid: ReactionEquationMap = {}

        for reaction_id in reaction_ids:
            equation = self.get_equation_for_reaction(reaction_id)
            if equation is None:
                equations_by_rid[reaction_id] = None
                continue

            if reaction_id not in directions:
                equations_by_rid[reaction_id] = equation
                continue

            left_ids, right_ids, module_arrow = directions[reaction_id]
            parsed = parse_equation(equation)
            oriented = orient_equation_to_module(parsed, left_ids, right_ids)
            equations_by_rid[reaction_id] = equation_to_text(oriented, arrow=module_arrow)

        return equations_by_rid

    def get_pathway_equations(
        self,
        pathway_id: str,
    ) -> dict[str, ReactionEquationMap]:
        """
        Build nested module/reaction equation mappings for a pathway.

        :param pathway_id:
            KEGG pathway identifier.
        :type pathway_id: str

        :returns:
            Mapping of the form ``{module_id: {reaction_id: equation}}``.
        :rtype: dict[str, dict[str, Optional[str]]]

        Example
        -------
        .. code-block:: python

            nested = extractor.get_pathway_equations("hsa00010")
        """
        modules = self.get_modules_from_pathway(pathway_id)
        return {
            module_id: self.get_module_equations(module_id) for module_id in modules
        }

    def get_compound_name(self, compound_id: str) -> Optional[str]:
        """
        Retrieve the primary KEGG compound name.

        When multiple synonyms are present in the ``NAME`` field, only the
        first entry is returned.

        :param compound_id:
            KEGG compound identifier such as ``"C00001"``.
        :type compound_id: str

        :returns:
            Primary compound name if available, otherwise ``None``.
        :rtype: Optional[str]

        Example
        -------
        .. code-block:: python

            name = extractor.get_compound_name("C00001")
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
            KEGG compound identifier.
        :type compound_id: str

        :returns:
            MOL block text when available, otherwise ``None``.
        :rtype: Optional[str]

        Example
        -------
        .. code-block:: python

            molblock = extractor.get_compound_molblock("C00001")
        """
        return self.client.get_optional_text(f"get/cpd:{compound_id}/mol")

    def build_compound_table(
        self,
        compound_ids: list[str],
    ) -> CompoundTable:
        """
        Build a compound table for a list of KEGG compound identifiers.

        Each returned record includes the KEGG compound identifier, the primary
        compound name, the optional MOL block, and a canonical SMILES string
        derived from the MOL block when RDKit parsing succeeds.

        :param compound_ids:
            KEGG compound identifiers.
        :type compound_ids: list[str]

        :returns:
            Compound table of the form
            ``{cid: {"id", "name", "smiles", "molblock"}}``.
        :rtype: dict[str, dict[str, Any]]

        Example
        -------
        .. code-block:: python

            compounds = extractor.build_compound_table(["C00001", "C00002"])
        """
        compounds: CompoundTable = {}

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

    def build_reaction_smiles_dict(
        self,
        parsed_by_rid: Mapping[str, Any],
        compounds_by_cid: Mapping[str, Mapping[str, Any]],
    ) -> tuple[ReactionSmilesMap, MissingByReaction]:
        """
        Build reaction SMILES strings for parsed KEGG equations.

        :param parsed_by_rid:
            Parsed equation objects keyed by reaction identifier.
        :type parsed_by_rid: Mapping[str, Any]
        :param compounds_by_cid:
            Compound table keyed by KEGG compound identifier.
        :type compounds_by_cid: Mapping[str, Mapping[str, Any]]

        :returns:
            Tuple ``(reaction_smiles_by_id, missing_by_id)``.
        :rtype: tuple[dict[str, str], dict[str, dict[str, list[str]]]]

        Example
        -------
        .. code-block:: python

            rsmi_by_rid, missing = extractor.build_reaction_smiles_dict(
                parsed_by_rid,
                compounds_by_cid,
            )
        """
        reaction_smiles: ReactionSmilesMap = {}
        missing_by_rid: MissingByReaction = {}

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
        reaction_smiles_by_id: Mapping[str, str],
    ) -> dict[str, Optional[str]]:
        """
        Atom-map reaction SMILES using RXNMapper.

        :param reaction_smiles_by_id:
            Mapping ``{reaction_id: reaction_smiles}``.
        :type reaction_smiles_by_id: Mapping[str, str]

        :returns:
            Mapping ``{reaction_id: mapped_reaction_smiles_or_none}``.
        :rtype: dict[str, Optional[str]]

        Example
        -------
        .. code-block:: python

            mapped = extractor.atom_map_reactions({"R00001": "CCO>>CC=O"})
        """
        mapper = self.mapper_cls() if self.mapper_cls is not None else None
        mapped_by_id: dict[str, Optional[str]] = {}

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

    def build_missing_compound_report(
        self,
        equations_by_rid: ReactionEquationMap,
        compounds_by_cid: Mapping[str, Mapping[str, Any]],
    ) -> JSONDict:
        """
        Build a report for compounds lacking SMILES.

        :param equations_by_rid:
            Reaction equations keyed by reaction identifier.
        :type equations_by_rid: dict[str, Optional[str]]
        :param compounds_by_cid:
            Compound records keyed by KEGG compound identifier.
        :type compounds_by_cid: Mapping[str, Mapping[str, Any]]

        :returns:
            Report containing missing compounds and per-reaction provenance.
        :rtype: dict[str, Any]

        Example
        -------
        .. code-block:: python

            report = extractor.build_missing_compound_report(
                equations_by_rid,
                compounds_by_cid,
            )
        """
        cid_to_rids: dict[str, set[str]] = {}

        for reaction_id, equation in equations_by_rid.items():
            if not equation:
                continue

            for compound_id in get_compound_ids_from_text(equation):
                cid_to_rids.setdefault(compound_id, set()).add(reaction_id)

        missing_compounds: list[dict[str, Any]] = []
        missing_ids: set[str] = set()
        involving_reactions: set[str] = set()

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

    def build_kegg_json(
        self,
        equations_by_rid: ReactionEquationMap,
        *,
        smiles_by_rid: Optional[Mapping[str, str]] = None,
        rules_by_rid: Optional[Mapping[str, Optional[str]]] = None,
        molecules_by_cid: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> JSONDict:
        """
        Build a compact KEGG JSON block with reactions and molecules.

        :param equations_by_rid:
            Reaction equations keyed by reaction identifier.
        :type equations_by_rid: dict[str, Optional[str]]
        :param smiles_by_rid:
            Optional reaction SMILES keyed by reaction identifier.
        :type smiles_by_rid: Optional[Mapping[str, str]]
        :param rules_by_rid:
            Optional atom-mapped reaction strings keyed by reaction identifier.
        :type rules_by_rid: Optional[Mapping[str, Optional[str]]]
        :param molecules_by_cid:
            Optional molecule table keyed by compound identifier.
        :type molecules_by_cid: Optional[Mapping[str, Mapping[str, Any]]]

        :returns:
            Dictionary with ``"reactions"`` and ``"molecules"`` entries.
        :rtype: dict[str, Any]

        Example
        -------
        .. code-block:: python

            data = extractor.build_kegg_json(equations_by_rid)
        """
        smiles_by_rid = dict(smiles_by_rid or {})
        rules_by_rid = dict(rules_by_rid or {})
        molecules_by_cid = dict(molecules_by_cid or {})

        all_compound_ids: set[str] = set()
        reactions: list[dict[str, Any]] = []

        for reaction_id in sorted(equations_by_rid.keys()):
            equation = equations_by_rid[reaction_id]
            if not equation:
                continue

            all_compound_ids.update(get_compound_ids_from_text(equation))
            reactions.append(
                {
                    "id": reaction_id,
                    "reaction": equation,
                    "rule": rules_by_rid.get(reaction_id),
                    "smiles": smiles_by_rid.get(reaction_id),
                }
            )

        molecules: list[dict[str, Any]] = []
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
    ) -> JSONDict:
        """
        Build a JSON block for a KEGG module.

        :param module_id:
            KEGG module ID.
        :type module_id: str
        :param with_compounds:
            Whether to resolve compound names, MOL blocks, and SMILES strings.
        :type with_compounds: bool
        :param with_atom_maps:
            Whether to compute atom-mapped reactions.
        :type with_atom_maps: bool
        :param save_as:
            Optional output path for writing the JSON block to disk.
        :type save_as: Optional[str]

        :returns:
            Module JSON dictionary.
        :rtype: dict[str, Any]

        Example
        -------
        .. code-block:: python

            data = extractor.build_module_json(
                "M00001",
                with_compounds=True,
                with_atom_maps=False,
            )
        """
        equations_by_rid = self.get_module_equations(module_id)
        compound_ids, parsed_by_rid = get_compound_ids_from_equations(equations_by_rid)

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

        data: JSONDict = {"module_id": module_id}
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

        self.save_json(data, save_as)

        return data

    def build_pathway_json(
        self,
        pathway_id: str,
        *,
        with_compounds: bool = True,
        with_atom_maps: bool = True,
        save_as: Optional[str] = None,
    ) -> JSONDict:
        """
        Build a JSON block for a KEGG pathway, organized by module.

        :param pathway_id:
            KEGG pathway ID.
        :type pathway_id: str
        :param with_compounds:
            Whether to resolve compound records.
        :type with_compounds: bool
        :param with_atom_maps:
            Whether to compute atom-mapped reactions.
        :type with_atom_maps: bool
        :param save_as:
            Optional output path for writing the JSON block to disk.
        :type save_as: Optional[str]

        :returns:
            Pathway JSON dictionary.
        :rtype: dict[str, Any]

        Example
        -------
        .. code-block:: python

            data = extractor.build_pathway_json(
                "hsa00010",
                with_compounds=True,
                with_atom_maps=False,
            )
        """
        modules = self.get_modules_from_pathway(pathway_id)
        by_module: dict[str, Any] = {}

        aggregate_missing_ids: set[str] = set()
        aggregate_reaction_ids: set[str] = set()

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

        data: JSONDict = {
            "pathway_id": pathway_id,
            "modules": modules,
            "by_module": by_module,
        }

        if with_compounds:
            data["missing"] = {
                "missing_compound_ids": sorted(aggregate_missing_ids),
                "reactions_involving_missing": sorted(aggregate_reaction_ids),
            }

        self.save_json(data, save_as)

        return data
