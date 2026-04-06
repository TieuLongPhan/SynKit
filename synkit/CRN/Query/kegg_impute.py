from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .kegg_extract import KEGGExtractor
from .kegg_parse import parse_equation, reaction_smiles_from_equation

MoleculeRecord = dict[str, Any]
ReactionRecord = dict[str, Any]
FixRecord = dict[str, str]


@dataclass
class KEGGImputer:
    """
    Impute missing compound SMILES and repair reaction records in KEGG-style
    module or pathway JSON blocks.

    The imputer supports two fix types through the same ``fixes`` argument:

    - molecule fixes, for example
      ``{"id": "C00404a", "smiles": "..."}``
    - reaction fixes, for example
      ``{"id": "R02189", "reaction": "C00404 + C00267 <=> C00404a + C00668"}``

    Reaction fixes are applied first, then molecule fixes are applied, and
    finally impacted reaction SMILES, atom-mapped rules, and missing-compound
    summaries are rebuilt.

    :param extractor:
        Optional high-level KEGG extractor used for atom-mapping utilities and
        missing-compound report generation. When omitted, a default
        :class:`KEGGExtractor` instance is created.
    :type extractor: Optional[KEGGExtractor]

    Example
    -------
    .. code-block:: python

        imputer = KEGGImputer()
        updated = imputer.impute_module(
            module_data,
            fixes=[
                {
                    "id": "R02189",
                    "reaction": "C00404 + C00267 <=> C00404a + C00668",
                },
                {
                    "id": "C00404a",
                    "name": "Polyphosphate fragment",
                    "smiles": "O=P(O)(O)OP(=O)(O)O",
                },
            ],
        )
    """

    extractor: Optional[KEGGExtractor] = None

    def __post_init__(self) -> None:
        if self.extractor is None:
            self.extractor = KEGGExtractor()

    @staticmethod
    def _restore_molecule_list(
        original_molecules: List[MoleculeRecord],
        molecules_by_id: Dict[str, MoleculeRecord],
        *,
        id_key: str = "id",
    ) -> List[MoleculeRecord]:
        """
        Restore a molecule list while preserving original order.

        Molecules present in ``molecules_by_id`` but absent from the original
        list are appended in sorted identifier order.

        :param original_molecules:
            Original molecule records.
        :type original_molecules: List[dict[str, Any]]
        :param molecules_by_id:
            Updated molecule mapping keyed by identifier.
        :type molecules_by_id: Dict[str, dict[str, Any]]
        :param id_key:
            Dictionary key used as the molecule identifier.
        :type id_key: str

        :returns:
            Restored molecule list.
        :rtype: List[dict[str, Any]]

        Example
        -------
        .. code-block:: python

            restored = KEGGImputer._restore_molecule_list(
                [{"id": "C00001"}],
                {"C00001": {"id": "C00001"}, "C00002": {"id": "C00002"}},
            )
        """
        original_ids = [
            record[id_key] for record in original_molecules if id_key in record
        ]
        restored: List[MoleculeRecord] = []

        for molecule_id in original_ids:
            if molecule_id in molecules_by_id:
                restored.append(molecules_by_id[molecule_id])

        original_id_set = set(original_ids)
        for molecule_id in sorted(molecules_by_id):
            if molecule_id not in original_id_set:
                restored.append(molecules_by_id[molecule_id])

        return restored

    @staticmethod
    def _split_fixes(
        fixes: List[FixRecord],
        *,
        reaction_id_key: str,
        equation_key: str,
    ) -> Tuple[List[FixRecord], List[FixRecord]]:
        """
        Split mixed fix records into reaction fixes and molecule fixes.

        A fix is treated as a reaction fix when it contains both the reaction
        identifier key and the equation key. All other fixes are treated as
        molecule fixes.

        :param fixes:
            Mixed fix records.
        :type fixes: List[dict[str, str]]
        :param reaction_id_key:
            Key used to identify reactions.
        :type reaction_id_key: str
        :param equation_key:
            Key used to store reaction equations.
        :type equation_key: str

        :returns:
            Tuple ``(reaction_fixes, molecule_fixes)``.
        :rtype: Tuple[List[dict[str, str]], List[dict[str, str]]]

        Example
        -------
        .. code-block:: python

            reaction_fixes, molecule_fixes = KEGGImputer._split_fixes(
                fixes,
                reaction_id_key="id",
                equation_key="reaction",
            )
        """
        reaction_fixes: List[FixRecord] = []
        molecule_fixes: List[FixRecord] = []

        for fix in fixes:
            if reaction_id_key in fix and equation_key in fix:
                reaction_fixes.append(fix)
            else:
                molecule_fixes.append(fix)

        return reaction_fixes, molecule_fixes

    @staticmethod
    def _apply_reaction_fixes(
        reactions: List[ReactionRecord],
        reaction_fixes: List[FixRecord],
        *,
        reaction_id_key: str,
        equation_key: str,
        reaction_smiles_key: str = "smiles",
        reaction_rule_key: str = "rule",
    ) -> Set[str]:
        """
        Apply reaction equation fixes in place.

        The target reaction equation is replaced for each matching reaction
        record. Existing SMILES and rule fields for edited reactions are reset
        so they can be rebuilt from the updated equation.

        :param reactions:
            Reaction records to update in place.
        :type reactions: List[dict[str, Any]]
        :param reaction_fixes:
            Reaction-level fix records.
        :type reaction_fixes: List[dict[str, str]]
        :param reaction_id_key:
            Dictionary key storing reaction identifiers.
        :type reaction_id_key: str
        :param equation_key:
            Dictionary key storing equation strings.
        :type equation_key: str
        :param reaction_smiles_key:
            Dictionary key used for reaction SMILES.
        :type reaction_smiles_key: str
        :param reaction_rule_key:
            Dictionary key used for atom-mapped rules.
        :type reaction_rule_key: str

        :returns:
            Set of edited reaction identifiers.
        :rtype: Set[str]

        :raises KeyError:
            If a reaction fix refers to a reaction identifier that is not
            present in the provided reaction list.

        Example
        -------
        .. code-block:: python

            edited = KEGGImputer._apply_reaction_fixes(
                reactions,
                [{"id": "R00001", "reaction": "C00001 => C00002"}],
                reaction_id_key="id",
                equation_key="reaction",
            )
        """
        reactions_by_id = {
            reaction.get(reaction_id_key): reaction
            for reaction in reactions
            if reaction.get(reaction_id_key)
        }

        edited_reaction_ids: Set[str] = set()

        for fix in reaction_fixes:
            reaction_id = fix.get(reaction_id_key)
            if not reaction_id:
                continue
            if reaction_id not in reactions_by_id:
                continue

            reaction = reactions_by_id[reaction_id]
            reaction[equation_key] = fix[equation_key]
            reaction[reaction_smiles_key] = None
            reaction[reaction_rule_key] = None
            edited_reaction_ids.add(reaction_id)

        return edited_reaction_ids

    @staticmethod
    def _apply_molecule_fixes(
        molecules_by_id: Dict[str, MoleculeRecord],
        molecule_fixes: List[FixRecord],
        *,
        molecule_id_key: str = "id",
    ) -> Set[str]:
        """
        Apply molecule fixes to a molecule mapping in place.

        Existing molecule records are updated, while missing identifiers are
        inserted as new molecule records.

        :param molecules_by_id:
            Molecule mapping keyed by identifier.
        :type molecules_by_id: Dict[str, dict[str, Any]]
        :param molecule_fixes:
            Molecule-level fix records.
        :type molecule_fixes: List[dict[str, str]]
        :param molecule_id_key:
            Dictionary key used as the molecule identifier.
        :type molecule_id_key: str

        :returns:
            Set of updated compound identifiers.
        :rtype: Set[str]

        Example
        -------
        .. code-block:: python

            updated_ids = KEGGImputer._apply_molecule_fixes(
                molecules_by_id,
                [{"id": "C00002", "smiles": "O"}],
            )
        """
        updated_compound_ids: Set[str] = set()

        for fix in molecule_fixes:
            compound_id = fix.get(molecule_id_key)
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
                if "smiles" in fix:
                    molecules_by_id[compound_id]["smiles"] = fix.get("smiles")

        return updated_compound_ids

    @staticmethod
    def _infer_impacted_reaction_ids(
        reactions: List[Dict[str, Any]],
        updated_compound_ids: Set[str],
        *,
        reaction_id_key: str,
        equation_key: str,
    ) -> Set[str]:
        """
        Infer which reactions are affected by updated compound identifiers.

        The method first consults the existing ``missing`` block. If that block
        already records which reactions involve the updated compounds, those
        reaction identifiers are reused directly. Otherwise, the method falls
        back to scanning reaction equation strings.

        :param reactions:
            Reaction records from a module block.
        :type reactions: List[Dict[str, Any]]
        :param updated_compound_ids:
            Compound identifiers whose records were changed by imputation.
        :type updated_compound_ids: Set[str]
        :param reaction_id_key:
            Dictionary key storing reaction identifiers.
        :type reaction_id_key: str
        :param equation_key:
            Dictionary key storing equation text.
        :type equation_key: str

        :returns:
            Set of impacted reaction identifiers.
        :rtype: Set[str]

        Example
        -------
        .. code-block:: python

            impacted = KEGGImputer._infer_impacted_reaction_ids(
                reactions,
                {"C00138"},
                reaction_id_key="id",
                equation_key="reaction",
            )
        """
        impacted: Set[str] = set()

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
        reaction_id_key: str = "id",
        equation_key: str = "reaction",
        reaction_smiles_key: str = "smiles",
        reaction_rule_key: str = "rule",
        molecule_id_key: str = "id",
    ) -> None:
        """
        Recompute reaction SMILES and mapped rules in place for impacted reactions.

        Only reactions listed in ``impacted_reaction_ids`` are rebuilt. Reaction
        SMILES are regenerated from the updated molecule table, then optional
        atom-mapped rules are refreshed via
        :meth:`KEGGExtractor.atom_map_reactions`.

        :param reactions:
            Reaction records to update in place.
        :type reactions: List[Dict[str, Any]]
        :param molecules:
            Molecule records containing current SMILES values.
        :type molecules: List[Dict[str, Any]]
        :param impacted_reaction_ids:
            Reaction identifiers requiring recomputation.
        :type impacted_reaction_ids: Set[str]
        :param reaction_id_key:
            Dictionary key storing reaction identifiers.
        :type reaction_id_key: str
        :param equation_key:
            Dictionary key storing reaction equation strings.
        :type equation_key: str
        :param reaction_smiles_key:
            Dictionary key used for reaction SMILES.
        :type reaction_smiles_key: str
        :param reaction_rule_key:
            Dictionary key used for atom-mapped reaction rules.
        :type reaction_rule_key: str
        :param molecule_id_key:
            Dictionary key storing molecule identifiers.
        :type molecule_id_key: str

        :returns:
            ``None``.
        :rtype: None

        Example
        -------
        .. code-block:: python

            imputer._rebuild_reaction_fields(
                reactions,
                molecules,
                {"R00001"},
                reaction_id_key="id",
                equation_key="reaction",
            )
        """
        if not impacted_reaction_ids:
            return

        molecules_by_id = {
            molecule[molecule_id_key]: molecule
            for molecule in molecules
            if molecule_id_key in molecule
        }
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

            parsed_equation = parse_equation(equation)
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
        Apply molecule and reaction fixes to a module JSON block.

        Reaction fixes are applied first, then molecule fixes are applied, then
        impacted reaction SMILES and atom-mapped rules are rebuilt, and finally
        the ``missing`` block is regenerated by delegating to
        :meth:`KEGGExtractor.build_missing_compound_report`.

        :param module_data:
            Module JSON dictionary.
        :type module_data: Dict[str, Any]
        :param fixes:
            List of mixed fix records. Reaction fixes must contain the reaction
            identifier key and the equation key. Molecule fixes use molecule
            identifiers and optional ``"name"`` / ``"smiles"`` fields.
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

        Example
        -------
        .. code-block:: python

            updated = imputer.impute_module(
                module_data,
                fixes=[
                    {
                        "id": "R02189",
                        "reaction": "C00404 + C00267 <=> C00404a + C00668",
                    },
                    {
                        "id": "C00404a",
                        "name": "Polyphosphate fragment",
                        "smiles": "O=P(O)(O)OP(=O)(O)O",
                    },
                ],
            )
        """
        new_data = copy.deepcopy(module_data)

        molecules = new_data.get("molecules", []) or []
        reactions = new_data.get("reactions", []) or []

        reaction_fixes, molecule_fixes = self._split_fixes(
            fixes,
            reaction_id_key=reaction_id_key,
            equation_key=equation_key,
        )

        edited_reaction_ids = self._apply_reaction_fixes(
            reactions,
            reaction_fixes,
            reaction_id_key=reaction_id_key,
            equation_key=equation_key,
            reaction_smiles_key=reaction_smiles_key,
            reaction_rule_key=reaction_rule_key,
        )

        molecules_by_id = {
            molecule[molecule_id_key]: molecule
            for molecule in molecules
            if molecule_id_key in molecule
        }

        updated_compound_ids = self._apply_molecule_fixes(
            molecules_by_id,
            molecule_fixes,
            molecule_id_key=molecule_id_key,
        )

        new_data["molecules"] = self._restore_molecule_list(
            molecules,
            molecules_by_id,
            id_key=molecule_id_key,
        )

        impacted_reaction_ids = set(edited_reaction_ids)
        impacted_reaction_ids.update(
            self._infer_impacted_reaction_ids(
                reactions,
                updated_compound_ids,
                reaction_id_key=reaction_id_key,
                equation_key=equation_key,
            )
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

        equations_by_rid = {
            reaction[reaction_id_key]: reaction.get(equation_key)
            for reaction in reactions
            if reaction.get(reaction_id_key)
        }
        compounds_by_cid = {
            molecule[molecule_id_key]: {
                "id": molecule.get(molecule_id_key),
                "name": molecule.get("name"),
                "smiles": molecule.get("smiles"),
            }
            for molecule in new_data["molecules"]
            if molecule.get(molecule_id_key)
        }

        new_data["missing"] = self.extractor.build_missing_compound_report(
            equations_by_rid,
            compounds_by_cid,
        )

        self.extractor.save_json(new_data, save_as)

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
        Apply molecule and reaction fixes across all modules in a pathway JSON
        block.

        Each module is processed independently through :meth:`impute_module`,
        then the pathway-level ``missing`` summary is rebuilt by aggregating the
        updated module summaries.

        :param pathway_data:
            Pathway JSON dictionary with ``"by_module"``.
        :type pathway_data: Dict[str, Any]
        :param fixes:
            Mixed reaction and molecule fix records.
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
            Updated pathway JSON dictionary.
        :rtype: Dict[str, Any]

        Example
        -------
        .. code-block:: python

            updated = imputer.impute_pathway(
                pathway_data,
                fixes=[
                    {
                        "id": "R02189",
                        "reaction": "C00404 + C00267 <=> C00404a + C00668",
                    },
                    {
                        "id": "C00404a",
                        "name": "Polyphosphate fragment",
                        "smiles": "O=P(O)(O)OP(=O)(O)O",
                    },
                ],
            )
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

        for block in by_module.values():
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

        self.extractor.save_json(new_pathway, save_as)

        return new_pathway
