from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _split_reaction_smiles(reaction_smiles: str) -> Tuple[List[str], List[str]]:
    """
    Split reaction SMILES into reactant and product molecule lists.

    :param reaction_smiles:
        Reaction SMILES string of the form ``"A.B>>C.D"``.
    :type reaction_smiles: str

    :returns:
        Tuple ``(reactants, products)``.
    :rtype: Tuple[List[str], List[str]]

    :raises ValueError:
        If the string does not contain ``">>"``.
    """
    reaction_smiles = reaction_smiles.strip()
    if ">>" not in reaction_smiles:
        raise ValueError(
            f"Not a reaction SMILES string (missing '>>'): {reaction_smiles}"
        )

    left, right = reaction_smiles.split(">>", 1)
    left = left.strip()
    right = right.strip()

    reactants = [token.strip() for token in left.split(".") if token.strip()] if left else []
    products = [token.strip() for token in right.split(".") if token.strip()] if right else []

    return reactants, products


def _excel_label(index: int) -> str:
    """
    Convert a zero-based index to Excel-style labels.

    Examples:
        - ``0 -> "A"``
        - ``25 -> "Z"``
        - ``26 -> "AA"``

    :param index:
        Zero-based integer index.
    :type index: int

    :returns:
        Alphabetic label.
    :rtype: str
    """
    if index < 0:
        raise ValueError("Index must be non-negative")

    label = ""
    value = index + 1

    while value > 0:
        value, remainder = divmod(value - 1, 26)
        label = chr(ord("A") + remainder) + label

    return label


def _normalize_reaction_side(side: str) -> List[str]:
    """
    Normalize one abstract reaction side by splitting on ``"+"`` and sorting tokens.

    :param side:
        Reaction side string.
    :type side: str

    :returns:
        Sorted token list.
    :rtype: List[str]
    """
    return sorted(token.strip() for token in side.split("+") if token.strip())


def deduplicate_abstract_reactions(reactions: List[str]) -> List[str]:
    """
    Remove identity reactions and duplicate abstract reactions.

    Reactant and product order are normalized internally for deduplication.

    :param reactions:
        Abstract reactions like ``"A+B>>C+D"``.
    :type reactions: List[str]

    :returns:
        Filtered reaction list.
    :rtype: List[str]
    """
    seen = set()
    filtered: List[str] = []

    for reaction in reactions:
        if ">>" not in reaction:
            continue

        left, right = reaction.split(">>", 1)
        reactants = _normalize_reaction_side(left)
        products = _normalize_reaction_side(right)

        if reactants == products:
            continue

        normalized = ">>".join(["+".join(reactants), "+".join(products)])
        if normalized not in seen:
            seen.add(normalized)
            filtered.append(reaction)

    return filtered


def _iter_reaction_records(
    data: Dict[str, Any],
) -> Iterable[Tuple[Dict[str, Any], str]]:
    """
    Iterate over reaction records from module-like or pathway-like JSON.

    Supported inputs:
        - module-like: ``{"reactions": [...]}`
        - pathway-like: ``{"by_module": {"M00001": {"reactions": [...]}, ...}}``

    :param data:
        Input data block.
    :type data: Dict[str, Any]

    :yields:
        Tuples ``(reaction_record, module_id_label)``.
    """
    if not data:
        return

    by_module = data.get("by_module")
    if isinstance(by_module, dict):
        for module_id, block in by_module.items():
            for reaction in block.get("reactions", []) or []:
                yield reaction, str(module_id)
        return

    for reaction in data.get("reactions", []) or []:
        yield reaction, ""


def _extract_reactions_and_templates(
    reactions: Optional[List[str]] = None,
    *,
    data: Optional[Dict[str, Any]] = None,
    templates: Optional[Dict[str, str]] = None,
    drop_missing_smiles_reactions: bool = True,
    prefix_module_in_reaction_id: bool = True,
) -> Tuple[List[str], Dict[str, str]]:
    """
    Extract reaction SMILES and reaction-template mapping from a raw list or JSON.

    :param reactions:
        Direct list of reaction SMILES.
    :type reactions: Optional[List[str]]

    :param data:
        Module-like or pathway-like JSON block.
    :type data: Optional[Dict[str, Any]]

    :param templates:
        Optional external template mapping.
    :type templates: Optional[Dict[str, str]]

    :param drop_missing_smiles_reactions:
        Whether to drop records with missing reaction SMILES.
    :type drop_missing_smiles_reactions: bool

    :param prefix_module_in_reaction_id:
        Whether to prefix pathway reaction IDs with module IDs.
    :type prefix_module_in_reaction_id: bool

    :returns:
        Tuple ``(reaction_smiles_list, template_pool)``.
    :rtype: Tuple[List[str], Dict[str, str]]
    """
    reactions = reactions or []
    template_pool = dict(templates or {})

    if reactions:
        return reactions, template_pool

    if not data:
        return [], template_pool

    extracted_reactions: List[str] = []
    extracted_templates: Dict[str, str] = {}

    for reaction_record, module_id in _iter_reaction_records(data):
        reaction_id = reaction_record.get("id") or reaction_record.get("kegg_id")
        reaction_smiles = (
            reaction_record.get("smiles")
            or reaction_record.get("reaction")
            or reaction_record.get("rxn_smiles")
        )
        template = (
            reaction_record.get("rule")
            or reaction_record.get("template")
            or reaction_record.get("smirks")
        )

        if not reaction_smiles:
            if drop_missing_smiles_reactions:
                continue
            reaction_smiles = ""

        extracted_reactions.append(reaction_smiles)

        if reaction_id is not None and template is not None:
            key = str(reaction_id)
            if module_id and prefix_module_in_reaction_id:
                key = f"{module_id}:{key}"
            extracted_templates[key] = str(template)

    extracted_templates.update(template_pool)
    return extracted_reactions, extracted_templates


@dataclass(frozen=True)
class AbstractReactionNetwork:
    """
    Abstracted symbolic reaction network.

    :param spool:
        Unique molecule pool in full representation.
    :type spool: List[str]

    :param reactions:
        Abstract reaction strings such as ``"A+B>>C+D"``.
    :type reactions: List[str]

    :param tpool:
        Optional reaction template mapping.
    :type tpool: Dict[str, str]

    :param ground_truth:
        Mapping from abstract labels to full molecules.
    :type ground_truth: Dict[str, str]

    :param molecule_to_label:
        Mapping from full molecules to abstract labels.
    :type molecule_to_label: Dict[str, str]
    """

    spool: List[str]
    reactions: List[str]
    tpool: Dict[str, str]
    ground_truth: Dict[str, str]
    molecule_to_label: Dict[str, str]


def abstract_reaction_network(
    reactions: Optional[List[str]] = None,
    *,
    data: Optional[Dict[str, Any]] = None,
    drop_missing_smiles_reactions: bool = True,
    deduplicate: bool = False,
    templates: Optional[Dict[str, str]] = None,
    order: str = "appearance",
    reactant_join: str = "+",
    product_join: str = "+",
    prefix_module_in_reaction_id: bool = True,
    save_as: Optional[str] = None,
) -> AbstractReactionNetwork:
    """
    Convert full reaction SMILES into abstract symbolic reactions.

    You may provide either a direct list of reaction SMILES or a module/pathway
    JSON block containing reaction records.

    :param reactions:
        Direct reaction SMILES list.
    :type reactions: Optional[List[str]]

    :param data:
        Module-like or pathway-like reaction JSON block.
    :type data: Optional[Dict[str, Any]]

    :param drop_missing_smiles_reactions:
        Whether to skip reaction records missing reaction SMILES.
    :type drop_missing_smiles_reactions: bool

    :param deduplicate:
        Whether to remove identity and duplicate abstract reactions.
    :type deduplicate: bool

    :param templates:
        Optional external template mapping.
    :type templates: Optional[Dict[str, str]]

    :param order:
        Molecule ordering strategy, one of ``"appearance"`` or ``"sorted"``.
    :type order: str

    :param reactant_join:
        Join token for abstract reactants.
    :type reactant_join: str

    :param product_join:
        Join token for abstract products.
    :type product_join: str

    :param prefix_module_in_reaction_id:
        Whether to prefix pathway reaction IDs with module IDs.
    :type prefix_module_in_reaction_id: bool

    :param save_as:
        Optional JSON output path.
    :type save_as: Optional[str]

    :returns:
        Abstract symbolic reaction network.
    :rtype: AbstractReactionNetwork
    """
    full_reactions, template_pool = _extract_reactions_and_templates(
        reactions,
        data=data,
        templates=templates,
        drop_missing_smiles_reactions=drop_missing_smiles_reactions,
        prefix_module_in_reaction_id=prefix_module_in_reaction_id,
    )

    parsed_reactions: List[Tuple[List[str], List[str]]] = []
    for reaction_smiles in full_reactions:
        reactants, products = _split_reaction_smiles(reaction_smiles)
        parsed_reactions.append((reactants, products))

    if order == "appearance":
        seen = set()
        spool: List[str] = []

        for reactants, products in parsed_reactions:
            for molecule in reactants + products:
                if molecule not in seen:
                    seen.add(molecule)
                    spool.append(molecule)
    elif order == "sorted":
        unique_molecules = set()
        for reactants, products in parsed_reactions:
            unique_molecules.update(reactants)
            unique_molecules.update(products)
        spool = sorted(unique_molecules)
    else:
        raise ValueError("order must be 'appearance' or 'sorted'")

    molecule_to_label = {
        molecule: _excel_label(index) for index, molecule in enumerate(spool)
    }
    ground_truth = {label: molecule for molecule, label in molecule_to_label.items()}

    abstracted_reactions: List[str] = []
    for reactants, products in parsed_reactions:
        left = reactant_join.join(molecule_to_label[mol] for mol in reactants) if reactants else ""
        right = product_join.join(molecule_to_label[mol] for mol in products) if products else ""
        abstracted_reactions.append(f"{left}>>{right}")

    if deduplicate:
        abstracted_reactions = deduplicate_abstract_reactions(abstracted_reactions)

    if save_as:
        payload = {
            "meta": {"name": save_as, "version": 1},
            "examples": [
                {
                    "spool": spool,
                    "reactions": abstracted_reactions,
                    "tpool": template_pool or {},
                    "ground_truth": ground_truth,
                }
            ],
        }
        with open(save_as, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4)

    return AbstractReactionNetwork(
        spool=spool,
        reactions=abstracted_reactions,
        tpool=template_pool or {},
        ground_truth=ground_truth,
        molecule_to_label=molecule_to_label,
    )