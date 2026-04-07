from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

PathLike = Union[str, Path]
ReactionSides = Tuple[List[str], List[str]]


def _split_reaction_smiles(reaction_smiles: str) -> ReactionSides:
    """
    Split a reaction SMILES string into reactant and product molecule lists.

    The expected format is ``"A.B>>C.D"``. Empty left or right sides are allowed.

    :param reaction_smiles:
        Reaction SMILES string.
    :type reaction_smiles: str

    :returns:
        Tuple of reactant and product molecule lists.
    :rtype: Tuple[List[str], List[str]]

    :raises ValueError:
        If the reaction string does not contain ``">>"``.

    Example
    -------
    .. code-block:: python

        reactants, products = _split_reaction_smiles("CCO.O>>CC=O")
    """
    value = reaction_smiles.strip()
    if ">>" not in value:
        raise ValueError(f"Invalid reaction SMILES (missing '>>'): {reaction_smiles}")

    left, right = value.split(">>", 1)
    reactants = [token.strip() for token in left.split(".") if token.strip()]
    products = [token.strip() for token in right.split(".") if token.strip()]
    return reactants, products


def _excel_label(index: int) -> str:
    """
    Convert a zero-based integer index into an Excel-style alphabetic label.

    Examples include ``0 -> "A"``, ``25 -> "Z"``, and ``26 -> "AA"``.

    :param index:
        Zero-based index.
    :type index: int

    :returns:
        Excel-style alphabetic label.
    :rtype: str

    :raises ValueError:
        If ``index`` is negative.

    Example
    -------
    .. code-block:: python

        label = _excel_label(27)   # "AB"
    """
    if index < 0:
        raise ValueError("Index must be non-negative")

    label = ""
    value = index + 1
    while value > 0:
        value, remainder = divmod(value - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label


def _normalize_abstract_side(side: str) -> List[str]:
    """
    Normalize one abstract reaction side by splitting on ``"+"`` and sorting tokens.

    :param side:
        Abstract reaction side.
    :type side: str

    :returns:
        Sorted abstract token list.
    :rtype: List[str]

    Example
    -------
    .. code-block:: python

        tokens = _normalize_abstract_side("B+A+C")
    """
    return sorted(token.strip() for token in side.split("+") if token.strip())


def _first_present(
    record: Mapping[str, Any],
    keys: Sequence[str],
) -> Optional[Any]:
    """
    Return the first non-``None`` value found in a mapping for the given keys.

    :param record:
        Input mapping.
    :type record: Mapping[str, Any]

    :param keys:
        Candidate keys to try in order.
    :type keys: Sequence[str]

    :returns:
        First matching value, or ``None`` if none are present.
    :rtype: Optional[Any]

    Example
    -------
    .. code-block:: python

        value = _first_present(record, ["smiles", "reaction", "rxn_smiles"])
    """
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def deduplicate_abstract_reactions(reactions: Sequence[str]) -> List[str]:
    """
    Remove identity reactions and duplicate abstract reactions.

    Reactant and product order are normalized internally before comparison.
    The original retained representative is the first encountered entry.

    :param reactions:
        Abstract reactions such as ``"A+B>>C+D"``.
    :type reactions: Sequence[str]

    :returns:
        Filtered abstract reactions.
    :rtype: List[str]

    Example
    -------
    .. code-block:: python

        filtered = deduplicate_abstract_reactions(
            ["A+B>>C", "B+A>>C", "A>>A"]
        )
    """
    seen: set[str] = set()
    filtered: List[str] = []

    for reaction in reactions:
        if ">>" not in reaction:
            continue

        left, right = reaction.split(">>", 1)
        reactants = _normalize_abstract_side(left)
        products = _normalize_abstract_side(right)

        if reactants == products:
            continue

        normalized = ">>".join(["+".join(reactants), "+".join(products)])
        if normalized in seen:
            continue

        seen.add(normalized)
        filtered.append(reaction)

    return filtered


@dataclass(frozen=True)
class AbstractReactionNetwork:
    """
    Symbolic abstraction of a reaction network.

    :param molecule_pool:
        Unique molecule pool in the original full representation.
    :type molecule_pool: List[str]

    :param reactions:
        Abstract symbolic reactions such as ``"A+B>>C+D"``.
    :type reactions: List[str]

    :param templates:
        Optional mapping from reaction identifiers to rule or template strings.
    :type templates: Dict[str, str]

    :param label_to_molecule:
        Mapping from abstract labels back to original molecule strings.
    :type label_to_molecule: Dict[str, str]
    """

    molecule_pool: List[str]
    reactions: List[str]
    templates: Dict[str, str] = field(default_factory=dict)
    label_to_molecule: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the abstract network to a plain dictionary.

        :returns:
            Dictionary representation of the abstract network.
        :rtype: Dict[str, Any]

        Example
        -------
        .. code-block:: python

            payload = network.to_dict()
        """
        return {
            "molecule_pool": list(self.molecule_pool),
            "reactions": list(self.reactions),
            "templates": dict(self.templates),
            "label_to_molecule": dict(self.label_to_molecule),
        }

    def to_json_payload(
        self, name: str = "abstract_reaction_network"
    ) -> Dict[str, Any]:
        """
        Convert the abstract network into a SynKit-style JSON payload.

        :param name:
            Name stored in the metadata block.
        :type name: str

        :returns:
            JSON-serializable payload.
        :rtype: Dict[str, Any]

        Example
        -------
        .. code-block:: python

            payload = network.to_json_payload(name="glycolysis_abstract")
        """
        return {
            "meta": {"name": name, "version": 1},
            "examples": [
                {
                    "molecule_pool": list(self.molecule_pool),
                    "reactions": list(self.reactions),
                    "templates": dict(self.templates),
                    "label_to_molecule": dict(self.label_to_molecule),
                }
            ],
        }

    def save_json(self, path: PathLike, *, name: Optional[str] = None) -> None:
        """
        Save the abstract network as a JSON file.

        :param path:
            Output JSON path.
        :type path: PathLike

        :param name:
            Optional metadata name. If omitted, the filename stem is used.
        :type name: Optional[str]

        :returns:
            ``None``
        :rtype: None

        Example
        -------
        .. code-block:: python

            network.save_json("abstract_network.json")
        """
        output_path = Path(path)
        payload = self.to_json_payload(name=name or output_path.stem)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4)


@dataclass
class AbstractReactionExtractor:
    """
    Build abstract symbolic reaction networks from reaction SMILES lists or
    SynKit-style module/pathway JSON blocks.

    This class supports configurable field names when extracting reactions from
    JSON-like input records.

    Example
    -------
    .. code-block:: python

        extractor = KEGGExtractor()
        data = extractor.build_module_json(
            "M00001",
            with_compounds=True,
            with_atom_maps=False,
        )

        abstractor = AbstractReactionExtractor()
        network = abstractor.build(
            data=data,
            deduplicate=True,
            order="appearance",
        )
    """

    def iter_reaction_records(
        self,
        data: Mapping[str, Any],
    ) -> Iterable[Tuple[Mapping[str, Any], str]]:
        """
        Iterate over reaction records from a module-like or pathway-like JSON block.

        Supported structures include:

        - module-like: ``{"reactions": [...]}`
        - pathway-like: ``{"by_module": {"M00001": {"reactions": [...]}, ...}}``

        :param data:
            Input JSON-like mapping.
        :type data: Mapping[str, Any]

        :yields:
            Tuples of ``(reaction_record, module_id)``.
        :rtype: Iterable[Tuple[Mapping[str, Any], str]]

        Example
        -------
        .. code-block:: python

            abstractor = AbstractReactionExtractor()
            for record, module_id in abstractor.iter_reaction_records(data):
                print(module_id, record.get("id"))
        """
        by_module = data.get("by_module")
        if isinstance(by_module, Mapping):
            for module_id, block in by_module.items():
                if not isinstance(block, Mapping):
                    continue
                for reaction in block.get("reactions", []) or []:
                    if isinstance(reaction, Mapping):
                        yield reaction, str(module_id)
            return

        for reaction in data.get("reactions", []) or []:
            if isinstance(reaction, Mapping):
                yield reaction, ""

    def extract_reactions_and_templates(
        self,
        reactions: Optional[Sequence[str]] = None,
        *,
        data: Optional[Mapping[str, Any]] = None,
        templates: Optional[Mapping[str, str]] = None,
        drop_missing_smiles_reactions: bool = True,
        prefix_module_in_reaction_id: bool = True,
        reaction_id_keys: Optional[Sequence[str]] = None,
        reaction_smiles_keys: Optional[Sequence[str]] = None,
        template_keys: Optional[Sequence[str]] = None,
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Extract reaction SMILES and rule-template mappings from raw inputs.

        Either a direct list of reaction SMILES or a JSON data block may be
        provided. If both are given, the explicit ``reactions`` list takes
        precedence for reaction extraction, while ``templates`` is still merged.

        When ``data`` is used, the user may customize which keys are searched
        for reaction identifiers, reaction SMILES strings, and templates.

        :param reactions:
            Direct reaction SMILES list.
        :type reactions: Optional[Sequence[str]]

        :param data:
            Module-like or pathway-like JSON block.
        :type data: Optional[Mapping[str, Any]]

        :param templates:
            Optional external mapping from reaction identifiers to templates.
        :type templates: Optional[Mapping[str, str]]

        :param drop_missing_smiles_reactions:
            Whether to skip records that do not contain a reaction SMILES string.
        :type drop_missing_smiles_reactions: bool

        :param prefix_module_in_reaction_id:
            Whether to prefix reaction identifiers with module IDs in pathway-style
            inputs.
        :type prefix_module_in_reaction_id: bool

        :param reaction_id_keys:
            Candidate keys used to find reaction identifiers in each reaction record.
            The keys are tried in order.
        :type reaction_id_keys: Optional[Sequence[str]]

        :param reaction_smiles_keys:
            Candidate keys used to find reaction SMILES strings in each reaction
            record. The keys are tried in order.
        :type reaction_smiles_keys: Optional[Sequence[str]]

        :param template_keys:
            Candidate keys used to find rule or template strings in each reaction
            record. The keys are tried in order.
        :type template_keys: Optional[Sequence[str]]

        :returns:
            Tuple of reaction SMILES list and template mapping.
        :rtype: Tuple[List[str], Dict[str, str]]

        Example
        -------
        .. code-block:: python

            extractor = KEGGExtractor()
            data = extractor.build_module_json(
                "M00001",
                with_compounds=True,
                with_atom_maps=False,
            )

            abstractor = AbstractReactionExtractor()
            reactions, templates = abstractor.extract_reactions_and_templates(
                data=data,
                reaction_id_keys=["id", "kegg_id", "rid"],
                reaction_smiles_keys=["smiles", "reaction", "rxn_smiles"],
                template_keys=["rule", "template", "smirks"],
            )
        """
        reaction_id_keys = list(reaction_id_keys or ["id", "kegg_id"])
        reaction_smiles_keys = list(
            reaction_smiles_keys or ["smiles", "reaction", "rxn_smiles"]
        )
        template_keys = list(template_keys or ["rule", "template", "smirks"])

        reaction_list = list(reactions or [])
        template_pool: Dict[str, str] = dict(templates or {})

        if reaction_list:
            return reaction_list, template_pool

        if not data:
            return [], template_pool

        extracted_reactions: List[str] = []
        extracted_templates: Dict[str, str] = {}

        for reaction_record, module_id in self.iter_reaction_records(data):
            reaction_id = _first_present(reaction_record, reaction_id_keys)
            reaction_smiles = _first_present(reaction_record, reaction_smiles_keys)
            template = _first_present(reaction_record, template_keys)

            if not reaction_smiles:
                if drop_missing_smiles_reactions:
                    continue
                reaction_smiles = ""

            extracted_reactions.append(str(reaction_smiles))

            if reaction_id is not None and template is not None:
                key = str(reaction_id)
                if module_id and prefix_module_in_reaction_id:
                    key = f"{module_id}:{key}"
                extracted_templates[key] = str(template)

        extracted_templates.update(template_pool)
        return extracted_reactions, extracted_templates

    def build_molecule_pool(
        self,
        parsed_reactions: Sequence[ReactionSides],
        *,
        order: str = "appearance",
    ) -> List[str]:
        """
        Build the unique molecule pool from parsed reactions.

        :param parsed_reactions:
            Parsed reaction sides as ``(reactants, products)`` tuples.
        :type parsed_reactions: Sequence[Tuple[List[str], List[str]]]

        :param order:
            Molecule ordering mode. Supported values are ``"appearance"`` and
            ``"sorted"``.
        :type order: str

        :returns:
            Ordered unique molecule pool.
        :rtype: List[str]

        :raises ValueError:
            If ``order`` is not supported.

        Example
        -------
        .. code-block:: python

            molecule_pool = abstractor.build_molecule_pool(
                parsed_reactions,
                order="appearance",
            )
        """
        if order == "appearance":
            seen: set[str] = set()
            molecule_pool: List[str] = []
            for reactants, products in parsed_reactions:
                for molecule in reactants + products:
                    if molecule not in seen:
                        seen.add(molecule)
                        molecule_pool.append(molecule)
            return molecule_pool

        if order == "sorted":
            unique_molecules: set[str] = set()
            for reactants, products in parsed_reactions:
                unique_molecules.update(reactants)
                unique_molecules.update(products)
            return sorted(unique_molecules)

        raise ValueError("order must be 'appearance' or 'sorted'")

    def build(
        self,
        reactions: Optional[Sequence[str]] = None,
        *,
        data: Optional[Mapping[str, Any]] = None,
        drop_missing_smiles_reactions: bool = True,
        deduplicate: bool = False,
        templates: Optional[Mapping[str, str]] = None,
        order: str = "appearance",
        reactant_join: str = "+",
        product_join: str = "+",
        prefix_module_in_reaction_id: bool = True,
        reaction_id_keys: Optional[Sequence[str]] = None,
        reaction_smiles_keys: Optional[Sequence[str]] = None,
        template_keys: Optional[Sequence[str]] = None,
        save_as: Optional[PathLike] = None,
    ) -> AbstractReactionNetwork:
        """
        Convert full reaction SMILES into an abstract symbolic reaction network.

        You may provide either a direct list of reaction SMILES or a module/pathway
        JSON block.

        If ``data`` is provided, field names for reaction identifiers, reaction
        SMILES strings, and templates may be customized.

        :param reactions:
            Direct reaction SMILES list.
        :type reactions: Optional[Sequence[str]]

        :param data:
            Module-like or pathway-like reaction JSON block.
        :type data: Optional[Mapping[str, Any]]

        :param drop_missing_smiles_reactions:
            Whether to skip records missing reaction SMILES.
        :type drop_missing_smiles_reactions: bool

        :param deduplicate:
            Whether to remove identity and duplicate abstract reactions.
        :type deduplicate: bool

        :param templates:
            Optional external mapping from reaction identifiers to templates.
        :type templates: Optional[Mapping[str, str]]

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

        :param reaction_id_keys:
            Candidate keys used to find reaction identifiers in each reaction record.
        :type reaction_id_keys: Optional[Sequence[str]]

        :param reaction_smiles_keys:
            Candidate keys used to find reaction SMILES strings in each reaction
            record.
        :type reaction_smiles_keys: Optional[Sequence[str]]

        :param template_keys:
            Candidate keys used to find rule or template strings in each reaction
            record.
        :type template_keys: Optional[Sequence[str]]

        :param save_as:
            Optional JSON output path.
        :type save_as: Optional[PathLike]

        :returns:
            Abstract symbolic reaction network.
        :rtype: AbstractReactionNetwork

        Example
        -------
        .. code-block:: python

            extractor = KEGGExtractor()
            data = extractor.build_module_json(
                "M00001",
                with_compounds=True,
                with_atom_maps=False,
            )

            abstractor = AbstractReactionExtractor()
            network = abstractor.build(
                data=data,
                deduplicate=True,
                order="appearance",
                reaction_id_keys=["id", "kegg_id", "rid"],
                reaction_smiles_keys=["smiles", "reaction", "rxn_smiles"],
                template_keys=["rule", "template", "smirks"],
                save_as="M00001_abstract.json",
            )
        """
        full_reactions, template_pool = self.extract_reactions_and_templates(
            reactions=reactions,
            data=data,
            templates=templates,
            drop_missing_smiles_reactions=drop_missing_smiles_reactions,
            prefix_module_in_reaction_id=prefix_module_in_reaction_id,
            reaction_id_keys=reaction_id_keys,
            reaction_smiles_keys=reaction_smiles_keys,
            template_keys=template_keys,
        )

        parsed_reactions: List[ReactionSides] = [
            _split_reaction_smiles(reaction_smiles)
            for reaction_smiles in full_reactions
        ]

        molecule_pool = self.build_molecule_pool(parsed_reactions, order=order)

        molecule_to_label = {
            molecule: _excel_label(index)
            for index, molecule in enumerate(molecule_pool)
        }
        label_to_molecule = {
            label: molecule for molecule, label in molecule_to_label.items()
        }

        abstracted_reactions: List[str] = []
        for reactants, products in parsed_reactions:
            left = reactant_join.join(molecule_to_label[mol] for mol in reactants)
            right = product_join.join(molecule_to_label[mol] for mol in products)
            abstracted_reactions.append(f"{left}>>{right}")

        if deduplicate:
            abstracted_reactions = deduplicate_abstract_reactions(abstracted_reactions)

        network = AbstractReactionNetwork(
            molecule_pool=molecule_pool,
            reactions=abstracted_reactions,
            templates=dict(template_pool),
            label_to_molecule=label_to_molecule,
        )

        if save_as is not None:
            network.save_json(save_as)

        return network
