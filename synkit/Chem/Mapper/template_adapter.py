"""Convert fully mapped exact-ITS representatives into executable rules.

The global-shell enumerator normally operates on heavy atoms.  A heavy-only
correspondence is intentionally insufficient here: assigning omitted explicit
hydrogens from the deposited map would leak the reference alignment into a
candidate template.  This adapter therefore accepts only a declared complete
atom correspondence and exposes source replay as a construction check.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Integral
from typing import Any, Iterable, Mapping, Sequence

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - exercised only without RDKit
    Chem = None


def _require_rdkit():
    if Chem is None:
        raise ImportError("RDKit is required for executable-template conversion")


def _split_reaction(reaction: str) -> tuple[str, str]:
    if not isinstance(reaction, str) or reaction.count(">>") != 1:
        raise ValueError("reaction must contain exactly one '>>' separator")
    reactants, products = reaction.split(">>")
    if not reactants or not products:
        raise ValueError("reaction must contain non-empty reactant and product sides")
    return reactants, products


def _mapped_molecule(smiles: str, *, side: str):
    _require_rdkit()
    # Keep mapped explicit hydrogens.  The mapper's full-atom contract uses
    # them as ordinary vertices, whereas RDKit's default parser may remove
    # them before atom-map inventory validation.
    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    molecule = Chem.MolFromSmiles(smiles, parameters)
    if molecule is None:
        raise ValueError(f"{side} side is not valid SMILES")
    maps = [int(atom.GetAtomMapNum()) for atom in molecule.GetAtoms()]
    if (
        not maps
        or any(atom_map <= 0 for atom_map in maps)
        or len(set(maps)) != len(maps)
    ):
        raise ValueError(f"{side} side must have one unique positive atom map per atom")
    return molecule, tuple(maps)


def _correspondence_pairs(correspondence):
    try:
        pairs = tuple(tuple(pair) for pair in correspondence)
    except TypeError as error:
        raise ValueError(
            "correspondence must contain pairs of integer atom-map IDs"
        ) from error
    if not pairs or any(len(pair) != 2 for pair in pairs):
        raise ValueError("each atom-map correspondence must have two entries")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
        for pair in pairs
        for value in pair
    ):
        raise ValueError("atom-map correspondence entries must be positive integers")
    pairs = tuple((int(left), int(right)) for left, right in pairs)
    if len({left for left, _ in pairs}) != len(pairs):
        raise ValueError("reactant atom maps occur more than once in correspondence")
    if len({right for _, right in pairs}) != len(pairs):
        raise ValueError("product atom maps occur more than once in correspondence")
    return pairs


def _validate_correspondence(
    correspondence: Iterable[Sequence[int]],
    reactant_maps: tuple[int, ...],
    product_maps: tuple[int, ...],
) -> dict[int, int]:
    pairs = _correspondence_pairs(correspondence)
    remapping = dict(pairs)
    if set(remapping) != set(reactant_maps) or set(remapping.values()) != set(
        product_maps
    ):
        raise ValueError(
            "correspondence must cover each reactant and product atom once"
        )
    return remapping


def mapped_reaction_from_correspondence(
    reaction: str,
    correspondence: Iterable[Sequence[int]],
) -> str:
    """Return a fully mapped reaction whose product uses the candidate mapping.

    ``correspondence`` maps original reactant atom-map identifiers to original
    product identifiers. The returned product is relabelled into the reactant
    identifier namespace, which is the pairing convention required by ITS and
    executable-rule construction.
    """
    reactants, products = _split_reaction(reaction)
    reactant, reactant_maps = _mapped_molecule(reactants, side="reactant")
    product, product_maps = _mapped_molecule(products, side="product")
    remapping = _validate_correspondence(correspondence, reactant_maps, product_maps)
    _validate_elements(reactant, product, remapping)
    inverse = {
        product_map: reactant_map for reactant_map, product_map in remapping.items()
    }
    for atom in product.GetAtoms():
        atom.SetAtomMapNum(inverse[int(atom.GetAtomMapNum())])
    product_smiles = Chem.MolToSmiles(product, canonical=True, isomericSmiles=True)
    return f"{reactants}>>{product_smiles}"


def executable_rule_from_correspondence(
    reaction: str,
    correspondence: Iterable[Sequence[int]],
    *,
    heavy_only: bool,
    name: str = "synister_exact_its_class",
):
    """Build a :class:`SynRule` from a complete representative mapping.

    Heavy-only shell outputs cannot safely furnish a full explicit-hydrogen
    rule, so callers must run a full-atom protocol before calling this method.
    """
    if type(heavy_only) is not bool:
        raise ValueError("heavy_only must explicitly declare Boolean scope")
    if heavy_only:
        raise ValueError(
            "heavy-only correspondence cannot construct a full executable rule"
        )
    from synkit.Rule import SynRule

    mapped_reaction = mapped_reaction_from_correspondence(reaction, correspondence)
    return SynRule.from_smart(
        mapped_reaction,
        name=name,
        format="tuple",
        implicit_h=False,
    )


def _canonical_product(smiles: str) -> str:
    _require_rdkit()
    molecule = Chem.MolFromSmiles(smiles, sanitize=True)
    if molecule is None:
        raise ValueError("product is not valid SMILES")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


@dataclass(frozen=True)
class SourceReplayResult:
    """Product-blindness-safe source construction replay outcome."""

    mapped_reaction: str
    expected_product: str
    generated_products: tuple[str, ...]
    recovered: bool


@dataclass(frozen=True)
class ProductRecoveryScore:
    """Target-only score of previously generated prospective candidates."""

    expected_product: str
    candidate_count: int
    recovered: bool


def prospective_products(reactants: str, rules: Iterable[Any]) -> tuple[str, ...]:
    """Apply executable rules to reactants without observing a target product.

    The returned values are canonical unmapped product SMILES.  This is the
    application half of a held-out transfer study; target-product comparison
    belongs in a separate scoring function and must occur only after these
    candidates have been saved.
    """
    if not isinstance(reactants, str) or not reactants:
        raise ValueError("reactants must be a non-empty SMILES string")
    from synkit.Synthesis.Reactor import SynReactor

    products = set()
    for rule in rules:
        reactor = SynReactor(
            reactants,
            rule,
            template_format="tuple",
            explicit_h=False,
            stereo_mode="ignore",
        )
        for candidate in reactor.smarts:
            products.add(_canonical_product(candidate.split(">>", 1)[1]))
    return tuple(sorted(products))


def score_product_recovery(
    candidates: Iterable[str], expected_product: str
) -> ProductRecoveryScore:
    """Score saved candidates against a held-out product in a separate stage."""
    normalized_candidates = tuple(
        sorted({_canonical_product(value) for value in candidates})
    )
    expected = _canonical_product(expected_product)
    return ProductRecoveryScore(
        expected_product=expected,
        candidate_count=len(normalized_candidates),
        recovered=expected in normalized_candidates,
    )


def replay_executable_rule_on_source(
    reaction: str,
    correspondence: Iterable[Sequence[int]],
    *,
    heavy_only: bool,
) -> SourceReplayResult:
    """Construct a rule and check it on its own source reactants.

    This helper is a conversion audit only. It is deliberately unsuitable for
    held-out scoring because it reads both sides of ``reaction`` to establish
    the expected source product.
    """
    correspondence = _correspondence_pairs(correspondence)
    mapped_reaction = mapped_reaction_from_correspondence(reaction, correspondence)
    rule = executable_rule_from_correspondence(
        reaction,
        correspondence,
        heavy_only=heavy_only,
    )
    reactants, products = _split_reaction(mapped_reaction)
    from synkit.Synthesis.Reactor import SynReactor

    reactor = SynReactor(
        reactants,
        rule,
        template_format="tuple",
        explicit_h=False,
        stereo_mode="ignore",
    )
    generated = tuple(
        sorted(
            {
                _canonical_product(candidate.split(">>", 1)[1])
                for candidate in reactor.smarts
            }
        )
    )
    expected = _canonical_product(products)
    return SourceReplayResult(
        mapped_reaction=mapped_reaction,
        expected_product=expected,
        generated_products=generated,
        recovered=expected in generated,
    )


def correspondence_from_export(
    class_record: Mapping[str, object],
) -> tuple[tuple[int, int], ...]:
    """Read one representative correspondence from a version-2 class record."""
    try:
        value = class_record["atom_map_correspondence"]
    except KeyError as error:
        raise ValueError("class record lacks atom_map_correspondence") from error
    if not isinstance(value, list):
        raise ValueError("atom_map_correspondence must be a JSON list")
    return _correspondence_pairs(value)


def _validate_elements(reactant, product, remapping):
    before = {atom.GetAtomMapNum(): atom.GetAtomicNum() for atom in reactant.GetAtoms()}
    after = {atom.GetAtomMapNum(): atom.GetAtomicNum() for atom in product.GetAtoms()}
    if any(before[left] != after[right] for left, right in remapping.items()):
        raise ValueError("correspondence must preserve atom elements")


@dataclass(frozen=True)
class ClassCorrespondence:
    """Reaction-bound correspondence from a declared complete ITS-class export."""

    reaction_sha256: str
    its_class_id: str
    heavy_only: bool
    reactant_atom_maps: tuple[int, ...]
    product_atom_maps: tuple[int, ...]
    correspondence: tuple[tuple[int, int], ...]


def _export_inventories(payload):
    inventories = []
    for key in ("reactant_atom_maps", "product_atom_maps"):
        values = payload.get(key)
        if (
            not isinstance(values, list)
            or not values
            or any(type(value) is not int or value <= 0 for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError("export contains invalid endpoint atom-map inventory")
        inventories.append(tuple(values))
    return inventories


def class_correspondence_from_export(
    payload: Mapping[str, object],
    reaction: str,
    its_class_id: str,
    *,
    allow_legacy: bool = False,
) -> ClassCorrespondence:
    """Validate export version, reaction identity, scope and endpoint coordinates.

    New library exports use schema version 1. Existing unversioned library
    results can be read explicitly with ``allow_legacy=True``. This checks the
    declared completion contract and the selected correspondence; independently
    verifying enumeration completeness requires the search certificate/audit.
    CLI application records wrap these library results under ``queries``.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("export must be a mapping")
    schema = payload.get("schema_version")
    if schema is None and allow_legacy:
        pass
    elif (
        type(schema) is not int
        or schema != 1
        or payload.get("kind") != "synister_mapped_reaction_its_alternatives"
    ):
        raise ValueError("unsupported correspondence export schema")
    reactants, products = _split_reaction(reaction)
    expected_hash = hashlib.sha256(reaction.encode("utf-8")).hexdigest()
    if payload.get("reaction_sha256") != expected_hash:
        raise ValueError("export is bound to a different reaction")
    heavy_only = payload.get("heavy_only")
    if type(heavy_only) is not bool:
        raise ValueError("export must declare Boolean heavy_only scope")
    shell = payload.get("shell")
    if not isinstance(shell, Mapping) or any(
        shell.get(key) is not True
        for key in ("complete", "shell_complete", "classification_complete")
    ):
        raise ValueError("export lacks complete enumeration and ITS classification")
    classes = shell.get("classes")
    if (
        not isinstance(classes, list)
        or not classes
        or any(not isinstance(record, Mapping) for record in classes)
    ):
        raise ValueError("export lacks ITS-class records")
    identifiers = [record.get("its_class_id") for record in classes]
    if any(
        not isinstance(identifier, str) or not identifier for identifier in identifiers
    ) or len(set(identifiers)) != len(identifiers):
        raise ValueError("export contains invalid or duplicate ITS-class identifiers")
    if its_class_id not in identifiers:
        raise ValueError("requested ITS class is absent from export")
    selected = classes[identifiers.index(its_class_id)]
    pairs = correspondence_from_export(selected)
    inventories = _export_inventories(payload)
    before_maps, after_maps = inventories
    _validate_correspondence(pairs, before_maps, after_maps)
    representative = selected.get("representative_mapping")
    if (
        not isinstance(representative, list)
        or len(representative) != len(before_maps)
        or any(type(value) is not int for value in representative)
        or sorted(representative) != list(range(len(after_maps)))
        or dict(pairs)
        != {before_maps[i]: after_maps[j] for i, j in enumerate(representative)}
    ):
        raise ValueError("correspondence differs from exported endpoint coordinates")
    molecules = [
        _mapped_molecule(smiles, side=side)[0]
        for smiles, side in ((reactants, "reactant"), (products, "product"))
    ]
    for molecule, declared in zip(molecules, inventories):
        actual = {
            atom.GetAtomMapNum()
            for atom in molecule.GetAtoms()
            if not heavy_only or atom.GetAtomicNum() != 1
        }
        if set(declared) != actual:
            raise ValueError(
                "endpoint inventory differs from the declared reaction scope"
            )
    _validate_elements(*molecules, dict(pairs))
    return ClassCorrespondence(
        expected_hash, its_class_id, heavy_only, before_maps, after_maps, pairs
    )


__all__ = [
    "SourceReplayResult",
    "ProductRecoveryScore",
    "correspondence_from_export",
    "ClassCorrespondence",
    "class_correspondence_from_export",
    "executable_rule_from_correspondence",
    "mapped_reaction_from_correspondence",
    "prospective_products",
    "score_product_recovery",
    "replay_executable_rule_on_source",
]
