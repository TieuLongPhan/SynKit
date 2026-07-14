"""Selection gate for mapped reactions with hydrogen in the reaction center.

This module deliberately uses a narrow definition suitable for regression and
conformance data.  An accepted reaction must be fully atom mapped, conserve
mapped atom identities, molecular formula, and total formal charge, and contain
at least one explicit mapped hydrogen incident to a changed bond.

The check is stricter than merely searching for ``"[H:"``.  A mapped hydrogen
which is present but remote from every bond change does not qualify.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula


@dataclass(frozen=True, order=True)
class MappedBondChange:
    """One mapped bond before and after a reaction."""

    atom_maps: tuple[int, int]
    before: float
    after: float

    @property
    def kind(self) -> str:
        if self.before == 0.0:
            return "FORMED"
        if self.after == 0.0:
            return "BROKEN"
        return "ORDER_CHANGED"


@dataclass(frozen=True)
class ExplicitHydrogenReactionAudit:
    """Structured result returned by :func:`audit_explicit_h_reaction`."""

    reaction: str
    errors: tuple[str, ...] = ()
    reactant_formula: str | None = None
    product_formula: str | None = None
    reactant_charge: int | None = None
    product_charge: int | None = None
    atom_maps: tuple[int, ...] = ()
    explicit_hydrogen_maps: tuple[int, ...] = ()
    changed_hydrogen_maps: tuple[int, ...] = ()
    reaction_center_maps: tuple[int, ...] = ()
    changed_bonds: tuple[MappedBondChange, ...] = ()

    @property
    def accepted(self) -> bool:
        """Whether the reaction satisfies every strict selection criterion."""

        return not self.errors


@dataclass(frozen=True)
class _ReactionSide:
    formula: str
    elemental_composition: tuple[tuple[tuple[int, int], int], ...]
    charge: int
    atoms_by_map: dict[int, tuple[int, int]]
    bonds_by_maps: dict[tuple[int, int], float]
    unmapped_atoms: tuple[int, ...]
    duplicate_maps: tuple[int, ...]


def reaction_smiles_from_annotated_text(text: str) -> str:
    """Return the reaction token from legacy ``SMIRKS arrow-code`` text.

    SynKit's polar and radical CSV files append electron-flow notation after a
    space.  Ordinary reaction SMILES contains no whitespace, so the first token
    is the complete reaction in both the annotated and unannotated forms.
    """

    stripped = str(text).strip()
    return stripped.split(maxsplit=1)[0] if stripped else ""


def _parse_side(smiles: str) -> _ReactionSide | None:
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    molecule = Chem.MolFromSmiles(smiles, parser)
    if molecule is None:
        return None

    atoms_by_map: dict[int, tuple[int, int]] = {}
    unmapped_atoms = []
    duplicate_maps = []
    for atom in molecule.GetAtoms():
        atom_map = int(atom.GetAtomMapNum())
        if atom_map <= 0:
            unmapped_atoms.append(atom.GetIdx())
            continue
        if atom_map in atoms_by_map:
            duplicate_maps.append(atom_map)
            continue
        # Charge, radical state, and hydrogen count may legitimately change.
        # Element and isotope may not change under an atom mapping.
        atoms_by_map[atom_map] = (atom.GetAtomicNum(), atom.GetIsotope())

    bonds_by_maps = {}
    for bond in molecule.GetBonds():
        begin_map = int(bond.GetBeginAtom().GetAtomMapNum())
        end_map = int(bond.GetEndAtom().GetAtomMapNum())
        if begin_map <= 0 or end_map <= 0:
            continue
        key = tuple(sorted((begin_map, end_map)))
        bonds_by_maps[key] = float(bond.GetBondTypeAsDouble())

    # CalcMolFormula includes a charge suffix (for example ``H4N+``).  Keep
    # that human-readable value in the report, but compare mass balance using
    # an isotope-aware elemental inventory so charge has its own error code.
    composition: Counter[tuple[int, int]] = Counter()
    for atom in molecule.GetAtoms():
        composition[(atom.GetAtomicNum(), atom.GetIsotope())] += 1
        if atom.GetAtomicNum() != 1:
            hydrogen_count = atom.GetTotalNumHs(includeNeighbors=False)
            if hydrogen_count:
                composition[(1, 0)] += int(hydrogen_count)

    return _ReactionSide(
        formula=CalcMolFormula(molecule),
        elemental_composition=tuple(sorted(composition.items())),
        charge=int(Chem.GetFormalCharge(molecule)),
        atoms_by_map=atoms_by_map,
        bonds_by_maps=bonds_by_maps,
        unmapped_atoms=tuple(unmapped_atoms),
        duplicate_maps=tuple(sorted(set(duplicate_maps))),
    )


def audit_explicit_h_reaction(text: str) -> ExplicitHydrogenReactionAudit:
    """Audit a mapped reaction for strict explicit-H-center suitability.

    Parameters
    ----------
    text:
        A mapped reaction SMILES, optionally followed by SynKit's legacy
        whitespace-separated electron-flow annotation.

    Returns
    -------
    ExplicitHydrogenReactionAudit
        A report with stable error codes and the detected mapped bond changes.
    """

    reaction = reaction_smiles_from_annotated_text(text)
    if reaction.count(">>") != 1:
        return ExplicitHydrogenReactionAudit(
            reaction,
            errors=("INVALID_REACTION_SEPARATOR",),
        )

    reactant_text, product_text = reaction.split(">>", 1)
    reactants = _parse_side(reactant_text)
    products = _parse_side(product_text)
    if reactants is None or products is None:
        errors = []
        if reactants is None:
            errors.append("REACTANT_PARSE_FAILED")
        if products is None:
            errors.append("PRODUCT_PARSE_FAILED")
        return ExplicitHydrogenReactionAudit(reaction, errors=tuple(errors))

    errors = []
    if reactants.unmapped_atoms:
        errors.append("UNMAPPED_REACTANT_ATOM")
    if products.unmapped_atoms:
        errors.append("UNMAPPED_PRODUCT_ATOM")
    if reactants.duplicate_maps:
        errors.append("DUPLICATE_REACTANT_MAP")
    if products.duplicate_maps:
        errors.append("DUPLICATE_PRODUCT_MAP")

    reactant_maps = set(reactants.atoms_by_map)
    product_maps = set(products.atoms_by_map)
    if reactant_maps != product_maps:
        errors.append("MAP_INVENTORY_MISMATCH")
    elif reactants.atoms_by_map != products.atoms_by_map:
        errors.append("MAPPED_ATOM_IDENTITY_MISMATCH")

    if reactants.elemental_composition != products.elemental_composition:
        errors.append("FORMULA_IMBALANCE")
    if reactants.charge != products.charge:
        errors.append("CHARGE_IMBALANCE")

    changed_bonds = []
    all_bonds = set(reactants.bonds_by_maps) | set(products.bonds_by_maps)
    for atom_maps in sorted(all_bonds):
        before = reactants.bonds_by_maps.get(atom_maps, 0.0)
        after = products.bonds_by_maps.get(atom_maps, 0.0)
        if before != after:
            changed_bonds.append(MappedBondChange(atom_maps, before, after))

    hydrogen_maps = tuple(
        sorted(
            atom_map
            for atom_map, identity in reactants.atoms_by_map.items()
            if identity[0] == 1
        )
    )
    if not hydrogen_maps:
        errors.append("NO_EXPLICIT_MAPPED_HYDROGEN")

    center_maps = tuple(
        sorted({atom_map for change in changed_bonds for atom_map in change.atom_maps})
    )
    changed_hydrogen_maps = tuple(sorted(set(hydrogen_maps) & set(center_maps)))
    if not changed_hydrogen_maps:
        errors.append("NO_CHANGED_EXPLICIT_HYDROGEN")

    return ExplicitHydrogenReactionAudit(
        reaction=reaction,
        errors=tuple(errors),
        reactant_formula=reactants.formula,
        product_formula=products.formula,
        reactant_charge=reactants.charge,
        product_charge=products.charge,
        atom_maps=tuple(sorted(reactant_maps & product_maps)),
        explicit_hydrogen_maps=hydrogen_maps,
        changed_hydrogen_maps=changed_hydrogen_maps,
        reaction_center_maps=center_maps,
        changed_bonds=tuple(changed_bonds),
    )


__all__ = [
    "ExplicitHydrogenReactionAudit",
    "MappedBondChange",
    "audit_explicit_h_reaction",
    "reaction_smiles_from_annotated_text",
]
