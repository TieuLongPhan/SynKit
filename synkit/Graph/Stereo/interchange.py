"""Loss-accounted descriptor interchange with StereoMolGraph.

StereoMolGraph uses ``None`` for any ligand absent from its molecular graph.
SynKit requires the chemical identity of that virtual ligand because hydrogen
and a lone pair produce different stereo dependencies.  This module therefore
allows ``None`` only at the transport boundary and requires either an exact
slot sidecar or chemical resolution from the source molecule/Lewis graph.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from rdkit import Chem

from .descriptors import (
    AtropBondStereo,
    OctahedralStereo,
    PlanarBondStereo,
    SquarePlanarStereo,
    StereoValue,
    TetrahedralStereo,
    TrigonalBipyramidalStereo,
    descriptor_id,
    parse_virtual_reference,
    virtual_reference,
)

StereoProjection = Literal["all", "rdkit_2d"]
VirtualReferenceResolver = Callable[[int, int], str | None]

_SYNKIT_TYPES = {
    "tetrahedral": TetrahedralStereo,
    "square_planar": SquarePlanarStereo,
    "trigonal_bipyramidal": TrigonalBipyramidalStereo,
    "octahedral": OctahedralStereo,
    "planar_bond": PlanarBondStereo,
    "atrop_bond": AtropBondStereo,
}
_EXTERNAL_CLASS_NAMES = {
    "Tetrahedral": "tetrahedral",
    "SquarePlanar": "square_planar",
    "TrigonalBipyramidal": "trigonal_bipyramidal",
    "Octahedral": "octahedral",
    "PlanarBond": "planar_bond",
    "AtropBond": "atrop_bond",
}
_ATOM_CLASSES = frozenset(
    {"tetrahedral", "square_planar", "trigonal_bipyramidal", "octahedral"}
)
_EXPECTED_LENGTHS = {
    "tetrahedral": 5,
    "square_planar": 5,
    "trigonal_bipyramidal": 6,
    "octahedral": 7,
    "planar_bond": 6,
    "atrop_bond": 6,
}


class StereoInterchangeError(ValueError):
    """Raised when descriptor interchange would require a silent guess."""


@dataclass(frozen=True)
class VirtualReferenceSidecar:
    """Exact typed identities for StereoMolGraph ``None`` transport slots."""

    descriptor_class: str
    locus: str
    references: tuple[tuple[int, str], ...] = ()

    def __post_init__(self) -> None:
        if self.descriptor_class not in _SYNKIT_TYPES:
            raise StereoInterchangeError(
                f"Unsupported sidecar descriptor class: {self.descriptor_class!r}."
            )
        slots = [slot for slot, _reference in self.references]
        if len(slots) != len(set(slots)):
            raise StereoInterchangeError("Virtual-reference sidecar slots must be unique.")
        for slot, reference in self.references:
            if slot < 0 or parse_virtual_reference(reference) is None:
                raise StereoInterchangeError(
                    f"Invalid virtual-reference sidecar entry: {(slot, reference)!r}."
                )

    def by_slot(self) -> Mapping[int, str]:
        return MappingProxyType(dict(self.references))


@dataclass(frozen=True)
class StereoInterchangeIssue:
    """One explicit exclusion or information loss at the bridge boundary."""

    code: str
    locus: str
    detail: str


@dataclass(frozen=True)
class StereoInterchangeReport:
    """Normalized registry plus an auditable projection/loss ledger."""

    descriptors: Mapping[str, StereoValue]
    exclusions: tuple[StereoInterchangeIssue, ...] = ()
    losses: tuple[StereoInterchangeIssue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "descriptors", MappingProxyType(dict(self.descriptors)))

    @property
    def lossless(self) -> bool:
        """Whether every descriptor inside the selected projection was resolved."""
        return not self.losses


def _stereomolgraph_descriptor_types() -> Mapping[str, type[Any]]:
    try:
        from stereomolgraph.stereodescriptors import (
            AtropBond,
            Octahedral,
            PlanarBond,
            SquarePlanar,
            Tetrahedral,
            TrigonalBipyramidal,
        )
    except ImportError as error:
        raise StereoInterchangeError(
            "StereoMolGraph is not installed; pass descriptor_types explicitly."
        ) from error
    return {
        "tetrahedral": Tetrahedral,
        "square_planar": SquarePlanar,
        "trigonal_bipyramidal": TrigonalBipyramidal,
        "octahedral": Octahedral,
        "planar_bond": PlanarBond,
        "atrop_bond": AtropBond,
    }


def _external_descriptor_class(descriptor: Any) -> str:
    external_name = type(descriptor).__name__
    try:
        return _EXTERNAL_CLASS_NAMES[external_name]
    except KeyError as error:
        raise StereoInterchangeError(
            f"Unsupported StereoMolGraph descriptor class: {external_name!r}."
        ) from error


def _locus_from_atoms(descriptor_class: str, atoms: tuple[Any, ...]) -> str:
    if descriptor_class in _ATOM_CLASSES:
        if not atoms or type(atoms[0]) is not int:
            raise StereoInterchangeError(
                "Atom-centered descriptor transport requires an integer center."
            )
        return f"atom:{atoms[0]}"
    if len(atoms) < 4 or type(atoms[2]) is not int or type(atoms[3]) is not int:
        raise StereoInterchangeError(
            "Bond-centered descriptor transport requires integer central atoms."
        )
    left, right = sorted((atoms[2], atoms[3]))
    return f"bond:{left}-{right}"


def _slot_owner(descriptor_class: str, atoms: tuple[Any, ...], slot: int) -> int:
    if descriptor_class in _ATOM_CLASSES:
        if slot == 0 or type(atoms[0]) is not int:
            raise StereoInterchangeError(
                "A virtual reference cannot replace an atom stereocenter."
            )
        return atoms[0]
    if slot in {0, 1} and type(atoms[2]) is int:
        return atoms[2]
    if slot in {4, 5} and type(atoms[3]) is int:
        return atoms[3]
    raise StereoInterchangeError(
        "A virtual reference cannot replace a central bond atom."
    )


def _external_transport_state(
    descriptor: Any,
    descriptor_class: str,
) -> tuple[tuple[Any, ...], Any]:
    try:
        atoms = tuple(descriptor.atoms)
        parity = descriptor.parity
    except (AttributeError, TypeError) as error:
        raise StereoInterchangeError(
            "StereoMolGraph descriptor transport requires atoms and parity."
        ) from error
    expected_length = _EXPECTED_LENGTHS[descriptor_class]
    if len(atoms) != expected_length:
        raise StereoInterchangeError(
            f"{descriptor_class} transport requires {expected_length} references; "
            f"observed {len(atoms)}."
        )
    return atoms, parity


def synkit_descriptor_to_stereomolgraph(
    descriptor: StereoValue,
    *,
    descriptor_types: Mapping[str, type[Any]] | None = None,
) -> tuple[Any, VirtualReferenceSidecar]:
    """Convert one descriptor and preserve typed virtual slots in a sidecar."""
    external_types = descriptor_types or _stereomolgraph_descriptor_types()
    try:
        external_type = external_types[descriptor.descriptor_class]
    except KeyError as error:
        raise StereoInterchangeError(
            f"No StereoMolGraph type for {descriptor.descriptor_class!r}."
        ) from error

    transported: list[int | None] = []
    references: list[tuple[int, str]] = []
    for slot, value in enumerate(descriptor.atoms):
        if type(value) is int:
            transported.append(value)
            continue
        virtual = parse_virtual_reference(value)
        if virtual is None:
            raise StereoInterchangeError(f"Invalid SynKit stereo reference: {value!r}.")
        transported.append(None)
        references.append((slot, str(virtual)))

    sidecar = VirtualReferenceSidecar(
        descriptor.descriptor_class,
        descriptor_id(descriptor),
        tuple(references),
    )
    return external_type(tuple(transported), descriptor.parity), sidecar


def stereomolgraph_descriptor_to_synkit(
    descriptor: Any,
    *,
    virtual_references: VirtualReferenceSidecar | None = None,
    reference_resolver: VirtualReferenceResolver | None = None,
    provenance: str | None = "stereomolgraph",
) -> StereoValue:
    """Convert one descriptor, rejecting every unresolved ``None`` slot."""
    descriptor_class = _external_descriptor_class(descriptor)
    atoms, parity = _external_transport_state(descriptor, descriptor_class)
    locus = _locus_from_atoms(descriptor_class, atoms)
    supplied: Mapping[int, str] = {}
    if virtual_references is not None:
        if (
            virtual_references.descriptor_class != descriptor_class
            or virtual_references.locus != locus
        ):
            raise StereoInterchangeError(
                "Virtual-reference sidecar does not belong to this descriptor."
            )
        supplied = virtual_references.by_slot()

    used_slots: set[int] = set()
    resolved: list[int | str] = []
    for slot, value in enumerate(atoms):
        if type(value) is int:
            resolved.append(value)
            continue
        if value is not None:
            raise StereoInterchangeError(
                f"StereoMolGraph references must be integers or None, got {value!r}."
            )
        owner = _slot_owner(descriptor_class, atoms, slot)
        reference = supplied.get(slot)
        if reference is None and reference_resolver is not None:
            reference = reference_resolver(owner, slot)
        virtual = parse_virtual_reference(reference)
        if virtual is None or virtual.center != owner:
            raise StereoInterchangeError(
                f"Unresolved virtual reference at {locus} slot {slot}; "
                "an owner-matched @H or @LP identity is required."
            )
        resolved.append(str(virtual))
        used_slots.add(slot)

    unused_slots = set(supplied) - used_slots
    if unused_slots:
        raise StereoInterchangeError(
            f"Sidecar contains non-virtual slots for {locus}: {sorted(unused_slots)}."
        )
    descriptor_type = _SYNKIT_TYPES[descriptor_class]
    try:
        return descriptor_type(  # type: ignore[call-arg,return-value]
            tuple(resolved),
            parity,
            provenance,
        )
    except (TypeError, ValueError) as error:
        raise StereoInterchangeError(
            f"Invalid {descriptor_class} descriptor at {locus}: {error}"
        ) from error


def _source_atoms_by_id(molecule: Chem.Mol) -> dict[int, Chem.Atom]:
    values: dict[int, Chem.Atom] = {}
    for atom in molecule.GetAtoms():
        atom_id = int(atom.GetAtomMapNum())
        if atom_id <= 0:
            raise StereoInterchangeError(
                "Direct StereoMolGraph interchange requires non-zero atom maps."
            )
        if atom_id in values:
            raise StereoInterchangeError(
                "Direct StereoMolGraph interchange requires unique atom maps."
            )
        values[atom_id] = atom
    return values


def _lewis_lone_pairs_by_id(lewis_graph: Any | None) -> dict[int, int]:
    if lewis_graph is None:
        return {}
    values: dict[int, int] = {}
    for node, attributes in lewis_graph.nodes(data=True):
        atom_id = attributes.get("atom_map") or node
        if type(atom_id) is int:
            values[atom_id] = int(
                attributes.get("estimated_lone_pairs", attributes.get("lone_pairs", 0))
            )
    return values


def _chemical_reference_resolver(
    atoms_by_id: Mapping[int, Chem.Atom],
    lone_pairs_by_id: Mapping[int, int],
    *,
    use_rdkit_lone_pair_fallback: bool,
) -> VirtualReferenceResolver:
    def resolve(owner: int, _slot: int) -> str | None:
        atom = atoms_by_id.get(owner)
        if atom is None:
            return None
        hidden_hydrogens = int(atom.GetNumExplicitHs()) + int(atom.GetNumImplicitHs())
        if hidden_hydrogens > 0:
            return virtual_reference("H", owner)
        if lone_pairs_by_id.get(owner, 0) > 0:
            return virtual_reference("LP", owner)
        if use_rdkit_lone_pair_fallback:
            from synkit.IO.mol_to_graph import MolToGraph

            if MolToGraph.estimate_lone_pairs(atom) > 0:
                return virtual_reference("LP", owner)
        return None

    return resolve


def _rdkit_2d_projection_exclusion(
    descriptor_class: str,
    atoms: tuple[Any, ...],
    locus: str,
    *,
    tetrahedral_loci: set[int],
    ez_loci: set[frozenset[int]],
) -> StereoInterchangeIssue | None:
    if descriptor_class in _ATOM_CLASSES:
        if descriptor_class != "tetrahedral":
            return StereoInterchangeIssue(
                "CLASS_OUTSIDE_RDKIT_2D_PROJECTION",
                locus,
                "Atom descriptor class is outside SynKit's RDKit 2D subset.",
            )
        if atoms[0] not in tetrahedral_loci:
            return StereoInterchangeIssue(
                "UNSPECIFIED_ATOM_STEREO_EXCLUDED",
                locus,
                "Tetrahedral topology was not stereospecified by RDKit.",
            )
        return None
    if descriptor_class != "planar_bond":
        return StereoInterchangeIssue(
            "CLASS_OUTSIDE_RDKIT_2D_PROJECTION",
            locus,
            "Bond descriptor class is outside SynKit's RDKit 2D subset.",
        )
    if frozenset(atoms[2:4]) in ez_loci:
        return None
    return StereoInterchangeIssue(
        "TOPOLOGY_OR_UNSPECIFIED_BOND_STEREO_EXCLUDED",
        locus,
        "Planar-bond topology was not stereospecified as E/Z by RDKit.",
    )


def stereomolgraph_registry_to_synkit(
    stereo_mol_graph: Any,
    *,
    source_molecule: Chem.Mol,
    lewis_graph: Any | None = None,
    projection: StereoProjection = "rdkit_2d",
) -> StereoInterchangeReport:
    """Import a normalized registry directly, without regenerating bond orders.

    ``projection='rdkit_2d'`` selects the shared RDKit-authored tetrahedral and
    E/Z boundary. StereoMolGraph's topology-inferred, unspecified, and richer
    descriptor loci are reported as exclusions rather than false mismatches.
    """
    if projection not in {"all", "rdkit_2d"}:
        raise ValueError("projection must be 'all' or 'rdkit_2d'.")

    molecule = Chem.Mol(source_molecule)
    Chem.AssignStereochemistry(molecule, cleanIt=False, force=True)
    atoms_by_id = _source_atoms_by_id(molecule)
    lone_pairs_by_id = _lewis_lone_pairs_by_id(lewis_graph)
    tetrahedral_tags = {
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    }
    tetrahedral_loci = {
        atom_id
        for atom_id, atom in atoms_by_id.items()
        if atom.GetChiralTag() in tetrahedral_tags
    }
    ez_loci = {
        frozenset(
            {
                int(bond.GetBeginAtom().GetAtomMapNum()),
                int(bond.GetEndAtom().GetAtomMapNum()),
            }
        )
        for bond in molecule.GetBonds()
        if bond.GetStereo() in {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ}
    }

    resolve_reference = _chemical_reference_resolver(
        atoms_by_id,
        lone_pairs_by_id,
        use_rdkit_lone_pair_fallback=lewis_graph is None,
    )

    external_descriptors = tuple(stereo_mol_graph.atom_stereo.values()) + tuple(
        stereo_mol_graph.bond_stereo.values()
    )
    imported: dict[str, StereoValue] = {}
    exclusions: list[StereoInterchangeIssue] = []
    losses: list[StereoInterchangeIssue] = []
    for external in external_descriptors:
        try:
            descriptor_class = _external_descriptor_class(external)
            atoms = tuple(external.atoms)
            locus = _locus_from_atoms(descriptor_class, atoms)
        except StereoInterchangeError as error:
            losses.append(
                StereoInterchangeIssue("UNSUPPORTED_DESCRIPTOR", "unknown", str(error))
            )
            continue

        if projection == "rdkit_2d":
            exclusion = _rdkit_2d_projection_exclusion(
                descriptor_class,
                atoms,
                locus,
                tetrahedral_loci=tetrahedral_loci,
                ez_loci=ez_loci,
            )
            if exclusion is not None:
                exclusions.append(exclusion)
                continue

        try:
            converted = stereomolgraph_descriptor_to_synkit(
                external,
                reference_resolver=resolve_reference,
                provenance="stereomolgraph:direct",
            )
        except StereoInterchangeError as error:
            losses.append(
                StereoInterchangeIssue("UNRESOLVED_REFERENCE", locus, str(error))
            )
            continue
        key = descriptor_id(converted)
        previous = imported.get(key)
        if previous is not None and previous != converted:
            losses.append(
                StereoInterchangeIssue(
                    "LOCUS_COLLISION",
                    key,
                    "Two non-equivalent descriptors project onto the same SynKit locus.",
                )
            )
            continue
        imported[key] = converted

    return StereoInterchangeReport(imported, tuple(exclusions), tuple(losses))
