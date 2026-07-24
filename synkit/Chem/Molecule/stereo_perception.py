"""Typed evidence for molecular stereo-element perception.

Perception answers whether the molecular input can support a stereo element.
It does not assign a configuration, a CIP label, configurational stability, or
whole-molecule chirality.  Keeping this boundary explicit prevents potential
topology from being promoted to an invented stereochemical state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Mapping

import networkx as nx
from networkx.algorithms.isomorphism import (
    GraphMatcher,
    categorical_edge_match,
    categorical_node_match,
)
from rdkit import Chem

from synkit.Graph.Stereo.descriptors import (
    Reference,
    TetrahedralStereo,
    virtual_reference,
)
from synkit.Graph.Stereo.supports import (
    AtomStereoSupport,
    AxisStereoSupport,
    BondStereoSupport,
    StereoSupport,
)


class StereoElementType(str, Enum):
    """Stereo-element classes currently recognized from molecular input."""

    TETRAHEDRAL = "tetrahedral"
    DOUBLE_BOND = "double_bond"
    CUMULENE_AXIS = "cumulene_axis"
    EXTENDED_CIS_TRANS = "extended_cis_trans"
    ATROP_AXIS = "atrop_axis"


class StereoConfigurationState(str, Enum):
    """Whether the input supplies a configuration for a perceived element."""

    SPECIFIED = "specified"
    UNSPECIFIED = "unspecified"


class TetrahedralConstitutionStatus(str, Enum):
    """Canonicalization conclusion for one tetrahedral carrier."""

    CONSTITUTIONALLY_DISTINCT = "constitutionally_distinct"
    STEREO_DEPENDENT_DISTINCT = "stereo_dependent_distinct"
    SYMMETRY_RELATED = "symmetry_related"


class TetrahedralFrameStatus(str, Enum):
    """Outcome of configuration-neutral tetrahedral frame construction."""

    CANONICAL = "canonical"
    SYMMETRY_RELATED = "symmetry_related"
    NEIGHBORHOOD_KEY_COLLISION = "neighborhood_key_collision"


@dataclass(frozen=True, order=True)
class LocalNeighborKey:
    """Recursive, configuration-neutral key for one ligand environment.

    ``digest`` is derived only from constitutional atom/bond attributes and
    recursively refined neighboring keys.  It never includes atom indices,
    supplied stereo, CIP properties, or coordinates.  Exact carrier-fixed
    automorphisms remain authoritative when two references share a key.
    """

    digest: str
    depth: int
    algorithm: str = "recursive_local_neighbor_sha256_v2"

    def __post_init__(self) -> None:
        if len(self.digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.digest
        ):
            raise ValueError("Local-neighbor keys require a SHA-256 hex digest.")
        if type(self.depth) is not int or self.depth < 0:
            raise ValueError("Local-neighbor key depth must be non-negative.")


@dataclass(frozen=True)
class LigandSymmetryClass:
    """Ligand slots equivalent under automorphisms fixing their owner."""

    references: tuple[Reference, ...]
    multiplicity: int
    neighborhood_key: LocalNeighborKey | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "references", tuple(self.references))
        if not self.references:
            raise ValueError("Ligand symmetry classes cannot be empty.")
        if type(self.multiplicity) is not int or self.multiplicity < 1:
            raise ValueError("Ligand symmetry multiplicity must be positive.")


@dataclass(frozen=True)
class TetrahedralConstitution:
    """Exact local-symmetry evidence for a tetrahedral carrier.

    ``ligand_classes`` is canonical up to atom relabelling: two material
    ligands share a class exactly when a rooted attributed-graph isomorphism
    fixing the center maps one to the other. At dependency depth zero only
    constitution participates. Later depths may include canonical stereo
    markers established at other centers; the focal center remains unmarked.
    Hidden hydrogens are one owner-scoped class with their actual multiplicity.
    """

    support: AtomStereoSupport
    ligand_classes: tuple[LigandSymmetryClass, ...]
    status: TetrahedralConstitutionStatus
    automorphism_witness_checks: int
    canonical_frame: tuple[Reference, ...] | None = None
    frame_status: TetrahedralFrameStatus = TetrahedralFrameStatus.SYMMETRY_RELATED
    refinement_depth: int = 0
    dependency_depth: int = 0
    stereo_marker_count: int = 0
    method: str = (
        "recursive_local_neighbor_sha256_v2" "+exact_rooted_isomorphism_tie_audit"
    )

    @property
    def ligand_count(self) -> int:
        return sum(item.multiplicity for item in self.ligand_classes)

    @property
    def confirms_stereogenic_center(self) -> bool:
        return self.status in {
            TetrahedralConstitutionStatus.CONSTITUTIONALLY_DISTINCT,
            TetrahedralConstitutionStatus.STEREO_DEPENDENT_DISTINCT,
        }

    @property
    def is_stereo_dependent(self) -> bool:
        return self.status is TetrahedralConstitutionStatus.STEREO_DEPENDENT_DISTINCT

    @property
    def has_canonical_frame(self) -> bool:
        return self.frame_status is TetrahedralFrameStatus.CANONICAL


_EXPECTED_SUPPORT = {
    StereoElementType.TETRAHEDRAL: AtomStereoSupport,
    StereoElementType.DOUBLE_BOND: BondStereoSupport,
    StereoElementType.CUMULENE_AXIS: AxisStereoSupport,
    StereoElementType.EXTENDED_CIS_TRANS: AxisStereoSupport,
    StereoElementType.ATROP_AXIS: AxisStereoSupport,
}


@dataclass(frozen=True)
class PotentialStereoElement:
    """One perceived stereo carrier, separate from configured stereo state.

    Atom references use zero-based RDKit indices.  ``source_identifier`` is a
    diagnostic pointer into the perception source, not part of chemical
    identity.  In particular, a returned axis carries no barrier or handedness
    claim.
    """

    element_type: StereoElementType
    support: StereoSupport
    configuration_state: StereoConfigurationState
    evidence_provenance: str
    source_identifier: str
    constitutional_evidence: TetrahedralConstitution | None = None
    configuration: TetrahedralStereo | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "element_type", StereoElementType(self.element_type))
        object.__setattr__(
            self,
            "configuration_state",
            StereoConfigurationState(self.configuration_state),
        )
        expected = _EXPECTED_SUPPORT[self.element_type]
        if not isinstance(self.support, expected):
            raise TypeError(
                f"{self.element_type.value} perception requires "
                f"{expected.__name__}, not {type(self.support).__name__}."
            )
        if not self.evidence_provenance:
            raise ValueError("Stereo perception provenance cannot be empty.")
        if not self.source_identifier:
            raise ValueError("Stereo perception source identifier cannot be empty.")
        if self.constitutional_evidence is not None:
            if self.element_type is not StereoElementType.TETRAHEDRAL:
                raise TypeError(
                    "Constitutional evidence is currently supported only for "
                    "tetrahedral elements."
                )
            if self.constitutional_evidence.support != self.support:
                raise ValueError(
                    "Constitutional evidence and perceived support disagree."
                )
        if self.configuration is not None:
            if self.element_type is not StereoElementType.TETRAHEDRAL:
                raise TypeError(
                    "Attached configuration is currently supported only for "
                    "tetrahedral elements."
                )
            if self.configuration.support != self.support:
                raise ValueError("Configuration and perceived support disagree.")
            if self.configuration_state is not StereoConfigurationState.SPECIFIED:
                raise ValueError(
                    "An attached configuration requires specified input evidence."
                )
            if self.constitutional_evidence is None:
                raise ValueError(
                    "Attached configuration requires constitutional evidence."
                )
            if self.constitutional_evidence.canonical_frame != self.configuration.atoms:
                raise ValueError(
                    "Attached configuration must use the constitutional frame."
                )

    @property
    def identifier(self) -> str:
        """Return a deterministic support identifier for diagnostics."""
        if isinstance(self.support, AtomStereoSupport):
            carrier = str(self.support.center)
        elif isinstance(self.support, BondStereoSupport):
            carrier = "-".join(str(value) for value in self.support.endpoints)
        else:
            carrier = "-".join(str(value) for value in self.support.path)
        return f"{self.element_type.value}:{carrier}"


@dataclass(frozen=True)
class TetrahedralPerceptionResult:
    """Fixed-point result for primary and stereo-dependent tetrahedral centers."""

    carrier_evidence: tuple[TetrahedralConstitution, ...]
    elements: tuple[PotentialStereoElement, ...]
    dependency_iterations: int
    stereo_markers: tuple[tuple[int, tuple[str, int]], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "carrier_evidence", tuple(self.carrier_evidence))
        object.__setattr__(self, "elements", tuple(self.elements))
        object.__setattr__(self, "stereo_markers", tuple(self.stereo_markers))
        if (
            type(self.dependency_iterations) is not int
            or self.dependency_iterations < 0
        ):
            raise ValueError("Dependency iteration count must be non-negative.")


def _configuration_state(info: object) -> StereoConfigurationState:
    specified = getattr(info, "specified", None)
    if specified == Chem.StereoSpecified.Specified:
        return StereoConfigurationState.SPECIFIED
    return StereoConfigurationState.UNSPECIFIED


def _cumulated_double_bonds(molecule: Chem.Mol) -> frozenset[int]:
    """Return double bonds that belong to a run of at least two double bonds."""
    double_bonds = [
        bond
        for bond in molecule.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
    ]
    incidence: dict[int, list[int]] = {}
    for bond in double_bonds:
        for atom in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()):
            incidence.setdefault(atom, []).append(bond.GetIdx())
    cumulated = {
        bond_index
        for bond_indices in incidence.values()
        if len(bond_indices) > 1
        for bond_index in bond_indices
    }
    return frozenset(cumulated)


def _hidden_hydrogen_count(atom: Chem.Atom) -> int:
    return int(atom.GetNumExplicitHs()) + int(atom.GetNumImplicitHs())


def _is_tetrahedral_carrier(atom: Chem.Atom) -> bool:
    """Return whether the represented constitution has four tetrahedral slots."""
    if atom.GetIsAromatic():
        return False
    if atom.GetHybridization() != Chem.HybridizationType.SP3:
        return False
    if any(bond.GetBondType() != Chem.BondType.SINGLE for bond in atom.GetBonds()):
        return False
    return atom.GetDegree() + _hidden_hydrogen_count(atom) == 4


def detect_tetrahedral_carriers(
    molecule: Chem.Mol,
) -> tuple[AtomStereoSupport, ...]:
    """Return broad tetrahedral carriers before ligand-symmetry refinement.

    This topology gate deliberately includes methane-like and repeated-ligand
    environments.  It answers whether four tetrahedral slots exist, not whether
    the four ligands are distinguishable.
    """
    if molecule is None:
        raise ValueError("Tetrahedral-carrier detection requires a molecule.")
    working = Chem.Mol(molecule)
    return tuple(
        AtomStereoSupport(atom.GetIdx())
        for atom in working.GetAtoms()
        if _is_tetrahedral_carrier(atom)
    )


def _constitutional_graph(
    molecule: Chem.Mol,
    center: int,
    stereo_markers: Mapping[int, tuple[str, int]] | None = None,
) -> nx.Graph:
    """Build the attributed component used for exact center stabilizers."""
    markers = stereo_markers or {}
    graph = nx.Graph()
    for atom in molecule.GetAtoms():
        index = atom.GetIdx()
        graph.add_node(
            index,
            atomic_number=atom.GetAtomicNum(),
            isotope=atom.GetIsotope(),
            formal_charge=atom.GetFormalCharge(),
            radical_electrons=atom.GetNumRadicalElectrons(),
            hcount=_hidden_hydrogen_count(atom),
            aromatic=atom.GetIsAromatic(),
            stereo_anchor=index == center,
            stereo_marker=None if index == center else markers.get(index),
        )
    for bond in molecule.GetBonds():
        graph.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_type=str(bond.GetBondType()),
            aromatic=bond.GetIsAromatic(),
        )
    component = nx.node_connected_component(graph, center)
    return graph.subgraph(component).copy()


def _local_key_digest(value: object) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _local_key_node_seed(attributes: Mapping[str, object]) -> tuple[object, ...]:
    """Return the constitution-only seed; map IDs and stereo are absent."""
    return (
        int(attributes["atomic_number"]),
        int(attributes["isotope"]),
        int(attributes["formal_charge"]),
        int(attributes["radical_electrons"]),
        int(attributes["hcount"]),
        bool(attributes["aromatic"]),
        bool(attributes["stereo_anchor"]),
        attributes["stereo_marker"],
    )


def _local_key_edge_seed(attributes: Mapping[str, object]) -> tuple[object, ...]:
    return (
        str(attributes["bond_type"]),
        bool(attributes["aromatic"]),
    )


def _local_key_graph(
    molecule: Chem.Mol,
    center: int,
    stereo_markers: Mapping[int, tuple[str, int]] | None = None,
) -> tuple[nx.Graph, tuple[object, ...]]:
    """Materialize center-owned hydrogens for local recursive refinement."""
    graph = _constitutional_graph(molecule, center, stereo_markers)
    hidden_hydrogens = _hidden_hydrogen_count(molecule.GetAtomWithIdx(center))
    virtual_nodes: list[object] = []
    if hidden_hydrogens:
        # The count is represented by explicit virtual leaves below, so it must
        # not also remain folded into the center seed.
        graph.nodes[center]["hcount"] = 0
        for occurrence in range(hidden_hydrogens):
            node = ("virtual_H", center, occurrence)
            virtual_nodes.append(node)
            graph.add_node(
                node,
                atomic_number=1,
                isotope=0,
                formal_charge=0,
                radical_electrons=0,
                hcount=0,
                aromatic=False,
                stereo_anchor=False,
                stereo_marker=None,
            )
            graph.add_edge(
                center,
                node,
                bond_type=str(Chem.BondType.SINGLE),
                aromatic=False,
            )
    return graph, tuple(virtual_nodes)


def tetrahedral_local_neighbor_keys(
    molecule: Chem.Mol,
    center: int,
    *,
    stereo_markers: Mapping[int, tuple[str, int]] | None = None,
) -> Mapping[Reference, LocalNeighborKey]:
    """Return recursive constitutional keys for a tetrahedral carrier's ligands.

    Refinement runs for the size of the complete connected component, including
    center-owned virtual hydrogens, so information can propagate across rings
    and through the full ligand environment.  SHA-256 values are ordering keys,
    not equality proof: exact center-stabilizer orbits audit every tie.
    """
    if molecule is None:
        raise ValueError("Tetrahedral local-key construction requires a molecule.")
    if type(center) is not int or center < 0 or center >= molecule.GetNumAtoms():
        raise ValueError(f"Tetrahedral center {center!r} is absent.")
    working = Chem.Mol(molecule)
    atom = working.GetAtomWithIdx(center)
    if not _is_tetrahedral_carrier(atom):
        raise ValueError(f"Atom {center} is not a supported tetrahedral carrier.")

    graph, virtual_nodes = _local_key_graph(
        working,
        center,
        stereo_markers,
    )
    colors = {
        node: _local_key_digest(
            ("local_neighbor_seed_v2", _local_key_node_seed(attributes))
        )
        for node, attributes in graph.nodes(data=True)
    }
    depth = len(graph)
    for _round in range(depth):
        colors = {
            node: _local_key_digest(
                (
                    "local_neighbor_refine_v2",
                    colors[node],
                    tuple(
                        sorted(
                            (
                                _local_key_edge_seed(graph.edges[node, neighbor]),
                                colors[neighbor],
                            )
                            for neighbor in graph.neighbors(node)
                        )
                    ),
                )
            )
            for node in graph
        }

    keys: dict[Reference, LocalNeighborKey] = {
        neighbor.GetIdx(): LocalNeighborKey(colors[neighbor.GetIdx()], depth)
        for neighbor in atom.GetNeighbors()
    }
    if virtual_nodes:
        virtual_digests = {colors[node] for node in virtual_nodes}
        if len(virtual_digests) != 1:
            raise RuntimeError(
                "Equivalent center-owned hydrogens received unequal keys."
            )
        keys[virtual_reference("H", center)] = LocalNeighborKey(
            virtual_digests.pop(),
            depth,
        )
    return keys


_CONSTITUTIONAL_NODE_ATTRIBUTES = (
    "atomic_number",
    "isotope",
    "formal_charge",
    "radical_electrons",
    "hcount",
    "aromatic",
    "stereo_anchor",
    "stereo_marker",
    "ligand_probe",
)
_CONSTITUTIONAL_EDGE_ATTRIBUTES = ("bond_type", "aromatic")


def _center_fixed_ligands_are_equivalent(
    graph: nx.Graph,
    left: int,
    right: int,
) -> bool:
    """Test one exact ligand-orbit relation without enumerating the group."""
    left_rooted = graph.copy()
    right_rooted = graph.copy()
    nx.set_node_attributes(left_rooted, False, "ligand_probe")
    nx.set_node_attributes(right_rooted, False, "ligand_probe")
    left_rooted.nodes[left]["ligand_probe"] = True
    right_rooted.nodes[right]["ligand_probe"] = True
    matcher = GraphMatcher(
        left_rooted,
        right_rooted,
        node_match=categorical_node_match(
            _CONSTITUTIONAL_NODE_ATTRIBUTES,
            (0, 0, 0, 0, 0, False, False, None, False),
        ),
        edge_match=categorical_edge_match(
            _CONSTITUTIONAL_EDGE_ATTRIBUTES,
            ("", False),
        ),
    )
    return matcher.is_isomorphic()


def _exact_material_ligand_classes(
    graph: nx.Graph,
    neighbors: set[int],
    local_keys: Mapping[Reference, LocalNeighborKey],
) -> tuple[list[LigandSymmetryClass], int]:
    """Partition only tied local-key groups by exact rooted isomorphism."""
    by_key: dict[LocalNeighborKey, list[int]] = {}
    for reference in neighbors:
        by_key.setdefault(local_keys[reference], []).append(reference)

    classes = []
    witness_checks = 0
    for key, references in by_key.items():
        parent = {reference: reference for reference in references}

        def find(reference: int) -> int:
            while parent[reference] != reference:
                parent[reference] = parent[parent[reference]]
                reference = parent[reference]
            return reference

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for left_position, left in enumerate(references):
            for right in references[left_position + 1 :]:
                witness_checks += 1
                if _center_fixed_ligands_are_equivalent(graph, left, right):
                    union(left, right)

        members: dict[int, list[int]] = {}
        for reference in references:
            members.setdefault(find(reference), []).append(reference)
        classes.extend(
            LigandSymmetryClass(
                tuple(sorted(values)),
                len(values),
                key,
            )
            for values in members.values()
        )
    return classes, witness_checks


def canonicalize_tetrahedral_constitution(
    molecule: Chem.Mol,
    center: int,
    *,
    stereo_markers: Mapping[int, tuple[str, int]] | None = None,
    dependency_depth: int = 0,
) -> TetrahedralConstitution:
    """Partition tetrahedral ligands under exact center-fixed isomorphisms.

    The depth-zero calculation intentionally ignores configured stereo and CIP.
    Later dependency rounds may read canonical markers from already resolved
    neighboring centers, but never configuration at the focal center. A
    ``constitutionally_distinct`` or ``stereo_dependent_distinct`` result
    confirms a tetrahedral stereogenic center. ``symmetry_related`` is not an
    achiral verdict; it means that the staged detector cannot confirm it.
    """
    if molecule is None:
        raise ValueError("Tetrahedral constitutional analysis requires a molecule.")
    if type(center) is not int or center < 0 or center >= molecule.GetNumAtoms():
        raise ValueError(f"Tetrahedral center {center!r} is absent.")
    if type(dependency_depth) is not int or dependency_depth < 0:
        raise ValueError("Tetrahedral dependency depth must be non-negative.")
    working = Chem.Mol(molecule)
    atom = working.GetAtomWithIdx(center)
    if not _is_tetrahedral_carrier(atom):
        raise ValueError(f"Atom {center} is not a supported tetrahedral carrier.")

    markers = {
        index: marker
        for index, marker in (stereo_markers or {}).items()
        if index != center
    }
    graph = _constitutional_graph(working, center, markers)
    local_keys = tetrahedral_local_neighbor_keys(
        working,
        center,
        stereo_markers=markers,
    )
    neighbors = {neighbor.GetIdx() for neighbor in atom.GetNeighbors()}
    classes, witness_checks = _exact_material_ligand_classes(
        graph,
        neighbors,
        local_keys,
    )
    hidden_hydrogens = _hidden_hydrogen_count(atom)
    if hidden_hydrogens:
        classes.append(
            LigandSymmetryClass(
                (virtual_reference("H", center),),
                hidden_hydrogens,
                local_keys[virtual_reference("H", center)],
            )
        )
    classes.sort(
        key=lambda item: (
            item.neighborhood_key,
            item.multiplicity,
            repr(item.references),
        )
    )
    status = (
        (
            TetrahedralConstitutionStatus.STEREO_DEPENDENT_DISTINCT
            if dependency_depth
            else TetrahedralConstitutionStatus.CONSTITUTIONALLY_DISTINCT
        )
        if len(classes) == 4 and all(item.multiplicity == 1 for item in classes)
        else TetrahedralConstitutionStatus.SYMMETRY_RELATED
    )
    class_keys = tuple(item.neighborhood_key for item in classes)
    if status is TetrahedralConstitutionStatus.SYMMETRY_RELATED:
        canonical_frame = None
        frame_status = TetrahedralFrameStatus.SYMMETRY_RELATED
    elif len(set(class_keys)) != len(class_keys):
        canonical_frame = None
        frame_status = TetrahedralFrameStatus.NEIGHBORHOOD_KEY_COLLISION
    else:
        canonical_frame = (
            center,
            *(item.references[0] for item in classes),
        )
        frame_status = TetrahedralFrameStatus.CANONICAL
    result = TetrahedralConstitution(
        support=AtomStereoSupport(center),
        ligand_classes=tuple(classes),
        status=status,
        automorphism_witness_checks=witness_checks,
        canonical_frame=canonical_frame,
        frame_status=frame_status,
        refinement_depth=next(iter(local_keys.values())).depth,
        dependency_depth=dependency_depth,
        stereo_marker_count=sum(
            index in graph and index != center for index in markers
        ),
    )
    if result.ligand_count != 4:
        raise RuntimeError("Tetrahedral constitutional evidence lost a ligand slot.")
    return result


def _reference_permutation_sign(
    values: tuple[Reference, ...],
    ordered: tuple[Reference, ...],
) -> int:
    positions = {value: index for index, value in enumerate(ordered)}
    permutation = tuple(positions[value] for value in values)
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def canonicalize_tetrahedral_configuration(
    molecule: Chem.Mol,
    descriptor: TetrahedralStereo,
    *,
    constitutional_evidence: TetrahedralConstitution | None = None,
) -> TetrahedralStereo:
    """Attach tetrahedral configuration to its constitution-derived frame.

    Frame construction is completed before this function reads parity.
    The returned parity is transported through the ligand permutation, so
    relative configuration is preserved without allowing it to influence the
    canonical ligand order.
    """
    if not isinstance(descriptor, TetrahedralStereo):
        raise TypeError(
            "Tetrahedral configuration attachment requires TetrahedralStereo."
        )
    if type(descriptor.center) is not int:
        raise ValueError("Tetrahedral configuration centers must be atom indices.")
    evidence = constitutional_evidence or canonicalize_tetrahedral_constitution(
        molecule,
        descriptor.center,
    )
    if evidence.support != descriptor.support:
        raise ValueError(
            "Tetrahedral constitutional evidence belongs to another center."
        )
    if evidence.canonical_frame is None:
        raise ValueError(
            "Tetrahedral configuration has no canonical constitutional frame: "
            f"{evidence.frame_status.value}."
        )
    canonical_references = tuple(evidence.canonical_frame[1:])
    supplied_references = tuple(descriptor.atoms[1:])
    if set(supplied_references) != set(canonical_references):
        raise ValueError(
            "Tetrahedral descriptor references do not match the constitutional frame."
        )
    parity = (
        None
        if descriptor.parity is None
        else descriptor.parity
        * _reference_permutation_sign(supplied_references, canonical_references)
    )
    return TetrahedralStereo(
        evidence.canonical_frame,  # type: ignore[arg-type]
        parity,
        descriptor.provenance,
    )


_TETRAHEDRAL_TAG_PARITY = {
    Chem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW: -1,
}


def _raw_tetrahedral_configuration(
    atom: Chem.Atom,
) -> TetrahedralStereo | None:
    """Read supplied RDKit orientation in zero-based local-reference order."""
    parity = _TETRAHEDRAL_TAG_PARITY.get(atom.GetChiralTag())
    if parity is None:
        return None
    center = atom.GetIdx()
    references: list[Reference] = [
        neighbor.GetIdx() for neighbor in atom.GetNeighbors()
    ]
    hidden_hydrogens = _hidden_hydrogen_count(atom)
    if len(references) == 3 and hidden_hydrogens == 1:
        references.append(virtual_reference("H", center))
    if len(references) != 4:
        raise ValueError(
            f"Configured tetrahedral center {center} does not expose four ligands."
        )
    return TetrahedralStereo(
        (center, *references),  # type: ignore[arg-type]
        parity,
        "rdkit",
    )


def analyze_tetrahedral_carriers(
    molecule: Chem.Mol,
) -> tuple[TetrahedralConstitution, ...]:
    """Return configuration-neutral evidence for every broad carrier."""
    return tuple(
        canonicalize_tetrahedral_constitution(molecule, support.center)
        for support in detect_tetrahedral_carriers(molecule)
    )


def _element_from_tetrahedral_evidence(
    molecule: Chem.Mol,
    evidence: TetrahedralConstitution,
) -> PotentialStereoElement:
    support = evidence.support
    if not evidence.confirms_stereogenic_center or not evidence.has_canonical_frame:
        raise ValueError("Tetrahedral evidence does not define a canonical element.")
    raw_configuration = _raw_tetrahedral_configuration(
        molecule.GetAtomWithIdx(support.center)
    )
    configuration = (
        canonicalize_tetrahedral_configuration(
            molecule,
            raw_configuration,
            constitutional_evidence=evidence,
        )
        if raw_configuration is not None
        else None
    )
    configuration_state = (
        StereoConfigurationState.SPECIFIED
        if raw_configuration is not None
        else StereoConfigurationState.UNSPECIFIED
    )
    provenance = (
        "synkit_stereo_dependent_recursive_neighborhood"
        if evidence.is_stereo_dependent
        else "synkit_recursive_neighborhood_and_exact_symmetry"
    )
    return PotentialStereoElement(
        StereoElementType.TETRAHEDRAL,
        support,
        configuration_state,
        provenance,
        f"Atom_Tetrahedral:{support.center}",
        evidence,
        configuration,
    )


def _canonical_stereo_marker(
    element: PotentialStereoElement,
) -> tuple[str, int] | None:
    configuration = element.configuration
    if configuration is None or configuration.parity is None:
        return None
    return configuration.descriptor_class, configuration.parity


def perceive_tetrahedral_stereo(
    molecule: Chem.Mol,
) -> TetrahedralPerceptionResult:
    """Resolve primary then dependent tetrahedral centers to a fixed point.

    Each dependency round reads only canonical configurations established in
    earlier rounds.  The focal center's own configuration is never a marker in
    its frame construction, and CIP properties are never read.
    """
    if molecule is None:
        raise ValueError("Tetrahedral perception requires a molecule.")
    working = Chem.Mol(molecule)
    initial = analyze_tetrahedral_carriers(working)
    evidence_by_center = {evidence.support.center: evidence for evidence in initial}
    elements: dict[int, PotentialStereoElement] = {}
    markers: dict[int, tuple[str, int]] = {}
    pending = set(evidence_by_center)

    for center, evidence in evidence_by_center.items():
        if not evidence.has_canonical_frame:
            continue
        element = _element_from_tetrahedral_evidence(working, evidence)
        elements[center] = element
        pending.discard(center)
        marker = _canonical_stereo_marker(element)
        if marker is not None:
            markers[center] = marker

    dependency_iterations = 0
    while pending and markers:
        dependency_iterations += 1
        marker_snapshot = dict(markers)
        resolved_this_round: list[
            tuple[int, TetrahedralConstitution, PotentialStereoElement]
        ] = []
        for center in sorted(pending):
            evidence = canonicalize_tetrahedral_constitution(
                working,
                center,
                stereo_markers=marker_snapshot,
                dependency_depth=dependency_iterations,
            )
            evidence_by_center[center] = evidence
            if not evidence.has_canonical_frame:
                continue
            element = _element_from_tetrahedral_evidence(working, evidence)
            resolved_this_round.append((center, evidence, element))

        new_marker_count = 0
        for center, _evidence, element in resolved_this_round:
            elements[center] = element
            pending.discard(center)
            marker = _canonical_stereo_marker(element)
            if marker is not None:
                markers[center] = marker
                new_marker_count += 1
        if new_marker_count == 0:
            break

    return TetrahedralPerceptionResult(
        carrier_evidence=tuple(
            evidence_by_center[center] for center in sorted(evidence_by_center)
        ),
        elements=tuple(elements[center] for center in sorted(elements)),
        dependency_iterations=dependency_iterations,
        stereo_markers=tuple(sorted(markers.items())),
    )


def detect_tetrahedral_elements(
    molecule: Chem.Mol,
) -> tuple[PotentialStereoElement, ...]:
    """Return primary and fixed-point stereo-dependent tetrahedral elements."""
    return perceive_tetrahedral_stereo(molecule).elements


def detect_constitutionally_distinct_tetrahedral_centers(
    molecule: Chem.Mol,
) -> tuple[TetrahedralConstitution, ...]:
    """Return tetrahedral centers confirmed by constitution alone."""
    return tuple(
        evidence
        for evidence in analyze_tetrahedral_carriers(molecule)
        if evidence.confirms_stereogenic_center
    )


def _rdkit_elements(molecule: Chem.Mol) -> tuple[PotentialStereoElement, ...]:
    """Project RDKit double-bond perception into typed support evidence."""
    cumulated_bonds = _cumulated_double_bonds(molecule)
    elements = []
    for info in Chem.FindPotentialStereo(molecule):
        centered_on = int(info.centeredOn)
        if info.type != Chem.StereoType.Bond_Double:
            continue
        if centered_on in cumulated_bonds:
            # A bond inside a cumulene is not an independent local E/Z element.
            continue
        bond = molecule.GetBondWithIdx(centered_on)
        elements.append(
            PotentialStereoElement(
                StereoElementType.DOUBLE_BOND,
                BondStereoSupport(
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                ),
                _configuration_state(info),
                "rdkit_find_potential_stereo",
                f"Bond_Double:{centered_on}",
            )
        )
    return tuple(elements)


def _cumulene_terminal_references(
    molecule: Chem.Mol,
    owner: int,
    path_neighbor: int,
) -> tuple[Reference, Reference] | None:
    atom = molecule.GetAtomWithIdx(owner)
    references: list[Reference] = [
        neighbor.GetIdx()
        for neighbor in atom.GetNeighbors()
        if neighbor.GetIdx() != path_neighbor
    ]
    hidden_hydrogens = int(atom.GetNumExplicitHs()) + int(atom.GetNumImplicitHs())
    references.extend(virtual_reference("H", owner) for _ in range(hidden_hydrogens))
    if len(references) != 2 or references[0] == references[1]:
        return None
    return references[0], references[1]


def _extended_cis_trans_elements(
    molecule: Chem.Mol,
) -> tuple[PotentialStereoElement, ...]:
    """Detect odd-bond extended cumulene E/Z supports without inventing state."""
    double_graph = nx.Graph()
    double_graph.add_edges_from(
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in molecule.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
    )
    elements = []
    for component in nx.connected_components(double_graph):
        path_graph = double_graph.subgraph(component)
        edge_count = path_graph.number_of_edges()
        if edge_count < 3 or edge_count % 2 != 1:
            continue
        if edge_count != path_graph.number_of_nodes() - 1:
            continue
        degrees = dict(path_graph.degree())
        if any(degree > 2 for degree in degrees.values()):
            continue
        ends = sorted(node for node, degree in degrees.items() if degree == 1)
        if len(ends) != 2:
            continue
        path = tuple(nx.shortest_path(path_graph, ends[0], ends[1]))
        if any(molecule.GetAtomWithIdx(node).GetDegree() != 2 for node in path[1:-1]):
            continue
        left = _cumulene_terminal_references(
            molecule,
            path[0],
            path[1],
        )
        right = _cumulene_terminal_references(
            molecule,
            path[-1],
            path[-2],
        )
        if left is None or right is None:
            continue
        support = AxisStereoSupport(path, (left, right))
        elements.append(
            PotentialStereoElement(
                StereoElementType.EXTENDED_CIS_TRANS,
                support,
                StereoConfigurationState.UNSPECIFIED,
                "synkit_odd_cumulene_path_perception",
                "Extended_CisTrans:" + "-".join(str(atom) for atom in path),
            )
        )
    return tuple(elements)


def _axis_elements(molecule: Chem.Mol) -> tuple[PotentialStereoElement, ...]:
    """Adapt the existing conservative axial-locus detector."""
    # Local import avoids making whole-molecule chirality a dependency of the
    # typed support module at import time.  The legacy API remains authoritative
    # for this additive migration slice.
    from .chirality import detect_potential_stereo_loci

    elements = []
    for locus in detect_potential_stereo_loci(molecule):
        elements.append(
            PotentialStereoElement(
                StereoElementType(locus.locus_type.value),
                locus.support,
                StereoConfigurationState.UNSPECIFIED,
                locus.evidence_provenance,
                locus.identifier,
            )
        )
    return tuple(elements)


def detect_potential_stereo_elements(
    molecule: Chem.Mol,
) -> tuple[PotentialStereoElement, ...]:
    """Detect supported atom, bond, and axis stereo carriers.

    The result is an intentionally non-promoting perception inventory.  It
    includes configured atom/bond elements but does not reconstruct their
    descriptor frames.  Cumulated double bonds are not emitted as independent
    E/Z elements. Even-bond cumulenes are emitted as axes and odd-bond
    extended cumulenes are emitted once as path-level cis/trans supports.
    """
    if molecule is None:
        raise ValueError("Stereo-element perception requires a molecule.")
    working = Chem.Mol(molecule)
    Chem.AssignStereochemistry(working, cleanIt=False, force=True)
    elements = (
        *detect_tetrahedral_elements(working),
        *_rdkit_elements(working),
        *_axis_elements(working),
        *_extended_cis_trans_elements(working),
    )
    return tuple(sorted(elements, key=lambda element: element.identifier))


__all__ = [
    "LigandSymmetryClass",
    "LocalNeighborKey",
    "PotentialStereoElement",
    "StereoConfigurationState",
    "StereoElementType",
    "TetrahedralConstitution",
    "TetrahedralConstitutionStatus",
    "TetrahedralFrameStatus",
    "TetrahedralPerceptionResult",
    "analyze_tetrahedral_carriers",
    "canonicalize_tetrahedral_configuration",
    "canonicalize_tetrahedral_constitution",
    "detect_constitutionally_distinct_tetrahedral_centers",
    "detect_potential_stereo_elements",
    "detect_tetrahedral_carriers",
    "detect_tetrahedral_elements",
    "perceive_tetrahedral_stereo",
    "tetrahedral_local_neighbor_keys",
]
