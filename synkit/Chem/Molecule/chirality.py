"""Whole-molecule chirality classification by mirror automorphism.

This module is intentionally independent of reaction rules and reaction
stereo transport.  It asks one molecular question: is a molecule isomorphic
to its mirror image?
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Mapping

import networkx as nx
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    GetStereoisomerCount,
    StereoEnumerationOptions,
)

from synkit.Graph.Stereo.descriptors import (
    Reference,
    TetrahedralStereo,
    descriptor_id,
    parse_virtual_reference,
    virtual_reference,
)
from synkit.Graph.Stereo.identity import descriptor_relative_form
from synkit.IO.mol_to_graph import MolToGraph

__all__ = [
    "MolecularChirality",
    "MolecularChiralityAssessment",
    "MolecularChiralityOutcome",
    "MolecularChiralityResult",
    "UnspecifiedMolecularStereoError",
    "assess_molecular_chirality",
    "classify_molecular_chirality",
    "clear_molecular_chirality_cache",
    "is_molecular_chiral",
]


class MolecularChirality(str, Enum):
    """Global relationship between a molecule and its mirror image."""

    ACHIRAL = "Achiral"
    CHIRAL = "Chiral"


class MolecularChiralityOutcome(str, Enum):
    """Configuration-aware conclusion for possibly underspecified input."""

    NECESSARILY_ACHIRAL = "necessarily_achiral"
    NECESSARILY_CHIRAL = "necessarily_chiral"
    CONFIGURATION_DEPENDENT = "configuration_dependent"
    UNSUPPORTED_OR_INCOMPLETE = "unsupported_or_incomplete"


class UnspecifiedMolecularStereoError(ValueError):
    """Raised when strict binary classification receives unresolved stereo."""

    def __init__(self, loci: tuple[str, ...]) -> None:
        self.loci = loci
        joined = ", ".join(loci)
        super().__init__(f"Molecular stereochemistry is underspecified at: {joined}")


@dataclass(frozen=True)
class MolecularChiralityResult:
    """Evidence returned by whole-molecule mirror classification."""

    classification: MolecularChirality
    mirror_isomorphism: tuple[tuple[int, int], ...] | None
    descriptor_count: int
    completed_tetrahedral_centers: tuple[int, ...]
    completed_extended_tetrahedral_axes: tuple[tuple[int, int], ...] = ()
    completed_biaryl_atrop_axes: tuple[tuple[int, int], ...] = ()
    identity_profile: str = "element-isotope-hydrogen-connectivity"
    decision_method: str = "exact_mirror_isomorphism"
    input_stereo_status: str = "specified"
    unspecified_stereo_loci: tuple[str, ...] = ()

    @property
    def is_chiral(self) -> bool:
        """Return ``True`` when no orientation-preserving mirror map exists."""
        return self.classification is MolecularChirality.CHIRAL


@dataclass(frozen=True, eq=False)
class _MolecularAxialStereo:
    """Molecule-only relative orientation along a topological axis.

    SynKit's public reaction descriptor model does not yet claim extended
    tetrahedral support. This private value supplies only the operations used
    by the whole-molecule mirror matcher, without widening that public
    reaction capability boundary.
    """

    atoms: tuple[Reference, Reference, int, int, Reference, Reference]
    parity: int
    descriptor_class: str
    provenance: str

    _INVERSION = (1, 0, 2, 3, 4, 5)
    _PERMUTATIONS = (
        (0, 1, 2, 3, 4, 5),
        (1, 0, 2, 3, 5, 4),
        (4, 5, 3, 2, 0, 1),
        (5, 4, 3, 2, 1, 0),
    )

    @property
    def dependencies(self) -> frozenset[int]:
        return frozenset(value for value in self.atoms if type(value) is int)

    def _canonical_form(self) -> tuple[Reference, ...]:
        working = self.atoms
        if self.parity == -1:
            working = tuple(working[index] for index in self._INVERSION)  # type: ignore[assignment]
        return min(
            (
                tuple(working[index] for index in permutation)
                for permutation in self._PERMUTATIONS
            ),
            key=repr,
        )

    def invert(self) -> "_MolecularAxialStereo":
        return _MolecularAxialStereo(
            self.atoms,
            -self.parity,
            self.descriptor_class,
            self.provenance,
        )

    def relabel(
        self,
        mapping: Mapping[int, int],
    ) -> "_MolecularAxialStereo":
        def relabel_reference(value: Reference) -> Reference:
            if type(value) is int:
                return mapping.get(value, value)
            virtual = parse_virtual_reference(value)
            if virtual is None:
                return value
            return virtual_reference(
                virtual.kind,
                mapping.get(virtual.center, virtual.center),
            )

        return _MolecularAxialStereo(
            tuple(relabel_reference(value) for value in self.atoms),  # type: ignore[arg-type]
            self.parity,
            self.descriptor_class,
            self.provenance,
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _MolecularAxialStereo)
            and self.descriptor_class == other.descriptor_class
            and self._canonical_form() == other._canonical_form()
        )

    def __hash__(self) -> int:
        return hash((self.descriptor_class, self._canonical_form()))


@dataclass(frozen=True)
class MolecularChiralityAssessment:
    """Configuration-aware result over every enumerated stereo completion."""

    outcome: MolecularChiralityOutcome
    observed_classifications: tuple[MolecularChirality, ...]
    input_stereo_status: str
    unspecified_stereo_loci: tuple[str, ...]
    unsupported_stereo_loci: tuple[str, ...]
    theoretical_isomer_upper_bound: int
    evaluated_isomer_count: int
    enumeration_complete: bool
    max_isomers: int
    representative_isomers: tuple[tuple[str, MolecularChirality], ...] = ()

    @property
    def is_definitive(self) -> bool:
        """Return whether the outcome is proven despite possible truncation."""
        return self.outcome is not MolecularChiralityOutcome.UNSUPPORTED_OR_INCOMPLETE


def _molecular_node_match(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    """Match StereoMolGraph's explicit-H topology without expanding H atoms."""
    return (
        left.get("element") == right.get("element")
        and int(left.get("isotope", 0)) == int(right.get("isotope", 0))
        and int(left.get("hcount", 0)) == int(right.get("hcount", 0))
        and left.get("_molecular_colour") == right.get("_molecular_colour")
    )


def _connectivity_edge_match(
    _left: Mapping[str, Any],
    _right: Mapping[str, Any],
) -> bool:
    """Use connectivity, not one selected Lewis/resonance bond assignment."""
    return True


def _intern_colours(signatures: Mapping[int, Any]) -> dict[int, int]:
    """Intern comparable signatures into deterministic compact integers."""
    palette = {
        signature: index
        for index, signature in enumerate(sorted(set(signatures.values()), key=repr))
    }
    return {node: palette[signature] for node, signature in signatures.items()}


def _molecular_node_colours(graph: Any) -> dict[int, int]:
    """Return map-independent 1-WL molecular identity colours."""
    colours = _intern_colours(
        {
            node: (
                attributes.get("element"),
                int(attributes.get("isotope", 0)),
                int(attributes.get("hcount", 0)),
            )
            for node, attributes in graph.nodes(data=True)
        }
    )
    for _iteration in range(max(1, len(graph))):
        refined = _intern_colours(
            {
                node: (
                    colours[node],
                    tuple(sorted(colours[neighbor] for neighbor in graph[node])),
                )
                for node in graph
            }
        )
        if refined == colours:
            break
        colours = refined
    return colours


def _molecular_stereo_form(graph: Any) -> tuple[Any, ...]:
    """Return a safe colour-refinement prefilter for mirror isomorphism."""
    registry = graph.graph.get("stereo_descriptors", {})
    if not registry:
        return ()
    colours = _molecular_node_colours(graph)

    def resolve(reference: int | str) -> tuple[str, Any]:
        if type(reference) is int:
            if reference not in colours:
                raise ValueError(f"Stereo reference {reference} is absent.")
            return "atom", colours[reference]
        virtual = parse_virtual_reference(reference)
        if virtual is None or virtual.center not in colours:
            raise ValueError(f"Invalid virtual stereo reference: {reference!r}.")
        return "virtual", (virtual.kind, colours[virtual.center])

    return tuple(
        sorted(
            (
                descriptor_relative_form(descriptor, resolve)
                for descriptor in registry.values()
            ),
            key=repr,
        )
    )


def _resolved_registry_matches(
    source: tuple[Any, ...],
    target: frozenset[Any],
    mapping: Mapping[int, int],
) -> bool:
    """Reject any fully transported descriptor absent from the target."""
    mapped_nodes = mapping.keys()
    for descriptor in source:
        if not descriptor.dependencies.issubset(mapped_nodes):
            continue
        if descriptor.relabel(mapping) not in target:
            return False
    return True


class _MolecularStereoGraphMatcher(nx.isomorphism.GraphMatcher):
    """VF2 matcher that checks complete local frames during expansion."""

    def __init__(self, left: Any, right: Any) -> None:
        nx.set_node_attributes(
            left,
            _molecular_node_colours(left),
            "_molecular_colour",
        )
        nx.set_node_attributes(
            right,
            _molecular_node_colours(right),
            "_molecular_colour",
        )
        super().__init__(
            left,
            right,
            node_match=_molecular_node_match,
            edge_match=_connectivity_edge_match,
        )
        self._left_registry = tuple(left.graph.get("stereo_descriptors", {}).values())
        self._right_registry = tuple(right.graph.get("stereo_descriptors", {}).values())
        self._left_configurations = frozenset(self._left_registry)
        self._right_configurations = frozenset(self._right_registry)

    def semantic_feasibility(self, left_node: int, right_node: int) -> bool:
        if not super().semantic_feasibility(left_node, right_node):
            return False
        forward = dict(self.core_1)
        reverse = dict(self.core_2)
        forward[left_node] = right_node
        reverse[right_node] = left_node
        return _resolved_registry_matches(
            self._left_registry,
            self._right_configurations,
            forward,
        ) and _resolved_registry_matches(
            self._right_registry,
            self._left_configurations,
            reverse,
        )


def _indexed_copy(molecule: Chem.Mol) -> Chem.Mol:
    if molecule is None:
        raise ValueError("Molecular chirality classification requires a molecule.")
    working = Chem.Mol(molecule)
    for atom in working.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    Chem.AssignStereochemistry(working, cleanIt=False, force=True)
    return working


def _unspecified_stereo_loci(molecule: Chem.Mol) -> tuple[str, ...]:
    return tuple(
        sorted(
            f"{info.type}:{int(info.centeredOn)}"
            for info in Chem.FindPotentialStereo(molecule)
            if info.specified != Chem.StereoSpecified.Specified
        )
    )


def _unsupported_stereo_loci(molecule: Chem.Mol) -> tuple[str, ...]:
    supported = {
        Chem.StereoType.Atom_Tetrahedral,
        Chem.StereoType.Bond_Double,
    }
    return tuple(
        sorted(
            f"{info.type}:{int(info.centeredOn)}"
            for info in Chem.FindPotentialStereo(molecule)
            if info.specified != Chem.StereoSpecified.Specified
            and info.type not in supported
        )
    )


def _complete_tetrahedral_topology(
    molecule: Chem.Mol,
    registry: dict[str, Any],
) -> tuple[int, ...]:
    """Add fixed probes at eligible unrepresented sp3 topologies.

    These probes are not claims that each atom is a stereocentre. Their
    arbitrary common orientation lets the whole-molecule automorphism test
    decide which local probes cancel by symmetry and which collectively make
    the molecular topology chiral. This is the role of ``stereo_complete`` in
    the published StereoMolGraph validation protocol.
    """
    represented = {
        descriptor.atoms[0]
        for descriptor in registry.values()
        if descriptor.descriptor_class == "tetrahedral"
    }
    completed = []
    for atom in molecule.GetAtoms():
        center = atom.GetIdx() + 1
        if center in represented:
            continue
        if atom.GetHybridization() != Chem.HybridizationType.SP3:
            continue
        references: list[int | str] = [
            neighbor.GetIdx() + 1 for neighbor in atom.GetNeighbors()
        ]
        hidden_hydrogens = int(atom.GetNumExplicitHs()) + int(atom.GetNumImplicitHs())
        if len(references) == 3 and hidden_hydrogens == 1:
            references.append(virtual_reference("H", center))
        elif len(references) != 4:
            continue
        descriptor = TetrahedralStereo(
            (center, *references),
            1,
            "molecular_chirality:stereo_complete",
        )
        registry[descriptor_id(descriptor)] = descriptor
        completed.append(center)
    return tuple(completed)


def _cumulene_end_references(
    atom: Chem.Atom,
    axis_neighbor: int,
) -> tuple[Reference, Reference] | None:
    """Return the two terminal ligands of one cumulene end, if explicit."""
    center = atom.GetIdx() + 1
    references: list[Reference] = [
        neighbor.GetIdx() + 1
        for neighbor in atom.GetNeighbors()
        if neighbor.GetIdx() != axis_neighbor
    ]
    hidden_hydrogens = int(atom.GetNumExplicitHs()) + int(atom.GetNumImplicitHs())
    references.extend(virtual_reference("H", center) for _ in range(hidden_hydrogens))
    if len(references) != 2 or references[0] == references[1]:
        return None
    return references[0], references[1]


def _complete_extended_tetrahedral_topology(
    molecule: Chem.Mol,
    registry: dict[str, Any],
) -> tuple[tuple[int, int], ...]:
    """Add relative probes for topologically recoverable cumulene axes.

    An even number of consecutive double bonds places the terminal ligand
    planes orthogonally. The local ``@`` token is discarded by current RDKit
    SMILES parsing, but whole-molecule chiral/achiral classification needs
    only one relative orientation: the mirror matcher decides whether that
    orientation can be superposed on its inverse.
    """
    double_graph = nx.Graph()
    double_graph.add_edges_from(
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in molecule.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
    )
    completed = []
    for component in nx.connected_components(double_graph):
        axis = double_graph.subgraph(component)
        edge_count = axis.number_of_edges()
        if edge_count < 2 or edge_count % 2 != 0:
            continue
        if edge_count != axis.number_of_nodes() - 1:
            continue
        degrees = dict(axis.degree())
        if any(degree > 2 for degree in degrees.values()):
            continue
        ends = sorted(node for node, degree in degrees.items() if degree == 1)
        if len(ends) != 2:
            continue
        path = nx.shortest_path(axis, ends[0], ends[1])
        if any(molecule.GetAtomWithIdx(node).GetDegree() != 2 for node in path[1:-1]):
            continue
        left_refs = _cumulene_end_references(
            molecule.GetAtomWithIdx(path[0]),
            path[1],
        )
        right_refs = _cumulene_end_references(
            molecule.GetAtomWithIdx(path[-1]),
            path[-2],
        )
        if left_refs is None or right_refs is None:
            continue
        left, right = path[0] + 1, path[-1] + 1
        descriptor = _MolecularAxialStereo(
            (*left_refs, left, right, *right_refs),
            1,
            "extended_tetrahedral",
            "molecular_chirality:cumulene_topology",
        )
        registry[f"extended:{min(left, right)}-{max(left, right)}"] = descriptor
        completed.append((left, right))
    return tuple(completed)


def _complete_biaryl_atrop_topology(
    molecule: Chem.Mol,
    registry: dict[str, Any],
) -> tuple[tuple[int, int], ...]:
    """Add provisional axial probes to inter-ring aromatic single bonds.

    Connectivity identifies a stereogenic biaryl axis but cannot establish
    its rotational barrier. These probes therefore support molecular
    stereogenicity analysis; they are not kinetic claims of configurational
    stability.
    """
    represented = {
        descriptor.bond
        for descriptor in registry.values()
        if descriptor.descriptor_class == "atrop_bond"
    }
    completed = []
    for bond in molecule.GetBonds():
        if bond.GetIsAromatic():
            continue
        left_atom = bond.GetBeginAtom()
        right_atom = bond.GetEndAtom()
        if not left_atom.GetIsAromatic() or not right_atom.GetIsAromatic():
            continue
        left, right = left_atom.GetIdx() + 1, right_atom.GetIdx() + 1
        if frozenset((left, right)) in represented:
            continue
        left_refs = tuple(
            neighbor.GetIdx() + 1
            for neighbor in left_atom.GetNeighbors()
            if neighbor.GetIdx() != right_atom.GetIdx()
        )
        right_refs = tuple(
            neighbor.GetIdx() + 1
            for neighbor in right_atom.GetNeighbors()
            if neighbor.GetIdx() != left_atom.GetIdx()
        )
        if len(left_refs) != 2 or len(right_refs) != 2:
            continue
        descriptor = _MolecularAxialStereo(
            (*left_refs, left, right, *right_refs),
            1,
            "biaryl_atrop",
            "molecular_chirality:biaryl_topology",
        )
        registry[f"biaryl:{min(left, right)}-{max(left, right)}"] = descriptor
        completed.append((left, right))
    return tuple(completed)


def classify_molecular_chirality(
    molecule: Chem.Mol,
    *,
    stereo_complete: bool = True,
    require_specified: bool = False,
) -> MolecularChiralityResult:
    """Classify a molecule as globally chiral or achiral.

    The molecular identity profile uses element, isotope, total hydrogen
    count, and connectivity. It otherwise follows the published
    StereoMolGraph validation protocol and deliberately avoids raw charge and
    bond-order fields from a single Lewis/resonance form. Assigned local
    descriptors are retained. When ``stereo_complete`` is true, eligible
    unrepresented sp3, even-cumulene, and biaryl-axis topologies receive
    provisional probes before all parity-bearing configurations are reflected.
    Exact stereo-aware graph isomorphism is the final authority.

    This is a molecule classifier. It does not extract, apply, or compare
    reaction rules. The identity profile is a graph-topology convention, not a
    quantum/geometric chirality proof: bond-order distinctions that do not
    change hydrogen topology can be collapsed, as in validation case VS170.
    ``stereo_complete`` supplies provisional topology probes; it cannot recover
    a stereochemical configuration erased from an input SMILES. In particular,
    a topological biaryl candidate does not prove a high rotational barrier,
    and ordinary SMILES cannot distinguish helicene handedness. Set
    ``require_specified`` to reject RDKit-recognized unresolved input instead
    of returning the provisional binary result.
    """
    working = _indexed_copy(molecule)
    unspecified = _unspecified_stereo_loci(working)
    if require_specified and unspecified:
        raise UnspecifiedMolecularStereoError(unspecified)
    graph = MolToGraph(attr_profile="minimal").transform(
        working,
        use_index_as_atom_map=True,
    )
    registry = dict(graph.graph.get("stereo_descriptors", {}))
    completed = (
        _complete_tetrahedral_topology(working, registry) if stereo_complete else ()
    )
    completed_extended = (
        _complete_extended_tetrahedral_topology(working, registry)
        if stereo_complete
        else ()
    )
    completed_biaryl = (
        _complete_biaryl_atrop_topology(working, registry) if stereo_complete else ()
    )
    graph.graph["stereo_descriptors"] = registry

    mirror = graph.copy()
    mirror.graph["stereo_descriptors"] = {
        key: descriptor.invert() if descriptor.parity in {-1, 1} else descriptor
        for key, descriptor in registry.items()
    }
    if _molecular_stereo_form(graph) != _molecular_stereo_form(mirror):
        return MolecularChiralityResult(
            classification=MolecularChirality.CHIRAL,
            mirror_isomorphism=None,
            descriptor_count=len(registry),
            completed_tetrahedral_centers=completed,
            completed_extended_tetrahedral_axes=completed_extended,
            completed_biaryl_atrop_axes=completed_biaryl,
            decision_method="stereo_colour_prefilter",
            input_stereo_status=("underspecified" if unspecified else "specified"),
            unspecified_stereo_loci=unspecified,
        )
    matcher = _MolecularStereoGraphMatcher(graph, mirror)
    mapping = dict(matcher.mapping) if matcher.is_isomorphic() else None
    classification = (
        MolecularChirality.ACHIRAL if mapping is not None else MolecularChirality.CHIRAL
    )
    return MolecularChiralityResult(
        classification=classification,
        mirror_isomorphism=(
            None if mapping is None else tuple(sorted(mapping.items()))
        ),
        descriptor_count=len(registry),
        completed_tetrahedral_centers=completed,
        completed_extended_tetrahedral_axes=completed_extended,
        completed_biaryl_atrop_axes=completed_biaryl,
        input_stereo_status=("underspecified" if unspecified else "specified"),
        unspecified_stereo_loci=unspecified,
    )


def is_molecular_chiral(
    molecule: Chem.Mol,
    *,
    stereo_complete: bool = True,
    require_specified: bool = False,
) -> bool:
    """Return the boolean whole-molecule chirality classification."""
    return classify_molecular_chirality(
        molecule,
        stereo_complete=stereo_complete,
        require_specified=require_specified,
    ).is_chiral


def _canonical_isomeric_smiles(molecule: Chem.Mol) -> str:
    working = Chem.Mol(molecule)
    for atom in working.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(working, canonical=True, isomericSmiles=True)


@lru_cache(maxsize=4096)
def _cached_isomer_classification(
    isomeric_smiles: str,
    stereo_complete: bool,
) -> MolecularChirality:
    molecule = Chem.MolFromSmiles(isomeric_smiles)
    if molecule is None:
        raise ValueError(f"RDKit rejected cached isomeric SMILES: {isomeric_smiles}")
    return classify_molecular_chirality(
        molecule,
        stereo_complete=stereo_complete,
        require_specified=False,
    ).classification


def clear_molecular_chirality_cache() -> None:
    """Clear the bounded cache used by configuration-aware enumeration."""
    _cached_isomer_classification.cache_clear()


def assess_molecular_chirality(
    molecule: Chem.Mol,
    *,
    max_isomers: int = 256,
    stereo_complete: bool = True,
    try_embedding: bool = False,
    use_cache: bool = True,
) -> MolecularChiralityAssessment:
    """Assess chirality across supported completions of unresolved stereo.

    RDKit-supported unassigned tetrahedral atoms and double bonds are
    enumerated. A mixed chiral/achiral population proves
    ``configuration_dependent`` even when the search cap truncates the full
    population. A one-sided truncated population is never promoted to a
    necessary conclusion. Unsupported unresolved stereo types also fail
    closed as ``unsupported_or_incomplete``.
    """
    if molecule is None:
        raise ValueError("Molecular chirality assessment requires a molecule.")
    if type(max_isomers) is not int or max_isomers < 1:
        raise ValueError("max_isomers must be a positive integer.")

    working = Chem.Mol(molecule)
    for atom in working.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.AssignStereochemistry(working, cleanIt=False, force=True)
    unspecified = _unspecified_stereo_loci(working)
    unsupported = _unsupported_stereo_loci(working)
    if unsupported:
        return MolecularChiralityAssessment(
            outcome=MolecularChiralityOutcome.UNSUPPORTED_OR_INCOMPLETE,
            observed_classifications=(),
            input_stereo_status="underspecified",
            unspecified_stereo_loci=unspecified,
            unsupported_stereo_loci=unsupported,
            theoretical_isomer_upper_bound=0,
            evaluated_isomer_count=0,
            enumeration_complete=False,
            max_isomers=max_isomers,
        )

    options = StereoEnumerationOptions(
        tryEmbedding=try_embedding,
        onlyUnassigned=True,
        maxIsomers=max_isomers,
        rand=0x5A17,
        unique=True,
    )
    theoretical = int(GetStereoisomerCount(working, options=options))
    exhaustive = theoretical <= max_isomers
    classifications: set[MolecularChirality] = set()
    representatives: dict[MolecularChirality, str] = {}
    evaluated = 0
    stopped_after_decisive_mixture = False
    for isomer in EnumerateStereoisomers(working, options=options):
        isomeric_smiles = _canonical_isomeric_smiles(isomer)
        classification = (
            _cached_isomer_classification(isomeric_smiles, stereo_complete)
            if use_cache
            else classify_molecular_chirality(
                isomer,
                stereo_complete=stereo_complete,
            ).classification
        )
        evaluated += 1
        classifications.add(classification)
        representatives.setdefault(classification, isomeric_smiles)
        if len(classifications) > 1:
            stopped_after_decisive_mixture = True
            break

    observed = tuple(sorted(classifications, key=lambda value: value.value))
    evidence = tuple(
        (representatives[classification], classification) for classification in observed
    )
    if len(classifications) > 1:
        outcome = MolecularChiralityOutcome.CONFIGURATION_DEPENDENT
    elif not exhaustive or not classifications:
        outcome = MolecularChiralityOutcome.UNSUPPORTED_OR_INCOMPLETE
    elif MolecularChirality.CHIRAL in classifications:
        outcome = MolecularChiralityOutcome.NECESSARILY_CHIRAL
    else:
        outcome = MolecularChiralityOutcome.NECESSARILY_ACHIRAL

    return MolecularChiralityAssessment(
        outcome=outcome,
        observed_classifications=observed,
        input_stereo_status=("underspecified" if unspecified else "specified"),
        unspecified_stereo_loci=unspecified,
        unsupported_stereo_loci=(),
        theoretical_isomer_upper_bound=theoretical,
        evaluated_isomer_count=evaluated,
        enumeration_complete=(exhaustive and not stopped_after_decisive_mixture),
        max_isomers=max_isomers,
        representative_isomers=evidence,
    )
