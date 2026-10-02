"""Independent, witnessed CIP ligand ranking over molecular constitution.

The implementation follows the hierarchical ordering in IUPAC Blue Book
P-92: Sequence Rule 1a is exhausted before duplicate-node Rule 1b, followed
by isotope Rule 2. Sequence Rule 3 and the reference-independent, single-pair
subset of Rules 4c/5 are witnessed explicitly; multi-unit Rule 4b continues
to fail closed. No RDKit CIPCode or CIPRank property is read.

Reference: https://iupac.qmul.ac.uk/BlueBook/P9.html#92010000
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from itertools import combinations
from typing import Any, Iterable

from rdkit import Chem

from synkit.Graph.Stereo import descriptor_id, parse_virtual_reference

Reference = int | str

_STEREO_HIGH_LABELS = frozenset({"R", "M", "Z", "r", "m", "z"})
_STEREO_LOW_LABELS = frozenset({"S", "P", "E", "s", "p", "e"})
_STEREO_LABEL_FAMILIES = {
    "R": "center",
    "S": "center",
    "r": "center",
    "s": "center",
    "M": "axis",
    "P": "axis",
    "m": "axis",
    "p": "axis",
    "Z": "planar",
    "E": "planar",
    "z": "planar",
    "e": "planar",
}
_REFLECTED_STEREO_LABEL = {
    "R": "S",
    "S": "R",
    "M": "P",
    "P": "M",
    "r": "r",
    "s": "s",
    "m": "m",
    "p": "p",
    "Z": "Z",
    "E": "E",
    "z": "z",
    "e": "e",
}


class CIPSequenceRule(str, Enum):
    RULE_1A_ATOMIC_NUMBER = "1a_atomic_number"
    RULE_1B_DUPLICATE_DISTANCE = "1b_duplicate_distance"
    RULE_2_ISOTOPE_MASS = "2_isotope_mass"
    RULE_3_SEQUENCE_GEOMETRY = "3_sequence_geometry"
    RULE_4_STEREOGENIC_UNIT = "4_stereogenic_unit"
    RULE_5_REFLECTION_VARIANT = "5_reflection_variant"


class CIPComparisonOutcome(str, Enum):
    LEFT_HIGHER = "left_higher"
    RIGHT_HIGHER = "right_higher"
    TIE = "tie"
    UNSUPPORTED = "unsupported"


class CIPTermination(str, Enum):
    EXHAUSTED = "exhausted"
    VIRTUAL_REFERENCE = "virtual_reference"
    DEPTH_CAP = "depth_cap"


@dataclass(frozen=True)
class CIPNodeEvidence:
    """One material, duplicate, or virtual node in a ligand digraph sphere."""

    depth: int
    atomic_number: float
    mass: float
    atom_index: int | None
    path: tuple[int, ...]
    duplicate: bool = False
    duplicate_source_depth: int | None = None
    duplicate_kind: str | None = None

    @property
    def atomic_key(self) -> float:
        return self.atomic_number

    @property
    def isotope_key(self) -> tuple[float, float]:
        return self.atomic_number, self.mass


@dataclass(frozen=True)
class CIPSphereEvidence:
    depth: int
    nodes: tuple[CIPNodeEvidence, ...]

    @property
    def atomic_signature(self) -> tuple[float, ...]:
        return tuple(
            sorted(
                (node.atomic_number for node in self.nodes),
                reverse=True,
            )
        )

    @property
    def isotope_signature(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            sorted(
                (node.isotope_key for node in self.nodes),
                reverse=True,
            )
        )

    @property
    def duplicate_signature(self) -> tuple[tuple[float, int], ...]:
        values = (
            (node.atomic_number, -int(node.duplicate_source_depth))
            for node in self.nodes
            if node.duplicate and node.duplicate_source_depth is not None
        )
        return tuple(sorted(values, reverse=True))


@dataclass(frozen=True)
class CIPStereogenicUnitEvidence:
    descriptor_class: str
    identifier: str
    parity: int | None
    label: str | None
    depth: int
    encountered_atoms: tuple[int, ...]


@dataclass(frozen=True)
class CIPLigandEvidence:
    center: int
    reference: Reference
    spheres: tuple[CIPSphereEvidence, ...]
    stereogenic_units: tuple[CIPStereogenicUnitEvidence, ...]
    unsupported_features: tuple[str, ...]
    termination: CIPTermination

    @property
    def digest(self) -> str:
        return sha256(repr(self).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CIPComparison:
    center: int
    left_reference: Reference
    right_reference: Reference
    outcome: CIPComparisonOutcome
    deciding_rule: CIPSequenceRule | None
    depth: int | None
    left_witness: tuple[Any, ...] | None
    right_witness: tuple[Any, ...] | None
    reason: str
    left_evidence: CIPLigandEvidence
    right_evidence: CIPLigandEvidence

    @property
    def decided(self) -> bool:
        return self.outcome in {
            CIPComparisonOutcome.LEFT_HIGHER,
            CIPComparisonOutcome.RIGHT_HIGHER,
        }


@dataclass(frozen=True)
class CIPRanking:
    center: int
    references: tuple[Reference, ...]
    ordered_references: tuple[Reference, ...]
    comparisons: tuple[CIPComparison, ...]
    complete: bool

    @property
    def digest(self) -> str:
        return sha256(repr(self).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _Occurrence:
    atom_index: int
    parent: int
    path: tuple[int, ...]
    depth: int


def _effective_mass(atom: Chem.Atom) -> float:
    isotope = int(atom.GetIsotope())
    if isotope:
        return float(
            Chem.GetPeriodicTable().GetMassForIsotope(
                atom.GetAtomicNum(),
                isotope,
            )
        )
    return float(Chem.GetPeriodicTable().GetAtomicWeight(atom.GetAtomicNum()))


def _node(
    molecule: Chem.Mol,
    atom_index: int,
    *,
    depth: int,
    path: tuple[int, ...],
    duplicate: bool = False,
    duplicate_source_depth: int | None = None,
    duplicate_kind: str | None = None,
) -> CIPNodeEvidence:
    atom = molecule.GetAtomWithIdx(atom_index)
    return CIPNodeEvidence(
        depth,
        atom.GetAtomicNum(),
        _effective_mass(atom),
        atom_index,
        path,
        duplicate,
        duplicate_source_depth,
        duplicate_kind,
    )


def _virtual_node(reference: str) -> CIPNodeEvidence:
    virtual = parse_virtual_reference(reference)
    if virtual is None:
        raise ValueError(f"Invalid virtual CIP reference: {reference!r}.")
    if virtual.kind == "H":
        mass = float(Chem.GetPeriodicTable().GetAtomicWeight(1))
        return CIPNodeEvidence(1, 1, mass, None, (), duplicate_kind="virtual_H")
    return CIPNodeEvidence(1, 0, 0.0, None, (), duplicate_kind="virtual_LP")


def _bond_multiplicity(bond: Chem.Bond) -> tuple[int, str | None]:
    if bond.GetIsAromatic():
        return 1, None
    order = int(round(float(bond.GetBondTypeAsDouble())))
    return max(1, order), None


def _mancude_duplicate_values(
    molecule: Chem.Mol,
    atom_index: int,
) -> tuple[float, float] | None:
    """Return the averaged duplicate node for a simple mancude ring atom.

    IUPAC P-92.1.4.4 averages the atomic number over the possible Kekulé
    double-bond positions.  A non-fused even aromatic ring has two such
    neighbours at every atom, so its exact mean is available without choosing
    an arbitrary Kekulé form.  Fused and odd aromatic systems remain explicit
    unsupported features until their full resonance ensemble is implemented.
    """
    atom = molecule.GetAtomWithIdx(atom_index)
    aromatic_neighbours = tuple(
        bond.GetOtherAtomIdx(atom_index)
        for bond in atom.GetBonds()
        if bond.GetIsAromatic()
    )
    rings = tuple(
        ring
        for ring in molecule.GetRingInfo().AtomRings()
        if atom_index in ring
        and all(molecule.GetAtomWithIdx(index).GetIsAromatic() for index in ring)
    )
    if len(aromatic_neighbours) != 2 or len(rings) != 1 or len(rings[0]) % 2:
        return None
    neighbours = tuple(molecule.GetAtomWithIdx(index) for index in aromatic_neighbours)
    return (
        sum(atom.GetAtomicNum() for atom in neighbours) / len(neighbours),
        sum(_effective_mass(atom) for atom in neighbours) / len(neighbours),
    )


def _sort_nodes(
    nodes: Iterable[CIPNodeEvidence],
) -> tuple[CIPNodeEvidence, ...]:
    return tuple(
        sorted(
            nodes,
            key=lambda node: (
                -node.atomic_number,
                -node.mass,
                not node.duplicate,
                (
                    node.duplicate_source_depth
                    if node.duplicate_source_depth is not None
                    else 10**9
                ),
                repr(node.path),
                node.atom_index if node.atom_index is not None else -1,
            ),
        )
    )


class CIPRanker:
    """Rank ligands without consulting toolkit-assigned CIP properties."""

    def __init__(
        self,
        molecule: Chem.Mol,
        *,
        configured_descriptors: Iterable[Any] = (),
        configured_labels: dict[str, str] | None = None,
        max_depth: int | None = None,
    ) -> None:
        if molecule is None:
            raise ValueError("CIP ranking requires a molecule.")
        self.molecule = Chem.Mol(molecule)
        self.configured_descriptors = tuple(configured_descriptors)
        self.configured_labels = dict(configured_labels or {})
        self._mancude_duplicate_cache: dict[
            int,
            tuple[float, float] | None,
        ] = {}
        self.max_depth = (
            max(4, self.molecule.GetNumAtoms() * 2 + 2)
            if max_depth is None
            else max_depth
        )
        if type(self.max_depth) is not int or self.max_depth < 1:
            raise ValueError("CIP maximum depth must be a positive integer.")

    def _validate_center_reference(
        self,
        center: int,
        reference: Reference,
    ) -> None:
        if center < 0 or center >= self.molecule.GetNumAtoms():
            raise ValueError(f"CIP center {center} is absent.")
        if type(reference) is int:
            if reference < 0 or reference >= self.molecule.GetNumAtoms():
                raise ValueError(f"CIP reference {reference} is absent.")
            if self.molecule.GetBondBetweenAtoms(center, reference) is None:
                raise ValueError(
                    f"CIP reference {reference} is not adjacent to " f"center {center}."
                )
            return
        virtual = parse_virtual_reference(reference)
        if virtual is None or virtual.center != center:
            raise ValueError(
                f"Virtual CIP reference {reference!r} is not owned " f"by {center}."
            )

    def reflected(self) -> CIPRanker:
        """Return the same constitutional ranker with reflected stereo labels."""
        return CIPRanker(
            self.molecule,
            configured_descriptors=self.configured_descriptors,
            configured_labels={
                identifier: _REFLECTED_STEREO_LABEL.get(label, label)
                for identifier, label in self.configured_labels.items()
            },
            max_depth=self.max_depth,
        )

    def _mancude_duplicate(
        self,
        atom_index: int,
        *,
        depth: int,
        path: tuple[int, ...],
    ) -> CIPNodeEvidence | None:
        if atom_index not in self._mancude_duplicate_cache:
            self._mancude_duplicate_cache[atom_index] = _mancude_duplicate_values(
                self.molecule, atom_index
            )
        values = self._mancude_duplicate_cache[atom_index]
        if values is None:
            return None
        atomic_number, mass = values
        return CIPNodeEvidence(
            depth,
            atomic_number,
            mass,
            None,
            path,
            duplicate=True,
            duplicate_source_depth=depth,
            duplicate_kind="mancude_average",
        )

    def _stereo_evidence(
        self, material_depths: dict[int, int]
    ) -> tuple[CIPStereogenicUnitEvidence, ...]:
        result = []
        for descriptor in self.configured_descriptors:
            encountered = tuple(sorted(descriptor.dependencies & set(material_depths)))
            if not encountered:
                continue
            identifier = descriptor_id(descriptor)
            result.append(
                CIPStereogenicUnitEvidence(
                    descriptor.descriptor_class,
                    identifier,
                    descriptor.parity,
                    self.configured_labels.get(identifier),
                    min(material_depths[atom] for atom in encountered),
                    encountered,
                )
            )
        return tuple(
            sorted(
                result,
                key=lambda item: (
                    item.depth,
                    item.descriptor_class,
                    item.identifier,
                ),
            )
        )

    def build_evidence(
        self,
        center: int,
        reference: Reference,
    ) -> CIPLigandEvidence:
        self._validate_center_reference(center, reference)
        if isinstance(reference, str):
            sphere = CIPSphereEvidence(1, (_virtual_node(reference),))
            return CIPLigandEvidence(
                center,
                reference,
                (sphere,),
                (),
                (),
                CIPTermination.VIRTUAL_REFERENCE,
            )

        first = _node(
            self.molecule,
            reference,
            depth=1,
            path=(center, reference),
        )
        spheres = [CIPSphereEvidence(1, (first,))]
        active = [_Occurrence(reference, center, (center, reference), 1)]
        material_depths = {reference: 1}
        unsupported: set[str] = set()
        termination = CIPTermination.EXHAUSTED

        while active:
            next_nodes: list[CIPNodeEvidence] = []
            next_active: list[_Occurrence] = []
            next_depth = active[0].depth + 1
            if next_depth > self.max_depth:
                termination = CIPTermination.DEPTH_CAP
                break
            for occurrence in active:
                atom = self.molecule.GetAtomWithIdx(occurrence.atom_index)
                hidden_h = int(atom.GetNumExplicitHs()) + int(atom.GetNumImplicitHs())
                next_nodes.extend(
                    CIPNodeEvidence(
                        next_depth,
                        1,
                        float(Chem.GetPeriodicTable().GetAtomicWeight(1)),
                        None,
                        occurrence.path,
                        duplicate_kind="implicit_H",
                    )
                    for _ in range(hidden_h)
                )
                if atom.GetIsAromatic():
                    mancude_duplicate = self._mancude_duplicate(
                        occurrence.atom_index,
                        depth=next_depth,
                        path=occurrence.path,
                    )
                    if mancude_duplicate is None:
                        unsupported.add("unsupported_mancude_duplicate_model")
                    else:
                        next_nodes.append(mancude_duplicate)
                for bond in atom.GetBonds():
                    neighbor = bond.GetOtherAtomIdx(occurrence.atom_index)
                    multiplicity, feature = _bond_multiplicity(bond)
                    if feature is not None:
                        unsupported.add(feature)
                    if neighbor != occurrence.parent:
                        if neighbor in occurrence.path:
                            source_depth = occurrence.path.index(neighbor)
                            next_nodes.append(
                                _node(
                                    self.molecule,
                                    neighbor,
                                    depth=next_depth,
                                    path=occurrence.path + (neighbor,),
                                    duplicate=True,
                                    duplicate_source_depth=source_depth,
                                    duplicate_kind="ring_closure",
                                )
                            )
                        else:
                            path = occurrence.path + (neighbor,)
                            next_nodes.append(
                                _node(
                                    self.molecule,
                                    neighbor,
                                    depth=next_depth,
                                    path=path,
                                )
                            )
                            next_active.append(
                                _Occurrence(
                                    neighbor,
                                    occurrence.atom_index,
                                    path,
                                    next_depth,
                                )
                            )
                            material_depths.setdefault(neighbor, next_depth)
                        source_depth = (
                            occurrence.path.index(neighbor)
                            if neighbor in occurrence.path
                            else next_depth
                        )
                        next_nodes.extend(
                            _node(
                                self.molecule,
                                neighbor,
                                depth=next_depth,
                                path=occurrence.path + (neighbor,),
                                duplicate=True,
                                duplicate_source_depth=source_depth,
                                duplicate_kind="multiple_bond",
                            )
                            for _ in range(multiplicity - 1)
                        )
            if next_nodes:
                spheres.append(CIPSphereEvidence(next_depth, _sort_nodes(next_nodes)))
            active = next_active

        stereo = self._stereo_evidence(material_depths)
        return CIPLigandEvidence(
            center,
            reference,
            tuple(spheres),
            stereo,
            tuple(sorted(unsupported)),
            termination,
        )

    @staticmethod
    def _padded(
        values: tuple[Any, ...],
        length: int,
        zero: Any,
    ) -> tuple[Any, ...]:
        return values + (zero,) * (length - len(values))

    @classmethod
    def _first_sphere_difference(
        cls,
        left: CIPLigandEvidence,
        right: CIPLigandEvidence,
        field: str,
        zero: Any,
    ) -> tuple[int, tuple[Any, ...], tuple[Any, ...]] | None:
        depth_count = max(len(left.spheres), len(right.spheres))
        for offset in range(depth_count):
            left_values = (
                getattr(left.spheres[offset], field)
                if offset < len(left.spheres)
                else ()
            )
            right_values = (
                getattr(right.spheres[offset], field)
                if offset < len(right.spheres)
                else ()
            )
            width = max(len(left_values), len(right_values))
            left_padded = cls._padded(left_values, width, zero)
            right_padded = cls._padded(right_values, width, zero)
            if left_padded != right_padded:
                return offset + 1, left_padded, right_padded
        return None

    @staticmethod
    def _outcome(left: tuple[Any, ...], right: tuple[Any, ...]) -> CIPComparisonOutcome:
        return (
            CIPComparisonOutcome.LEFT_HIGHER
            if left > right
            else CIPComparisonOutcome.RIGHT_HIGHER
        )

    @staticmethod
    def _resolved_stereo_units(
        evidence: CIPLigandEvidence,
    ) -> tuple[CIPStereogenicUnitEvidence, ...] | None:
        if any(unit.label is None for unit in evidence.stereogenic_units):
            return None
        return evidence.stereogenic_units

    def _stereo_difference(
        self,
        left: CIPLigandEvidence,
        right: CIPLigandEvidence,
    ) -> (
        tuple[
            CIPSequenceRule,
            tuple[Any, ...],
            tuple[Any, ...],
            str,
        ]
        | None
    ):
        left_units = self._resolved_stereo_units(left)
        right_units = self._resolved_stereo_units(right)
        if left_units is None or right_units is None:
            return None
        # A single corresponding pair is the closed, reference-independent
        # subset of Rules 4c/5.  Multi-unit Rule 4b requires the full
        # reference-descriptor pairing algorithm and therefore still fails
        # closed below.
        if len(left_units) != 1 or len(right_units) != 1:
            return None
        left_unit, right_unit = left_units[0], right_units[0]
        left_label, right_label = str(left_unit.label), str(right_unit.label)
        if (
            left_unit.depth != right_unit.depth
            or _STEREO_LABEL_FAMILIES.get(left_label)
            != _STEREO_LABEL_FAMILIES.get(right_label)
            or (left_label in _STEREO_HIGH_LABELS)
            == (right_label in _STEREO_HIGH_LABELS)
        ):
            return None
        left_witness = (
            -left_unit.depth,
            _STEREO_LABEL_FAMILIES[left_label],
            int(left_label in _STEREO_HIGH_LABELS),
        )
        right_witness = (
            -right_unit.depth,
            _STEREO_LABEL_FAMILIES[right_label],
            int(right_label in _STEREO_HIGH_LABELS),
        )
        if (
            _STEREO_LABEL_FAMILIES[left_label] == "planar"
            and left_label.isupper()
            and right_label.isupper()
        ):
            return (
                CIPSequenceRule.RULE_3_SEQUENCE_GEOMETRY,
                left_witness,
                right_witness,
                "Sequence Rule 3 ranks seqCis/Z before seqTrans/E.",
            )
        if left_label.islower() and right_label.islower():
            return (
                CIPSequenceRule.RULE_4_STEREOGENIC_UNIT,
                left_witness,
                right_witness,
                "Sequence Rule 4c ranks r before s and m before p.",
            )
        if left_label.isupper() and right_label.isupper():
            return (
                CIPSequenceRule.RULE_5_REFLECTION_VARIANT,
                left_witness,
                right_witness,
                "Sequence Rule 5 ranks R/M/seqCis before "
                "S/P/seqTrans enantiomorphs.",
            )
        return None

    def compare(
        self,
        center: int,
        left_reference: Reference,
        right_reference: Reference,
        *,
        left_evidence: CIPLigandEvidence | None = None,
        right_evidence: CIPLigandEvidence | None = None,
    ) -> CIPComparison:
        left = left_evidence or self.build_evidence(center, left_reference)
        right = right_evidence or self.build_evidence(center, right_reference)
        atomic = self._first_sphere_difference(
            left,
            right,
            "atomic_signature",
            0,
        )
        if atomic is not None:
            depth, left_witness, right_witness = atomic
            return CIPComparison(
                center,
                left_reference,
                right_reference,
                self._outcome(left_witness, right_witness),
                CIPSequenceRule.RULE_1A_ATOMIC_NUMBER,
                depth,
                left_witness,
                right_witness,
                "First exhaustive digraph difference in atomic number.",
                left,
                right,
            )

        duplicate = self._first_sphere_difference(
            left, right, "duplicate_signature", (0, 0)
        )
        if duplicate is not None:
            depth, left_witness, right_witness = duplicate
            return CIPComparison(
                center,
                left_reference,
                right_reference,
                self._outcome(left_witness, right_witness),
                CIPSequenceRule.RULE_1B_DUPLICATE_DISTANCE,
                depth,
                left_witness,
                right_witness,
                "Nearer corresponding duplicate atom node has precedence.",
                left,
                right,
            )

        isotope = self._first_sphere_difference(
            left, right, "isotope_signature", (0, 0.0)
        )
        if isotope is not None:
            depth, left_witness, right_witness = isotope
            return CIPComparison(
                center,
                left_reference,
                right_reference,
                self._outcome(left_witness, right_witness),
                CIPSequenceRule.RULE_2_ISOTOPE_MASS,
                depth,
                left_witness,
                right_witness,
                "First exhaustive digraph difference in atomic mass.",
                left,
                right,
            )

        stereo = self._stereo_difference(left, right)
        if stereo is not None:
            rule, left_witness, right_witness, reason = stereo
            return CIPComparison(
                center,
                left_reference,
                right_reference,
                self._outcome(left_witness, right_witness),
                rule,
                None,
                left_witness,
                right_witness,
                reason,
                left,
                right,
            )

        if (
            left.termination is CIPTermination.DEPTH_CAP
            or right.termination is CIPTermination.DEPTH_CAP
        ):
            reason = "Ligand exploration reached the explicit depth cap."
        elif left.unsupported_features or right.unsupported_features:
            reason = "Tied ligands require an unsupported mancude duplicate model."
        elif left.stereogenic_units or right.stereogenic_units:
            reason = (
                "Tied ligands require Sequence Rules 3-5 " "stereogenic-unit ordering."
            )
        else:
            return CIPComparison(
                center,
                left_reference,
                right_reference,
                CIPComparisonOutcome.TIE,
                None,
                None,
                None,
                None,
                "Ligands are constitutionally and isotopically " "indistinguishable.",
                left,
                right,
            )
        return CIPComparison(
            center,
            left_reference,
            right_reference,
            CIPComparisonOutcome.UNSUPPORTED,
            (
                CIPSequenceRule.RULE_4_STEREOGENIC_UNIT
                if left.stereogenic_units or right.stereogenic_units
                else None
            ),
            None,
            None,
            None,
            reason,
            left,
            right,
        )

    def rank(self, center: int, references: Iterable[Reference]) -> CIPRanking:
        values = tuple(references)
        if len(values) != len(set(values)):
            raise ValueError("CIP ranking references must be distinct.")
        comparisons = tuple(
            self.compare(center, left, right) for left, right in combinations(values, 2)
        )
        complete = all(comparison.decided for comparison in comparisons)
        if not complete:
            return CIPRanking(center, values, (), comparisons, False)
        wins = {reference: 0 for reference in values}
        for comparison in comparisons:
            winner = (
                comparison.left_reference
                if comparison.outcome is CIPComparisonOutcome.LEFT_HIGHER
                else comparison.right_reference
            )
            wins[winner] += 1
        ordered = tuple(sorted(values, key=lambda value: (-wins[value], repr(value))))
        if len(set(wins.values())) != len(values):
            return CIPRanking(center, values, (), comparisons, False)
        return CIPRanking(center, values, ordered, comparisons, True)


__all__ = [
    "CIPComparison",
    "CIPComparisonOutcome",
    "CIPLigandEvidence",
    "CIPNodeEvidence",
    "CIPRanker",
    "CIPRanking",
    "CIPSequenceRule",
    "CIPSphereEvidence",
    "CIPStereogenicUnitEvidence",
    "CIPTermination",
]
