"""Coupled stereochemical relations carried by executable reaction rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .descriptors import PlanarBondStereo, Reference, TetrahedralStereo


@dataclass(frozen=True)
class StereoCoupling:
    """Relate two local stereo changes as one chemical operation.

    Independent endpoint descriptors encode concrete molecular geometry, but
    do not state that two delivered ligands belong to the same syn/anti event.
    The rule stores this coupling instead; application derives lossless
    endpoint descriptors for the applied ITS and RDKit reconstruction.
    """

    kind: str
    relation: str
    centers: tuple[int, int]
    ligands: tuple[int, int]

    def __post_init__(self) -> None:
        kind = self.kind.upper()
        relation = self.relation.upper()
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "relation", relation)
        allowed_kinds = {"VICINAL_ADDITION", "VICINAL_ELIMINATION"}
        if kind not in allowed_kinds:
            raise ValueError(
                "Stereo coupling kind must be 'VICINAL_ADDITION' or "
                "'VICINAL_ELIMINATION'."
            )
        if relation not in {"SYN", "ANTI"}:
            raise ValueError("Stereo coupling relation must be 'SYN' or 'ANTI'.")
        if len(self.centers) != 2 or len(set(self.centers)) != 2:
            raise ValueError("Stereo coupling requires two distinct centers.")
        if len(self.ligands) != 2 or len(set(self.ligands)) != 2:
            raise ValueError("Stereo coupling requires two distinct ligands.")
        if not all(isinstance(value, int) and value > 0 for value in self.dependencies):
            raise ValueError("Stereo coupling references must be positive atom maps.")

    @property
    def dependencies(self) -> tuple[int, ...]:
        """Return all mapped atoms needed to interpret this relation."""
        return (*self.centers, *self.ligands)

    @property
    def target(self) -> str:
        """Return the normalized bond target joining the coupled centers."""
        left, right = sorted(self.centers)
        return f"bond:{left}-{right}"

    @classmethod
    def vicinal_addition(
        cls,
        relation: str,
        *,
        centers: tuple[int, int],
        ligands: tuple[int, int],
    ) -> "StereoCoupling":
        """Construct a syn/anti vicinal-addition declaration."""
        return cls("VICINAL_ADDITION", relation, centers, ligands)

    @classmethod
    def from_value(
        cls,
        value: "StereoCoupling | Mapping[str, Any]",
    ) -> "StereoCoupling":
        """Normalize the public object/dictionary forms."""
        if isinstance(value, cls):
            return value
        return cls(
            kind=str(value.get("kind", "VICINAL_ADDITION")),
            relation=str(value["relation"]),
            centers=tuple(int(item) for item in value["centers"]),  # type: ignore[arg-type]
            ligands=tuple(int(item) for item in value["ligands"]),  # type: ignore[arg-type]
        )

    def reverse(self) -> "StereoCoupling":
        """Return the directionally reversed operation with the same relation."""
        reverse_kind = {
            "VICINAL_ADDITION": "VICINAL_ELIMINATION",
            "VICINAL_ELIMINATION": "VICINAL_ADDITION",
        }[self.kind]
        return StereoCoupling(reverse_kind, self.relation, self.centers, self.ligands)

    def relabel(self, mapping: Mapping[int, int]) -> "StereoCoupling":
        """Translate rule atom maps into one application mapping."""
        return StereoCoupling(
            self.kind,
            self.relation,
            tuple(mapping.get(value, value) for value in self.centers),  # type: ignore[arg-type]
            tuple(mapping.get(value, value) for value in self.ligands),  # type: ignore[arg-type]
        )

    def planar_product_relation(self, descriptor: PlanarBondStereo) -> str:
        """Derive syn/anti delivery from a formed planar-bond descriptor.

        At each endpoint, the planar descriptor contains two ordered ligand
        slots. Delivered ligands occupying equal slots are syn; opposite slots
        are anti. The result is invariant to whole-bond reversal and the
        simultaneous endpoint swap allowed by :class:`PlanarBondStereo`.
        """
        left_center, right_center = descriptor.atoms[2:4]
        if (left_center, right_center) == self.centers:
            left_ligand, right_ligand = self.ligands
            left_refs = descriptor.atoms[:2]
            right_refs = descriptor.atoms[4:]
        elif (right_center, left_center) == self.centers:
            left_ligand, right_ligand = self.ligands
            left_refs = descriptor.atoms[4:]
            right_refs = descriptor.atoms[:2]
        else:
            raise ValueError(
                f"Planar descriptor {descriptor.atoms[2:4]!r} does not use "
                f"coupled centers {self.centers!r}."
            )
        if left_ligand not in left_refs or right_ligand not in right_refs:
            raise ValueError(
                "Each coupled ligand must occur at its corresponding planar "
                "bond endpoint."
            )
        left_slot = left_refs.index(left_ligand)
        right_slot = right_refs.index(right_ligand)
        return "SYN" if left_slot == right_slot else "ANTI"

    def planar_product_descriptor(
        self,
        left_reference: Reference,
        right_reference: Reference,
        *,
        provenance: str | None = "rule-coupling",
    ) -> PlanarBondStereo:
        """Construct the alkene geometry formed by syn/anti delivery."""
        left_center, right_center = self.centers
        left_ligand, right_ligand = self.ligands
        right_refs = (
            (right_reference, right_ligand)
            if self.relation == "SYN"
            else (right_ligand, right_reference)
        )
        return PlanarBondStereo(
            (
                left_reference,
                left_ligand,
                left_center,
                right_center,
                *right_refs,
            ),
            0,
            provenance,
        )

    def tetrahedral_product_pairs(
        self,
        reactant: PlanarBondStereo,
        *,
        provenance: str | None = "rule-coupling",
    ) -> tuple[
        tuple[TetrahedralStereo, TetrahedralStereo],
        tuple[TetrahedralStereo, TetrahedralStereo],
    ]:
        """Construct the two correlated face outcomes of alkene addition.

        The reactant planar descriptor supplies an oriented reference frame.
        Inverting both product centers together changes the attacked face but
        preserves the declared syn/anti relationship. It is deliberately one
        paired branch, never a Cartesian product of independent centers.
        """
        left_center, right_center = reactant.atoms[2:4]
        if (left_center, right_center) == self.centers:
            left_refs = reactant.atoms[:2]
            right_refs = reactant.atoms[4:]
        elif (right_center, left_center) == self.centers:
            left_refs = reactant.atoms[4:]
            right_refs = reactant.atoms[:2]
        else:
            raise ValueError(
                f"Planar descriptor {reactant.atoms[2:4]!r} does not use "
                f"coupled centers {self.centers!r}."
            )

        left_center, right_center = self.centers
        left_ligand, right_ligand = self.ligands
        left = TetrahedralStereo(
            (
                left_center,
                left_refs[0],
                left_ligand,
                right_center,
                left_refs[1],
            ),
            -1,
            provenance,
        )
        right = TetrahedralStereo(
            (
                right_center,
                left_center,
                right_ligand,
                right_refs[1],
                right_refs[0],
            ),
            1 if self.relation == "SYN" else -1,
            provenance,
        )
        return (left, right), (left.invert(), right.invert())

    def to_dict(self) -> dict[str, Any]:
        """Return the compact JSON/GML-friendly representation."""
        return {
            "kind": self.kind,
            "relation": self.relation,
            "centers": list(self.centers),
            "ligands": list(self.ligands),
        }

    def signature(self) -> tuple[Any, ...]:
        """Return a stable rule-identity signature."""
        return self.kind, self.relation, self.centers, self.ligands
