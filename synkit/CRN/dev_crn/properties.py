from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Any, Mapping
from collections import Counter
import hashlib

from .constants import NodeKind, EdgeRole


# -----------------------------
# Utility (internal)
# -----------------------------
def _sha12(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


# -----------------------------
# Species
# -----------------------------
@dataclass
class SpeciesProperty:
    """
    Canonical description of a chemical species (node).

    The class is *fluent*: mutating helpers return ``self`` so you can chain calls.
    Access results via attributes or :py:meth:`to_dict`.

    :param identifier: User-facing identifier (e.g., "A", "CH4").
    :param smiles: Canonical SMILES if available.
    :param inchi: InChI string if available.
    :param charge: Formal charge.
    :param radical: Radical electron count.
    :param mass: Optional molecular mass.
    :param annotations: Free-form metadata.
    """

    identifier: str
    smiles: Optional[str] = None
    inchi: Optional[str] = None
    charge: Optional[int] = None
    radical: Optional[int] = None
    mass: Optional[float] = None
    annotations: Dict[str, Any] = field(default_factory=dict)

    # ---- Fluent mutators ----
    def set_smiles(self, s: Optional[str]) -> "SpeciesProperty":
        """Set SMILES and return ``self``."""
        self.smiles = s
        return self

    def set_inchi(self, s: Optional[str]) -> "SpeciesProperty":
        """Set InChI and return ``self``."""
        self.inchi = s
        return self

    def set_charge(self, q: Optional[int]) -> "SpeciesProperty":
        """Set charge and return ``self``."""
        self.charge = q
        return self

    def set_radical(self, r: Optional[int]) -> "SpeciesProperty":
        """Set radical count and return ``self``."""
        self.radical = r
        return self

    def set_mass(self, m: Optional[float]) -> "SpeciesProperty":
        """Set mass and return ``self``."""
        self.mass = m
        return self

    def add_annotations(self, extra: Mapping[str, Any]) -> "SpeciesProperty":
        """Shallow-merge annotations and return ``self``."""
        self.annotations.update(dict(extra))
        return self

    # ---- Derived keys / serialization ----
    @property
    def canonical_id(self) -> str:
        """
        Stable key combining preferred canonical fields.

        :returns: A composited key like ``"CH4|a1b2c3d4e5f6"``.
        """
        if self.smiles:
            base = f"smiles:{self.smiles}"
        elif self.inchi:
            base = f"inchi:{self.inchi}"
        else:
            base = f"id:{self.identifier}"
        return f"{self.identifier}|{_sha12(base)}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict."""
        d = asdict(self)
        d["_canonical_id"] = self.canonical_id
        return d

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "SpeciesProperty":
        """Create from a serialized dict."""
        return SpeciesProperty(
            identifier=str(d.get("identifier", "UNK")),
            smiles=d.get("smiles"),
            inchi=d.get("inchi"),
            charge=d.get("charge"),
            radical=d.get("radical"),
            mass=d.get("mass"),
            annotations=dict(d.get("annotations", {})),
        )

    def __repr__(self) -> str:  # simple; keep complexity low
        return f"SpeciesProperty({self.identifier!r}, smiles={self.smiles!r})"


# -----------------------------
# Reaction stoichiometry + meta
# -----------------------------
@dataclass
class ReactionProperty:
    """
    Immutable(-ish) reaction property bundle: stoichiometry + meta.

    Stoichiometry uses the convention:
    - reactants have **negative** coefficients
    - products  have **positive** coefficients

    :param rxn_id: Stable reaction identifier.
    :param stoichiometry: Mapping canonical species id -> coefficient.
    :param reversible: Whether the step is reversible.
    :param kinetics: Optional kinetics model parameters.
    :param conditions: Optional experimental/computational conditions.
    :param annotations: Free-form metadata.
    """

    rxn_id: str
    stoichiometry: Dict[str, int] = field(default_factory=dict)
    reversible: bool = False
    kinetics: Optional[Dict[str, Any]] = None
    conditions: Optional[Dict[str, Any]] = None
    annotations: Dict[str, Any] = field(default_factory=dict)

    # ---- Fluent mutators ----
    def set_reversible(self, flag: bool) -> "ReactionProperty":
        self.reversible = bool(flag)
        return self

    def set_kinetics(self, params: Optional[Mapping[str, Any]]) -> "ReactionProperty":
        self.kinetics = None if params is None else dict(params)
        return self

    def set_conditions(self, params: Optional[Mapping[str, Any]]) -> "ReactionProperty":
        self.conditions = None if params is None else dict(params)
        return self

    def add_annotations(self, extra: Mapping[str, Any]) -> "ReactionProperty":
        self.annotations.update(dict(extra))
        return self

    def set_coeff(self, species_key: str, coeff: int) -> "ReactionProperty":
        """
        Set the stoichiometric coefficient for a species.

        :param species_key: Canonical species key.
        :param coeff: Negative for reactant, positive for product, 0 to remove.
        """
        if coeff == 0:
            self.stoichiometry.pop(species_key, None)
        else:
            self.stoichiometry[species_key] = int(coeff)
        return self

    # ---- Derived views ----
    @property
    def reactants(self) -> Counter:
        """Multiset of reactants (positive counts)."""
        return Counter({k: -v for k, v in self.stoichiometry.items() if v < 0})

    @property
    def products(self) -> Counter:
        """Multiset of products (positive counts)."""
        return Counter({k: v for k, v in self.stoichiometry.items() if v > 0})

    @property
    def net_change(self) -> Counter:
        """Products − reactants."""
        net = Counter(self.products)
        net.subtract(self.reactants)
        for k in list(net):
            if net[k] == 0:
                del net[k]
        return net

    # ---- Serialization ----
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rxn_id": self.rxn_id,
            "stoichiometry": dict(self.stoichiometry),
            "reversible": self.reversible,
            "kinetics": self.kinetics,
            "conditions": self.conditions,
            "annotations": self.annotations,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ReactionProperty":
        return ReactionProperty(
            rxn_id=str(d["rxn_id"]),
            stoichiometry={
                str(k): int(v) for k, v in dict(d.get("stoichiometry", {})).items()
            },
            reversible=bool(d.get("reversible", False)),
            kinetics=(dict(d["kinetics"]) if d.get("kinetics") else None),
            conditions=(dict(d["conditions"]) if d.get("conditions") else None),
            annotations=dict(d.get("annotations", {})),
        )

    def __repr__(self) -> str:
        return f"ReactionProperty({self.rxn_id!r}, n={len(self.stoichiometry)})"


# -----------------------------
# Graph node/edge descriptors
# -----------------------------
@dataclass
class NodeProperty:
    """
    Node metadata for a bipartite CRN graph visualization or export.

    :param kind: Node kind (species/reaction).
    :param key: Unique node key (e.g., species canonical id or 'p<ID>').
    :param label: Display label.
    :param data: Arbitrary metadata.
    """

    kind: NodeKind
    key: str
    label: str
    data: Dict[str, Any] = field(default_factory=dict)

    def add_data(self, extra: Mapping[str, Any]) -> "NodeProperty":
        """Shallow-merge metadata and return ``self``."""
        self.data.update(dict(extra))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "key": self.key,
            "label": self.label,
            "data": dict(self.data),
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "NodeProperty":
        return NodeProperty(
            kind=NodeKind(str(d["kind"])),
            key=str(d["key"]),
            label=str(d["label"]),
            data=dict(d.get("data", {})),
        )

    def __repr__(self) -> str:
        return f"NodeProperty({self.kind.value}, {self.key!r})"


@dataclass
class EdgeProperty:
    """
    Edge metadata for bipartite CRN graphs (species↔reaction).

    :param src: Source node key.
    :param dst: Target node key.
    :param role: Edge role (reactant/product).
    :param stoich: Stoichiometric coefficient (abs value usually >= 1).
    :param data: Arbitrary metadata.
    """

    src: str
    dst: str
    role: EdgeRole
    stoich: int = 1
    data: Dict[str, Any] = field(default_factory=dict)

    def add_data(self, extra: Mapping[str, Any]) -> "EdgeProperty":
        """Shallow-merge metadata and return ``self``."""
        self.data.update(dict(extra))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "role": self.role.value,
            "stoich": int(self.stoich),
            "data": dict(self.data),
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "EdgeProperty":
        return EdgeProperty(
            src=str(d["src"]),
            dst=str(d["dst"]),
            role=EdgeRole(str(d["role"])),
            stoich=int(d.get("stoich", 1)),
            data=dict(d.get("data", {})),
        )

    def __repr__(self) -> str:
        return f"EdgeProperty({self.role.value}, {self.src!r}->{self.dst!r}, stoich={self.stoich})"
