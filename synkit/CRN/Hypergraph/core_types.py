# CRN/core_types.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CRNSpecies:
    """
    A single chemical species in a reaction network.

    :param name: Human-readable identifier (e.g. 'A', 'B', 'ATP').
    :type name: str
    :param metadata: Optional arbitrary metadata, such as compartment,
                     charge, or links to RDKit molecules.
    :type metadata: Dict[str, Any]
    """

    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CRNReaction:
    """
    A single reaction hyperedge.

    Stoichiometry uses species *indices* (integers indexing the CRNNetwork.species list).

    :param reactants: Mapping species index -> stoichiometric coefficient (consumed).
    :type reactants: Dict[int, float]
    :param products: Mapping species index -> stoichiometric coefficient (produced).
    :type products: Dict[int, float]
    :param reversible: Whether the reaction is considered reversible.
    :type reversible: bool
    :param metadata: Optional metadata such as kinetic law, rate constants,
                     annotations, etc.
    :type metadata: Dict[str, Any]
    """

    reactants: Dict[int, float]
    products: Dict[int, float]
    reversible: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        def fmt(side: Dict[int, float]) -> str:
            return (
                " + ".join(
                    [
                        f"{int(v)}*S{int(i)}" if v != 1 else f"S{int(i)}"
                        for i, v in sorted(side.items())
                    ]
                )
                or "∅"
            )

        left = fmt(self.reactants)
        right = fmt(self.products)
        rev = "<->" if self.reversible else "->"
        return f"{left} {rev} {right}"


# ---------------------------------------------------------------------------
# Abstract CRN interface (no circular imports)
# ---------------------------------------------------------------------------


class CRNLike(ABC):
    """
    Abstract interface for all CRN-like objects.

    Implementations must provide:
      - stoichiometric_matrix(sparse=False) -> np.ndarray (n_species x n_reactions)
      - species -> List[str] (ordered species labels)
      - reactions -> List[Any] (ordered reaction objects; CRNReaction is recommended)
    """

    # -----------------------------
    # Essential abstract methods
    # -----------------------------
    @abstractmethod
    def stoichiometric_matrix(self, *, sparse: bool = False) -> Any:
        """
        Return stoichiometric/incidence matrix.

        :param sparse: If True, return a sparse-friendly representation if supported.
        :type sparse: bool
        :returns: Dense numpy.ndarray (n_species x n_reactions) or sparse mapping representation.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def species(self) -> List[str]:
        """
        Ordered list of species names.

        :returns: List of species labels.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def reactions(self) -> List[Any]:
        """
        Ordered list of reaction objects.

        :returns: List of reaction objects (implementation-defined).
        """
        raise NotImplementedError

    # -----------------------------
    # Optional helpers
    # -----------------------------
    def n_species(self) -> int:
        """Return number of species."""
        return len(self.species)

    def n_reactions(self) -> int:
        """Return number of reactions."""
        return len(self.reactions)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        try:
            return f"{cls}(n_species={len(self.species)}, n_reactions={len(self.reactions)})"
        except Exception:
            return f"{cls}(?)"


# ---------------------------------------------------------------------------
# Concrete (legacy) CRNNetwork implementing CRNLike
# ---------------------------------------------------------------------------


@dataclass
class CRNNetwork(CRNLike):
    """
    Concrete, minimal chemical reaction network container.

    Species are stored as :class:`CRNSpecies` objects; reactions as
    :class:`CRNReaction`.

    :param species: Ordered list of CRNSpecies objects.
    :type species: List[CRNSpecies]
    :param reactions: Ordered list of CRNReaction objects.
    :type reactions: List[CRNReaction]
    """

    species: List[CRNSpecies]
    reactions: List[CRNReaction]

    # ------------------------------------------------------------------
    # CRNLike API
    # ------------------------------------------------------------------
    def stoichiometric_matrix(self, *, sparse: bool = False) -> np.ndarray:
        """
        Build the species x reaction stoichiometric matrix S.

        :param sparse: Ignored for the dense-only legacy representation.
        :type sparse: bool
        :returns: Stoichiometric matrix of shape (n_species, n_reactions).
        :rtype: numpy.ndarray
        """
        n_s = len(self.species)
        n_r = len(self.reactions)
        S = np.zeros((n_s, n_r), dtype=float)

        for j, r in enumerate(self.reactions):
            for i, coeff in r.products.items():
                S[int(i), j] += float(coeff)
            for i, coeff in r.reactants.items():
                S[int(i), j] -= float(coeff)
        return S

    @property
    def species(self) -> List[str]:  # type: ignore[override]
        """
        Return ordered list of species names.

        :returns: List of species names (strings).
        :rtype: List[str]
        """
        # the dataclass stores CRNSpecies objects; expose their names for CRNLike contract
        return [sp.name for sp in self._species_objects]

    @species.setter
    def species(self, value: List[CRNSpecies]) -> None:  # allow dataclass assignment
        # internal storage kept as list of CRNSpecies objects
        self._species_objects = list(value)

    @property
    def reactions(self) -> List[CRNReaction]:  # type: ignore[override]
        """
        Return ordered list of CRNReaction objects.

        :returns: List of CRNReaction.
        :rtype: List[CRNReaction]
        """
        return self._reaction_objects

    @reactions.setter
    def reactions(self, value: List[CRNReaction]) -> None:
        self._reaction_objects = list(value)

    # ------------------------------------------------------------------
    # convenience constructors / helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_reaction_dicts(
        cls,
        species: Sequence[str],
        reactions: Sequence[Tuple[Dict[int, float], Dict[int, float], Optional[bool]]],
    ) -> "CRNNetwork":
        """
        Build CRNNetwork from simple species names and reaction dicts.

        :param species: Ordered sequence of species labels.
        :type species: Sequence[str]
        :param reactions: Sequence of tuples (reactants, products, reversible)
                          where reactants/products are dicts index->coeff.
        :type reactions: Sequence[Tuple[Dict[int,float], Dict[int,float], Optional[bool]]]
        :returns: CRNNetwork instance.
        :rtype: CRNNetwork

        .. code-block:: python

            species = ["A","B","C"]
            reactions = [
                ({0:1,1:1}, {2:1}, False),  # A + B -> C
                ({2:1}, {0:1}, False),      # C -> A
            ]
            net = CRNNetwork.from_reaction_dicts(species, reactions)
        """
        sp_objs = [CRNSpecies(name=s) for s in species]
        rx_objs: List[CRNReaction] = []
        for rx in reactions:
            if len(rx) == 3:
                reactants, products, rev = rx
            else:
                reactants, products = rx
                rev = False
            rx_objs.append(
                CRNReaction(
                    reactants=dict(reactants),
                    products=dict(products),
                    reversible=bool(rev),
                )
            )
        return cls(species=sp_objs, reactions=rx_objs)

    # direct accessors for code that wants objects
    @property
    def species_objects(self) -> List[CRNSpecies]:
        """Return CRNSpecies objects (internal representation)."""
        return list(self._species_objects)

    @property
    def reaction_objects(self) -> List[CRNReaction]:
        """Return CRNReaction objects (internal representation)."""
        return list(self._reaction_objects)

    # Pretty repr
    def __repr__(self) -> str:
        return f"CRNNetwork(n_species={len(self._species_objects)}, n_reactions={len(self._reaction_objects)})"
