from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


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

    :param reactants: Mapping species index -> stoichiometric coefficient.
    :type reactants: Dict[int, float]
    :param products: Mapping species index -> stoichiometric coefficient.
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


@dataclass
class CRNNetwork:
    """
    Chemical Reaction Network data structure.

    :param species: Ordered list of species.
    :type species: List[CRNSpecies]
    :param reactions: Ordered list of reactions.
    :type reactions: List[CRNReaction]
    """

    species: List[CRNSpecies]
    reactions: List[CRNReaction]

    def stoichiometric_matrix(self) -> np.ndarray:
        """
        Build the species x reaction stoichiometric matrix N.

        :returns: Stoichiometric matrix with shape (n_species, n_reactions).
        :rtype: numpy.ndarray
        """
        n_s = len(self.species)
        n_r = len(self.reactions)
        N = np.zeros((n_s, n_r), dtype=float)

        for j, r in enumerate(self.reactions):
            for i, coeff in r.products.items():
                N[i, j] += coeff
            for i, coeff in r.reactants.items():
                N[i, j] -= coeff
        return N
