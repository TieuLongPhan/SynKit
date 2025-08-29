from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

from ..Chem.Reaction.standardize import Standardize

from .utils import split_components


@dataclass(frozen=True)
class Reaction:
    """
    Canonical representation of a reaction, with both original and standardized forms.

    :param id: Stable index of the reaction in the source list.
    :type id: int
    :param original_raw: Original input reaction SMILES (keeps atom maps).
    :type original_raw: str
    :param canonical_raw: Standardized reaction SMILES (used for string matching).
    :type canonical_raw: str
    :param reactants_can: Canonical reactant multiset (split by '.').
    :type reactants_can: Counter[str]
    :param products_can: Canonical product multiset (split by '.').
    :type products_can: Counter[str]
    """

    id: int
    original_raw: str
    canonical_raw: str
    reactants_can: Counter[str] = field(default_factory=Counter)
    products_can: Counter[str] = field(default_factory=Counter)

    @classmethod
    def from_raw(
        cls,
        raw: str,
        idx: int,
        standardizer: Optional[Standardize] = None,
        remove_aam: bool = True,
    ) -> "Reaction":
        """
        Build a :class:`Reaction` from an input rsmi string.

        :param raw: Input reaction SMILES (may contain atom maps).
        :type raw: str
        :param idx: Stable index for the reaction in the dataset.
        :type idx: int
        :param standardizer: Optional :class:`..Chem.Reaction.standardize.Standardize` instance.
        :type standardizer: Optional[Standardize]
        :param remove_aam: Whether to remove atom mapping in canonical form.
        :type remove_aam: bool
        :raises ValueError: If the reaction is missing the ``>>`` delimiter.
        :return: Constructed :class:`Reaction`.
        :rtype: Reaction
        """
        canonical = raw
        if standardizer is not None:
            try:
                canonical = standardizer.fit(raw, remove_aam=remove_aam)
            except Exception:
                canonical = raw

        if ">>" not in canonical:
            raise ValueError(f"Invalid reaction SMILES (missing >>): {canonical!r}")

        left, right = canonical.split(">>", 1)
        reactants = Counter(split_components(left))
        products = Counter(split_components(right))

        return cls(
            id=idx,
            original_raw=raw,
            canonical_raw=canonical,
            reactants_can=reactants,
            products_can=products,
        )

    def __repr__(self) -> str:
        left = ".".join(sorted(self.reactants_can.elements()))
        right = ".".join(sorted(self.products_can.elements()))
        return f"R{self.id}: {left} >> {right}"
