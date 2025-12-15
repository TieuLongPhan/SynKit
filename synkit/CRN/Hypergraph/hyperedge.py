from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Mapping, Iterable, Tuple, Set
import copy

from .rxn import RXNSide


@dataclass
class HyperEdge:
    """
    Reaction hyperedge container: reactants -> products.

    :param id: Unique edge identifier (string).
    :type id: str
    :param reactants: Reactant multiset. May be an ``RXNSide`` instance, a
                      mapping ``{species: count}``, an iterable of species labels,
                      or an iterable of ``(species, count)`` pairs.
    :type reactants: Union[RXNSide, Mapping[str, int], Iterable[str], Iterable[Tuple[str, int]]]
    :param products: Product multiset (same accepted types as ``reactants``).
    :type products: Union[RXNSide, Mapping[str, int], Iterable[str], Iterable[Tuple[str, int]]]
    :param rule: Rule/label associated with the reaction (e.g., template id).
                 Defaults to ``"r"``.
    :type rule: str

    Examples
    --------
    .. code-block:: python

        from synkit.CRN.Hypergraph.hyperedge import HyperEdge

        e = HyperEdge("e1", {"A": 2, "B": 1}, {"C": 1}, rule="r1")
        assert e.id == "e1"
        assert e.reactants.to_dict() == {"A": 2, "B": 1}
        assert e.products.to_dict() == {"C": 1}
    """

    id: str
    reactants: Union[
        RXNSide, Mapping[str, int], Iterable[str], Iterable[Tuple[str, int]]
    ]
    products: Union[
        RXNSide, Mapping[str, int], Iterable[str], Iterable[Tuple[str, int]]
    ]
    rule: str = "r"

    def __post_init__(self) -> None:
        if not isinstance(self.reactants, RXNSide):
            self.reactants = RXNSide.from_any(self.reactants)
        if not isinstance(self.products, RXNSide):
            self.products = RXNSide.from_any(self.products)

    def species(self) -> Set[str]:
        """Return species participating in this reaction (union of reactants & products)."""
        return self.reactants.species() | self.products.species()

    def is_trivial(self) -> bool:
        """True if reactants and products are identical multisets."""
        return self.reactants.to_dict() == self.products.to_dict()

    def arity(self, include_coeff: bool = False):
        """Return (n_reactants, n_products) under chosen convention."""
        return self.reactants.arity(include_coeff), self.products.arity(include_coeff)

    def copy(self) -> "HyperEdge":
        """Deep copy of the edge."""
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return f"{self.id}: {self.reactants} >> {self.products}  (rule={self.rule})"
