from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from collections import Counter

from .exceptions import InvalidReactionError, StandardizationError
from .utils import split_components, normalize_counter


# Optional external standardizer—replace with your actual import.
try:
    # e.g., from synkit or your own module
    from ..Chem.Reaction.standardize import Standardize  # type: ignore
except Exception:  # pragma: no cover
    Standardize = None  # type: ignore


@dataclass
class Reaction:
    """
    Canonical representation of a reaction with convenience builders.

    Fluent API
    ----------
    - Use :py:meth:`standardize` to set the canonical form.
    - Use :py:meth:`build` once to populate reactants/products.
    - Query derived data via properties (:pyattr:`reactants_can`, :pyattr:`products_can`, etc.).

    :param id: Stable index of the reaction.
    :param original_raw: Original reaction SMILES (keeps atom maps).
    """

    id: int
    original_raw: str
    canonical_raw: Optional[str] = None
    _reactants_can: Counter = field(default_factory=Counter, init=False, repr=False)
    _products_can: Counter = field(default_factory=Counter, init=False, repr=False)

    # ---- Fluent builders ----
    def standardize(
        self, standardizer: Optional["Standardize"] = None, *, remove_aam: bool = True
    ) -> "Reaction":
        """
        Standardize ``original_raw`` to ``canonical_raw``.

        :param standardizer: Optional standardizer instance.
        :param remove_aam: Whether to drop atom maps.
        :returns: ``self`` for chaining.
        :raises StandardizationError: If standardization fails irrecoverably.
        """
        if standardizer is None:
            self.canonical_raw = self.original_raw
            return self
        try:
            self.canonical_raw = standardizer.fit(
                self.original_raw, remove_aam=remove_aam
            )
        except Exception as exc:  # keep it robust; caller can still proceed on raw
            raise StandardizationError(
                f"Failed to standardize reaction {self.id}: {exc}"
            ) from exc
        return self

    def build(self) -> "Reaction":
        """
        Parse canonical/raw string into reactant/product multisets.

        :returns: ``self`` for chaining.
        :raises InvalidReactionError: If the string is malformed.
        """
        rs = self.canonical_raw or self.original_raw
        if ">>" not in rs:
            raise InvalidReactionError(
                f"Invalid reaction string (missing '>>'): {rs!r}"
            )
        left, right = rs.split(">>", 1)
        self._reactants_can = Counter(split_components(left))
        self._products_can = Counter(split_components(right))
        return self

    # ---- Properties (read-only views) ----
    @property
    def reactants_can(self) -> Counter:
        """Reactant multiset (canonical tokens)."""
        return Counter(self._reactants_can)

    @property
    def products_can(self) -> Counter:
        """Product multiset (canonical tokens)."""
        return Counter(self._products_can)

    @property
    def net_change(self) -> Counter:
        """Products − Reactants (zero entries removed)."""
        net = Counter(self.products_can)
        net.subtract(self.reactants_can)
        return normalize_counter(net)

    # ---- Lightweight state application primitives ----
    def can_fire_forward(
        self, state: Counter, min_overlap: int = 1
    ) -> Tuple[bool, Counter]:
        """
        Check if the reaction can fire forward given ``state``.

        :param state: Current state.
        :param min_overlap: Minimum matched instances required.
        :returns: (ok, matched_subset).
        """
        matched = self.reactants_can & state
        return (sum(matched.values()) >= min_overlap, matched)

    def apply_forward(
        self, state: Counter, matched: Optional[Counter] = None
    ) -> Counter:
        """
        Apply the reaction forward to ``state``.

        :param state: Current state.
        :param matched: Subset to consume (defaults to maximal overlap).
        :returns: New state.
        """
        use = matched if matched is not None else (self.reactants_can & state)
        new_state = state - use
        new_state += self.products_can
        return normalize_counter(new_state)

    def can_fire_backward(
        self, state: Counter, min_overlap: int = 1
    ) -> Tuple[bool, Counter]:
        """Check backward applicability; see :py:meth:`can_fire_forward`."""
        matched = self.products_can & state
        return (sum(matched.values()) >= min_overlap, matched)

    def apply_backward(
        self, state: Counter, matched_products: Optional[Counter] = None
    ) -> Counter:
        """Apply backward; see :py:meth:`apply_forward`."""
        use = (
            matched_products
            if matched_products is not None
            else (self.products_can & state)
        )
        new_state = state - use
        new_state += self.reactants_can
        return normalize_counter(new_state)

    # ---- Serialization ----
    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            "id": self.id,
            "original_raw": self.original_raw,
            "canonical_raw": self.canonical_raw or self.original_raw,
            "reactants_can": dict(self._reactants_can),
            "products_can": dict(self._products_can),
        }

    @staticmethod
    def from_dict(d: Dict) -> "Reaction":
        r = Reaction(
            id=int(d["id"]),
            original_raw=str(d["original_raw"]),
            canonical_raw=str(d.get("canonical_raw", d["original_raw"])),
        )
        r._reactants_can = Counter(d.get("reactants_can", {}))
        r._products_can = Counter(d.get("products_can", {}))
        return r

    def __repr__(self) -> str:
        left = ".".join(sorted(self._reactants_can.elements()))
        right = ".".join(sorted(self._products_can.elements()))
        return f"Reaction(id={self.id}, {left} >> {right})"
