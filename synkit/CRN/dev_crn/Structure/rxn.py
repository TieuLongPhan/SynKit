from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Set,
    ItemsView,
    KeysView,
    ValuesView,
)
import copy
import re

SideInput = Union[
    Mapping[str, int],
    Iterable[str],
    Iterable[Tuple[str, int]],
    None,
]


@dataclass
class RXNSide(MutableMapping[str, int]):
    """
    Stoichiometric multiset for one reaction side (LHS or RHS).

    This class normalizes many input formats into a mapping
    ``{species_label: positive_integer_count}``.

    Supported side-string examples
    ------------------------------
    - ``"2A + B"``
    - ``"2*A + B"``
    - ``"2 A + B"``
    - ``"2A.B"``
    - ``"B.3C"``
    - ``"A.A"``
    - ``"B.C.C.C"``
    - ``"2A + B.C"``
    - ``"∅"`` or ``""`` for empty

    Interpretation rules
    --------------------
    - ``+`` and ``.`` are both treated as separators between terms.
    - ``2A`` means species ``A`` with coefficient 2.
    - ``A.A`` means two occurrences of species ``A``.
    - ``B.C.C.C`` means ``B + 3C``.
    - Zero or negative counts are dropped during normalization.

    :param data:
        Initial content. May be:
        - a mapping ``species -> count``
        - an iterable of species labels
        - an iterable of ``(species, count)`` pairs
        - ``None`` for an empty side
    :type data: SideInput

    .. code-block:: python

        s1 = RXNSide.from_str("2A + B")
        assert s1.to_dict() == {"A": 2, "B": 1}

        s2 = RXNSide.from_str("A.A >> B.C.C.C".split(">>")[0])
        assert s2.to_dict() == {"A": 2}

        s3 = RXNSide.from_str("B.3C")
        assert s3.to_dict() == {"B": 1, "C": 3}

        s4 = RXNSide.from_any(["A", "B", "A"])
        assert s4.to_dict() == {"A": 2, "B": 1}
    """

    data: Dict[str, int] = field(default_factory=dict)

    _EMPTY_TOKENS = {"", "∅", "0", "null", "None"}
    _TERM_SPLIT_RE = re.compile(r"\s*(?:\+|\.)\s*")
    _PREFIX_COEFF_RE = re.compile(r"^\s*(\d+)\s*\*?\s*(.+?)\s*$")
    _COMPACT_COEFF_RE = re.compile(r"^\s*(\d+)([A-Za-z].*?)\s*$")

    def __post_init__(self) -> None:
        self.data = self._normalize_any(self.data)

    # ------------------------------------------------------------------
    # normalization helpers
    # ------------------------------------------------------------------
    @classmethod
    def _normalize_any(cls, obj: SideInput) -> Dict[str, int]:
        """
        Normalize mapping / iterable / None into ``{species: positive_count}``.
        """
        out: Dict[str, int] = {}
        if obj is None:
            return out

        if isinstance(obj, Mapping):
            for key, value in obj.items():
                cls._add_term(out, str(key), int(value))
            return out

        for item in obj:
            if isinstance(item, tuple) and len(item) == 2:
                species, count = item
                cls._add_term(out, str(species), int(count))
            else:
                species = str(item).strip()
                if species and species not in cls._EMPTY_TOKENS:
                    cls._add_term(out, species, 1)

        return out

    @staticmethod
    def _add_term(out: Dict[str, int], species: str, count: int) -> None:
        """
        Add one normalized term into an output dictionary.
        """
        sp = str(species).strip()
        c = int(count)
        if not sp or c <= 0:
            return
        out[sp] = out.get(sp, 0) + c

    @classmethod
    def _parse_one_term(cls, token: str) -> Tuple[str, int]:
        """
        Parse one side term.

        Accepted forms include:
        - ``A``
        - ``2A``
        - ``2*A``
        - ``2 A``
        - ``Fe``
        - ``10Fe``
        - ``Cl2``
        - ``3Cl2``

        :param token: One term from a side string.
        :type token: str
        :returns: ``(species, count)``
        :rtype: Tuple[str, int]
        """
        raw = token.strip()
        if raw in cls._EMPTY_TOKENS:
            return ("", 0)

        # Case 1: explicit separated coefficient, e.g. "2 A" or "2*A"
        m = cls._PREFIX_COEFF_RE.match(raw)
        if m:
            c = int(m.group(1))
            sp = m.group(2).strip()

            # Prevent misreading bare numbers as coefficients without species
            if sp and sp not in cls._EMPTY_TOKENS:
                return (sp, c)

        # Case 2: compact coefficient, e.g. "2A", "10Fe", "3Cl2"
        m = cls._COMPACT_COEFF_RE.match(raw)
        if m:
            c = int(m.group(1))
            sp = m.group(2).strip()
            if sp:
                return (sp, c)

        # Case 3: bare species
        return (raw, 1)

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_any(cls, obj: SideInput) -> RXNSide:
        """
        Build from mapping / iterable / None with normalization.

        :param obj: Side-like input.
        :type obj: SideInput
        :returns: Normalized side.
        :rtype: RXNSide
        """
        return cls(cls._normalize_any(obj))

    @classmethod
    def from_str(cls, side: str) -> RXNSide:
        """
        Parse one reaction side.

        Both ``+`` and ``.`` are accepted as separators between terms.
        This allows both coefficient-style and repeated-species style.

        Examples
        --------
        .. code-block:: python

            RXNSide.from_str("2A + B").to_dict()
            # {"A": 2, "B": 1}

            RXNSide.from_str("B.3C").to_dict()
            # {"B": 1, "C": 3}

            RXNSide.from_str("A.A").to_dict()
            # {"A": 2}

            RXNSide.from_str("B.C.C.C").to_dict()
            # {"B": 1, "C": 3}

            RXNSide.from_str("2A + B.C").to_dict()
            # {"A": 2, "B": 1, "C": 1}

        :param side: Side string.
        :type side: str
        :returns: Parsed side.
        :rtype: RXNSide
        """
        if side is None:
            return cls()

        text = str(side).strip()
        if text in cls._EMPTY_TOKENS:
            return cls()

        pieces = [p.strip() for p in cls._TERM_SPLIT_RE.split(text) if p.strip()]
        out: Dict[str, int] = {}

        for piece in pieces:
            sp, c = cls._parse_one_term(piece)
            cls._add_term(out, sp, c)

        return cls(out)

    # ------------------------------------------------------------------
    # mapping interface
    # ------------------------------------------------------------------
    def __getitem__(self, key: str) -> int:
        return self.data[key]

    def __setitem__(self, key: str, value: int) -> None:
        self.set(key, value)

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __contains__(self, key: object) -> bool:
        return key in self.data

    def items(self) -> ItemsView[str, int]:
        return self.data.items()

    def keys(self) -> KeysView[str]:
        return self.data.keys()

    def values(self) -> ValuesView[int]:
        return self.data.values()

    def get(self, key: str, default: Optional[int] = None) -> Optional[int]:
        return self.data.get(key, default)

    def pop(self, key: str, default: Optional[int] = None) -> Optional[int]:
        return self.data.pop(key, default)  # type: ignore[return-value]

    def clear(self) -> None:
        self.data.clear()

    def set(self, species: str, count: int) -> None:
        """
        Set a species coefficient. Non-positive values remove the species.
        """
        sp = str(species).strip()
        c = int(count)
        if not sp:
            return
        if c <= 0:
            self.data.pop(sp, None)
        else:
            self.data[sp] = c

    def update(
        self,
        other: Union[Mapping[str, int], Iterable[Tuple[str, int]], Iterable[str]],
        **kwargs: int,
    ) -> None:
        """
        Add counts from another side-like object.

        Unlike ``dict.update``, this performs stoichiometric addition.

        .. code-block:: python

            s = RXNSide.from_str("A + 2B")
            s.update({"A": 1, "C": 3})
            assert s.to_dict() == {"A": 2, "B": 2, "C": 3}
        """
        norm = self._normalize_any(other)
        for k, v in norm.items():
            self.data[k] = self.data.get(k, 0) + v

        if kwargs:
            for k, v in kwargs.items():
                self.data[str(k)] = self.data.get(str(k), 0) + int(v)

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, int]:
        """
        Export as a plain dict.
        """
        return dict(self.data)

    def copy(self) -> RXNSide:
        """
        Deep copy this side.
        """
        return RXNSide(copy.deepcopy(self.data))

    def species(self) -> Set[str]:
        """
        Return the set of present species.
        """
        return set(self.data.keys())

    def coeff(self, species: str) -> int:
        """
        Return the stoichiometric coefficient of a species, or 0 if absent.
        """
        return int(self.data.get(str(species), 0))

    def incr(self, species: str, by: int = 1) -> None:
        """
        Increment a species count by an integer amount.

        Removes the species if the resulting count is non-positive.
        """
        sp = str(species).strip()
        if not sp:
            return
        new_count = int(self.data.get(sp, 0)) + int(by)
        if new_count <= 0:
            self.data.pop(sp, None)
        else:
            self.data[sp] = new_count

    def remove(self, species: str, by: int = 1) -> None:
        """
        Decrement a species count.
        """
        self.incr(species, -int(by))

    def arity(self, include_coeff: bool = False) -> int:
        """
        Count molecules on this side.

        :param include_coeff:
            If ``False``, count distinct present terms.
            If ``True``, sum stoichiometric coefficients.
        :type include_coeff: bool
        :returns: Arity under the chosen convention.
        :rtype: int
        """
        if include_coeff:
            return sum(self.data.values())
        return len(self.data)

    def is_empty(self) -> bool:
        """
        Return ``True`` if this side has no species.
        """
        return not self.data

    def expand(self) -> List[str]:
        """
        Expand into a flat list respecting stoichiometry.

        Example:
        ``{"A": 2, "B": 1} -> ["A", "A", "B"]``
        """
        out: List[str] = []
        for sp in sorted(self.data):
            out.extend([sp] * self.data[sp])
        return out

    def sorted_items(self) -> List[Tuple[str, int]]:
        """
        Return sorted ``(species, count)`` pairs.
        """
        return sorted(self.data.items(), key=lambda kv: kv[0])

    # ------------------------------------------------------------------
    # formatting
    # ------------------------------------------------------------------
    def to_string(
        self,
        sep: str = " + ",
        repeated: bool = False,
        sort: bool = True,
    ) -> str:
        """
        Format this side as a string.

        :param sep: Separator between terms.
        :type sep: str
        :param repeated:
            If ``False``, use coefficient style: ``2A + B``.
            If ``True``, use repeated style: ``A.A.B`` if ``sep="."``.
        :type repeated: bool
        :param sort: Whether to sort species labels.
        :type sort: bool
        :returns: String representation.
        :rtype: str

        .. code-block:: python

            s = RXNSide.from_str("2A + B")
            assert s.to_string() == "2A + B"
            assert s.to_string(sep=".", repeated=True) == "A.A.B"
        """
        if not self.data:
            return "∅"

        items = self.sorted_items() if sort else list(self.data.items())

        parts: List[str] = []
        if repeated:
            for sp, c in items:
                parts.extend([sp] * int(c))
        else:
            for sp, c in items:
                parts.append(sp if c == 1 else f"{c}{sp}")

        return sep.join(parts)

    def __repr__(self) -> str:
        return self.to_string()

    def __str__(self) -> str:
        return self.to_string()

    # ------------------------------------------------------------------
    # algebra
    # ------------------------------------------------------------------
    def __add__(self, other: RXNSide) -> RXNSide:
        if not isinstance(other, RXNSide):
            return NotImplemented
        out = self.copy()
        out.update(other.data)
        return out

    def __iadd__(self, other: RXNSide) -> RXNSide:
        if not isinstance(other, RXNSide):
            return NotImplemented
        self.update(other.data)
        return self

    def __sub__(self, other: RXNSide) -> RXNSide:
        if not isinstance(other, RXNSide):
            return NotImplemented
        out = self.copy()
        for sp, c in other.items():
            out.incr(sp, -c)
        return out

    def __isub__(self, other: RXNSide) -> RXNSide:
        if not isinstance(other, RXNSide):
            return NotImplemented
        for sp, c in other.items():
            self.incr(sp, -c)
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RXNSide):
            return False
        return self.data == other.data
