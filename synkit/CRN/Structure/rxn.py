from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union, Set
import copy
import re


@dataclass
class RXNSide:
    """
    Stoichiometric multiset for one reaction side (LHS or RHS).

    The class normalizes many possible input formats into a mapping
    ``{species_label: count}`` where counts are positive integers and
    species labels are strings.

    :param data: Initial content. May be any of:
                 - a mapping ``species -> count``
                 - an iterable of species labels (each counts as +1)
                 - an iterable of ``(species, count)`` pairs
                 - ``None`` (defaults to empty side)
    :type data: Union[Mapping[str, int], Iterable[str], Iterable[Tuple[str, int]], None]

    :returns: RXNSide instance with normalized counts

    Examples
    --------
    .. code-block:: python

        from synkit.CRN.Hypergraph.rxn import RXNSide

        # parse from string
        s = RXNSide.from_str("2A + B")
        assert s.to_dict() == {"A": 2, "B": 1}

        # build from iterable
        s2 = RXNSide.from_any(["A", "B", "A"])
        assert s2.to_dict() == {"A": 2, "B": 1}

        # zero/negative counts are dropped
        s3 = RXNSide.from_any({"A": 0, "B": -1, "C": 3})
        assert s3.to_dict() == {"C": 3}
    """

    data: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data:
            self.data = RXNSide._normalize_any(self.data)

    # ---- normalization helpers ----
    @staticmethod
    def _normalize_any(
        obj: Union[Mapping[str, int], Iterable[str], Iterable[Tuple[str, int]]],
    ) -> Dict[str, int]:
        out: Dict[str, int] = {}
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                s = str(k)
                c = int(v)
                if c > 0:
                    out[s] = out.get(s, 0) + c
            return out
        for item in obj:
            if isinstance(item, tuple) and len(item) == 2:
                s, c = item
                s = str(s)
                c = int(c)
                if c > 0:
                    out[s] = out.get(s, 0) + c
            else:
                s = str(item)
                if s:
                    out[s] = out.get(s, 0) + 1
        return out

    # ---- constructors ----
    @classmethod
    def from_any(
        cls, obj: Union[Mapping[str, int], Iterable[str], Iterable[Tuple[str, int]]]
    ) -> RXNSide:
        """
        Build from mapping/iterable with normalization.

        :param obj: Mapping or iterable (labels or ``(label, count)`` pairs).
        :type obj: Union[Mapping[str, int], Iterable[str], Iterable[Tuple[str, int]]]
        :returns: Normalized side.
        :rtype: RXNSide
        """
        return cls(cls._normalize_any(obj))

    @classmethod
    def from_str(cls, side: str) -> RXNSide:
        """
        Parse a side like ``"2A + B"`` or ``"10Fe+2Cl2"``.

        Supported patterns include ``'2A+B'``, ``'2 A + B'``, ``'2*A+B'``, ``'A+B'``.
        ``'∅'`` or empty string returns an empty side.

        :param side: String for one reaction side (LHS or RHS).
        :type side: str
        :returns: Parsed and normalized side.
        :rtype: RXNSide

        Example
        -------
        .. code-block:: python

            RXNSide.from_str("2A + B")  # -> {"A":2, "B":1}
        """
        side = side.strip()
        if side == "" or side == "∅":
            return cls()

        parts = [p.strip() for p in side.split("+") if p.strip()]
        out: Dict[str, int] = {}

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Normalize "2*A" -> "2 A"
            part = part.replace("*", " ").strip()
            toks = part.split()

            if len(toks) == 1:
                token = toks[0]

                # Case 1: pattern like "2A", "10Fe", "3Cl2"
                m = re.match(r"^(\d+)([A-Za-z].*)$", token)
                if m:
                    c = int(m.group(1))
                    sp = m.group(2)
                    if c > 0:
                        out[sp] = out.get(sp, 0) + c

                # Case 2: bare species "A"
                else:
                    out[token] = out.get(token, 0) + 1
            else:
                # Case 3: tokens like ["2","A"], ["3","Fe"], etc.
                try:
                    c = int(toks[0])
                    sp = " ".join(toks[1:])
                    if c > 0:
                        out[sp] = out.get(sp, 0) + c
                except ValueError:
                    # Fallback: treat the whole chunk as one species label
                    sp = " ".join(toks)
                    out[sp] = out.get(sp, 0) + 1
        return cls(out)

    # ---- mapping-like API ----
    def __getitem__(self, key: str) -> int:
        return self.data[key]

    def __setitem__(self, key: str, value: int) -> None:
        c = int(value)
        if c <= 0:
            self.data.pop(str(key), None)
        else:
            self.data[str(key)] = c

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __contains__(self, key: object) -> bool:
        return key in self.data

    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def get(self, key: str, default: Optional[int] = None) -> Optional[int]:
        return self.data.get(key, default)

    def pop(self, key: str, default: Optional[int] = None) -> Optional[int]:
        return self.data.pop(key, default)  # type: ignore[return-value]

    def update(
        self, other: Union[Mapping[str, int], Iterable[Tuple[str, int]]]
    ) -> None:
        for k, v in RXNSide._normalize_any(other).items():
            self.data[k] = self.data.get(k, 0) + v

    # ---- utilities ----
    def to_dict(self) -> Dict[str, int]:
        """
        Export as a plain dict.

        :returns: ``{species: count}`` with counts >= 0.
        :rtype: Dict[str, int]
        """
        return dict(self.data)

    def copy(self) -> RXNSide:
        """
        Deep copy.

        :returns: A deep-copied side.
        :rtype: RXNSide
        """
        return RXNSide(copy.deepcopy(self.data))

    def species(self) -> Set[str]:
        """
        Species present on this side.

        :returns: Set of species labels.
        :rtype: Set[str]
        """
        return set(self.data.keys())

    def incr(self, species: str, by: int = 1) -> None:
        """
        Increment a species count (removes entry when it reaches 0).

        :param species: Species label.
        :type species: str
        :param by: Increment amount (can be negative).
        :type by: int
        """
        s = str(species)
        c = int(self.data.get(s, 0)) + int(by)
        if c <= 0:
            self.data.pop(s, None)
        else:
            self.data[s] = c

    def arity(self, include_coeff: bool = False) -> int:
        """
        Count the number of molecules on this side under two conventions.

        If ``include_coeff`` is ``False`` (default), each ‘+’-separated term
        counts as 1 regardless of its coefficient (e.g. ``2A`` and ``A`` both
        contribute 1). If ``include_coeff`` is ``True``, the integer
        coefficients are summed (e.g. ``2A+B`` contributes 3).

        :param include_coeff: Whether to sum coefficients (``True``) or count
                              unique terms (``False``).
        :type include_coeff: bool
        :returns: Arity of this side under the chosen convention.
        :rtype: int
        """
        if not self.data:
            return 0
        if include_coeff:
            return sum(int(c) for c in self.data.values() if int(c) > 0)
        # each distinct species present counts as 1 if its coefficient > 0
        return sum(1 for c in self.data.values() if int(c) > 0)

    def expand(self) -> List[str]:
        """
        Expand this side to a flat list of species labels respecting stoichiometry.

        Example: ``{A:2, B:1} -> ["A", "A", "B"]``

        :returns: Expanded list of species labels with repetitions per coefficient.
        :rtype: List[str]
        """
        out: List[str] = []
        for sp, c in self.data.items():
            out.extend([sp] * int(c))
        return out

    def __repr__(self) -> str:
        if not self.data:
            return "∅"
        parts = []
        for s in sorted(self.data.keys()):
            c = self.data[s]
            parts.append(f"{s}" if c == 1 else f"{c}{s}")
        return " + ".join(parts)
