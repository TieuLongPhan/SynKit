from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Set
import itertools as it
import re

from synkit.dev_crn.reaction import Reaction  # type: ignore
from synkit.dev_crn.network import ReactionNetwork  # type: ignore
from synkit.dev_crn.exceptions import CRNError  # type: ignore

_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)$")


def _parse_formula(formula: str) -> Dict[str, int]:
    """
    Parse a plain chemical formula into element counts.

    The parser accepts compact formulas without parentheses (e.g., ``"C2H6O"``),
    accumulates duplicated element symbols, and treats missing counts as ``1``.

    :param formula: Chemical formula string to parse.
    :type formula: str
    :returns: Mapping from element symbol to integer count (only nonzero counts).
    :rtype: Dict[str, int]
    :raises ValueError: If the input is empty, not a string, or contains an invalid token.
    """
    if not formula or not isinstance(formula, str):
        raise ValueError("Empty formula")
    out: Dict[str, int] = {}
    for sym, num in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if not _TOKEN.match(f"{sym}{num}"):
            raise ValueError(f"Invalid token: {sym}{num}")
        out[sym] = out.get(sym, 0) + (int(num) if num else 1)
    if not out:
        raise ValueError(f"Failed to parse: {formula}")
    return out


def _counts_key(counts: Dict[str, int]) -> Tuple[Tuple[str, int], ...]:
    """
    Convert an element-count mapping into a canonical, hashable key.

    :param counts: Element counts for a species or a multiset sum.
    :type counts: Dict[str, int]
    :returns: Sorted tuple of ``(element, count)`` pairs for nonzero counts.
    :rtype: Tuple[Tuple[str, int], ...]
    """
    return tuple(sorted((k, v) for k, v in counts.items() if v))


def _multiset_from_tuple(t: Tuple[str, ...]) -> Dict[str, int]:
    """
    Build a name→multiplicity multiset from a tuple of species names.

    :param t: Tuple of species names (possibly with repeats).
    :type t: Tuple[str, ...]
    :returns: Mapping name→multiplicity.
    :rtype: Dict[str, int]
    """
    d: Dict[str, int] = {}
    for x in t:
        d[x] = d.get(x, 0) + 1
    return d


def _sum_counts(
    spec_counts: Dict[str, Dict[str, int]],
    multiset: Dict[str, int],
) -> Dict[str, int]:
    """
    Sum element counts of a multiset of species.

    :param spec_counts: Precomputed element-counts for each species label.
    :type spec_counts: Dict[str, Dict[str, int]]
    :param multiset: Name→multiplicity mapping representing the LHS.
    :type multiset: Dict[str, int]
    :returns: Total element counts of the multiset (nonzero only).
    :rtype: Dict[str, int]
    """
    res: Dict[str, int] = {}
    for name, m in multiset.items():
        for el, n in spec_counts[name].items():
            res[el] = res.get(el, 0) + n * m
    return {k: v for k, v in res.items() if v}


def _fmt_side(side: Dict[str, int]) -> str:
    """
    Human-readable formatter for one reaction side.

    :param side: Name→coefficient mapping (``{species: coeff}``).
    :type side: Dict[str, int]
    :returns: Pretty side string, e.g., ``"2 A + B"`` or ``"∅"``.
    :rtype: str
    """
    if not side:
        return "∅"
    parts = []
    for k in sorted(side):
        v = side[k]
        parts.append(f"{v} {k}" if v != 1 else k)
    return " + ".join(parts)


# ----------------------------- CRNFormula ------------------------------
class CRNFormula:
    """
    Construct a :class:`ReactionNetwork` from a collection of **formulas**.

    By default this enumerates **synthesis (1..k→1)** reactions only and,
    if requested, adds the **reverse decompositions (1←1..k)**. Optionally,
    1→1 isomerizations can be included for species that share element counts
    but differ in labels.

    .. note::
       This class **does not** enumerate general ``m→n`` redistributions by
       default (a common source of unwanted combinatorics).

    :param max_reactants: Maximum number of molecules on the LHS for synthesis
        enumeration. Coefficients arise as multiplicities in the multiset.
    :type max_reactants: int, optional
    :param include_reverse: If ``True``, add decomposition reactions for each
        enumerated synthesis.
    :type include_reverse: bool, optional
    :param include_isomers: If ``True``, add isomerization edges (1→1) between
        species with identical element counts but distinct labels.
    :type include_isomers: bool, optional
    :param allow_overlap: If ``False``, forbid any species name from appearing
        on both sides of a reaction.
    :type allow_overlap: bool, optional
    :param deduplicate: If ``True``, suppress stoichiometrically identical
        reactions (based on lhs/rhs name-coefficient signatures).
    :type deduplicate: bool, optional
    :param id_start: Starting identifier for generated :class:`Reaction` objects.
    :type id_start: int, optional
    """

    def __init__(
        self,
        max_reactants: int = 3,
        include_reverse: bool = True,
        include_isomers: bool = False,
        allow_overlap: bool = False,
        deduplicate: bool = True,
        id_start: int = 0,
    ) -> None:
        self.max_reactants = int(max(1, max_reactants))
        self.include_reverse = bool(include_reverse)
        self.include_isomers = bool(include_isomers)
        self.allow_overlap = bool(allow_overlap)
        self.deduplicate = bool(deduplicate)
        self._id_next = int(id_start)

        self._species: List[str] = []
        self._counts: Dict[str, Dict[str, int]] = {}
        self._rx_tuples: List[Tuple[Dict[str, int], Dict[str, int], str]] = []
        self._errors: List[Tuple[int, Any, str]] = []
        self._net: Optional[ReactionNetwork] = None

    # ----------------- fluent configuration -----------------
    def set_search(
        self,
        max_reactants: Optional[int] = None,
        include_reverse: Optional[bool] = None,
        include_isomers: Optional[bool] = None,
        allow_overlap: Optional[bool] = None,
        deduplicate: Optional[bool] = None,
    ) -> "CRNFormula":
        """
        Update search parameters in a fluent style.

        All arguments are optional; unspecified parameters retain their
        current values.

        :param max_reactants: New maximum LHS arity for synthesis enumeration.
        :type max_reactants: int, optional
        :param include_reverse: Whether to include reverse decompositions.
        :type include_reverse: bool, optional
        :param include_isomers: Whether to include 1→1 isomerizations.
        :type include_isomers: bool, optional
        :param allow_overlap: Whether to allow species overlap across sides.
        :type allow_overlap: bool, optional
        :param deduplicate: Whether to deduplicate stoichiometrically identical reactions.
        :type deduplicate: bool, optional
        :returns: This instance for chaining.
        :rtype: CRNFormula
        """
        if max_reactants is not None:
            self.max_reactants = int(max(1, max_reactants))
        if include_reverse is not None:
            self.include_reverse = bool(include_reverse)
        if include_isomers is not None:
            self.include_isomers = bool(include_isomers)
        if allow_overlap is not None:
            self.allow_overlap = bool(allow_overlap)
        if deduplicate is not None:
            self.deduplicate = bool(deduplicate)
        return self

    def clear(self) -> "CRNFormula":
        """
        Clear internal species, reactions, errors, and cached network.

        :returns: This instance for chaining.
        :rtype: CRNFormula
        """
        self._species.clear()
        self._counts.clear()
        self._rx_tuples.clear()
        self._errors.clear()
        self._net = None
        self._id_next = 0
        return self

    # ----------------- ingestion -----------------
    def process_list(self, formulas: Sequence[str]) -> "CRNFormula":
        """
        Ingest a list of formula strings (labels equal to formulas by default).

        Duplicate labels are made unique by suffixing ``#2``, ``#3``, etc.
        Parsing errors are recorded in :pyattr:`errors`.

        :param formulas: Sequence of chemical formulas, e.g., ``['CHN', 'C2H2N2']``.
        :type formulas: Sequence[str]
        :returns: This instance for chaining.
        :rtype: CRNFormula
        """
        self._species = []
        self._counts = {}
        self._errors = []
        seen: Set[str] = set()
        for idx, f in enumerate(formulas):
            try:
                name = str(f)
                if name in seen:
                    i = 2
                    base = name
                    while f"{base}#{i}" in seen:
                        i += 1
                    name = f"{base}#{i}"
                self._species.append(name)
                self._counts[name] = _parse_formula(str(f))
                seen.add(name)
            except Exception as exc:  # noqa: BLE001
                self._errors.append((idx, f, str(exc)))
        return self

    def process_list_dict(
        self,
        records: Sequence[Dict[str, Any]],
        formula_key: str = "formula",
        id_key: Optional[str] = None,
    ) -> "CRNFormula":
        """
        Ingest a list of records containing formulas and optional IDs.

        If ``id_key`` is provided and present in a record, its value is used as
        the species label; otherwise the formula string is used. Duplicate labels
        are made unique by suffixing ``#2``, ``#3``, etc. Parsing errors are
        recorded in :pyattr:`errors`.

        :param records: Sequence of dictionaries, each containing at least ``formula_key``.
        :type records: Sequence[Dict[str, Any]]
        :param formula_key: Dictionary key under which the formula string is stored.
        :type formula_key: str, optional
        :param id_key: Optional dictionary key providing a stable species label.
        :type id_key: str, optional
        :returns: This instance for chaining.
        :rtype: CRNFormula
        """
        self._species = []
        self._counts = {}
        self._errors = []
        seen: Set[str] = set()
        for idx, rec in enumerate(records):
            try:
                f = str(rec[formula_key])
                name = str(rec[id_key]) if id_key and id_key in rec else f
                if name in seen:
                    i = 2
                    base = name
                    while f"{base}#{i}" in seen:
                        i += 1
                    name = f"{base}#{i}"
                self._species.append(name)
                self._counts[name] = _parse_formula(f)
                seen.add(name)
            except Exception as exc:  # noqa: BLE001
                self._errors.append((idx, rec, str(exc)))
        return self

    # ----------------- core build -----------------
    def build(self) -> "CRNFormula":
        """
        Enumerate reactions and construct a :class:`ReactionNetwork`.

        The enumeration includes:
          * **Synthesis**: all multisets of size ``1..max_reactants`` on the LHS
            that sum (by element counts) to a single product on the RHS.
          * **Reverse** (optional): decomposition reactions for each synthesis.
          * **Isomerizations** (optional): 1→1 edges for species sharing counts
            but having different labels.

        Stoichiometrically identical edges (same name→coeff maps) are deduplicated
        if :pyattr:`deduplicate` is ``True``.

        :returns: This instance for chaining.
        :rtype: CRNFormula
        :raises CRNError: If :class:`Reaction` objects cannot be constructed.
        """
        self._rx_tuples.clear()
        self._net = None

        # Map element-count signatures to species names (may be >1 for isomers)
        by_counts: Dict[Tuple[Tuple[str, int], ...], List[str]] = {}
        for nm, cnt in self._counts.items():
            by_counts.setdefault(_counts_key(cnt), []).append(nm)

        # Enumerate LHS multisets (coefficients via multiplicities)
        lhs_multisets: List[Dict[str, int]] = []
        for rsize in range(1, self.max_reactants + 1):
            for combo in it.combinations_with_replacement(self._species, rsize):
                lhs_multisets.append(_multiset_from_tuple(combo))

        # Synthesis: LHS → single product with matching element counts
        dedup_seen: Set[
            Tuple[Tuple[Tuple[str, int], ...], Tuple[Tuple[str, int], ...]]
        ] = set()
        for lhs in lhs_multisets:
            lhs_counts = _sum_counts(self._counts, lhs)
            key = _counts_key(lhs_counts)
            candidates = by_counts.get(key, [])
            for prod in candidates:
                if not self.allow_overlap and prod in lhs:
                    continue
                rhs = {prod: 1}
                sig = (tuple(sorted(lhs.items())), tuple(sorted(rhs.items())))
                if self.deduplicate and sig in dedup_seen:
                    continue
                dedup_seen.add(sig)
                orig = f"{_fmt_side(lhs)} -> {_fmt_side(rhs)}"
                self._rx_tuples.append((lhs, rhs, orig))
                if self.include_reverse:
                    lhs_r, rhs_r = rhs, lhs
                    sig_r = (tuple(sorted(lhs_r.items())), tuple(sorted(rhs_r.items())))
                    if not (self.deduplicate and sig_r in dedup_seen):
                        dedup_seen.add(sig_r)
                        orig_r = f"{_fmt_side(lhs_r)} -> {_fmt_side(rhs_r)}"
                        self._rx_tuples.append((lhs_r, rhs_r, orig_r))

        # Optional: isomerizations 1→1 for identical counts but distinct labels
        if self.include_isomers:
            for names in by_counts.values():
                if len(names) < 2:
                    continue
                for a, b in it.permutations(names, 2):
                    if not self.allow_overlap and a == b:
                        continue
                    lhs = {a: 1}
                    rhs = {b: 1}
                    sig = (tuple(sorted(lhs.items())), tuple(sorted(rhs.items())))
                    if self.deduplicate and sig in dedup_seen:
                        continue
                    dedup_seen.add(sig)
                    orig = f"{_fmt_side(lhs)} -> {_fmt_side(rhs)}"
                    self._rx_tuples.append((lhs, rhs, orig))

        # Build Reaction objects and wrap in ReactionNetwork
        rx_objs: List[Reaction] = []
        for lhs, rhs, orig in self._rx_tuples:
            payload = {
                "id": self._id_next,
                "original_raw": orig,
                "reactants_can": dict(lhs),
                "products_can": dict(rhs),
            }
            try:
                rx = Reaction.from_dict(payload)  # preferred (synkit-compatible)
            except Exception:
                try:
                    rx = Reaction(**payload)  # fallback
                except Exception as exc:
                    raise CRNError("Failed to construct Reaction") from exc
            rx_objs.append(rx)
            self._id_next += 1

        try:
            self._net = ReactionNetwork(rx_objs)
        except Exception:
            self._net = ReactionNetwork()
            for r in rx_objs:
                self._net.add_reaction(r)

        return self

    # ----------------- properties -----------------
    @property
    def species(self) -> List[str]:
        """
        Species labels ingested so far.

        :returns: List of species names (labels).
        :rtype: List[str]
        """
        return list(self._species)

    @property
    def reactions(self) -> List[Tuple[Dict[str, int], Dict[str, int], str]]:
        """
        Raw reaction tuples produced by :py:meth:`build`.

        Each tuple contains ``(lhs_dict, rhs_dict, original_raw)`` in build order.

        :returns: List of raw reaction triples.
        :rtype: List[Tuple[Dict[str, int], Dict[str, int], str]]
        """
        return list(self._rx_tuples)

    @property
    def network(self) -> Optional[ReactionNetwork]:
        """
        The constructed :class:`ReactionNetwork` (after :py:meth:`build`).

        :returns: ReactionNetwork instance or ``None`` if not built yet.
        :rtype: Optional[ReactionNetwork]
        """
        return self._net

    @property
    def errors(self) -> List[Tuple[int, Any, str]]:
        """
        Errors collected during ingestion.

        Each entry is a triple ``(index, payload, message)`` describing the
        offending record and the associated error message.

        :returns: List of ingestion errors.
        :rtype: List[Tuple[int, Any, str]]
        """
        return list(self._errors)

    # ----------------- dunders / misc -----------------
    def __repr__(self) -> str:  # pragma: no cover (cosmetic)
        """
        Developer-friendly summary string.

        :returns: Short string containing species/reaction counts and settings.
        :rtype: str
        """
        n = len(self._species)
        r = len(self._rx_tuples)
        return (
            f"CRNFormula(|S|={n}, |R|={r}, mode='synthesis±reverse', "
            f"max_reactants={self.max_reactants})"
        )
