from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
    Sequence,
)

import copy
import re
import numpy as np
from collections import defaultdict, deque

import networkx as nx


@dataclass
class RXNSide:
    """
    Stoichiometric multiset for one reaction side (LHS or RHS).

    Normalization:
      - species stored as ``str``
      - counts coerced to ``int``, entries with ``count <= 0`` are dropped.

    Exposes a mapping-like API so it can replace ``Dict[str, int]`` seamlessly.

    :param data: Initial content (optional). May be:
                 * a mapping ``species -> count``,
                 * an iterable of species labels (each counts as +1),
                 * an iterable of ``(species, count)`` pairs.
    :type data: Union[Mapping[str, int], Iterable[str], Iterable[Tuple[str, int]], None]
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


@dataclass
class HyperEdge:
    """
    One hyperedge representing a (possibly stoichiometric) reaction:

        reactants (species → count)  ->  products (species → count)

    :param id: Unique edge identifier.
    :type id: str
    :param reactants: Reactant side (species → count).
    :type reactants: Union[RXNSide, Mapping[str, int], Iterable[str], Iterable[Tuple[str, int]]]
    :param products: Product side (species → count).
    :type products: Union[RXNSide, Mapping[str, int], Iterable[str], Iterable[Tuple[str, int]]]
    :param rule: Rule/label associated with the reaction (e.g., template id).
    :type rule: str
    """

    id: str
    reactants: RXNSide
    products: RXNSide
    rule: str = "r"

    def __post_init__(self) -> None:
        if not isinstance(self.reactants, RXNSide):
            self.reactants = RXNSide.from_any(self.reactants)
        if not isinstance(self.products, RXNSide):
            self.products = RXNSide.from_any(self.products)

    def species(self) -> Set[str]:
        """
        All species participating in this edge.

        :returns: Species present in reactants or products.
        :rtype: Set[str]
        """
        return self.reactants.species() | self.products.species()

    def is_trivial(self) -> bool:
        """
        Check if the reaction is stoichiometrically trivial.

        :returns: ``True`` if reactants and products are identical multisets.
        :rtype: bool
        """
        return self.reactants.to_dict() == self.products.to_dict()

    def arity(self, include_coeff: bool = False) -> Tuple[int, int]:
        """
        Total number of molecules on each side under the chosen convention.

        If ``include_coeff`` is ``False`` (default), each term counts as 1
        (``2A`` ≡ ``A``). If ``include_coeff`` is ``True``, coefficients are summed.

        :param include_coeff: Whether to sum coefficients for each side.
        :type include_coeff: bool
        :returns: ``(n_reactants, n_products)`` under the chosen convention.
        :rtype: Tuple[int, int]
        """
        return self.reactants.arity(include_coeff), self.products.arity(include_coeff)

    def copy(self) -> HyperEdge:
        """
        Deep copy.

        :returns: A deep-copied ``HyperEdge``.
        :rtype: HyperEdge
        """
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return f"{self.id}: {self.reactants} >> {self.products}  (rule={self.rule})"


class CRNHyperGraph:
    """
    Directed hypergraph where edges map many reactants -> many products.

    Species are identified by string keys; edge ids are strings.
    """

    def __init__(self) -> None:
        self.species: Set[str] = set()
        self.edges: Dict[str, HyperEdge] = {}  # id -> HyperEdge

        # per-rule counters used to produce readable ids like "r_1", "mechA_2"
        self._rule_counters: Dict[str, int] = defaultdict(int)

        # quick indices: species -> set of in/out edge ids
        self.species_to_in_edges: Dict[str, Set[str]] = defaultdict(set)
        self.species_to_out_edges: Dict[str, Set[str]] = defaultdict(set)

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------
    def _next_edge_id_for_rule(self, rule: str) -> str:
        """
        Produce a unique edge id using the rule as prefix (e.g. 'r_1').

        :param rule: Rule prefix.
        :type rule: str
        :returns: New unique edge identifier.
        :rtype: str
        """
        cnt = self._rule_counters.get(rule, 0) + 1
        self._rule_counters[rule] = cnt
        return f"{rule}_{cnt}"

    # ------------------------------------------------------------------
    # Parsing / adding / removing
    # ------------------------------------------------------------------
    def add_rxn(
        self,
        reactant_side: Union[
            RXNSide, Mapping[str, int], Iterable[str], Iterable[Tuple[str, int]]
        ],
        product_side: Union[
            RXNSide, Mapping[str, int], Iterable[str], Iterable[Tuple[str, int]]
        ],
        rule: Optional[str] = None,
        edge_id: Optional[str] = None,
    ) -> HyperEdge:
        """
        Add a hyperedge from ``reactant_side`` -> ``product_side``.

        :param reactant_side: Reactant multiset (species → count).
        :type reactant_side: Union[RXNSide, Mapping[str, int], Iterable[str], Iterable[Tuple[str, int]]]
        :param product_side: Product multiset (species → count).
        :type product_side: Union[RXNSide, Mapping[str, int], Iterable[str], Iterable[Tuple[str, int]]]
        :param rule: Rule/label; defaults to ``"r"``.
        :type rule: Optional[str]
        :param edge_id: Custom edge id. If ``None``, a new id is generated.
        :type edge_id: Optional[str]
        :returns: Created hyperedge.
        :rtype: HyperEdge
        :raises ValueError: If both sides are empty.
        :raises KeyError: If ``edge_id`` already exists.
        """
        r_side = (
            reactant_side
            if isinstance(reactant_side, RXNSide)
            else RXNSide.from_any(reactant_side)
        )
        p_side = (
            product_side
            if isinstance(product_side, RXNSide)
            else RXNSide.from_any(product_side)
        )

        rule = rule if rule is not None else "r"

        if edge_id is None:
            edge_id = self._next_edge_id_for_rule(rule)
        else:
            if edge_id in self.edges:
                raise KeyError(f"Edge id {edge_id!r} already exists")

        if not r_side.data and not p_side.data:
            raise ValueError("Reaction must have at least one reactant or product")

        e = HyperEdge(id=edge_id, reactants=r_side, products=p_side, rule=rule)
        self.edges[edge_id] = e

        # register species and indices
        for s in e.species():
            self.species.add(s)
            _ = self.species_to_in_edges[s]  # ensure keys exist
            _ = self.species_to_out_edges[s]

        for s in e.reactants.keys():
            self.species_to_out_edges[s].add(edge_id)
        for s in e.products.keys():
            self.species_to_in_edges[s].add(edge_id)
        return e

    def add_rxn_from_str(
        self,
        reaction: str,
        rule: Optional[str] = None,
        *,
        parse_rule_from_suffix: bool = True,
    ) -> HyperEdge:
        """
        Parse a reaction string into a hyperedge and add it to the graph.

        The function accepts strings like ``"A+B>>C"`` and optionally parses a
        trailing meta section of the form ``" | rule=R1"`` when
        ``parse_rule_from_suffix`` is enabled. If both an explicit ``rule`` is
        provided and a suffix is present, the explicit rule is used (the suffix is
        ignored). To allow the suffix to take precedence, call with
        ``rule=None`` and leave ``parse_rule_from_suffix=True``.

        :param reaction: Reaction string in ``LHS>>RHS`` format, optionally followed
                        by a meta suffix like ``" | rule=Rk"``.
        :type reaction: str
        :param rule: Explicit rule label to attach to the created hyperedge. If
                    ``None`` and ``parse_rule_from_suffix`` is ``True``, the rule
                    is read from the suffix if present; otherwise the caller's
                    default applies upstream.
        :type rule: Optional[str]
        :param parse_rule_from_suffix: If ``True``, parse a trailing
                                    ``" | rule=..."`` suffix from ``reaction``
                                    when ``rule`` is not provided.
        :type parse_rule_from_suffix: bool, keyword-only
        :returns: The newly created hyperedge.
        :rtype: HyperEdge
        :raises ValueError: If the reaction string does not contain the ``">>"`` separator.
        """
        rule_local = rule
        core = reaction

        if parse_rule_from_suffix and "|" in reaction:
            core, meta = reaction.split("|", 1)
            meta = meta.strip()
            m = re.search(r"rule\s*=\s*([^\s]+)", meta)
            if m and rule_local is None:
                rule_local = m.group(1)

        core = core.strip()
        if ">>" not in core:
            raise ValueError(f"Invalid reaction format (missing '>>'): {reaction!r}")
        left, right = core.split(">>", 1)
        reactants = RXNSide.from_str(left)
        products = RXNSide.from_str(right)
        return self.add_rxn(reactants, products, rule=rule_local)

    def parse_rxns(
        self,
        reactions: Union[
            Iterable[str],
            Iterable[Tuple[str, Optional[str]]],
            Mapping[str, Optional[str]],
        ],
        *,
        default_rule: str = "r",
        parse_rule_from_suffix: bool = True,
        rules: Optional[Sequence[Optional[str]]] = None,
        prefer_suffix: bool = False,
    ) -> "CRNHyperGraph":
        """
        Build a graph from reaction strings with flexible rule sources.

        Supports two complementary ways to supply reaction rules:
        (1) inline suffix in the reaction string (e.g., ``"A+B>>C | rule=R1"``),
        (2) external rule specification via tuples, a mapping, or a parallel
        ``rules`` list. If both an external rule and a suffix rule are present
        for the same reaction, precedence is controlled by ``prefer_suffix``.

        :param reactions: Input reactions. Accepted forms:
                        - ``Iterable[str]`` (e.g., ``["A+B>>C", "C>>D | rule=R2"]``).
                            If ``rules`` is provided, it is zipped with the strings
                            as explicit per-line rules.
                        - ``Iterable[Tuple[str, Optional[str]]]`` (e.g.,
                            ``[("A+B>>C","R1"), ("C>>D", None)]``).
                        - ``Mapping[str, Optional[str]]`` (e.g.,
                            ``{"A+B>>C": "R1", "C>>D": None}``).
        :type reactions: Union[Iterable[str], Iterable[Tuple[str, Optional[str]]], Mapping[str, Optional[str]]]
        :param default_rule: Rule used when neither an explicit rule nor a suffix
                            rule is provided.
        :type default_rule: str, keyword-only
        :param parse_rule_from_suffix: If ``True``, parse ``" | rule=..."`` suffixes
                                    from reaction strings (effective when a per-line
                                    explicit rule is not taking precedence).
        :type parse_rule_from_suffix: bool, keyword-only
        :param rules: Parallel rules for ``Iterable[str]`` input. Length must match
                    the number of reaction strings.
        :type rules: Optional[Sequence[Optional[str]]], keyword-only
        :param prefer_suffix: Conflict policy when both explicit and suffix rules are
                            present. If ``False`` (default), the explicit rule takes
                            precedence; if ``True``, the suffix rule takes precedence.
        :type prefer_suffix: bool, keyword-only
        :returns: A new graph instance populated with the given reactions.
        :rtype: CRNHyperGraph
        :raises ValueError: If a reaction string lacks the ``">>"`` separator, or if a
                            parallel ``rules`` sequence length does not match the number
                            of reactions.
        :notes: When explicit rule wins (``prefer_suffix=False``), the call to
                :meth:`add_rxn_from_str` disables suffix parsing to avoid accidental
                override. When suffix is preferred (``prefer_suffix=True``), the method
                first attempts to parse the suffix; if none is found, it falls back to
                the explicit rule (or ``default_rule`` when no explicit rule is provided).
        """

        # Normalize 'reactions' into iterable of (line, explicit_rule)
        if isinstance(reactions, Mapping):
            items: Iterable[Tuple[str, Optional[str]]] = reactions.items()
        else:
            if rules is not None:
                rxn_list = list(reactions)
                if len(rxn_list) != len(rules):
                    raise ValueError(
                        f"'rules' length ({len(rules)}) does not match number of "
                        f"reactions ({len(rxn_list)})."
                    )
                items = ((line, rules[i]) for i, line in enumerate(rxn_list))
            else:

                def _iter_items() -> Iterable[Tuple[str, Optional[str]]]:
                    for rec in reactions:
                        if isinstance(rec, tuple) and len(rec) >= 2:
                            yield (str(rec[0]), rec[1])
                        else:
                            yield (str(rec), None)

                items = _iter_items()

        for line, explicit_rule in items:
            if explicit_rule is not None:
                if prefer_suffix and parse_rule_from_suffix:
                    # Prefer suffix if present; else use explicit
                    if re.search(r"\|\s*rule\s*=\s*[^\s]+", line):
                        self.add_rxn_from_str(
                            line, rule=None, parse_rule_from_suffix=True
                        )
                    else:
                        self.add_rxn_from_str(
                            line, rule=explicit_rule, parse_rule_from_suffix=False
                        )
                else:
                    # Explicit wins; avoid suffix override
                    self.add_rxn_from_str(
                        line, rule=explicit_rule, parse_rule_from_suffix=False
                    )
            else:
                # No explicit rule: allow suffix, else default_rule
                self.add_rxn_from_str(
                    line,
                    rule=default_rule,
                    parse_rule_from_suffix=parse_rule_from_suffix,
                )
        return self

    def remove_rxn(self, edge_id: str) -> None:
        """
        Remove a hyperedge by id and update indices.

        :param edge_id: Identifier of the edge to remove.
        :type edge_id: str
        :raises KeyError: If ``edge_id`` is not present.
        """
        if edge_id not in self.edges:
            raise KeyError(edge_id)
        e = self.edges.pop(edge_id)
        for s in list(e.reactants.keys()):
            self.species_to_out_edges[s].discard(edge_id)
            if not self.species_to_in_edges[s] and not self.species_to_out_edges[s]:
                self.species.discard(s)
                self.species_to_in_edges.pop(s, None)
                self.species_to_out_edges.pop(s, None)
        for s in list(e.products.keys()):
            self.species_to_in_edges[s].discard(edge_id)
            if not self.species_to_in_edges[s] and not self.species_to_out_edges[s]:
                self.species.discard(s)
                self.species_to_in_edges.pop(s, None)
                self.species_to_out_edges.pop(s, None)

    def remove_species(self, species: str, prune_orphans: bool = True) -> None:
        """
        Remove a species from all reactions (adjusting stoichiometry).

        Reactions that become empty on both sides are removed. When
        ``prune_orphans`` is ``True``, species left with no incidence are dropped.

        :param species: Species label to remove.
        :type species: str
        :param prune_orphans: If ``True``, drop species nodes with no incident edges.
        :type prune_orphans: bool
        :raises KeyError: If the species is not present.
        """
        if species not in self.species:
            raise KeyError(species)
        to_remove_edges = set()
        for eid in list(self.species_to_in_edges.get(species, [])):
            e = self.edges[eid]
            e.products.pop(species, None)
            self.species_to_in_edges[species].discard(eid)
            if not e.reactants.data and not e.products.data:
                to_remove_edges.add(eid)
        for eid in list(self.species_to_out_edges.get(species, [])):
            e = self.edges[eid]
            e.reactants.pop(species, None)
            self.species_to_out_edges[species].discard(eid)
            if not e.reactants.data and not e.products.data:
                to_remove_edges.add(eid)
        for eid in to_remove_edges:
            self.remove_rxn(eid)
        if prune_orphans:
            if not self.species_to_in_edges.get(
                species
            ) and not self.species_to_out_edges.get(species):
                self.species.discard(species)
                self.species_to_in_edges.pop(species, None)
                self.species_to_out_edges.pop(species, None)

    # ------------------------------------------------------------------
    # Query / utility
    # ------------------------------------------------------------------
    def get_edge(self, edge_id: str) -> HyperEdge:
        """
        Get a hyperedge by id.

        :param edge_id: Identifier of the edge.
        :type edge_id: str
        :returns: The corresponding hyperedge.
        :rtype: HyperEdge
        :raises KeyError: If the edge does not exist.
        """
        return self.edges[edge_id]

    def species_list(self) -> List[str]:
        """
        List all species in sorted order.

        :returns: Sorted list of species labels.
        :rtype: List[str]
        """
        return sorted(self.species)

    def edge_list(self) -> List[HyperEdge]:
        """
        List all hyperedges.

        :returns: List of hyperedges (unspecified order).
        :rtype: List[HyperEdge]
        """
        return list(self.edges.values())

    def __len__(self) -> int:
        """
        Number of hyperedges.

        :returns: Edge count.
        :rtype: int
        """
        return len(self.edges)

    def __contains__(self, item: str) -> bool:
        """
        Membership test for species or edges.

        :param item: Species label or edge id.
        :type item: str
        :returns: ``True`` if present as species or edge id, else ``False``.
        :rtype: bool
        """
        return item in self.species or item in self.edges

    def __iter__(self) -> Iterator[HyperEdge]:
        """
        Iterate over hyperedges.

        :returns: Iterator over hyperedges.
        :rtype: Iterator[HyperEdge]
        """
        return iter(self.edge_list())

    def copy(self) -> CRNHyperGraph:
        """
        Deep copy of the hypergraph.

        :returns: A deep-copied graph.
        :rtype: CRNHyperGraph
        """
        return copy.deepcopy(self)

    def merge(self, other: Any, prefix_edges: bool = True) -> None:
        """
        Merge another hypergraph-like object into this one.

        Requires that ``other`` exposes ``edge_list()`` and its edges have
        fields ``id``, ``rule``, ``reactants``, ``products``.

        :param other: Other hypergraph-like object.
        :type other: Any
        :param prefix_edges: If ``True``, regenerate edge ids to avoid collisions.
        :type prefix_edges: bool
        :raises TypeError: If ``other`` lacks ``edge_list()``.
        """
        if not hasattr(other, "edge_list"):
            raise TypeError("other must expose edge_list() for merge")

        for e in other.edge_list():  # type: ignore[attr-defined]
            new_id = getattr(e, "id", None)
            rule = getattr(e, "rule", "r")
            if prefix_edges or new_id is None or new_id in self.edges:
                new_id = self._next_edge_id_for_rule(rule)
            self.add_rxn(
                (
                    e.reactants
                    if isinstance(e.reactants, RXNSide)
                    else RXNSide.from_any(e.reactants)
                ),
                (
                    e.products
                    if isinstance(e.products, RXNSide)
                    else RXNSide.from_any(e.products)
                ),
                rule=rule,
                edge_id=new_id,
            )

    # ------------------------------------------------------------------
    # Graph exports
    # ------------------------------------------------------------------
    def to_bipartite(
        self,
        *,
        species_prefix: str = "S:",
        reaction_prefix: str = "R:",
        bipartite_values: Tuple[int, int] = (0, 1),
        include_stoich: bool = True,
        include_role: bool = True,
        include_isolated_species: bool = True,
        integer_ids: bool = True,
        include_species_attr: bool = False,
        include_edge_id_attr: bool = False,
    ) -> nx.DiGraph:
        """
        Export as a bipartite NetworkX directed graph.

        Node-ID modes:

        * ``integer_ids=True``:
          species nodes are ``1..N``, reaction nodes are ``N+1..N+M``.
          Nodes carry ``label`` and ``kind``. Optional attributes may be added.
        * ``integer_ids=False``:
          species node ids are ``f"{species_prefix}{name}"``, reaction node ids
          are ``f"{reaction_prefix}{edge_id}"``.

        :param species_prefix: Prefix for species node ids when ``integer_ids=False``.
        :type species_prefix: str, keyword-only
        :param reaction_prefix: Prefix for reaction node ids when ``integer_ids=False``.
        :type reaction_prefix: str, keyword-only
        :param bipartite_values: Bipartite attribute values for (species, reaction).
        :type bipartite_values: Tuple[int, int], keyword-only
        :param include_stoich: If ``True``, add ``stoich`` edge attribute.
        :type include_stoich: bool, keyword-only
        :param include_role: If ``True``, add ``role`` edge attribute (``"reactant"``/``"product"``).
        :type include_role: bool, keyword-only
        :param include_isolated_species: If ``True``, include species with no incident edges.
        :type include_isolated_species: bool, keyword-only
        :param integer_ids: If ``True``, use consecutive integer node ids.
        :type integer_ids: bool, keyword-only
        :param include_species_attr: If ``True``, add ``species`` attribute to species nodes.
        :type include_species_attr: bool, keyword-only
        :param include_edge_id_attr: If ``True``, add ``edge_id`` attribute to reaction nodes.
        :type include_edge_id_attr: bool, keyword-only
        :returns: Bipartite directed graph with species and reaction nodes.
        :rtype: nx.DiGraph
        """
        G = nx.DiGraph()
        species_val, reaction_val = bipartite_values

        # which species to include
        if include_isolated_species:
            species_iter = sorted(self.species)
        else:
            species_iter = sorted(
                {
                    s
                    for s in self.species
                    if (
                        self.species_to_in_edges.get(s)
                        or self.species_to_out_edges.get(s)
                    )
                }
            )

        def make_sp_attrs(s: str) -> Dict[str, Any]:
            attrs: Dict[str, Any] = {
                "bipartite": species_val,
                "label": s,
                "kind": "species",
            }
            if include_species_attr:
                attrs["species"] = s
            return attrs

        def make_rxn_attrs(eid: str, rule: str) -> Dict[str, Any]:
            attrs: Dict[str, Any] = {
                "bipartite": reaction_val,
                "label": rule,
                "kind": "reaction",
            }
            if include_edge_id_attr:
                attrs["edge_id"] = eid
            return attrs

        def make_e_attrs(role: Optional[str], stoich: Optional[int]) -> Dict[str, Any]:
            attrs: Dict[str, Any] = {}
            if include_stoich and stoich is not None:
                attrs["stoich"] = int(stoich)
            if include_role and role is not None:
                attrs["role"] = role
            return attrs

        species_map: Dict[str, Any] = {}
        next_id = 1

        def add_sp_node(s: str) -> Any:
            nonlocal next_id
            if s in species_map:
                return species_map[s]
            if integer_ids:
                nid = next_id
                next_id += 1
            else:
                nid = f"{species_prefix}{s}"
            species_map[s] = nid
            if not G.has_node(nid):
                G.add_node(nid, **make_sp_attrs(s))
            return nid

        def add_rxn_node(eid: str, rule: str) -> Any:
            nonlocal next_id
            if integer_ids:
                nid = next_id
                next_id += 1
            else:
                nid = f"{reaction_prefix}{eid}"
            G.add_node(nid, **make_rxn_attrs(eid, rule))
            return nid

        # initial species nodes
        for s in species_iter:
            add_sp_node(s)

        # reactions + incidence arcs
        for eid, e in sorted(self.edges.items()):
            rnode = add_rxn_node(eid, e.rule)

            # reactant arcs: species -> reaction
            for s, c in e.reactants.items():
                u = add_sp_node(s)
                G.add_edge(u, rnode, **make_e_attrs("reactant", c))

            # product arcs: reaction -> species
            for s, c in e.products.items():
                v = add_sp_node(s)
                G.add_edge(rnode, v, **make_e_attrs("product", c))

        return G

    def to_species_graph(self) -> nx.DiGraph:
        """
        Collapse hyperedges to a directed species→species graph.

        For each hyperedge, create edges from each reactant to each product and
        aggregate attributes across multiple reactions.

        Edge attributes:
          * ``via``: set of edge ids contributing to the species→species arc
          * ``rules``: set of rule names contributing to the arc
          * ``min_stoich``: minimum ``min(reactant_count, product_count)`` observed

        :returns: Directed species-to-species graph with aggregated attributes.
        :rtype: nx.DiGraph
        """
        G = nx.DiGraph()
        for s in self.species:
            G.add_node(s, label=s, kind="species")
        for eid, e in self.edges.items():
            for r, rc in e.reactants.items():
                for p, pc in e.products.items():
                    if G.has_edge(r, p):
                        data = G[r][p]
                        data["via"].add(eid)
                        data["rules"].add(e.rule)
                        data["min_stoich"] = min(data["min_stoich"], min(rc, pc))
                    else:
                        G.add_edge(
                            r,
                            p,
                            via={eid},
                            rules={e.rule},
                            min_stoich=min(rc, pc),
                        )
        return G

    # ------------------------------------------------------------------
    # Incidence / stoichiometry
    # ------------------------------------------------------------------
    def incidence_matrix(
        self,
        *,
        sparse: bool = True,
    ) -> Union[
        Tuple[List[str], List[str], Dict[Tuple[str, str], int]],
        Tuple[List[str], List[str], np.ndarray],
    ]:
        """
        Construct the stoichiometric incidence matrix (sparse or dense).

        If ``sparse=True`` (default), returns a sparse dict:
        ``mapping[(species, edge_id)] = signed_count`` with reactants negative,
        products positive. If ``sparse=False``, returns a dense integer array of
        shape ``(n_species, n_edges)`` in the third position.

        :param sparse: If ``True``, return sparse mapping; else return dense matrix.
        :type sparse: bool, keyword-only
        :returns: Either
                  * ``(species_order, edge_order, mapping)`` (sparse), or
                  * ``(species_order, edge_order, matrix)`` (dense).
        :rtype: Union[Tuple[List[str], List[str], Dict[Tuple[str, str], int]],
                      Tuple[List[str], List[str], np.ndarray]]
        """
        species_order = self.species_list()
        edge_order = sorted(self.edges.keys())

        if sparse:
            mapping: Dict[Tuple[str, str], int] = {}
            for eid in edge_order:
                e = self.edges[eid]
                for s, c in e.reactants.items():
                    mapping[(s, eid)] = mapping.get((s, eid), 0) - int(c)
                for s, c in e.products.items():
                    mapping[(s, eid)] = mapping.get((s, eid), 0) + int(c)
            return species_order, edge_order, mapping
        else:
            mat = np.zeros((len(species_order), len(edge_order)), dtype=int)
            s_idx = {s: i for i, s in enumerate(species_order)}
            for j, eid in enumerate(edge_order):
                e = self.edges[eid]
                for s, c in e.reactants.items():
                    mat[s_idx[s], j] -= int(c)
                for s, c in e.products.items():
                    mat[s_idx[s], j] += int(c)
            return species_order, edge_order, mat

    def stoichiometric_matrix(
        self,
        *,
        sparse: bool = True,
    ) -> Union[
        Tuple[List[str], List[str], Dict[Tuple[str, str], int]],
        Tuple[List[str], List[str], np.ndarray],
    ]:
        """
        Construct the stoichiometric (incidence) matrix.

        This is an alias of :meth:`incidence_matrix` for convenience. See that
        method for details on construction and sign conventions.

        :param sparse: Whether to return a sparse mapping (``True``) or a dense
                    matrix (``False``). Passed through to
                    :meth:`incidence_matrix`.
        :type sparse: bool
        :returns: If ``sparse`` is ``True``, returns
                ``(species_order, edge_order, mapping)`` where
                ``mapping[(species, edge_id)] = signed_count``.
                If ``sparse`` is ``False``, returns
                ``(species_order, edge_order, matrix)``, where
                ``matrix[i][j]`` is the signed count for
                ``species_order[i]`` and ``edge_order[j]``.
        :rtype: Union[
            Tuple[List[str], List[str], Dict[Tuple[str, str], int]],
            Tuple[List[str], List[str], List[List[int]]]
        ]
        :see also: :meth:`incidence_matrix`
        """
        return self.incidence_matrix(sparse=sparse)

    # ------------------------------------------------------------------
    # Traversal / path finding
    # ------------------------------------------------------------------
    def neighbors(self, species: str) -> Set[str]:
        """
        One-step product neighbors from a species.

        Returns all products reachable via any reaction where the species
        appears as a reactant.

        :param species: Species label serving as the source.
        :type species: str
        :returns: Set of product species reachable in one reaction step.
        :rtype: Set[str]
        :raises KeyError: If the species is not present.
        """
        if species not in self.species:
            raise KeyError(species)
        out: Set[str] = set()
        for eid in self.species_to_out_edges.get(species, ()):
            e = self.edges[eid]
            out.update(e.products.keys())
        return out

    def paths(
        self,
        source: str,
        target: str,
        max_hops: int = 4,
        max_paths: Optional[int] = None,
    ) -> List[List[str]]:
        """
        Enumerate simple species→species paths up to a hop limit.

        :param source: Start species.
        :type source: str
        :param target: Target species.
        :type target: str
        :param max_hops: Maximum number of edges (species steps) allowed.
        :type max_hops: int
        :param max_paths: If provided, stop after returning this many paths.
        :type max_paths: Optional[int]
        :returns: List of paths, each a list of species labels.
        :rtype: List[List[str]]
        :raises KeyError: If ``source`` or ``target`` is not present.
        """
        if source not in self.species or target not in self.species:
            raise KeyError("source/target not in hypergraph")
        paths: List[List[str]] = []
        q = deque([[source]])
        while q:
            path = q.popleft()
            if len(path) - 1 > max_hops:
                continue
            last = path[-1]
            if last == target:
                paths.append(path)
                if max_paths is not None and len(paths) >= max_paths:
                    break
                continue
            for nbr in sorted(self.neighbors(last)):
                if nbr in path:
                    continue
                q.append(path + [nbr])
        return paths

    # ------------------------------------------------------------------
    # Pretty-print helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_side(side: Dict[str, int]) -> str:
        """
        Pretty-format a stoichiometric mapping.

        :param side: Mapping ``species -> count``.
        :type side: Dict[str, int]
        :returns: Human-readable text like ``"2A + B"`` or ``"∅"``.
        :rtype: str
        """
        if not side:
            return "∅"
        parts = []
        for s in sorted(side.keys()):
            c = side[s]
            parts.append(f"{c}{s}" if c != 1 else s)
        return " + ".join(parts)

    def species_summary(
        self, species: Optional[Iterable[str]] = None
    ) -> List[Tuple[str, List[str], List[str]]]:
        """
        Summarize species incidence (incoming/outgoing edge ids).

        :param species: Optional iterable to restrict the report to selected species.
        :type species: Optional[Iterable[str]]
        :returns: List of tuples ``(species, in_edge_ids, out_edge_ids)``.
        :rtype: List[Tuple[str, List[str], List[str]]]
        """
        species_iter = species if species is not None else sorted(self.species)
        out = []
        for s in species_iter:
            ins = sorted(self.species_to_in_edges.get(s, []))
            outs = sorted(self.species_to_out_edges.get(s, []))
            out.append((s, ins, outs))
        return out

    def print_species(
        self,
        species: Optional[Iterable[str]] = None,
        show_counts: bool = True,
    ) -> None:
        """
        Pretty-print species incidence summary.

        :param species: Optional iterable of species to print; defaults to all.
        :type species: Optional[Iterable[str]]
        :param show_counts: If ``True``, include counts alongside edge lists.
        :type show_counts: bool
        :returns: ``None``.
        :rtype: None
        """
        rows = self.species_summary(species=species)
        if not rows:
            print("No species.")
            return
        longest = max((len(s) for s, _, _ in rows), default=7)
        header = f"{'Species'.ljust(longest)}   In-edges          Out-edges"
        print(header)
        print("-" * len(header))
        for s, ins, outs in rows:
            ins_str = ", ".join(ins) if ins else "—"
            outs_str = ", ".join(outs) if outs else "—"
            if show_counts:
                ic = len(ins)
                oc = len(outs)
                print(
                    f"{s.ljust(longest)}         "
                    f"[{ic:2d}] {ins_str:<10}   "
                    f"[{oc:2d}] {outs_str}"
                )
            else:
                print(f"{s.ljust(longest)}          {ins_str:<10}        {outs_str}")

    def edge_summary(
        self, edge_ids: Optional[Iterable[str]] = None
    ) -> List[Tuple[str, str, Dict[str, int], Dict[str, int]]]:
        """
        Summarize edges with rule and stoichiometry.

        :param edge_ids: Optional iterable of edge ids to restrict the report.
        :type edge_ids: Optional[Iterable[str]]
        :returns: List of tuples ``(edge_id, rule, reactants_dict, products_dict)``.
        :rtype: List[Tuple[str, str, Dict[str, int], Dict[str, int]]]
        """
        ids = list(edge_ids) if edge_ids is not None else sorted(self.edges.keys())
        out = []
        for eid in ids:
            e = self.edges[eid]
            out.append((eid, e.rule, e.reactants.to_dict(), e.products.to_dict()))
        return out

    def print_edges(
        self,
        edge_ids: Optional[Iterable[str]] = None,
        show_stoich: bool = True,
    ) -> None:
        """
        Pretty-print edge list with reactant/product sides.

        :param edge_ids: Optional iterable of edge ids to print; defaults to all.
        :type edge_ids: Optional[Iterable[str]]
        :param show_stoich: If ``True``, include stoichiometric counts; else names only.
        :type show_stoich: bool
        :returns: ``None``.
        :rtype: None
        """
        rows = self.edge_summary(edge_ids)
        if not rows:
            print("No edges.")
            return
        print("Edge id   Rule   Reactants >> Products")
        print("-" * 60)
        for eid, rule, reactants, products in rows:
            if show_stoich:
                left = self._format_side(reactants)
                right = self._format_side(products)
            else:
                left = ", ".join(sorted(reactants.keys()) or ["∅"])
                right = ", ".join(sorted(products.keys()) or ["∅"])
            print(f"{eid:<8}  {rule:<6} {left}  >>  {right}")

    def print_attrs(
        self,
        *,
        species_prefix: str = "S:",
        reaction_prefix: str = "R:",
        include_nodes: bool = True,
        include_edges: bool = True,
        max_rows: Optional[int] = 50,
        integer_ids: bool = True,
        include_species_attr: bool = False,
        include_edge_id_attr: bool = False,
    ) -> None:
        """
        Print node/edge attributes of the bipartite NetworkX graph.

        :param species_prefix: Prefix for species node ids when ``integer_ids=False``.
        :type species_prefix: str, keyword-only
        :param reaction_prefix: Prefix for reaction node ids when ``integer_ids=False``.
        :type reaction_prefix: str, keyword-only
        :param include_nodes: If ``True``, print nodes with attributes.
        :type include_nodes: bool, keyword-only
        :param include_edges: If ``True``, print edges with attributes.
        :type include_edges: bool, keyword-only
        :param max_rows: Maximum rows to print for nodes/edges (``None`` for unlimited).
        :type max_rows: Optional[int], keyword-only
        :param integer_ids: If ``True``, use integer ids in the temporary graph.
        :type integer_ids: bool, keyword-only
        :param include_species_attr: If ``True``, add ``species`` attribute to species nodes.
        :type include_species_attr: bool, keyword-only
        :param include_edge_id_attr: If ``True``, add ``edge_id`` attribute to reaction nodes.
        :type include_edge_id_attr: bool, keyword-only
        :returns: ``None``.
        :rtype: None
        """
        G = self.to_bipartite(
            species_prefix=species_prefix,
            reaction_prefix=reaction_prefix,
            integer_ids=integer_ids,
            include_species_attr=include_species_attr,
            include_edge_id_attr=include_edge_id_attr,
        )
        if include_nodes:
            print("Nodes:")
            print("-" * 40)
            for i, (n, attrs) in enumerate(G.nodes(data=True)):
                if max_rows is not None and i >= max_rows:
                    print(f"... ({len(G.nodes()) - max_rows} more)")
                    break
                print(f"{n}: {attrs}")
            print()
        if include_edges:
            print("Edges:")
            print("-" * 40)
            for i, (u, v, attrs) in enumerate(G.edges(data=True)):
                if max_rows is not None and i >= max_rows:
                    print(f"... ({len(G.edges()) - max_rows} more)")
                    break
                print(f"{u} >> {v}: {attrs}")
            print()

    def __repr__(self) -> str:
        """Human-readable multi-line summary of the hypergraph."""
        lines = ["CRNHyperGraph:"]
        for e in self.edge_list():
            lines.append("  " + repr(e))
        lines.append("Species: " + ", ".join(sorted(self.species)))
        return "\n".join(lines)
