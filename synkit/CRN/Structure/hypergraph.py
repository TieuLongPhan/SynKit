from __future__ import annotations

from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
from collections import defaultdict, deque
import copy
import re
import numpy as np

from .rxn import RXNSide
from .hyperedge import HyperEdge


class CRNHyperGraph:
    """
    Directed hypergraph representation of a chemical reaction network.

    Responsibilities:
      - add/remove reactions
      - species/edge bookkeeping and indices
      - incidence / stoichiometric matrix construction
      - simple traversal (neighbors, paths)
      - merging & copying

    The class focuses on topological / stoichiometric representation; higher-level
    structural analyses (deficiency, injectivity, siphons, etc.) should be implemented
    in separate ``props`` modules that import this hypergraph for data.

    :param: (constructed empty; use methods to populate)
    :type: CRNHyperGraph

    Examples
    --------
    .. code-block:: python

        from synkit.CRN.Hypergraph.hypergraph import CRNHyperGraph

        H = CRNHyperGraph()
        # parse a few human-readable reactions
        H.parse_rxns(["A + B >> C", "C >> A"])
        # get species and edges
        species = H.species_list()
        edges = H.edge_list()
        # incidence mapping (sparse)
        species_order, edge_order, mapping = H.incidence_matrix(sparse=True)
        # dense stoichiometric matrix
        _, _, mat = H.incidence_matrix(sparse=False)
    """

    def __init__(self) -> None:
        self.species: Set[str] = set()
        self.edges: Dict[str, HyperEdge] = {}
        self._rule_counters: Dict[str, int] = defaultdict(int)
        self.species_to_in_edges: Dict[str, Set[str]] = defaultdict(set)
        self.species_to_out_edges: Dict[str, Set[str]] = defaultdict(set)
        self.species_to_mol: Dict[str, Any] = {}

    # -----------------------
    # id generation
    # -----------------------
    def _next_edge_id_for_rule(self, rule: str) -> str:
        cnt = self._rule_counters.get(rule, 0) + 1
        self._rule_counters[rule] = cnt
        return f"{rule}_{cnt}"

    # -----------------------
    # add / remove / parse
    # -----------------------
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
        Add a reaction edge to the hypergraph.

        :param reactant_side: mapping/iterable or RXNSide for reactants
        :type reactant_side: Union[RXNSide, Mapping[str, int], Iterable[str], Iterable[Tuple[str, int]]]
        :param product_side: mapping/iterable or RXNSide for products
        :type product_side: Union[RXNSide, Mapping[str, int], Iterable[str], Iterable[Tuple[str, int]]]
        :param rule: optional rule/label; defaults to 'r'
        :type rule: Optional[str]
        :param edge_id: optional explicit id; if omitted an id is generated
        :type edge_id: Optional[str]
        :returns: created HyperEdge
        :rtype: HyperEdge
        :raises ValueError: if both sides are empty
        :raises KeyError: if edge_id already exists
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
        rule = rule or "r"
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
            _ = self.species_to_in_edges[s]
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
        Parse a reaction string like "A+B>>C" and add the reaction.

        Optionally parse a trailing suffix " | rule=R1" to set the rule when
        parse_rule_from_suffix=True and rule is None.

        :param reaction: reaction string
        :type reaction: str
        :param rule: explicit rule label or None
        :type rule: Optional[str]
        :param parse_rule_from_suffix: whether to parse suffix for rule
        :type parse_rule_from_suffix: bool
        :returns: created HyperEdge
        :rtype: HyperEdge
        :raises ValueError: if ">>" separator missing
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

        :param reactions: Input reactions. Accepted forms:
                        - Iterable[str] (if `rules` is provided, it is zipped)
                        - Iterable[Tuple[str, Optional[str]]]
                        - Mapping[str, Optional[str]]
        :type reactions: Union[Iterable[str], Iterable[Tuple[str, Optional[str]]], Mapping[str, Optional[str]]]
        :param default_rule: Rule used when neither an explicit rule nor a suffix is provided.
        :type default_rule: str
        :param parse_rule_from_suffix: If True, parse "| rule=..." suffixes from reaction strings.
        :type parse_rule_from_suffix: bool
        :param rules: Parallel rules for Iterable[str] input.
        :type rules: Optional[Sequence[Optional[str]]]
        :param prefer_suffix: If True, suffix rule takes precedence over an explicit per-line rule.
        :type prefer_suffix: bool
        :returns: A new graph instance (self) populated with the given reactions.
        :rtype: CRNHyperGraph
        :raises ValueError: If a reaction string lacks the ">>" separator, or if `rules` length mismatches.
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
            # Case: explicit per-line rule provided
            if explicit_rule is not None:
                if prefer_suffix and parse_rule_from_suffix:
                    # If suffix present, allow it to override by asking add_rxn_from_str
                    # to parse suffix (pass rule=None). Otherwise use explicit rule.
                    if re.search(r"\|\s*rule\s*=\s*[^\s]+", line):
                        self.add_rxn_from_str(
                            line, rule=None, parse_rule_from_suffix=True
                        )
                    else:
                        self.add_rxn_from_str(
                            line, rule=explicit_rule, parse_rule_from_suffix=False
                        )
                else:
                    # Explicit wins: provide it and avoid suffix parsing (prevents accidental override)
                    self.add_rxn_from_str(
                        line, rule=explicit_rule, parse_rule_from_suffix=False
                    )
            else:
                # No explicit per-line rule provided:
                # If parse_rule_from_suffix is True, allow the suffix to set the rule (pass rule=None).
                # If no suffix present, add_rxn_from_str will fall back to default_rule.
                if parse_rule_from_suffix:
                    self.add_rxn_from_str(line, rule=None, parse_rule_from_suffix=True)
                else:
                    # Don't parse suffixes; use default_rule
                    self.add_rxn_from_str(
                        line, rule=default_rule, parse_rule_from_suffix=False
                    )

        return self

    def remove_rxn(self, edge_id: str) -> None:
        """
        Remove edge by id and update indices.

        :param edge_id: id to remove
        :type edge_id: str
        :raises KeyError: if not found
        """
        if edge_id not in self.edges:
            raise KeyError(edge_id)
        e = self.edges.pop(edge_id)
        for s in list(e.reactants.keys()):
            self.species_to_out_edges[s].discard(edge_id)
            if not self.species_to_in_edges.get(
                s
            ) and not self.species_to_out_edges.get(s):
                self.species.discard(s)
                self.species_to_in_edges.pop(s, None)
                self.species_to_out_edges.pop(s, None)
                self.species_to_mol.pop(s, None)
        for s in list(e.products.keys()):
            self.species_to_in_edges[s].discard(edge_id)
            if not self.species_to_in_edges.get(
                s
            ) and not self.species_to_out_edges.get(s):
                self.species.discard(s)
                self.species_to_in_edges.pop(s, None)
                self.species_to_out_edges.pop(s, None)
                self.species_to_mol.pop(s, None)

    def remove_species(self, species: str, *, prune_orphans: bool = True) -> None:
        """
        Remove species from all reactions (adjust stoichiometry). Remove empty reactions.

        :param species: label of species
        :type species: str
        :param prune_orphans: if True, drop species entries with no incidence
        :type prune_orphans: bool
        :raises KeyError: if species absent
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
                self.species_to_mol.pop(species, None)

    # -----------------------
    # query / utilities
    # -----------------------
    def get_edge(self, edge_id: str) -> HyperEdge:
        return self.edges[edge_id]

    def species_list(self) -> List[str]:
        return sorted(self.species)

    def edge_list(self) -> List[HyperEdge]:
        return list(self.edges.values())

    def __len__(self) -> int:
        return len(self.edges)

    def __contains__(self, item: str) -> bool:
        return item in self.species or item in self.edges

    def __iter__(self) -> Iterator[HyperEdge]:
        return iter(self.edge_list())

    def copy(self) -> "CRNHyperGraph":
        return copy.deepcopy(self)

    def merge(self, other: Any, prefix_edges: bool = True) -> None:
        """
        Merge another hypergraph-like object into this one.

        :param other: must provide edge_list() returning edges with attributes id, rule, reactants, products
        :type other: Any
        :param prefix_edges: if True regenerate edge ids to avoid collisions
        :type prefix_edges: bool
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

    # -----------------------
    # traversal / path finding
    # -----------------------
    def neighbors(self, species: str) -> Set[str]:
        """
        One-step product neighbors from a species.

        :param species: source species label
        :type species: str
        :returns: set of product species reachable in one reaction
        :rtype: Set[str]
        :raises KeyError: if species absent
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
        Enumerate simple species->species paths up to hop limit (BFS).

        :param source: start species
        :type source: str
        :param target: target species
        :type target: str
        :param max_hops: maximum number of edges
        :type max_hops: int
        :param max_paths: optionally stop after this many paths
        :type max_paths: Optional[int]
        :returns: list of paths (each path is a list of species labels)
        :rtype: List[List[str]]
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

    # -----------------------
    # incidence / stoichiometry
    # -----------------------
    def incidence_matrix(
        self,
        *,
        sparse: bool = True,
    ) -> Union[
        Tuple[List[str], List[str], Dict[Tuple[str, str], int]],
        Tuple[List[str], List[str], np.ndarray],
    ]:
        """
        Construct incidence/stoichiometric matrix.

        If sparse=True (default), returns (species_order, edge_order, mapping)
        where mapping[(species, edge_id)] = signed_count (reactants negative, products positive).

        If sparse=False returns (species_order, edge_order, matrix) as dense np.ndarray.

        :param sparse: whether to return sparse mapping or dense matrix
        :type sparse: bool
        :returns: tuple (species_order, edge_order, mapping/matrix)
        :rtype: Union[Tuple[List[str], List[str], Dict[Tuple[str, str], int]], Tuple[List[str], List[str], np.ndarray]]
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

    def stoichiometric_matrix(self, *, sparse: bool = True):
        """Alias of incidence_matrix for clarity."""
        return self.incidence_matrix(sparse=sparse)

    # -----------------------
    # species–molecule mapping
    # -----------------------
    def set_mol_map(
        self,
        mapping: Mapping[str, Any],
        *,
        strict: bool = True,
        clear_existing: bool = False,
    ) -> None:
        """
        Set or update the mapping from species labels to molecule identifiers.

        :param mapping: mapping from species label to molecule
        :type mapping: Mapping[str, Any]
        :param strict: if True, raise if mapping contains species not in the hypergraph
        :type strict: bool
        :param clear_existing: if True, clear existing entries before applying mapping
        :type clear_existing: bool
        :raises KeyError: if strict is True and mapping contains unknown species
        """
        unknown = set(mapping) - self.species
        if strict and unknown:
            raise KeyError(
                "set_mol_map: mapping contains species not in hypergraph: "
                f"{sorted(unknown)}"
            )

        if clear_existing:
            self.species_to_mol.clear()

        for s, mol in mapping.items():
            if s in self.species:
                self.species_to_mol[s] = mol

    def assign_mol(self, species: str, mol: Any) -> None:
        """
        Assign or update the molecule identifier for a single species.

        :param species: species label
        :type species: str
        :param mol_id: molecule identifier (e.g. int, str, or other hashable type)
        :type mol_id: Any
        :raises KeyError: if species is not present in the hypergraph
        """
        if species not in self.species:
            raise KeyError(f"Unknown species {species!r}")
        self.species_to_mol[species] = mol

    def get_mol(self, species: str) -> Any:
        """
        Get the molecule identifier for a species.

        :param species: species label
        :type species: str
        :returns: molecule identifier associated with the species
        :rtype: Any
        :raises KeyError: if species is not present or has no molecule assigned
        """
        if species not in self.species:
            raise KeyError(f"Unknown species {species!r}")
        if species not in self.species_to_mol:
            raise KeyError(f"No molecule assigned for species {species!r}")
        return self.species_to_mol[species]

    def __repr__(self) -> str:
        lines = ["CRNHyperGraph:"]

        def _edge_key(e) -> tuple:
            """
            Sort edges by edge_id like 'r_1', 'r_2', 'r_10'.
            Fallback to repr(e) if we cannot find a suitable id.
            """
            eid = getattr(e, "edge_id", None) or getattr(e, "id", None)
            if isinstance(eid, str):
                # split into non-digit prefix + numeric suffix
                prefix = "".join(ch for ch in eid if not ch.isdigit())
                digits = "".join(ch for ch in eid if ch.isdigit())
                try:
                    n = int(digits) if digits else 0
                except ValueError:
                    n = 0
                return (prefix, n)
            # fallback: group all such edges together, sorted by repr
            return ("", repr(e))

        for e in sorted(self.edge_list(), key=_edge_key):
            lines.append("  " + repr(e))

        lines.append("Species: " + ", ".join(sorted(self.species)))

        # Optional: show species→molecule mapping when available
        if getattr(self, "species_to_mol", None):
            pairs = [
                f"{s} → {self.species_to_mol[s]}" for s in sorted(self.species_to_mol)
            ]
            lines.append("Species → mol: " + ", ".join(pairs))

        return "\n".join(lines)
