from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple

import networkx as nx
import numpy as np

from .reaction import Reaction
from .rule import Rule
from .species import Species
from ._graph_io import (
    _add_reaction_edges_to_graph,
    _build_reaction_from_graph_node,
    _build_species_table_from_graph,
    _collect_bipartite_nodes,
    _make_internal_id_maps,
    _reaction_node_attrs,
    _resolve_reaction_node_id,
    _resolve_species_node_id,
    _species_node_attrs,
    _validate_bipartite_node_sets,
)
from ._matrices import _dense_from_entries, _sparse_from_entries
from ._parse import (
    ID_STYLES,
    _build_reactions_from_parsed_strings,
    _build_rules_table_from_reaction_strings,
    _build_species_table_from_labels,
    _default_parse_side_text,
    _mint_reaction_string_ids,
    _parse_reaction_string_entry,
    _species_order_from_parsed_reactions,
)

__all__ = ["ID_STYLES", "SynCRN"]


@dataclass
class SynCRN:
    """Canonical reaction-system object for SynKit-CRN.

    Official representations exposed by this object
    ------------------------------------------------
    - ``SynCRN``: master reaction-system object
    - ``SynCRN.to_digraph()``: species--reaction bipartite graph view
    - ``SynCRN.to_stoichiometric_matrices()``: matrix view for stoichiometric analysis
    - ``SynCRN.to_petrinet()``: pre/post incidence view for pathways
    - ``SynCRN.to_equations()``: human-readable reaction list

    Design notes
    ------------
    Input digraph nodes with ``kind="rule"`` are preserved exactly on round-trip
    via ``Reaction.source_kind`` and ``Reaction.source_attrs``. They are also
    normalized as concrete reaction instances for downstream computation.

    :param species:
        Mapping from internal species ids to Species records.
    :type species: Dict[str, Species]

    :param reactions:
        Mapping from internal reaction ids to Reaction records.
    :type reactions: Dict[str, Reaction]

    :param rules:
        Mapping from internal rule ids to Rule records.
    :type rules: Dict[str, Rule]

    :param graph_attrs:
        Original graph-level attributes from the input digraph.
    :type graph_attrs: Dict[str, Any]

    :param metadata:
        Additional canonical metadata for the SynCRN object.
    :type metadata: Dict[str, Any]

    .. rubric:: Example

    .. code-block:: python

        syn = SynCRN.from_reaction_strings(["A>>B", "B>>C"])
        print(syn.n_species)
        print(syn.to_equations())
    """

    species: Dict[str, Species] = field(default_factory=dict)
    reactions: Dict[str, Reaction] = field(default_factory=dict)
    rules: Dict[str, Rule] = field(default_factory=dict)
    graph_attrs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_digraph(
        cls,
        crn: nx.DiGraph,
        *,
        species_kind: str = "species",
        reaction_kinds: Tuple[str, ...] = ("reaction", "rule"),
        species_prefix: str = "s_",
        reaction_prefix: str = "r_",
        rule_prefix: str = "rule_",
        strict: bool = True,
        id_style: str = "prefixed",
    ) -> "SynCRN":
        """Build a canonical SynCRN object from a species--reaction bipartite digraph.

        The current SynKit-CRN graph frequently stores concrete reaction-instance
        nodes with ``kind="rule"``. This constructor preserves that original kind
        in ``Reaction.source_kind`` while also converting such nodes into concrete
        reaction instances for downstream computation.

        Internal ids follow the same policy as
        :meth:`from_reaction_strings`: ``id_style="prefixed"`` mints ``s_1`` /
        ``r_1`` / ``rule_1``, while ``id_style="numeric"`` mints ``1..n`` for
        species and ``n+1..n+m`` for reactions.

        :param crn:
            Directed bipartite graph with species and reaction-like nodes.
        :type crn: nx.DiGraph

        :param species_kind:
            Node-kind value used for species nodes.
        :type species_kind: str

        :param reaction_kinds:
            Node-kind values used for reaction-instance nodes.
        :type reaction_kinds: Tuple[str, ...]

        :param species_prefix:
            Prefix for generated internal species ids.
        :type species_prefix: str

        :param reaction_prefix:
            Prefix for generated internal reaction ids.
        :type reaction_prefix: str

        :param rule_prefix:
            Prefix for generated internal rule ids.
        :type rule_prefix: str

        :param strict:
            Whether malformed graph structure should raise an error.
        :type strict: bool

        :param id_style:
            Internal-id policy, either ``"prefixed"`` (default) or ``"numeric"``.
        :type id_style: str

        :return:
            Canonical SynCRN object.
        :rtype: SynCRN

        :raises TypeError:
            If ``crn`` is not an ``nx.DiGraph``.

        :raises ValueError:
            If ``id_style`` is not a recognised policy.

        .. rubric:: Example

        .. code-block:: python

            syn = SynCRN.from_digraph(crn)
            print(syn.to_equations())
        """
        if not isinstance(crn, nx.DiGraph):
            raise TypeError(f"crn must be nx.DiGraph, got {type(crn).__name__}")

        species_nodes, reaction_nodes = _collect_bipartite_nodes(
            crn,
            species_kind=species_kind,
            reaction_kinds=reaction_kinds,
        )
        _validate_bipartite_node_sets(
            species_nodes=species_nodes,
            reaction_nodes=reaction_nodes,
            strict=strict,
        )

        species_node_to_id, reaction_node_to_id = _make_internal_id_maps(
            species_nodes=species_nodes,
            reaction_nodes=reaction_nodes,
            species_prefix=species_prefix,
            reaction_prefix=reaction_prefix,
            id_style=id_style,
        )

        species = _build_species_table_from_graph(
            crn,
            species_nodes=species_nodes,
            species_node_to_id=species_node_to_id,
        )

        rules: Dict[str, Rule] = {}
        rule_key_to_id: Dict[Tuple[Optional[int], Optional[str]], str] = {}
        reactions: Dict[str, Reaction] = {}

        for rnode in reaction_nodes:
            rid = reaction_node_to_id[rnode]
            reactions[rid] = _build_reaction_from_graph_node(
                crn,
                rnode=rnode,
                rid=rid,
                species_node_to_id=species_node_to_id,
                rules=rules,
                rule_key_to_id=rule_key_to_id,
                rule_prefix=rule_prefix,
                strict=strict,
            )

        return cls(
            species=species,
            reactions=reactions,
            rules=rules,
            graph_attrs=dict(crn.graph),
            metadata={
                "source_graph_type": type(crn).__name__,
                "id_style": id_style,
            },
        )

    @property
    def species_ids(self) -> List[str]:
        """Return the internal species order.

        :return:
            Ordered list of species ids.
        :rtype: List[str]

        .. rubric:: Example

        .. code-block:: python

            print(syn.species_ids)
        """
        return list(self.species.keys())

    @property
    def reaction_ids(self) -> List[str]:
        """Return the internal reaction order.

        :return:
            Ordered list of reaction ids.
        :rtype: List[str]

        .. rubric:: Example

        .. code-block:: python

            print(syn.reaction_ids)
        """
        return list(self.reactions.keys())

    @property
    def rule_ids(self) -> List[str]:
        """Return the internal rule order.

        :return:
            Ordered list of rule ids.
        :rtype: List[str]

        .. rubric:: Example

        .. code-block:: python

            print(syn.rule_ids)
        """
        return list(self.rules.keys())

    @property
    def n_species(self) -> int:
        """Return the number of species.

        :return:
            Number of species.
        :rtype: int
        """
        return len(self.species)

    @property
    def n_reactions(self) -> int:
        """Return the number of reactions.

        :return:
            Number of reactions.
        :rtype: int
        """
        return len(self.reactions)

    @property
    def n_rules(self) -> int:
        """Return the number of unique abstract rules.

        :return:
            Number of rules.
        :rtype: int
        """
        return len(self.rules)

    def __repr__(self) -> str:
        """Return a compact developer-facing representation.

        :return:
            Summary representation string.
        :rtype: str
        """
        return (
            f"SynCRN(n_species={self.n_species}, "
            f"n_reactions={self.n_reactions}, n_rules={self.n_rules})"
        )

    def __str__(self) -> str:
        """Return a human-readable text summary.

        :return:
            Multiline description string.
        :rtype: str
        """
        return self.describe(include_species=True, species="label")

    def _species_token(self, species_id: str, mode: str = "label") -> str:
        """Resolve how a species should be displayed.

        Supported modes are ``"id"``, ``"label"``, ``"smiles"``, and ``"source"``.

        :param species_id:
            Internal species id.
        :type species_id: str

        :param mode:
            Species display mode.
        :type mode: str

        :return:
            Display token for the species.
        :rtype: str

        :raises ValueError:
            If the display mode is unsupported.

        .. rubric:: Example

        .. code-block:: python

            syn._species_token("1", mode="label")
        """
        sp = self.species[species_id]
        if mode == "id":
            return sp.id
        if mode == "label":
            return sp.label
        if mode == "smiles":
            return sp.smiles or sp.label
        if mode == "source":
            return str(sp.source_node_id)
        raise ValueError("species mode must be one of: id, label, smiles, source")

    def format_reaction(
        self,
        reaction_id: str,
        *,
        species: str = "label",
        include_id: bool = True,
        include_rule: bool = False,
        include_step: bool = False,
        arrow: str = ">>",
    ) -> str:
        """Format one reaction as text.

        :param reaction_id:
            Internal reaction id.
        :type reaction_id: str

        :param species:
            Species display mode.
        :type species: str

        :param include_id:
            Whether to include the internal reaction id.
        :type include_id: bool

        :param include_rule:
            Whether to include rule provenance.
        :type include_rule: bool

        :param include_step:
            Whether to include step provenance.
        :type include_step: bool

        :param arrow:
            Arrow string between lhs and rhs.
        :type arrow: str

        :return:
            Human-readable reaction string.
        :rtype: str

        .. rubric:: Example

        .. code-block:: python

            syn.format_reaction("r_1", species="label", include_rule=True)
        """
        rxn = self.reactions[reaction_id]
        return rxn.format(
            lambda sid: self._species_token(sid, species),
            include_id=include_id,
            include_rule=include_rule,
            include_step=include_step,
            arrow=arrow,
        )

    def to_equations(
        self,
        *,
        species: str = "label",
        include_id: bool = True,
        include_rule: bool = False,
        include_step: bool = False,
        arrow: str = ">>",
    ) -> List[str]:
        """Return the network as a list of formatted reaction equations.

        :param species:
            Species display mode.
        :type species: str

        :param include_id:
            Whether to include internal reaction ids.
        :type include_id: bool

        :param include_rule:
            Whether to include rule provenance.
        :type include_rule: bool

        :param include_step:
            Whether to include step provenance.
        :type include_step: bool

        :param arrow:
            Arrow string between lhs and rhs.
        :type arrow: str

        :return:
            List of formatted reaction equations.
        :rtype: List[str]

        .. rubric:: Example

        .. code-block:: python

            eqs = syn.to_equations(species="smiles", include_rule=True)
            print("\\n".join(eqs))
        """
        return [
            self.format_reaction(
                rid,
                species=species,
                include_id=include_id,
                include_rule=include_rule,
                include_step=include_step,
                arrow=arrow,
            )
            for rid in self.reaction_ids
        ]

    def describe(
        self,
        *,
        include_species: bool = False,
        species: str = "label",
    ) -> str:
        """Return a human-readable multiline description of the network.

        :param include_species:
            Whether to append a final line listing species names.
        :type include_species: bool

        :param species:
            Species display mode used in the text summary.
        :type species: str

        :return:
            Multiline text description.
        :rtype: str

        .. rubric:: Example

        .. code-block:: python

            print(syn.describe(include_species=True, species="label"))
        """
        lines = [f"SynCRN: {self.n_species} species, {self.n_reactions} reactions"]

        for rid in self.reaction_ids:
            rxn = self.reactions[rid]
            lhs = rxn.format_side(
                rxn.lhs, lambda sid: self._species_token(sid, species)
            )
            rhs = rxn.format_side(
                rxn.rhs, lambda sid: self._species_token(sid, species)
            )

            line = f"  {rxn.id}: {lhs} >> {rhs}"
            if rxn.rule_index is not None:
                line += f" rule {rxn.rule_index}"

            lines.append(line)

        if include_species:
            names = [self._species_token(sid, species) for sid in self.species_ids]
            lines.append("Species: " + ", ".join(names))

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Return a nested JSON-like dictionary representation.

        :return:
            Full SynCRN object as a dictionary.
        :rtype: Dict[str, Any]

        .. rubric:: Example

        .. code-block:: python

            data = syn.to_dict()
            print(data["species"].keys())
        """
        return {
            "graph_attrs": dict(self.graph_attrs),
            "metadata": dict(self.metadata),
            "species": {sid: sp.to_dict() for sid, sp in self.species.items()},
            "rules": {rid: rule.to_dict() for rid, rule in self.rules.items()},
            "reactions": {rid: rxn.to_dict() for rid, rxn in self.reactions.items()},
        }

    def to_stoichiometric_matrices(
        self,
        *,
        sparse: bool = False,
        dtype: Any = None,
    ) -> Dict[str, Any]:
        """Construct stoichiometric matrices in canonical species and reaction order.

        The returned dictionary contains:

        - ``species_order``
        - ``reaction_order``
        - ``S_minus``: reactant-incidence matrix
        - ``S_plus``: product-incidence matrix
        - ``S``: net stoichiometric matrix

        Matrices are ``numpy.ndarray`` of shape ``(n_species, n_reactions)``, so
        they can be handed straight to :mod:`numpy.linalg` and to
        :mod:`synkit.CRN.Props`. Reaction networks are sparse — a species takes
        part in a handful of reactions regardless of network size — so pass
        ``sparse=True`` for large networks to get ``scipy.sparse.csr_array``
        matrices built from the nonzeros only, without ever allocating
        ``n_species x n_reactions`` cells.

        :param sparse:
            Whether to return ``scipy.sparse.csr_array`` instead of dense
            ``numpy.ndarray`` matrices.
        :type sparse: bool

        :param dtype:
            Element dtype. Defaults to ``numpy.int64`` when every coefficient is
            integral and ``numpy.float64`` otherwise.
        :type dtype: Any

        :return:
            Stoichiometric matrix view of the network.
        :rtype: Dict[str, Any]

        .. rubric:: Example

        .. code-block:: python

            mats = syn.to_stoichiometric_matrices()
            print(mats["species_order"])
            print(mats["reaction_order"])
            print(mats["S"])

            big = syn.to_stoichiometric_matrices(sparse=True)
            print(big["S"].nnz)
        """
        species_order = self.species_ids
        reaction_order = self.reaction_ids

        n = len(species_order)
        m = len(reaction_order)

        sidx = {sid: i for i, sid in enumerate(species_order)}
        ridx = {rid: j for j, rid in enumerate(reaction_order)}

        minus_entries: List[Tuple[int, int, Any]] = []
        plus_entries: List[Tuple[int, int, Any]] = []

        for rid, rxn in self.reactions.items():
            j = ridx[rid]
            for sid, coeff in rxn.lhs.items():
                minus_entries.append((sidx[sid], j, coeff))
            for sid, coeff in rxn.rhs.items():
                plus_entries.append((sidx[sid], j, coeff))

        if dtype is None:
            values = [v for _, _, v in minus_entries] + [v for _, _, v in plus_entries]
            integral = all(
                isinstance(v, (int, np.integer)) or float(v).is_integer()
                for v in values
            )
            dtype = np.int64 if integral else np.float64

        if sparse:
            s_minus = _sparse_from_entries(minus_entries, shape=(n, m), dtype=dtype)
            s_plus = _sparse_from_entries(plus_entries, shape=(n, m), dtype=dtype)
        else:
            s_minus = _dense_from_entries(minus_entries, shape=(n, m), dtype=dtype)
            s_plus = _dense_from_entries(plus_entries, shape=(n, m), dtype=dtype)

        return {
            "species_order": species_order,
            "reaction_order": reaction_order,
            "S_minus": s_minus,
            "S_plus": s_plus,
            "S": s_plus - s_minus,
        }

    def to_petrinet(self) -> Dict[str, Any]:
        """Return a Petri-net style pre/post incidence view.

        The returned dictionary contains:

        - ``places``: species ids
        - ``transitions``: reaction ids
        - ``pre``: input incidence map
        - ``post``: output incidence map

        :return:
            Petri-net incidence representation.
        :rtype: Dict[str, Any]

        .. rubric:: Example

        .. code-block:: python

            pn = syn.to_petrinet()
            print(pn["pre"])
            print(pn["post"])
        """
        pre: Dict[str, Dict[str, int]] = {sid: {} for sid in self.species_ids}
        post: Dict[str, Dict[str, int]] = {sid: {} for sid in self.species_ids}

        for rid, rxn in self.reactions.items():
            for sid, coeff in rxn.lhs.items():
                pre[sid][rid] = coeff
            for sid, coeff in rxn.rhs.items():
                post[sid][rid] = coeff

        return {
            "places": list(self.species_ids),
            "transitions": list(self.reaction_ids),
            "pre": pre,
            "post": post,
        }

    def to_digraph(
        self,
        *,
        node_ids: str = "source",
        reaction_kind: Optional[str] = None,
        include_internal_ids: bool = True,
    ) -> nx.DiGraph:
        """Reconstruct a species--reaction bipartite digraph.

        By default, this method preserves original node ids and original
        reaction-node kinds. This means that input reaction nodes with
        ``kind="rule"`` will still appear as ``kind="rule"`` after round-trip.

        :param node_ids:
            ``"source"`` preserves original node ids, while ``"internal"``
            uses canonical ids such as ``s_1`` and ``r_1``.
        :type node_ids: str

        :param reaction_kind:
            Optional override for reconstructed reaction-node kind.
        :type reaction_kind: Optional[str]

        :param include_internal_ids:
            Whether to attach ``syncrn_id`` and ``source_node_id`` as node attributes.
        :type include_internal_ids: bool

        :return:
            Reconstructed bipartite digraph.
        :rtype: nx.DiGraph

        :raises ValueError:
            If ``node_ids`` is not ``"source"`` or ``"internal"``.

        .. rubric:: Example

        .. code-block:: python

            g2 = syn.to_digraph()
            g3 = syn.to_digraph(node_ids="internal", reaction_kind="reaction")
        """
        if node_ids not in {"source", "internal"}:
            raise ValueError("node_ids must be 'source' or 'internal'")

        g = nx.DiGraph()
        g.graph.update(self.graph_attrs)
        g.graph.update(self.metadata)

        species_node_map: Dict[str, Hashable] = {}
        reaction_node_map: Dict[str, Hashable] = {}

        for sid, sp in self.species.items():
            nid = _resolve_species_node_id(sp, node_ids=node_ids)
            g.add_node(
                nid,
                **_species_node_attrs(sp, include_internal_ids=include_internal_ids),
            )
            species_node_map[sid] = nid

        for rid, rxn in self.reactions.items():
            nid = _resolve_reaction_node_id(rxn, node_ids=node_ids)
            g.add_node(
                nid,
                **_reaction_node_attrs(
                    rxn,
                    reaction_kind=reaction_kind,
                    include_internal_ids=include_internal_ids,
                ),
            )
            reaction_node_map[rid] = nid

        for rid, rxn in self.reactions.items():
            _add_reaction_edges_to_graph(
                g,
                rxn=rxn,
                reaction_node=reaction_node_map[rid],
                species_node_map=species_node_map,
            )

        return g

    @classmethod
    def from_reaction_strings(
        cls,
        rxns: List[str],
        rules: Optional[List[Optional[str]]] = None,
        *,
        parser: Optional[Callable[[str], Any]] = None,
        strict: bool = True,
        id_style: str = "prefixed",
        species_prefix: str = "s_",
        reaction_prefix: str = "r_",
        rule_prefix: str = "rule_",
    ) -> "SynCRN":
        """Build a SynCRN object directly from reaction strings.

        Reactions and rules are interpreted pairwise, so ``rxns[i]`` corresponds
        to ``rules[i]``.

        ID policy
        ---------
        By default this constructor mints the same prefixed ids as
        :meth:`from_digraph`, so code written against one constructor works
        against the other:

        - Species ids: ``"s_1"``, ``"s_2"``, ...
        - Reaction ids: ``"r_1"``, ``"r_2"``, ...
        - Rule ids: ``"rule_1"``, ``"rule_2"``, ...

        Passing ``id_style="numeric"`` restores the legacy single-namespace
        scheme, where species are ``"1".."n"`` and reaction ids continue at
        ``"n+1"``. That scheme is ambiguous — a species label can coincide with
        a reaction id — and is kept only for backwards compatibility.

        Species are indexed by first appearance in the input reactions.

        :param rxns:
            List of reaction strings such as ``"2A>>B+3C"``.
        :type rxns: List[str]

        :param rules:
            Optional list of rule strings pairwise aligned with ``rxns``.
        :type rules: Optional[List[Optional[str]]]

        :param parser:
            Optional side parser. It should accept one side string such as
            ``"2A+B"`` and return either a mapping, an object with
            ``to_dict()``, or an object with ``items()``.
        :type parser: Optional[Callable[[str], Any]]

        :param strict:
            Whether malformed reaction strings or empty sides should raise an error.
        :type strict: bool

        :param id_style:
            Internal-id policy, either ``"prefixed"`` (default) or ``"numeric"``.
        :type id_style: str

        :param species_prefix:
            Prefix for generated species ids under ``id_style="prefixed"``.
        :type species_prefix: str

        :param reaction_prefix:
            Prefix for generated reaction ids under ``id_style="prefixed"``.
        :type reaction_prefix: str

        :param rule_prefix:
            Prefix for generated rule ids under ``id_style="prefixed"``.
        :type rule_prefix: str

        :return:
            Canonical SynCRN object.
        :rtype: SynCRN

        :raises TypeError:
            If ``rxns`` or ``rules`` have invalid types.

        :raises ValueError:
            If reaction strings are malformed, rules are not pairwise aligned,
            or ``id_style`` is unknown.

        .. rubric:: Example

        .. code-block:: python

            syn = SynCRN.from_reaction_strings(["2A>>B+3C"])
            print(syn.species_ids)      # ['s_1', 's_2', 's_3']
            print(syn.reaction_ids)     # ['r_1']

            legacy = SynCRN.from_reaction_strings(["2A>>B+3C"], id_style="numeric")
            print(legacy.species_ids)   # ['1', '2', '3']
        """
        if id_style not in ID_STYLES:
            raise ValueError(
                f"id_style must be one of {sorted(ID_STYLES)}, got {id_style!r}"
            )
        if not isinstance(rxns, (list, tuple)) or not all(
            isinstance(x, str) for x in rxns
        ):
            raise TypeError("rxns must be a list or tuple of reaction strings")

        if len(rxns) == 0:
            return cls(
                species={},
                reactions={},
                rules={},
                graph_attrs={},
                metadata={
                    "source": "reaction_strings",
                    "n_input_reactions": 0,
                    "has_pairwise_rules": False,
                    "id_style": id_style,
                },
            )

        has_rules = rules is not None and len(rules) > 0
        if has_rules:
            if not isinstance(rules, (list, tuple)):
                raise TypeError("rules must be a list or tuple when provided")
            if len(rules) != len(rxns):
                raise ValueError(
                    f"rules must be pairwise with rxns: got {len(rxns)} reactions and "
                    f"{len(rules)} rules"
                )
            if not all(r is None or isinstance(r, str) for r in rules):
                raise TypeError("rules entries must be strings or None")

        parse_side = parser if parser is not None else _default_parse_side_text

        parsed_rxns = [
            _parse_reaction_string_entry(
                rxn_text=rxn_text,
                rxn_index=i,
                rule_repr=rules[i] if has_rules else None,
                has_rules=has_rules,
                parse_side=parse_side,
                strict=strict,
            )
            for i, rxn_text in enumerate(rxns)
        ]

        species_order = _species_order_from_parsed_reactions(parsed_rxns)
        minted = _mint_reaction_string_ids(
            n_species=len(species_order),
            n_reactions=len(parsed_rxns),
            id_style=id_style,
            species_prefix=species_prefix,
            reaction_prefix=reaction_prefix,
            rule_prefix=rule_prefix,
        )
        label_to_sid = dict(zip(species_order, minted["species_ids"]))

        species = _build_species_table_from_labels(
            species_order,
            species_ids=minted["species_ids"],
            species_source_ids=minted["species_source_ids"],
        )
        rules_table = _build_rules_table_from_reaction_strings(
            parsed_rxns,
            rule_ids=minted["rule_ids"],
            has_rules=has_rules,
        )
        reactions = _build_reactions_from_parsed_strings(
            parsed_rxns,
            label_to_sid=label_to_sid,
            reaction_ids=minted["reaction_ids"],
            reaction_source_ids=minted["reaction_source_ids"],
            rule_ids=minted["rule_ids"],
            has_rules=has_rules,
        )

        return cls(
            species=species,
            reactions=reactions,
            rules=rules_table,
            graph_attrs={},
            metadata={
                "source": "reaction_strings",
                "n_input_reactions": len(rxns),
                "has_pairwise_rules": has_rules,
                "id_style": id_style,
            },
        )
