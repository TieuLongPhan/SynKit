from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Tuple,
)

import networkx as nx

Place = str
TransitionId = str
Marking = Mapping[Place, int]
Multiset = Mapping[str, int]


@dataclass(frozen=True)
class SynCRNIncidence:
    """
    Canonical species--reaction incidence view extracted from a SynCRN-like object.

    This dataclass stores a normalized incidence representation that can be
    constructed either from a SynCRN-style object exposing ``species`` and
    ``reactions`` mappings, or from a bipartite :class:`networkx.DiGraph`
    returned by ``SynCRN.to_digraph()``.

    The canonical representation separates:

    - species order
    - reaction order
    - species and reaction labels
    - pre-incidence stoichiometry
    - post-incidence stoichiometry
    - source-node provenance metadata

    :param species_order:
        Ordered list of canonical species identifiers.
    :type species_order: List[str]
    :param reaction_order:
        Ordered list of canonical reaction identifiers.
    :type reaction_order: List[str]
    :param species_labels:
        Mapping from species identifier to display label.
    :type species_labels: Dict[str, str]
    :param reaction_labels:
        Mapping from reaction identifier to display label.
    :type reaction_labels: Dict[str, str]
    :param pre:
        Mapping ``reaction_id -> {species_id: stoichiometric_coefficient}``
        for reactants / input arcs.
    :type pre: Dict[str, Dict[str, int]]
    :param post:
        Mapping ``reaction_id -> {species_id: stoichiometric_coefficient}``
        for products / output arcs.
    :type post: Dict[str, Dict[str, int]]
    :param species_source_node_ids:
        Mapping from canonical species identifier to original source node id.
    :type species_source_node_ids: Dict[str, Hashable]
    :param reaction_source_node_ids:
        Mapping from canonical reaction identifier to original source node id.
    :type reaction_source_node_ids: Dict[str, Hashable]
    :param graph_attrs:
        Graph-level metadata copied from the source object or source graph.
    :type graph_attrs: Dict[str, Any]
    :param metadata:
        Additional extraction metadata.
    :type metadata: Dict[str, Any]

    Example
    -------
    .. code-block:: python

        incidence = SynCRNIncidence(
            species_order=["A", "B", "C"],
            reaction_order=["r1"],
            species_labels={"A": "A", "B": "B", "C": "C"},
            reaction_labels={"r1": "A + B -> C"},
            pre={"r1": {"A": 1, "B": 1}},
            post={"r1": {"C": 1}},
        )
    """

    species_order: List[str]
    reaction_order: List[str]
    species_labels: Dict[str, str]
    reaction_labels: Dict[str, str]
    pre: Dict[str, Dict[str, int]]
    post: Dict[str, Dict[str, int]]
    species_source_node_ids: Dict[str, Hashable] = field(default_factory=dict)
    reaction_source_node_ids: Dict[str, Hashable] = field(default_factory=dict)
    graph_attrs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Transition:
    """
    Petri-net transition with stoichiometric input/output multisets.

    A transition corresponds to one reaction node in the source SynCRN-like
    representation.

    :param tid:
        Canonical transition identifier.
    :type tid: TransitionId
    :param pre:
        Input multiset mapping ``place -> coefficient``.
    :type pre: Dict[Place, int]
    :param post:
        Output multiset mapping ``place -> coefficient``.
    :type post: Dict[Place, int]
    :param label:
        Optional display label.
    :type label: Optional[str]
    :param source_reaction_id:
        Optional source reaction identifier from the original CRN object.
    :type source_reaction_id: Optional[str]
    :param metadata:
        Optional transition metadata dictionary.
    :type metadata: Dict[str, Any]

    Example
    -------
    .. code-block:: python

        t = Transition(
            tid="r1",
            pre={"A": 1, "B": 2},
            post={"C": 1},
            label="A + 2B -> C",
        )
    """

    tid: TransitionId
    pre: Dict[Place, int]
    post: Dict[Place, int]
    label: Optional[str] = None
    source_reaction_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        """
        Return a compact debug representation.

        :returns:
            Readable transition representation.
        :rtype: str
        """
        extra = f", label={self.label!r}" if self.label is not None else ""
        return f"Transition({self.tid!r}, pre={self.pre}, post={self.post}{extra})"


def _safe_int(x: Any, default: int = 0) -> int:
    """
    Convert a value to ``int`` with fallback.

    :param x:
        Value to convert.
    :type x: Any
    :param default:
        Fallback value used if conversion fails.
    :type default: int
    :returns:
        Converted integer value.
    :rtype: int
    """
    try:
        return int(x)
    except Exception:
        return int(default)


def _coerce_stoich(x: Any) -> int:
    """
    Validate and normalize a stoichiometric coefficient.

    The value is converted to ``int`` and must be strictly positive.

    :param x:
        Candidate stoichiometric coefficient.
    :type x: Any
    :returns:
        Positive stoichiometric coefficient.
    :rtype: int
    :raises ValueError:
        If the resulting value is not strictly positive.

    Example
    -------
    .. code-block:: python

        _coerce_stoich(2)      # 2
        _coerce_stoich("3")    # 3
    """
    val = _safe_int(x, default=1)
    if val <= 0:
        raise ValueError(f"Stoichiometric coefficient must be positive, got {x!r}")
    return val


def _naturalish_key(x: Any) -> Tuple[int, str, str]:
    """
    Build a stable sorting key for mixed identifiers.

    Numeric-looking strings are sorted numerically first, while all other
    objects are sorted by type name and representation.

    :param x:
        Object used as an identifier.
    :type x: Any
    :returns:
        Tuple suitable for deterministic ordering.
    :rtype: Tuple[int, str, str]
    """
    sx = str(x)
    if sx.isdigit():
        return (0, f"{int(sx):020d}", sx)
    return (1, type(x).__name__, repr(x))


def _graph_node_kind(attrs: Mapping[str, Any]) -> str:
    """
    Normalize the ``kind`` attribute of a graph node.

    :param attrs:
        Node attribute mapping.
    :type attrs: Mapping[str, Any]
    :returns:
        Lowercased and stripped node kind.
    :rtype: str
    """
    return str(attrs.get("kind", "")).strip().lower()


def _edge_side_from_graph(
    u: Hashable,
    v: Hashable,
    eattrs: Mapping[str, Any],
    reaction_node: Hashable,
) -> str:
    """
    Infer whether an incidence edge belongs to the left or right side.

    The function first checks the explicit edge attribute ``role``:

    - ``"reactant"`` -> ``"lhs"``
    - ``"product"`` -> ``"rhs"``

    If no explicit role is present, direction relative to the reaction node
    is used as a fallback.

    :param u:
        Source node of the directed edge.
    :type u: Hashable
    :param v:
        Target node of the directed edge.
    :type v: Hashable
    :param eattrs:
        Edge attribute mapping.
    :type eattrs: Mapping[str, Any]
    :param reaction_node:
        Node corresponding to the reaction / transition.
    :type reaction_node: Hashable
    :returns:
        Either ``"lhs"`` or ``"rhs"``.
    :rtype: str
    """
    role = eattrs.get("role")
    if role == "reactant":
        return "lhs"
    if role == "product":
        return "rhs"
    return "lhs" if v == reaction_node else "rhs"


def _extract_from_syncrn_object(crn: Any) -> SynCRNIncidence:
    """
    Extract canonical incidence information from a SynCRN-like object.

    The input object is expected to expose:

    - ``species``: mapping of species id -> species-like object
    - ``reactions``: mapping of reaction id -> reaction-like object

    Species-like objects may expose ``label`` and ``source_node_id``.
    Reaction-like objects may expose ``label``, ``source_node_id``, ``lhs``,
    and ``rhs``.

    :param crn:
        SynCRN-like object.
    :type crn: Any
    :returns:
        Canonical incidence representation.
    :rtype: SynCRNIncidence

    Example
    -------
    .. code-block:: python

        incidence = _extract_from_syncrn_object(crn)
        print(incidence.pre)
        print(incidence.post)
    """
    species_order = [str(x) for x in getattr(crn, "species", {}).keys()]
    reaction_order = [str(x) for x in getattr(crn, "reactions", {}).keys()]

    species_labels: Dict[str, str] = {}
    reaction_labels: Dict[str, str] = {}
    species_source_node_ids: Dict[str, Hashable] = {}
    reaction_source_node_ids: Dict[str, Hashable] = {}
    pre: Dict[str, Dict[str, int]] = {}
    post: Dict[str, Dict[str, int]] = {}

    for sid, sp in getattr(crn, "species", {}).items():
        sid_s = str(sid)
        species_labels[sid_s] = str(getattr(sp, "label", sid_s))
        species_source_node_ids[sid_s] = getattr(sp, "source_node_id", sid)

    for rid, rxn in getattr(crn, "reactions", {}).items():
        rid_s = str(rid)
        reaction_labels[rid_s] = str(getattr(rxn, "label", rid_s))
        reaction_source_node_ids[rid_s] = getattr(rxn, "source_node_id", rid)
        pre[rid_s] = {
            str(sid): int(coeff) for sid, coeff in getattr(rxn, "lhs", {}).items()
        }
        post[rid_s] = {
            str(sid): int(coeff) for sid, coeff in getattr(rxn, "rhs", {}).items()
        }

    return SynCRNIncidence(
        species_order=species_order,
        reaction_order=reaction_order,
        species_labels=species_labels,
        reaction_labels=reaction_labels,
        pre=pre,
        post=post,
        species_source_node_ids=species_source_node_ids,
        reaction_source_node_ids=reaction_source_node_ids,
        graph_attrs=dict(getattr(crn, "graph_attrs", {}) or {}),
        metadata=dict(getattr(crn, "metadata", {}) or {}),
    )


def _partition_syncrn_nodes(
    crn: nx.DiGraph,
) -> Tuple[List[Hashable], List[Hashable]]:
    """
    Partition graph nodes into species nodes and reaction nodes.

    Nodes with ``kind == "species"`` are treated as species nodes.
    Nodes with ``kind in {"reaction", "rule"}`` are treated as reaction nodes.

    :param crn:
        SynCRN bipartite digraph.
    :type crn: nx.DiGraph
    :returns:
        Pair ``(species_nodes, reaction_nodes)``.
    :rtype: Tuple[List[Hashable], List[Hashable]]
    """
    species_nodes: List[Hashable] = []
    reaction_nodes: List[Hashable] = []

    for node, attrs in crn.nodes(data=True):
        kind = _graph_node_kind(attrs)
        if kind == "species":
            species_nodes.append(node)
        elif kind in {"reaction", "rule"}:
            reaction_nodes.append(node)

    species_nodes.sort(key=lambda n: _naturalish_key(crn.nodes[n].get("syncrn_id", n)))
    reaction_nodes.sort(key=lambda n: _naturalish_key(crn.nodes[n].get("syncrn_id", n)))
    return species_nodes, reaction_nodes


def _build_species_index(
    crn: nx.DiGraph,
    species_nodes: Iterable[Hashable],
) -> Tuple[List[str], Dict[str, str], Dict[str, Hashable], Dict[Hashable, str]]:
    """
    Build canonical species metadata from graph nodes.

    :param crn:
        Source graph.
    :type crn: nx.DiGraph
    :param species_nodes:
        Iterable of species node ids.
    :type species_nodes: Iterable[Hashable]
    :returns:
        Tuple containing species order, labels, source node ids, and reverse lookup.
    :rtype: Tuple[List[str], Dict[str, str], Dict[str, Hashable], Dict[Hashable, str]]
    """
    species_order: List[str] = []
    species_labels: Dict[str, str] = {}
    species_source_node_ids: Dict[str, Hashable] = {}
    species_node_to_id: Dict[Hashable, str] = {}

    for node in species_nodes:
        attrs = dict(crn.nodes[node])
        sid = str(attrs.get("syncrn_id", node))
        species_order.append(sid)
        species_labels[sid] = str(attrs.get("label", sid))
        species_source_node_ids[sid] = attrs.get("source_node_id", node)
        species_node_to_id[node] = sid

    return (
        species_order,
        species_labels,
        species_source_node_ids,
        species_node_to_id,
    )


def _build_reaction_index(
    crn: nx.DiGraph,
    reaction_nodes: Iterable[Hashable],
) -> Tuple[List[str], Dict[str, str], Dict[str, Hashable], Dict[Hashable, str]]:
    """
    Build canonical reaction metadata from graph nodes.

    :param crn:
        Source graph.
    :type crn: nx.DiGraph
    :param reaction_nodes:
        Iterable of reaction node ids.
    :type reaction_nodes: Iterable[Hashable]
    :returns:
        Tuple containing reaction order, labels, source node ids, and reverse lookup.
    :rtype: Tuple[List[str], Dict[str, str], Dict[str, Hashable], Dict[Hashable, str]]
    """
    reaction_order: List[str] = []
    reaction_labels: Dict[str, str] = {}
    reaction_source_node_ids: Dict[str, Hashable] = {}
    reaction_node_to_id: Dict[Hashable, str] = {}

    for node in reaction_nodes:
        attrs = dict(crn.nodes[node])
        rid = str(attrs.get("syncrn_id", node))
        reaction_order.append(rid)
        reaction_labels[rid] = str(attrs.get("label", rid))
        reaction_source_node_ids[rid] = attrs.get("source_node_id", node)
        reaction_node_to_id[node] = rid

    return (
        reaction_order,
        reaction_labels,
        reaction_source_node_ids,
        reaction_node_to_id,
    )


def _accumulate_incidence_from_edges(
    crn: nx.DiGraph,
    reaction_node: Hashable,
    species_node_to_id: Mapping[Hashable, str],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Accumulate input and output stoichiometry for one reaction node.

    Both incoming and outgoing graph edges are inspected. The edge role is
    inferred via :func:`_edge_side_from_graph`.

    :param crn:
        Source graph.
    :type crn: nx.DiGraph
    :param reaction_node:
        Reaction node whose local incidence is extracted.
    :type reaction_node: Hashable
    :param species_node_to_id:
        Mapping from species graph node to canonical species id.
    :type species_node_to_id: Mapping[Hashable, str]
    :returns:
        Pair ``(pre, post)`` for the reaction.
    :rtype: Tuple[Dict[str, int], Dict[str, int]]
    """
    lhs_counter: Counter[str] = Counter()
    rhs_counter: Counter[str] = Counter()

    for u, v, eattrs in crn.in_edges(reaction_node, data=True):
        if u not in species_node_to_id:
            continue
        sid = species_node_to_id[u]
        side = _edge_side_from_graph(u, v, eattrs, reaction_node)
        stoich = _coerce_stoich(eattrs.get("stoich", 1))
        if side == "lhs":
            lhs_counter[sid] += stoich
        else:
            rhs_counter[sid] += stoich

    for u, v, eattrs in crn.out_edges(reaction_node, data=True):
        if v not in species_node_to_id:
            continue
        sid = species_node_to_id[v]
        side = _edge_side_from_graph(u, v, eattrs, reaction_node)
        stoich = _coerce_stoich(eattrs.get("stoich", 1))
        if side == "lhs":
            lhs_counter[sid] += stoich
        else:
            rhs_counter[sid] += stoich

    return dict(lhs_counter), dict(rhs_counter)


def _build_pre_post_from_graph(
    crn: nx.DiGraph,
    reaction_nodes: Iterable[Hashable],
    reaction_node_to_id: Mapping[Hashable, str],
    species_node_to_id: Mapping[Hashable, str],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    """
    Build reaction-wise pre and post incidence maps from a graph.

    :param crn:
        Source graph.
    :type crn: nx.DiGraph
    :param reaction_nodes:
        Iterable of reaction graph nodes.
    :type reaction_nodes: Iterable[Hashable]
    :param reaction_node_to_id:
        Mapping from reaction graph node to canonical reaction id.
    :type reaction_node_to_id: Mapping[Hashable, str]
    :param species_node_to_id:
        Mapping from species graph node to canonical species id.
    :type species_node_to_id: Mapping[Hashable, str]
    :returns:
        Pair ``(pre, post)`` keyed by reaction id.
    :rtype: Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]
    """
    pre: Dict[str, Dict[str, int]] = {}
    post: Dict[str, Dict[str, int]] = {}

    for rnode in reaction_nodes:
        rid = reaction_node_to_id[rnode]
        pre[rid], post[rid] = _accumulate_incidence_from_edges(
            crn=crn,
            reaction_node=rnode,
            species_node_to_id=species_node_to_id,
        )

    return pre, post


def _extract_from_syncrn_digraph(crn: nx.DiGraph) -> SynCRNIncidence:
    """
    Extract canonical incidence information from a SynCRN bipartite digraph.

    The graph is expected to use node attribute ``kind`` with values:

    - ``"species"`` for species nodes
    - ``"reaction"`` or ``"rule"`` for reaction nodes

    Optional node attributes include:

    - ``syncrn_id``
    - ``label``
    - ``source_node_id``

    Optional edge attributes include:

    - ``stoich``
    - ``role`` with values ``"reactant"`` or ``"product"``

    :param crn:
        SynCRN bipartite directed graph.
    :type crn: nx.DiGraph
    :returns:
        Canonical incidence representation extracted from the graph.
    :rtype: SynCRNIncidence
    :raises TypeError:
        If ``crn`` is not a :class:`networkx.DiGraph`.

    Example
    -------
    .. code-block:: python

        g = syn.to_digraph()
        incidence = _extract_from_syncrn_digraph(g)
        print(incidence.species_order)
        print(incidence.reaction_order)
    """
    if not isinstance(crn, nx.DiGraph):
        raise TypeError(f"Expected nx.DiGraph, got {type(crn).__name__}")

    species_nodes, reaction_nodes = _partition_syncrn_nodes(crn)

    (
        species_order,
        species_labels,
        species_source_node_ids,
        species_node_to_id,
    ) = _build_species_index(crn, species_nodes)

    (
        reaction_order,
        reaction_labels,
        reaction_source_node_ids,
        reaction_node_to_id,
    ) = _build_reaction_index(crn, reaction_nodes)

    pre, post = _build_pre_post_from_graph(
        crn=crn,
        reaction_nodes=reaction_nodes,
        reaction_node_to_id=reaction_node_to_id,
        species_node_to_id=species_node_to_id,
    )

    return SynCRNIncidence(
        species_order=species_order,
        reaction_order=reaction_order,
        species_labels=species_labels,
        reaction_labels=reaction_labels,
        pre=pre,
        post=post,
        species_source_node_ids=species_source_node_ids,
        reaction_source_node_ids=reaction_source_node_ids,
        graph_attrs=dict(crn.graph),
        metadata={"source_graph_type": type(crn).__name__},
    )


def extract_syncrn_incidence(crn: Any) -> SynCRNIncidence:
    """
    Extract a canonical incidence view from a SynCRN-like object or digraph.

    The function accepts three input styles:

    - a :class:`networkx.DiGraph` in SynCRN bipartite format
    - a SynCRN-like object exposing ``species`` and ``reactions``
    - an object exposing ``to_digraph()`` that returns such a graph

    :param crn:
        SynCRN-like object or bipartite digraph.
    :type crn: Any
    :returns:
        Canonical incidence representation.
    :rtype: SynCRNIncidence
    :raises TypeError:
        If the input cannot be interpreted as a supported SynCRN source.

    Example
    -------
    .. code-block:: python

        incidence = extract_syncrn_incidence(crn)
        print(incidence.pre)
        print(incidence.post)
    """
    if isinstance(crn, nx.DiGraph):
        return _extract_from_syncrn_digraph(crn)

    if hasattr(crn, "species") and hasattr(crn, "reactions"):
        return _extract_from_syncrn_object(crn)

    if hasattr(crn, "to_digraph"):
        g = crn.to_digraph()
        if isinstance(g, nx.DiGraph):
            return _extract_from_syncrn_digraph(g)

    raise TypeError(
        "crn must be a SynCRN-like object with species/reactions or an nx.DiGraph "
        "in SynCRN bipartite format"
    )


class PetriNet:
    """
    Minimal Petri net container with marking semantics and SynCRN metadata.

    Places correspond to species and transitions correspond to reactions.
    The class stores a lightweight Petri-net representation with utilities
    for adding places and transitions, checking enabledness, firing
    transitions, and converting between mapping-based and tuple-based
    markings.

    Attributes
    ----------
    places:
        Set of place identifiers.
    transitions:
        Mapping from transition id to :class:`Transition`.
    place_labels:
        Optional human-readable labels for places.
    transition_labels:
        Optional human-readable labels for transitions.
    place_source_node_ids:
        Provenance mapping for places.
    transition_source_node_ids:
        Provenance mapping for transitions.
    graph_attrs:
        Graph-level metadata copied from the source.
    metadata:
        Additional metadata.

    Example
    -------
    .. code-block:: python

        net = PetriNet()
        net.add_place("A")
        net.add_place("B")
        net.add_transition("r1", pre={"A": 1}, post={"B": 1})

        m0 = {"A": 1}
        assert net.enabled(m0, "r1")
        m1 = net.fire(m0, "r1")
        print(m1)   # {'A': 0, 'B': 1}
    """

    def __init__(self) -> None:
        """
        Initialize an empty Petri net.

        :returns:
            None
        :rtype: None
        """
        self.places: Set[Place] = set()
        self.transitions: Dict[TransitionId, Transition] = {}
        self._place_index: Dict[Place, int] = {}
        self._transition_index: Dict[TransitionId, int] = {}
        self.place_labels: Dict[Place, str] = {}
        self.transition_labels: Dict[TransitionId, str] = {}
        self.place_source_node_ids: Dict[Place, Hashable] = {}
        self.transition_source_node_ids: Dict[TransitionId, Hashable] = {}
        self.graph_attrs: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

    @classmethod
    def from_syncrn(cls, crn: Any) -> "PetriNet":
        """
        Build a Petri-net view directly from SynCRN incidence data.

        :param crn:
            SynCRN-like object or SynCRN bipartite digraph.
        :type crn: Any
        :returns:
            Petri net constructed from the canonical incidence view.
        :rtype: PetriNet

        Example
        -------
        .. code-block:: python

            net = PetriNet.from_syncrn(crn)
            print(net.place_order)
            print(net.transition_order)
        """
        incidence = extract_syncrn_incidence(crn)
        net = cls()
        net.graph_attrs.update(incidence.graph_attrs)
        net.metadata.update(incidence.metadata)
        net.metadata["source"] = "syncrn"

        for sid in incidence.species_order:
            net.add_place(
                sid,
                label=incidence.species_labels.get(sid, sid),
                source_node_id=incidence.species_source_node_ids.get(sid),
            )

        for rid in incidence.reaction_order:
            net.add_transition(
                rid,
                pre=incidence.pre.get(rid, {}),
                post=incidence.post.get(rid, {}),
                label=incidence.reaction_labels.get(rid, rid),
                source_reaction_id=rid,
            )
            if rid in incidence.reaction_source_node_ids:
                net.transition_source_node_ids[rid] = (
                    incidence.reaction_source_node_ids[rid]
                )

        return net

    @property
    def place_order(self) -> List[Place]:
        """
        Return places in insertion order.

        :returns:
            Ordered place identifiers.
        :rtype: List[Place]
        """
        return list(self._place_index.keys())

    @property
    def transition_order(self) -> List[TransitionId]:
        """
        Return transitions in insertion order.

        :returns:
            Ordered transition identifiers.
        :rtype: List[TransitionId]
        """
        return list(self._transition_index.keys())

    def add_place(
        self,
        p: Place,
        *,
        label: Optional[str] = None,
        source_node_id: Optional[Hashable] = None,
    ) -> None:
        """
        Add a place to the Petri net.

        If the place already exists, only optional metadata is updated.

        :param p:
            Place identifier.
        :type p: Place
        :param label:
            Optional display label.
        :type label: Optional[str]
        :param source_node_id:
            Optional provenance node id from the source CRN.
        :type source_node_id: Optional[Hashable]
        :returns:
            None
        :rtype: None
        """
        if p not in self.places:
            self.places.add(p)
            self._place_index[p] = len(self._place_index)
        if label is not None:
            self.place_labels[p] = str(label)
        if source_node_id is not None:
            self.place_source_node_ids[p] = source_node_id

    def add_transition(
        self,
        tid: TransitionId,
        pre: Mapping[Place, int],
        post: Mapping[Place, int],
        *,
        label: Optional[str] = None,
        source_reaction_id: Optional[str] = None,
        metadata: Optional[MutableMapping[str, Any]] = None,
    ) -> None:
        """
        Add or replace a transition in the Petri net.

        Zero or invalid non-positive weights are filtered out before storage.
        Any places referenced by ``pre`` or ``post`` are created automatically.

        :param tid:
            Transition identifier.
        :type tid: TransitionId
        :param pre:
            Input arc multiset mapping ``place -> coefficient``.
        :type pre: Mapping[Place, int]
        :param post:
            Output arc multiset mapping ``place -> coefficient``.
        :type post: Mapping[Place, int]
        :param label:
            Optional display label.
        :type label: Optional[str]
        :param source_reaction_id:
            Optional source reaction identifier.
        :type source_reaction_id: Optional[str]
        :param metadata:
            Optional metadata attached to the transition.
        :type metadata: Optional[MutableMapping[str, Any]]
        :returns:
            None
        :rtype: None

        Example
        -------
        .. code-block:: python

            net.add_transition(
                "r1",
                pre={"A": 2, "B": 1},
                post={"C": 1},
                label="2A + B -> C",
            )
        """
        clean_pre = {
            str(p): _coerce_stoich(w) for p, w in pre.items() if _safe_int(w, 0) > 0
        }
        clean_post = {
            str(p): _coerce_stoich(w) for p, w in post.items() if _safe_int(w, 0) > 0
        }

        for p in set(clean_pre) | set(clean_post):
            self.add_place(p)

        tid = str(tid)
        if tid not in self._transition_index:
            self._transition_index[tid] = len(self._transition_index)

        self.transitions[tid] = Transition(
            tid=tid,
            pre=clean_pre,
            post=clean_post,
            label=None if label is None else str(label),
            source_reaction_id=(
                None if source_reaction_id is None else str(source_reaction_id)
            ),
            metadata=dict(metadata or {}),
        )
        if label is not None:
            self.transition_labels[tid] = str(label)

    def enabled(self, marking: Marking, tid: TransitionId) -> bool:
        """
        Test whether a transition is enabled under a marking.

        A transition is enabled if every required input place has at least the
        corresponding token count.

        :param marking:
            Current marking as ``place -> token_count``.
        :type marking: Marking
        :param tid:
            Transition identifier.
        :type tid: TransitionId
        :returns:
            ``True`` if the transition is enabled, else ``False``.
        :rtype: bool
        """
        t = self.transitions[tid]
        return all(int(marking.get(p, 0)) >= w for p, w in t.pre.items())

    def fire(self, marking: Marking, tid: TransitionId) -> Dict[Place, int]:
        """
        Fire a transition and return the successor marking.

        This function does not itself check enabledness. If the transition is
        not enabled, negative token counts may appear in the result.

        :param marking:
            Current marking.
        :type marking: Marking
        :param tid:
            Transition identifier to fire.
        :type tid: TransitionId
        :returns:
            Successor marking after consuming ``pre`` and producing ``post``.
        :rtype: Dict[Place, int]

        Example
        -------
        .. code-block:: python

            m0 = {"A": 2, "B": 1}
            m1 = net.fire(m0, "r1")
        """
        t = self.transitions[tid]
        nxt = {str(p): int(v) for p, v in marking.items()}
        for p, w in t.pre.items():
            nxt[p] = nxt.get(p, 0) - w
        for p, w in t.post.items():
            nxt[p] = nxt.get(p, 0) + w
        return nxt

    def marking_to_tuple(self, m: Marking) -> Tuple[int, ...]:
        """
        Convert a mapping-based marking into a tuple in place order.

        :param m:
            Marking mapping.
        :type m: Marking
        :returns:
            Tuple of token counts aligned to :attr:`place_order`.
        :rtype: Tuple[int, ...]
        """
        arr = [0] * len(self._place_index)
        for p, idx in self._place_index.items():
            arr[idx] = int(m.get(p, 0))
        return tuple(arr)

    def tuple_to_marking(self, values: Iterable[int]) -> Dict[Place, int]:
        """
        Convert a tuple-like token vector into a sparse marking mapping.

        Zero entries are omitted from the returned dictionary.

        :param values:
            Iterable of token counts aligned to :attr:`place_order`.
        :type values: Iterable[int]
        :returns:
            Sparse marking mapping.
        :rtype: Dict[Place, int]
        """
        vals = list(values)
        return {
            p: int(vals[idx])
            for p, idx in self._place_index.items()
            if idx < len(vals) and int(vals[idx]) != 0
        }

    def place_name(self, p: Place) -> str:
        """
        Return the display label of a place if available.

        :param p:
            Place identifier.
        :type p: Place
        :returns:
            Display label or the identifier itself.
        :rtype: str
        """
        return self.place_labels.get(p, p)

    def transition_name(self, tid: TransitionId) -> str:
        """
        Return the display label of a transition if available.

        :param tid:
            Transition identifier.
        :type tid: TransitionId
        :returns:
            Display label or the identifier itself.
        :rtype: str
        """
        return self.transition_labels.get(tid, tid)

    def to_pre_post(self) -> Dict[str, Any]:
        """
        Export the Petri net as place-indexed pre/post adjacency maps.

        The returned structure is often convenient for reachability, firing,
        or incidence-based downstream algorithms.

        :returns:
            Dictionary containing places, transitions, labels, pre/post maps,
            graph attributes, and metadata.
        :rtype: Dict[str, Any]

        Example
        -------
        .. code-block:: python

            data = net.to_pre_post()
            print(data["pre"])
            print(data["post"])
        """
        pre = {p: {} for p in self.place_order}
        post = {p: {} for p in self.place_order}
        for tid in self.transition_order:
            t = self.transitions[tid]
            for p, w in t.pre.items():
                pre.setdefault(p, {})[tid] = w
            for p, w in t.post.items():
                post.setdefault(p, {})[tid] = w
        return {
            "places": self.place_order,
            "transitions": self.transition_order,
            "place_labels": dict(self.place_labels),
            "transition_labels": dict(self.transition_labels),
            "pre": pre,
            "post": post,
            "graph_attrs": dict(self.graph_attrs),
            "metadata": dict(self.metadata),
        }

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        """
        Return a compact debug representation.

        :returns:
            Readable Petri net summary.
        :rtype: str
        """
        return (
            f"PetriNet(n_places={len(self.places)}, "
            f"n_transitions={len(self.transitions)})"
        )
