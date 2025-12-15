from __future__ import annotations

"""
Pathway realizability utilities.

This module implements conversion of integer hyperflows to Petri nets,
several realizability checks (fast sufficient König test, bounded BFS
reachability), and helpers for scaled- and borrow-realizability.

High-level entry point: :class:`PathwayRealizability`.

Typical workflow
----------------

1. Start from a :class:`CRNHyperGraph`::

       from synkit.CRN.Hypergraph.hypergraph import CRNHyperGraph

       hg = CRNHyperGraph()
       hg.parse_rxns(["A >> B", "B >> C"])

2. Convert the hypergraph into the simple tuple format used here via
   :func:`hypergraph_to_pr_inputs`.

3. Load the data into :class:`PathwayRealizability`, build the Petri net,
   and call one of the realizability checks.

References
----------
- Andersen et al. — Pathway realizability via Petri nets and integer flows.
- Murata (1989), Proc. IEEE — Petri nets: Properties, analysis and applications.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

from collections import deque, defaultdict
import itertools
import json

import networkx as nx

from synkit.CRN.Hypergraph.hypergraph import CRNHyperGraph

# shared Petri container + type aliases
from synkit.CRN.Petri import PetriNet, Place, TransitionId, Multiset


# ---------------------------------------------------------------------------
# PathwayRealizability core
# ---------------------------------------------------------------------------


@dataclass
class RealizabilityConfig:
    """
    Configuration container for bounded search heuristics.

    :param max_states: Maximum number of distinct markings visited in BFS.
    :type max_states: int
    :param max_depth: Maximum length of firing sequence explored in BFS.
    :type max_depth: int
    """

    max_states: int = 100_000
    max_depth: int = 10_000


class PathwayRealizability:
    """
    High level pathway realizability utilities.

    Use case
    --------
    1. Construct an instance (optionally with a :class:`RealizabilityConfig`).
    2. Load hypergraph + flow via :meth:`load_hypergraph_and_flow`.
    3. Build the extended Petri net via :meth:`build_petri_net_from_flow`.
    4. Call one of the checks:

       * :meth:`is_realizable` (bounded BFS reachability),
       * :meth:`is_realizable_via_konig` (fast sufficient DAG test),
       * :meth:`is_scaled_realizable`,
       * :meth:`is_borrow_realizable`.

    Hypergraph format
    -----------------
    The internal representation is deliberately simple to make adapters
    from other CRN formats easy:

    * ``vertices`` — iterable of species identifiers (strings).
    * ``edges`` — mapping ``edge_id -> (tail_multiset, head_multiset)``,
      where each multiset is a dict ``{species: multiplicity}``.
    * ``flow`` — mapping ``edge_id -> integer multiplicity``.

    :param config: Optional configuration for bounded BFS search.
    :type config: RealizabilityConfig or None

    Examples
    --------
    .. code-block:: python

        from synkit.CRN.Hypergraph.hypergraph import CRNHyperGraph
        from synkit.CRN.Props.realizability import (
            hypergraph_to_pr_inputs, PathwayRealizability
        )

        hg = CRNHyperGraph()
        hg.parse_rxns([">>A", "A>>B", "B>>"])
        vertices, edges, flow = hypergraph_to_pr_inputs(hg)

        pr = PathwayRealizability()
        pr.load_hypergraph_and_flow(vertices, edges, flow)
        pr.build_petri_net_from_flow()

        ok, cert = pr.is_realizable()
        print("Realizable?", ok, "Firing sequence:", cert)
    """

    def __init__(self, config: Optional[RealizabilityConfig] = None) -> None:
        self.vertices: Set[str] = set()
        self.edges: Dict[str, Tuple[Multiset, Multiset]] = {}
        self.flow: Dict[str, int] = {}
        self._petri: Optional[PetriNet] = None
        self._initial_marking: Optional[Dict[Place, int]] = None
        self._target_marking: Optional[Dict[Place, int]] = None
        self._certificate: Optional[List[TransitionId]] = None
        self._config = config or RealizabilityConfig()

    # ------------------------------------------------------------------
    # Data loading + Petri net construction
    # ------------------------------------------------------------------

    def load_hypergraph_and_flow(
        self,
        vertices: Iterable[str],
        edges: Mapping[str, Tuple[Multiset, Multiset]],
        flow: Mapping[str, int],
    ) -> "PathwayRealizability":
        """
        Load a hypergraph + flow into the object.

        :param vertices: Species identifiers.
        :type vertices: Iterable[str]
        :param edges: Mapping ``edge_id -> (tail_multiset, head_multiset)``,
            where each multiset is a dict-like mapping ``{species: count}``.
        :type edges: Mapping[str, Tuple[Mapping[str, int], Mapping[str, int]]]
        :param flow: Mapping ``edge_id -> integer multiplicity`` (may be
            zero for unused edges).
        :type flow: Mapping[str, int]
        :returns: ``self`` (for fluent chaining).
        :rtype: PathwayRealizability
        """
        self.vertices = set(vertices)
        self.edges = {
            eid: (dict(tail), dict(head)) for eid, (tail, head) in edges.items()
        }
        self.flow = {eid: int(flow.get(eid, 0)) for eid in self.edges}
        # invalidate previous Petri net, if any
        self._petri = None
        self._initial_marking = None
        self._target_marking = None
        self._certificate = None
        return self

    def build_petri_net_from_flow(self) -> "PathwayRealizability":
        """
        Convert the loaded hypergraph + flow into the extended Petri net
        :math:`(N, M_0, M_T)` following Andersen et al.:

        * For each species :math:`v`, we create a place ``v``.
        * For each edge :math:`e`, we create:
          - a transition ``t_e``,
          - an external "supply" place ``__ext__e`` (tokens = flow(e)),
          - a target place ``__target__e`` (tokens required in the target
            marking = flow(e)).

        Tokens in species places start at 0 and must end at 0 (unless
        borrow-realizability is used).

        :returns: ``self`` (for fluent chaining).
        :rtype: PathwayRealizability
        :raises RuntimeError: if no hypergraph was previously loaded.
        """
        if not self.edges:
            raise RuntimeError("No hypergraph loaded; call load_hypergraph_and_flow().")

        net = PetriNet()
        M0: Dict[Place, int] = defaultdict(int)
        MT: Dict[Place, int] = defaultdict(int)

        # species places, initially 0 tokens and final 0 tokens
        for v in self.vertices:
            net.add_place(v)
            M0[v] = 0
            MT[v] = 0

        # transitions + external/target places
        for eid, (tail, head) in self.edges.items():
            t_id: TransitionId = eid

            # pre/post on species places
            pre = {v: int(w) for v, w in tail.items() if int(w) > 0}
            post = {v: int(w) for v, w in head.items() if int(w) > 0}

            # external place (supply) ve
            ve = f"__ext__{eid}"
            net.add_place(ve)
            pre_with_supply = dict(pre)
            pre_with_supply[ve] = 1

            # target place ve+ (collect firings)
            ve_t = f"__target__{eid}"
            net.add_place(ve_t)
            post_with_target = dict(post)
            post_with_target[ve_t] = 1

            net.add_transition(t_id, pre_with_supply, post_with_target)

            # initial/target tokens for edge-specific places
            fval = int(self.flow.get(eid, 0))
            M0[ve] = fval
            MT[ve_t] = fval

        self._petri = net
        self._initial_marking = dict(M0)
        self._target_marking = dict(MT)
        self._certificate = None
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def petri(self) -> PetriNet:
        """Return the built Petri net or raise if not yet built."""
        if self._petri is None:
            raise RuntimeError("Petri net not built; call build_petri_net_from_flow().")
        return self._petri

    @property
    def initial_marking(self) -> Dict[Place, int]:
        """Return the initial marking M0."""
        if self._initial_marking is None:
            raise RuntimeError("Initial marking missing; build Petri net first.")
        return self._initial_marking

    @property
    def target_marking(self) -> Dict[Place, int]:
        """Return the target marking MT."""
        if self._target_marking is None:
            raise RuntimeError("Target marking missing; build Petri net first.")
        return self._target_marking

    @property
    def certificate(self) -> Optional[List[TransitionId]]:
        """
        Return the last firing sequence found by :meth:`is_realizable`
        (or ``None`` if no certificate is available).
        """
        return self._certificate

    # ------------------------------------------------------------------
    # Bounded BFS reachability
    # ------------------------------------------------------------------

    def is_realizable(
        self,
        max_states: Optional[int] = None,
        max_depth: Optional[int] = None,
    ) -> Tuple[bool, Optional[List[TransitionId]]]:
        """
        Bounded BFS reachability search from :math:`M_0` to :math:`M_T`.

        Returns a pair ``(reachable, firing_sequence_or_None)``. If
        ``reachable`` is ``True``, the second component is a certificate
        firing sequence (list of transition IDs). If ``False``, it is
        ``None`` and the result should be interpreted as "unknown within
        the given bounds".

        :param max_states: Maximum number of distinct markings to visit.
            Defaults to :attr:`RealizabilityConfig.max_states`.
        :type max_states: int or None
        :param max_depth: Maximum length of firing sequence to explore.
            Defaults to :attr:`RealizabilityConfig.max_depth`.
        :type max_depth: int or None
        :returns: Tuple ``(ok, certificate_or_None)``.
        :rtype: Tuple[bool, Optional[List[str]]]
        """
        net = self.petri
        M0 = dict(self.initial_marking)
        MT = dict(self.target_marking)

        max_states = max_states if max_states is not None else self._config.max_states
        max_depth = max_depth if max_depth is not None else self._config.max_depth

        # quick check: M0 == MT
        if all(M0.get(p, 0) == MT.get(p, 0) for p in set(MT) | set(M0)):
            self._certificate = []
            return True, []

        start = net.marking_to_tuple(M0)
        target = net.marking_to_tuple(MT)

        q: deque[Tuple[Tuple[int, ...], List[TransitionId]]] = deque()
        q.append((start, []))
        visited = {start}
        states = 0

        while q:
            mtuple, seq = q.popleft()
            states += 1
            if states > max_states:
                break
            if len(seq) > max_depth:
                continue

            # reconstruct marking dict
            marking = {p: mtuple[net._place_index[p]] for p in net._place_index}

            # try firing enabled transitions
            for tid in net.transitions:
                if net.enabled(marking, tid):
                    new_mark = net.fire(marking, tid)
                    new_tuple = net.marking_to_tuple(new_mark)
                    if new_tuple == target:
                        seq2 = seq + [tid]
                        self._certificate = seq2
                        return True, seq2
                    if new_tuple not in visited:
                        visited.add(new_tuple)
                        q.append((new_tuple, seq + [tid]))

        # not found within bounds
        self._certificate = None
        return False, None

    # ------------------------------------------------------------------
    # König representation sufficient test
    # ------------------------------------------------------------------

    def is_realizable_via_konig(self) -> bool:
        """
        Fast sufficient DAG test via the **König representation**.

        If the König representation of the flow-induced subhypergraph is
        acyclic, then the pathway is guaranteed to be realizable. If this
        returns ``False``, the flow may still be realizable; the test is
        conservative.

        :returns: ``True`` if König DAG test passes, otherwise ``False``.
        :rtype: bool
        """
        # flow-induced subhypergraph V0, E0
        V0: Set[str] = set()
        E0: Dict[str, Tuple[Multiset, Multiset]] = {}
        for eid, fval in self.flow.items():
            if fval == 0:
                continue
            tail, head = self.edges[eid]
            E0[eid] = (tail, head)
            V0.update(tail.keys())
            V0.update(head.keys())

        G = nx.DiGraph()
        for v in V0:
            G.add_node(("v", v))
        for eid in E0:
            G.add_node(("e", eid))
        for eid, (tail, head) in E0.items():
            for v in tail:
                G.add_edge(("v", v), ("e", eid))
            for v in head:
                G.add_edge(("e", eid), ("v", v))

        return nx.is_directed_acyclic_graph(G)

    # ------------------------------------------------------------------
    # Scaled realizability
    # ------------------------------------------------------------------

    def is_scaled_realizable(self, k_max: int = 4) -> Tuple[bool, Optional[int]]:
        """
        Check **scaled-realizability** by trying integer factors ``k``.

        For each ``k`` in ``{1, ..., k_max}``:

        1. Multiply all edge flows by ``k``.
        2. Rebuild the Petri net.
        3. Run :meth:`is_realizable`.

        If a realizable scaling is found, returns ``(True, k)``. If none
        is found up to ``k_max``, returns ``(False, None)``.

        :param k_max: Maximum scaling factor to test.
        :type k_max: int
        :returns: Tuple ``(ok, k_or_None)``.
        :rtype: Tuple[bool, Optional[int]]
        """
        saved_flow = dict(self.flow)
        for k in range(1, k_max + 1):
            self.flow = {eid: k * int(v) for eid, v in saved_flow.items()}
            self.build_petri_net_from_flow()
            ok, _ = self.is_realizable()
            if ok:
                # restore original flow and net
                self.flow = saved_flow
                self.build_petri_net_from_flow()
                return True, k
        self.flow = saved_flow
        self.build_petri_net_from_flow()
        return False, None

    # ------------------------------------------------------------------
    # Borrow realizability
    # ------------------------------------------------------------------

    def is_borrow_realizable(
        self,
        max_borrow_each: int = 2,
    ) -> Tuple[bool, Optional[Mapping[str, int]]]:
        """
        Brute-force **borrow-realizability** search.

        We allow borrowing up to ``max_borrow_each`` tokens for each
        species (symmetric: we require the same number tokens to be
        returned at the end). For each candidate borrow vector ``b``,
        we:

        1. Add ``b(s)`` tokens to species place ``s`` in both the
           initial and target marking.
        2. Run :meth:`is_realizable`.
        3. Restore markings and continue if not realizable.

        This is exponential in the number of species; use only for very
        small examples or unit tests.

        :param max_borrow_each: Maximum tokens that can be borrowed per
            species.
        :type max_borrow_each: int
        :returns: Tuple ``(ok, b_or_None)``, where ``b`` is the borrow
            vector (species -> tokens) if found.
        :rtype: Tuple[bool, Optional[Mapping[str, int]]]
        """
        if self._initial_marking is None or self._target_marking is None:
            self.build_petri_net_from_flow()

        species = sorted(self.vertices)
        saved_initial = dict(self._initial_marking)
        saved_target = dict(self._target_marking)

        for comb in itertools.product(range(max_borrow_each + 1), repeat=len(species)):
            b = {s: comb[i] for i, s in enumerate(species)}
            # rebuild net to ensure consistency
            self.build_petri_net_from_flow()
            for s, val in b.items():
                if val:
                    self._initial_marking[s] = self._initial_marking.get(s, 0) + val
                    self._target_marking[s] = self._target_marking.get(s, 0) + val
            ok, _ = self.is_realizable()
            # restore
            self._initial_marking = dict(saved_initial)
            self._target_marking = dict(saved_target)
            if ok:
                return True, b

        return False, None

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_pnml(self, fn: str) -> "PathwayRealizability":
        """
        Export the built Petri net to a tiny JSON-like PNML-ish format.

        This is primarily for debugging and post-processing; for real
        model checking you would normally generate proper PNML or LoLA
        input files.

        :param fn: Output filename for the JSON structure.
        :type fn: str
        :returns: ``self``.
        :rtype: PathwayRealizability
        """
        self.build_petri_net_from_flow()
        net = self._petri
        data = {
            "places": sorted(list(net.places)),
            "transitions": {
                tid: {"pre": t.pre, "post": t.post}
                for tid, t in net.transitions.items()
            },
            "initial": self._initial_marking,
            "target": self._target_marking,
        }
        with open(fn, "w") as fh:
            json.dump(data, fh, indent=2)
        return self

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return (
            f"<PathwayRealizability vertices={len(self.vertices)} "
            f"edges={len(self.edges)}>"
        )


# ---------------------------------------------------------------------------
# Adapters and small harness (for CRNHyperGraph)
# ---------------------------------------------------------------------------


def _side_to_dict(side: object) -> Dict[str, int]:
    """
    Convert a :class:`RXNSide`-like or mapping-like object to ``dict[str,int]``.

    The function is deliberately defensive and supports:

    * plain dicts,
    * objects exposing ``.items()`` (e.g. :class:`RXNSide`),
    * objects exposing ``.data`` (dict-like),
    * iterables of tokens (interpreted as multiplicity 1).

    :param side: Input side description (reactants or products).
    :type side: object
    :returns: Plain dictionary mapping species labels to integer counts.
    :rtype: Dict[str, int]
    """
    if side is None:
        return {}
    if isinstance(side, dict):
        return {str(k): int(v) for k, v in side.items()}

    # Try mapping-like behaviour
    try:
        return {str(k): int(v) for k, v in side.items()}  # type: ignore[attr-defined]
    except Exception:
        pass

    # Try `.data` attribute (e.g. RXNSide)
    d = getattr(side, "data", None)
    if isinstance(d, dict):
        return {str(k): int(v) for k, v in d.items()}

    # Fallback: treat as iterable of species labels
    try:
        return {str(x): 1 for x in side}  # type: ignore[arg-type]
    except Exception:
        return {}


def hypergraph_to_pr_inputs(
    hg: CRNHyperGraph,
    flow: Optional[Mapping[str, int]] = None,
) -> Tuple[List[str], Dict[str, Tuple[Dict[str, int], Dict[str, int]]], Dict[str, int]]:
    """
    Convert :class:`CRNHyperGraph` into the tuple format used by
    :class:`PathwayRealizability`.

    Outputs
    -------
    * ``vertices``: list of species names (strings).
    * ``edges_map``: ``{edge_id: (tail_dict, head_dict)}``.
    * ``flow_map``: ``{edge_id: int}``.

    If ``flow`` is ``None``, the default is 1 for every edge.

    :param hg: Hypergraph describing the CRN.
    :type hg: CRNHyperGraph
    :param flow: Optional flow map (edge_id -> multiplicity).
    :type flow: Mapping[str, int] or None
    :returns: Tuple ``(vertices, edges_map, flow_map)``.
    :rtype: Tuple[list[str], dict, dict]

    Examples
    --------
    .. code-block:: python

        vertices, edges, flow = hypergraph_to_pr_inputs(hg)
        pr = PathwayRealizability().load_hypergraph_and_flow(vertices, edges, flow)
    """
    vertices = list(hg.species_list())
    edges: Dict[str, Tuple[Dict[str, int], Dict[str, int]]] = {}
    flow_map: Dict[str, int] = {}

    for e in hg.edge_list():
        eid = getattr(e, "id", None) or getattr(e, "edge_id", None) or str(e)
        r_side = _side_to_dict(getattr(e, "reactants", getattr(e, "lhs", None)))
        p_side = _side_to_dict(getattr(e, "products", getattr(e, "rhs", None)))
        edges[eid] = (r_side, p_side)
        if flow is not None and eid in flow:
            flow_map[eid] = int(flow[eid])
        else:
            flow_map[eid] = 1

    return vertices, edges, flow_map


def run_realizability_from_rxn_strings(
    rxns: Iterable[str],
    flow: Optional[Mapping[str, int]] = None,
    verbose: bool = True,
) -> Tuple[PathwayRealizability, Dict[str, object]]:
    """
    Convenience harness:

    * Build a :class:`CRNHyperGraph` from reaction strings.
    * Convert to PathwayRealizability inputs.
    * Build the Petri net and run König + BFS realizability checks.

    :param rxns: Iterable of reaction strings (``"A + B >> C"`` etc.).
    :type rxns: Iterable[str]
    :param flow: Optional flow map (edge_id -> multiplicity). If omitted,
        all edges receive flow 1 in order of creation.
    :type flow: Mapping[str, int] or None
    :param verbose: If ``True``, print a small summary to stdout.
    :type verbose: bool
    :returns: Tuple ``(pr, info)`` where ``pr`` is the configured
        :class:`PathwayRealizability` instance and ``info`` is a dict
        with keys ``"konig"``, ``"bfs"``, ``"certificate"``.
    :rtype: Tuple[PathwayRealizability, Dict[str, object]]

    Examples
    --------
    .. code-block:: python

        rxns = [">>A", "A>>B", "B>>"]
        pr, info = run_realizability_from_rxn_strings(rxns)
        print(info)
    """
    hg = CRNHyperGraph()
    hg.parse_rxns(list(rxns))

    vertices, edges, flow_map = hypergraph_to_pr_inputs(hg, flow=flow)
    pr = PathwayRealizability()
    pr.load_hypergraph_and_flow(vertices=vertices, edges=edges, flow=flow_map)
    pr.build_petri_net_from_flow()

    if verbose:
        print("Edges:")
        for eid, (t, h) in edges.items():
            print(f"  {eid}: {t} >> {h}")

    konig_ok = pr.is_realizable_via_konig()
    bfs_ok, cert = pr.is_realizable()
    if verbose:
        print("König sufficient test:", konig_ok)
        print("BFS realizable:", bfs_ok)
        print("Firing certificate:", cert)

    info: Dict[str, object] = {
        "konig": konig_ok,
        "bfs": bfs_ok,
        "certificate": cert,
    }
    return pr, info
