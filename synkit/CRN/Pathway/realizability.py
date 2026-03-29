from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

import itertools
import json

import networkx as nx

from ..Petrinet import Multiset, PetriNet, Place, TransitionId
from ._adapter import tokenize_syncrn_incidence


@dataclass
class RealizabilityConfig:
    """
    Configuration for bounded realizability search.

    :param max_states:
        Maximum number of BFS states explored during exact realizability
        search.
    :type max_states: int
    :param max_depth:
        Maximum firing-sequence depth explored during exact realizability
        search.
    :type max_depth: int

    Example
    -------
    .. code-block:: python

        cfg = RealizabilityConfig(
            max_states=200_000,
            max_depth=5_000,
        )
    """

    max_states: int = 100_000
    max_depth: int = 10_000


@dataclass
class RealizabilitySummary:
    """
    Small serializable summary of the active realizability instance.

    :param n_species:
        Number of species currently loaded in the instance.
    :type n_species: int
    :param n_reactions:
        Number of reactions currently loaded in the instance.
    :type n_reactions: int
    :param active_flow:
        Active reaction flow restricted to reactions with positive requested
        firing count.
    :type active_flow: Dict[str, int]
    :param initial_marking:
        Initial marking supplied by the user before Petri-net augmentation.
    :type initial_marking: Dict[str, int]
    :param goal_exact:
        Exact target marking constraints currently imposed on the auxiliary
        Petri-net places.
    :type goal_exact: Dict[str, int]
    :param goal_atleast:
        Lower-bound target marking constraints currently imposed on the
        auxiliary Petri-net places.
    :type goal_atleast: Dict[str, int]

    Example
    -------
    .. code-block:: python

        summary = pr.summary()
        print(summary.n_species)
        print(summary.active_flow)
    """

    n_species: int
    n_reactions: int
    active_flow: Dict[str, int] = field(default_factory=dict)
    initial_marking: Dict[str, int] = field(default_factory=dict)
    goal_exact: Dict[str, int] = field(default_factory=dict)
    goal_atleast: Dict[str, int] = field(default_factory=dict)


class PathwayRealizability:
    """
    Exact flow-realizability utilities for SynCRN-like inputs.

    A pathway flow is realizable if there exists an ordering of reaction
    firings such that:

    - each selected reaction is fired exactly the requested number of times,
    - the execution starts from the supplied initial marking,
    - every firing respects stoichiometric token consumption and production.

    Unspecified reactions default to flow 0.

    The class supports:

    - loading directly from tokenized hypergraph-style inputs,
    - loading from a SynCRN-like object,
    - exact bounded BFS realizability testing,
    - a sufficient acyclicity test via a König-style construction,
    - scaled realizability,
    - borrow realizability.

    :param config:
        Optional configuration for bounded realizability search. If ``None``,
        a default :class:`RealizabilityConfig` is used.
    :type config: Optional[RealizabilityConfig]

    Example
    -------
    .. code-block:: python

        pr = PathwayRealizability().load_syncrn_and_flow(
            syn,
            flow={"12": 1, "13": 1},
            species="label",
            reaction="id",
        )
        pr.build_petri_net_from_flow()
        ok, cert = pr.is_realizable()
        print(ok, cert)
    """

    def __init__(self, config: Optional[RealizabilityConfig] = None) -> None:
        """
        Initialize an empty pathway realizability instance.

        :param config:
            Optional configuration for bounded realizability search.
        :type config: Optional[RealizabilityConfig]
        :returns:
            None.
        :rtype: None
        """
        self.vertices: Set[str] = set()
        self.edges: Dict[str, Tuple[Multiset, Multiset]] = {}
        self.flow: Dict[str, int] = {}
        self.species_token_to_id: Dict[str, str] = {}
        self.reaction_token_to_id: Dict[str, str] = {}
        self.species_id_to_token: Dict[str, str] = {}
        self.reaction_id_to_token: Dict[str, str] = {}
        self._provided_initial_marking: Dict[str, int] = {}
        self._petri: Optional[PetriNet] = None
        self._initial_marking: Optional[Dict[Place, int]] = None
        self._goal_exact: Dict[Place, int] = {}
        self._goal_atleast: Dict[Place, int] = {}
        self._certificate: Optional[List[TransitionId]] = None
        self._config = config or RealizabilityConfig()

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_multiset(data: Optional[Mapping[str, int]]) -> Dict[str, int]:
        """
        Normalize a multiset mapping by dropping zero entries.

        Negative multiplicities are rejected.

        :param data:
            Input multiset or ``None``.
        :type data: Optional[Mapping[str, int]]
        :returns:
            Cleaned multiset containing only strictly positive entries.
        :rtype: Dict[str, int]
        :raises ValueError:
            If a negative multiplicity is encountered.
        """
        if data is None:
            return {}
        out: Dict[str, int] = {}
        for k, v in data.items():
            n = int(v)
            if n < 0:
                raise ValueError(f"Negative multiplicity not allowed: {k!r} -> {n}")
            if n > 0:
                out[str(k)] = n
        return out

    def load_hypergraph_and_flow(
        self,
        vertices: Iterable[str],
        edges: Mapping[str, Tuple[Mapping[str, int], Mapping[str, int]]],
        flow: Mapping[str, int],
        *,
        initial_marking: Optional[Mapping[str, int]] = None,
    ) -> "PathwayRealizability":
        """
        Load tokenized hypergraph data and a requested pathway flow.

        The ``edges`` mapping is interpreted as
        ``reaction_id -> (tail_multiset, head_multiset)``. Reactions not
        present in ``flow`` default to 0. Negative flow values are excluded
        here and validated again later during Petri-net construction.

        :param vertices:
            Iterable of species tokens.
        :type vertices: Iterable[str]
        :param edges:
            Mapping from reaction token to ``(tail, head)`` multisets.
        :type edges: Mapping[str, Tuple[Mapping[str, int], Mapping[str, int]]]
        :param flow:
            Requested reaction firing counts.
        :type flow: Mapping[str, int]
        :param initial_marking:
            Optional initial species marking.
        :type initial_marking: Optional[Mapping[str, int]]
        :returns:
            The current instance.
        :rtype: PathwayRealizability

        Example
        -------
        .. code-block:: python

            pr = PathwayRealizability().load_hypergraph_and_flow(
                vertices=["A", "B", "C"],
                edges={
                    "r1": ({"A": 1}, {"B": 1}),
                    "r2": ({"B": 1}, {"C": 1}),
                },
                flow={"r1": 1, "r2": 1},
                initial_marking={"A": 1},
            )
        """
        self.vertices = {str(v) for v in vertices}
        self.edges = {
            str(eid): (
                {str(s): int(v) for s, v in tail.items() if int(v) > 0},
                {str(s): int(v) for s, v in head.items() if int(v) > 0},
            )
            for eid, (tail, head) in edges.items()
        }
        self.flow = {
            str(eid): int(flow.get(eid, 0))
            for eid in self.edges
            if int(flow.get(eid, 0)) >= 0
        }
        self.species_token_to_id = {v: v for v in self.vertices}
        self.reaction_token_to_id = {eid: eid for eid in self.edges}
        self.species_id_to_token = {v: v for v in self.vertices}
        self.reaction_id_to_token = {eid: eid for eid in self.edges}
        self._provided_initial_marking = self._clean_multiset(initial_marking)
        self._petri = None
        self._initial_marking = None
        self._goal_exact = {}
        self._goal_atleast = {}
        self._certificate = None
        return self

    def _resolve_flow_tokens(
        self,
        incidence_reaction_order: List[str],
        reaction_token_to_id: Mapping[str, str],
        reaction_labels: Mapping[str, str],
        reaction_source_node_ids: Mapping[str, object],
        provided: Mapping[str, int],
    ) -> Dict[str, int]:
        """
        Resolve user-supplied flow keys onto reaction tokens.

        Keys may match reaction token, internal id, label, or source node id.
        Missing reactions default to 0.

        :param incidence_reaction_order:
            Canonical reaction-id order from the incidence view.
        :type incidence_reaction_order: List[str]
        :param reaction_token_to_id:
            Mapping from reaction token to internal reaction id.
        :type reaction_token_to_id: Mapping[str, str]
        :param reaction_labels:
            Mapping from reaction id to human-readable label.
        :type reaction_labels: Mapping[str, str]
        :param reaction_source_node_ids:
            Mapping from reaction id to source node id.
        :type reaction_source_node_ids: Mapping[str, object]
        :param provided:
            User-provided flow mapping.
        :type provided: Mapping[str, int]
        :returns:
            Flow map keyed by reaction token.
        :rtype: Dict[str, int]
        """
        flow_map: Dict[str, int] = {}
        id_to_token = {rid: tok for tok, rid in reaction_token_to_id.items()}
        for rid in incidence_reaction_order:
            rtoken = id_to_token[rid]
            label = reaction_labels.get(rid, rid)
            source = str(reaction_source_node_ids.get(rid, rid))
            if rtoken in provided:
                flow_map[rtoken] = int(provided[rtoken])
            elif rid in provided:
                flow_map[rtoken] = int(provided[rid])
            elif label in provided:
                flow_map[rtoken] = int(provided[label])
            elif source in provided:
                flow_map[rtoken] = int(provided[source])
            else:
                flow_map[rtoken] = 0
        return flow_map

    def _resolve_species_marking_tokens(
        self,
        incidence_species_order: List[str],
        species_token_to_id: Mapping[str, str],
        species_labels: Mapping[str, str],
        species_source_node_ids: Mapping[str, object],
        provided: Mapping[str, int],
    ) -> Dict[str, int]:
        """
        Resolve user-supplied initial marking keys onto species tokens.

        Keys may match species token, internal id, label, or source node id.
        Missing species default to 0 and are dropped from the returned mapping.

        :param incidence_species_order:
            Canonical species-id order from the incidence view.
        :type incidence_species_order: List[str]
        :param species_token_to_id:
            Mapping from species token to internal species id.
        :type species_token_to_id: Mapping[str, str]
        :param species_labels:
            Mapping from species id to human-readable label.
        :type species_labels: Mapping[str, str]
        :param species_source_node_ids:
            Mapping from species id to source node id.
        :type species_source_node_ids: Mapping[str, object]
        :param provided:
            User-provided initial marking.
        :type provided: Mapping[str, int]
        :returns:
            Initial marking keyed by species token.
        :rtype: Dict[str, int]
        """
        marking_map: Dict[str, int] = {}
        id_to_token = {sid: tok for tok, sid in species_token_to_id.items()}
        for sid in incidence_species_order:
            stok = id_to_token[sid]
            label = species_labels.get(sid, sid)
            source = str(species_source_node_ids.get(sid, sid))
            if stok in provided:
                marking_map[stok] = int(provided[stok])
            elif sid in provided:
                marking_map[stok] = int(provided[sid])
            elif label in provided:
                marking_map[stok] = int(provided[label])
            elif source in provided:
                marking_map[stok] = int(provided[source])
            else:
                marking_map[stok] = 0
        return {k: v for k, v in marking_map.items() if int(v) > 0}

    def load_syncrn_and_flow(
        self,
        crn: object,
        flow: Optional[Mapping[str, int]] = None,
        *,
        initial_marking: Optional[Mapping[str, int]] = None,
        species: str = "label",
        reaction: str = "id",
    ) -> "PathwayRealizability":
        """
        Load a SynCRN-like object together with a requested pathway flow.

        The SynCRN object is tokenized via :func:`tokenize_syncrn_incidence`.
        User-supplied flow and marking keys are resolved flexibly against token,
        internal id, label, and source node id.

        :param crn:
            SynCRN-like object.
        :type crn: object
        :param flow:
            Requested reaction firing counts.
        :type flow: Optional[Mapping[str, int]]
        :param initial_marking:
            Optional initial species marking.
        :type initial_marking: Optional[Mapping[str, int]]
        :param species:
            Species tokenization mode.
        :type species: str
        :param reaction:
            Reaction tokenization mode.
        :type reaction: str
        :returns:
            The current instance.
        :rtype: PathwayRealizability

        Example
        -------
        .. code-block:: python

            pr = PathwayRealizability().load_syncrn_and_flow(
                syn,
                flow={"12": 1, "13": 1},
                initial_marking={"A": 2},
                species="label",
                reaction="id",
            )
        """
        vertices, edges, species_token_to_id, reaction_token_to_id, incidence = (
            tokenize_syncrn_incidence(crn, species=species, reaction=reaction)
        )

        provided_flow = (
            {} if flow is None else {str(k): int(v) for k, v in flow.items()}
        )
        flow_map = self._resolve_flow_tokens(
            incidence.reaction_order,
            reaction_token_to_id,
            incidence.reaction_labels,
            incidence.reaction_source_node_ids,
            provided_flow,
        )

        provided_marking = (
            {}
            if initial_marking is None
            else {str(k): int(v) for k, v in initial_marking.items()}
        )
        marking_map = self._resolve_species_marking_tokens(
            incidence.species_order,
            species_token_to_id,
            incidence.species_labels,
            incidence.species_source_node_ids,
            provided_marking,
        )

        self.load_hypergraph_and_flow(
            vertices,
            edges,
            flow_map,
            initial_marking=marking_map,
        )
        self.species_token_to_id = species_token_to_id
        self.reaction_token_to_id = reaction_token_to_id
        self.species_id_to_token = {
            sid: tok for tok, sid in species_token_to_id.items()
        }
        self.reaction_id_to_token = {
            rid: tok for tok, rid in reaction_token_to_id.items()
        }
        return self

    # ------------------------------------------------------------------
    # Petri net construction
    # ------------------------------------------------------------------

    def set_initial_marking(self, marking: Mapping[str, int]) -> "PathwayRealizability":
        """
        Replace the user-provided initial marking.

        This resets any previously built Petri-net instance, goals, and cached
        firing certificate.

        :param marking:
            New initial marking.
        :type marking: Mapping[str, int]
        :returns:
            The current instance.
        :rtype: PathwayRealizability
        """
        self._provided_initial_marking = self._clean_multiset(marking)
        self._petri = None
        self._initial_marking = None
        self._goal_exact = {}
        self._goal_atleast = {}
        self._certificate = None
        return self

    def build_petri_net_from_flow(self) -> "PathwayRealizability":
        """
        Build an augmented Petri net encoding the requested pathway flow.

        For every active reaction ``eid`` with requested flow ``f > 0``, two
        auxiliary places are created:

        - ``__ext__{eid}``: supply place initialized with ``f`` tokens,
        - ``__target__{eid}``: target place expected to accumulate ``f`` tokens.

        The reaction transition consumes one token from the supply place on each
        firing and produces one token in the target place on each firing. This
        enforces exact realization of the requested number of firings.

        :returns:
            The current instance.
        :rtype: PathwayRealizability
        :raises RuntimeError:
            If no pathway data has been loaded.
        :raises ValueError:
            If a negative flow value is encountered.

        Example
        -------
        .. code-block:: python

            pr = PathwayRealizability().load_syncrn_and_flow(
                syn,
                flow={"12": 1, "13": 1},
            )
            pr.build_petri_net_from_flow()
            print(pr.initial_marking)
            print(pr.goal_exact)
        """
        if not self.edges:
            raise RuntimeError(
                "No pathway data loaded; call load_syncrn_and_flow() or load_hypergraph_and_flow()."
            )

        net = PetriNet()
        m0: Dict[Place, int] = defaultdict(int)
        goal_exact: Dict[Place, int] = {}

        for v in sorted(self.vertices):
            net.add_place(v, label=v, source_node_id=v)
            m0[v] = int(self._provided_initial_marking.get(v, 0))

        for eid in sorted(self.edges):
            fval = int(self.flow.get(eid, 0))
            if fval < 0:
                raise ValueError(f"Flow must be non-negative, got {eid!r} -> {fval}")
            if fval == 0:
                continue

            tail, head = self.edges[eid]
            pre = {v: int(w) for v, w in tail.items() if int(w) > 0}
            post = {v: int(w) for v, w in head.items() if int(w) > 0}

            ext = f"__ext__{eid}"
            tgt = f"__target__{eid}"
            net.add_place(ext, label=ext)
            net.add_place(tgt, label=tgt)

            pre_with_supply = dict(pre)
            pre_with_supply[ext] = 1
            post_with_target = dict(post)
            post_with_target[tgt] = 1

            net.add_transition(
                eid,
                pre_with_supply,
                post_with_target,
                label=eid,
                source_reaction_id=self.reaction_token_to_id.get(eid, eid),
            )

            m0[ext] = fval
            goal_exact[ext] = 0
            goal_exact[tgt] = fval
            m0[tgt] = 0

        self._petri = net
        self._initial_marking = dict(m0)
        self._goal_exact = goal_exact
        self._goal_atleast = {}
        self._certificate = None
        return self

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def petri(self) -> PetriNet:
        """
        Return the active Petri net.

        :returns:
            Augmented Petri net for realizability checking.
        :rtype: PetriNet
        :raises RuntimeError:
            If the Petri net has not yet been built.
        """
        if self._petri is None:
            raise RuntimeError("Petri net not built; call build_petri_net_from_flow().")
        return self._petri

    @property
    def initial_marking(self) -> Dict[Place, int]:
        """
        Return the active initial marking of the augmented Petri net.

        :returns:
            Initial marking.
        :rtype: Dict[Place, int]
        :raises RuntimeError:
            If the initial marking has not yet been initialized.
        """
        if self._initial_marking is None:
            raise RuntimeError("Initial marking missing; build Petri net first.")
        return self._initial_marking

    @property
    def goal_exact(self) -> Dict[Place, int]:
        """
        Return exact target marking constraints.

        :returns:
            Exact goal marking constraints.
        :rtype: Dict[Place, int]
        """
        return dict(self._goal_exact)

    @property
    def goal_atleast(self) -> Dict[Place, int]:
        """
        Return lower-bound target marking constraints.

        :returns:
            Lower-bound goal marking constraints.
        :rtype: Dict[Place, int]
        """
        return dict(self._goal_atleast)

    @property
    def certificate(self) -> Optional[List[TransitionId]]:
        """
        Return the most recently found firing certificate.

        :returns:
            Realizing firing sequence, or ``None`` if none is cached.
        :rtype: Optional[List[TransitionId]]
        """
        return self._certificate

    def summary(self) -> RealizabilitySummary:
        """
        Return a compact summary of the current realizability instance.

        :returns:
            Serializable instance summary.
        :rtype: RealizabilitySummary

        Example
        -------
        .. code-block:: python

            info = pr.summary()
            print(info.active_flow)
        """
        return RealizabilitySummary(
            n_species=len(self.vertices),
            n_reactions=len(self.edges),
            active_flow={k: v for k, v in self.flow.items() if int(v) > 0},
            initial_marking=dict(self._provided_initial_marking),
            goal_exact=dict(self._goal_exact),
            goal_atleast=dict(self._goal_atleast),
        )

    # ------------------------------------------------------------------
    # goal helpers
    # ------------------------------------------------------------------

    def _goal_reached(self, marking: Mapping[str, int]) -> bool:
        """
        Test whether a marking satisfies the active goal constraints.

        Exact constraints in ``_goal_exact`` must match exactly, while lower
        bounds in ``_goal_atleast`` must be met or exceeded.

        :param marking:
            Candidate marking.
        :type marking: Mapping[str, int]
        :returns:
            ``True`` if the marking satisfies all active goals.
        :rtype: bool
        """
        for p, val in self._goal_exact.items():
            if int(marking.get(p, 0)) != int(val):
                return False
        for p, val in self._goal_atleast.items():
            if int(marking.get(p, 0)) < int(val):
                return False
        return True

    # ------------------------------------------------------------------
    # bounded BFS
    # ------------------------------------------------------------------

    def is_realizable(
        self,
        max_states: Optional[int] = None,
        max_depth: Optional[int] = None,
    ) -> Tuple[bool, Optional[List[TransitionId]]]:
        """
        Test exact realizability by bounded breadth-first search.

        The search explores reachable Petri-net markings while recording firing
        sequences. A certificate is returned when a goal-satisfying marking is
        found.

        :param max_states:
            Optional override for the maximum number of explored states.
        :type max_states: Optional[int]
        :param max_depth:
            Optional override for the maximum explored firing depth.
        :type max_depth: Optional[int]
        :returns:
            Pair ``(is_realizable, certificate)`` where ``certificate`` is a
            realizing transition sequence if one is found.
        :rtype: Tuple[bool, Optional[List[TransitionId]]]

        Example
        -------
        .. code-block:: python

            ok, cert = pr.is_realizable(max_states=50000, max_depth=2000)
            print(ok)
            print(cert)
        """
        net = self.petri
        m0 = dict(self.initial_marking)

        max_states = max_states if max_states is not None else self._config.max_states
        max_depth = max_depth if max_depth is not None else self._config.max_depth

        if self._goal_reached(m0):
            self._certificate = []
            return True, []

        start = net.marking_to_tuple(m0)
        q: deque[Tuple[Tuple[int, ...], List[TransitionId]]] = deque([(start, [])])
        visited = {start}
        states = 0

        while q:
            mtuple, seq = q.popleft()
            states += 1
            if states > max_states:
                break
            if len(seq) > max_depth:
                continue

            marking = net.tuple_to_marking(mtuple)
            for tid in net.transition_order:
                if not net.enabled(marking, tid):
                    continue
                new_mark = net.fire(marking, tid)
                if self._goal_reached(new_mark):
                    seq2 = seq + [tid]
                    self._certificate = seq2
                    return True, seq2
                new_tuple = net.marking_to_tuple(new_mark)
                if new_tuple not in visited:
                    visited.add(new_tuple)
                    q.append((new_tuple, seq + [tid]))

        self._certificate = None
        return False, None

    # ------------------------------------------------------------------
    # König sufficient test
    # ------------------------------------------------------------------

    def is_realizable_via_konig(self) -> bool:
        """
        Apply a König-style sufficient acyclicity test.

        A directed bipartite dependency graph is built using:

        - species vertices ``("v", species)``
        - active-reaction vertices ``("e", reaction)``

        with edges:

        - species -> reaction for reactant incidence,
        - reaction -> species for product incidence.

        If this dependency graph is acyclic, the active flow is certified
        realizable by this sufficient test.

        :returns:
            ``True`` if the sufficient acyclicity condition holds, else
            ``False``.
        :rtype: bool

        Example
        -------
        .. code-block:: python

            if pr.is_realizable_via_konig():
                print("Guaranteed realizable by sufficient test")
        """
        active_edges = {
            eid: self.edges[eid] for eid, fval in self.flow.items() if int(fval) > 0
        }
        vertices: Set[str] = set()
        for tail, head in active_edges.values():
            vertices.update(tail.keys())
            vertices.update(head.keys())

        g = nx.DiGraph()
        for v in vertices:
            g.add_node(("v", v))
        for eid in active_edges:
            g.add_node(("e", eid))
        for eid, (tail, head) in active_edges.items():
            for v in tail:
                g.add_edge(("v", v), ("e", eid))
            for v in head:
                g.add_edge(("e", eid), ("v", v))
        return nx.is_directed_acyclic_graph(g)

    # ------------------------------------------------------------------
    # scaled / borrow realizability
    # ------------------------------------------------------------------

    def is_scaled_realizable(self, k_max: int = 4) -> Tuple[bool, Optional[int]]:
        """
        Test whether a scaled version of the requested flow is realizable.

        For each integer ``k`` from 1 to ``k_max``, the requested flow is scaled
        to ``k * flow`` and tested for exact realizability.

        :param k_max:
            Maximum scale factor to try.
        :type k_max: int
        :returns:
            Pair ``(success, k)`` where ``k`` is the first successful scale
            factor, or ``None`` if no tested scale succeeds.
        :rtype: Tuple[bool, Optional[int]]

        Example
        -------
        .. code-block:: python

            ok, k = pr.is_scaled_realizable(k_max=5)
            print(ok, k)
        """
        saved_flow = dict(self.flow)
        for k in range(1, k_max + 1):
            self.flow = {eid: k * int(v) for eid, v in saved_flow.items()}
            self.build_petri_net_from_flow()
            ok, _ = self.is_realizable()
            if ok:
                self.flow = saved_flow
                self.build_petri_net_from_flow()
                return True, k
        self.flow = saved_flow
        self.build_petri_net_from_flow()
        return False, None

    def is_borrow_realizable(
        self,
        max_borrow_each: int = 2,
    ) -> Tuple[bool, Optional[Mapping[str, int]]]:
        """
        Test realizability when temporary borrowed initial tokens are allowed.

        Every combination of per-species borrowed tokens from 0 up to
        ``max_borrow_each`` is tried. For a candidate borrow multiset, the
        borrowed tokens are added to the initial marking and then imposed again
        as lower-bound final goals in ``goal_atleast`` so that the borrowed
        material must be returned.

        :param max_borrow_each:
            Maximum borrowed amount tested independently for each species.
        :type max_borrow_each: int
        :returns:
            Pair ``(success, borrow)`` where ``borrow`` is the first successful
            borrow multiset, or ``None`` if no tested borrow succeeds.
        :rtype: Tuple[bool, Optional[Mapping[str, int]]]

        Example
        -------
        .. code-block:: python

            ok, borrow = pr.is_borrow_realizable(max_borrow_each=1)
            print(ok, borrow)
        """
        saved_initial = dict(self._provided_initial_marking)
        species = sorted(self.vertices)

        for comb in itertools.product(range(max_borrow_each + 1), repeat=len(species)):
            borrow = {s: comb[i] for i, s in enumerate(species) if comb[i] > 0}
            self._provided_initial_marking = dict(saved_initial)
            for s, val in borrow.items():
                self._provided_initial_marking[s] = (
                    self._provided_initial_marking.get(s, 0) + val
                )
            self.build_petri_net_from_flow()
            self._goal_atleast = dict(borrow)
            ok, _ = self.is_realizable()
            if ok:
                self._provided_initial_marking = saved_initial
                self.build_petri_net_from_flow()
                return True, borrow

        self._provided_initial_marking = saved_initial
        self.build_petri_net_from_flow()
        return False, None

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------

    def export_pnml(self, fn: str) -> "PathwayRealizability":
        """
        Export the current augmented Petri net to a JSON-based PNML-like file.

        The exported structure contains places, transitions, initial marking,
        and active goal constraints. Despite the method name, the file written
        here is JSON rather than XML PNML.

        :param fn:
            Output filename.
        :type fn: str
        :returns:
            The current instance.
        :rtype: PathwayRealizability

        Example
        -------
        .. code-block:: python

            pr.export_pnml("pathway_realizability.json")
        """
        self.build_petri_net_from_flow()
        net = self.petri
        data = {
            "places": net.place_order,
            "transitions": {
                tid: {"pre": t.pre, "post": t.post}
                for tid, t in net.transitions.items()
            },
            "initial_marking": self.initial_marking,
            "goal_exact": self._goal_exact,
            "goal_atleast": self._goal_atleast,
        }
        with open(fn, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        return self

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        """
        Return a compact representation.

        :returns:
            Readable summary string.
        :rtype: str
        """
        return (
            f"<PathwayRealizability species={len(self.vertices)} "
            f"reactions={len(self.edges)}>"
        )


def syncrn_to_pr_inputs(
    crn: object,
    flow: Optional[Mapping[str, int]] = None,
    *,
    initial_marking: Optional[Mapping[str, int]] = None,
    species: str = "label",
    reaction: str = "id",
) -> Tuple[
    List[str],
    Dict[str, Tuple[Dict[str, int], Dict[str, int]]],
    Dict[str, int],
    Dict[str, int],
]:
    """
    Convert a SynCRN-like object into tokenized pathway-realizability inputs.

    The returned tuple contains:

    - species tokens,
    - reaction-tokenized hyperedges,
    - resolved flow map keyed by reaction token,
    - resolved initial marking keyed by species token.

    :param crn:
        SynCRN-like object.
    :type crn: object
    :param flow:
        Optional requested reaction firing counts.
    :type flow: Optional[Mapping[str, int]]
    :param initial_marking:
        Optional initial marking.
    :type initial_marking: Optional[Mapping[str, int]]
    :param species:
        Species tokenization mode.
    :type species: str
    :param reaction:
        Reaction tokenization mode.
    :type reaction: str
    :returns:
        Tuple ``(vertices, edges, flow_map, marking_map)``.
    :rtype:
        Tuple[
            List[str],
            Dict[str, Tuple[Dict[str, int], Dict[str, int]]],
            Dict[str, int],
            Dict[str, int],
        ]

    Example
    -------
    .. code-block:: python

        vertices, edges, flow_map, marking_map = syncrn_to_pr_inputs(
            syn,
            flow={"12": 1},
            initial_marking={"A": 2},
            species="label",
            reaction="id",
        )
    """
    vertices, edges, species_token_to_id, reaction_token_to_id, incidence = (
        tokenize_syncrn_incidence(
            crn,
            species=species,
            reaction=reaction,
        )
    )

    dummy = PathwayRealizability()
    provided_flow = {} if flow is None else {str(k): int(v) for k, v in flow.items()}
    flow_map = dummy._resolve_flow_tokens(
        incidence.reaction_order,
        reaction_token_to_id,
        incidence.reaction_labels,
        incidence.reaction_source_node_ids,
        provided_flow,
    )
    provided_marking = (
        {}
        if initial_marking is None
        else {str(k): int(v) for k, v in initial_marking.items()}
    )
    marking_map = dummy._resolve_species_marking_tokens(
        incidence.species_order,
        species_token_to_id,
        incidence.species_labels,
        incidence.species_source_node_ids,
        provided_marking,
    )
    return vertices, edges, flow_map, marking_map


def run_realizability_from_syncrn(
    crn: object,
    flow: Optional[Mapping[str, int]] = None,
    *,
    initial_marking: Optional[Mapping[str, int]] = None,
    species: str = "label",
    reaction: str = "id",
    verbose: bool = True,
) -> Tuple[PathwayRealizability, Dict[str, object]]:
    """
    Run König-style and exact BFS realizability tests on a SynCRN-like object.

    This convenience wrapper:

    1. loads the SynCRN object and requested flow,
    2. builds the augmented Petri net,
    3. applies the sufficient König-style acyclicity test,
    4. runs exact bounded BFS realizability search,
    5. optionally prints a human-readable summary.

    :param crn:
        SynCRN-like object.
    :type crn: object
    :param flow:
        Optional requested reaction firing counts.
    :type flow: Optional[Mapping[str, int]]
    :param initial_marking:
        Optional initial marking.
    :type initial_marking: Optional[Mapping[str, int]]
    :param species:
        Species tokenization mode.
    :type species: str
    :param reaction:
        Reaction tokenization mode.
    :type reaction: str
    :param verbose:
        Whether to print a textual summary.
    :type verbose: bool
    :returns:
        Pair ``(pathway_realizability_instance, info_dict)``.
    :rtype: Tuple[PathwayRealizability, Dict[str, object]]

    Example
    -------
    .. code-block:: python

        pr, info = run_realizability_from_syncrn(
            syn,
            flow={"12": 1, "13": 1},
            initial_marking={"A": 2},
            verbose=True,
        )
        print(info["konig"])
        print(info["bfs"])
        print(info["certificate"])
    """
    pr = PathwayRealizability().load_syncrn_and_flow(
        crn,
        flow=flow,
        initial_marking=initial_marking,
        species=species,
        reaction=reaction,
    )
    pr.build_petri_net_from_flow()

    konig_ok = pr.is_realizable_via_konig()
    bfs_ok, cert = pr.is_realizable()

    if verbose:
        print("Active flow:")
        for eid, val in sorted(pr.flow.items()):
            if int(val) > 0:
                print(f"  {eid}: {val}")
        print("Initial marking:", dict(sorted(pr._provided_initial_marking.items())))
        print("König sufficient test:", konig_ok)
        print("BFS realizable:", bfs_ok)
        print("Firing certificate:", cert)

    info: Dict[str, object] = {
        "konig": konig_ok,
        "bfs": bfs_ok,
        "certificate": cert,
        "summary": pr.summary(),
    }
    return pr, info
