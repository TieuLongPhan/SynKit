from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

import json

from ._adapter import tokenize_syncrn_incidence

Multiset = Dict[str, int]


@dataclass
class ReachabilityConfig:
    """
    Configuration for forward reachability traversal.

    This configuration controls how many propagation layers are explored and
    whether the traversal should terminate as soon as no newly reachable
    species are discovered.

    :param max_layers:
        Maximum number of forward layers to compute.
    :type max_layers: int
    :param stop_when_no_new_species:
        If ``True``, stop once a layer produces no newly reachable species,
        even if reactions are still enabled.
    :type stop_when_no_new_species: bool

    Example
    -------
    .. code-block:: python

        cfg = ReachabilityConfig(
            max_layers=100,
            stop_when_no_new_species=True,
        )
    """

    max_layers: int = 10_000
    stop_when_no_new_species: bool = True


@dataclass
class ReachabilityLayer:
    """
    One forward reachability layer.

    A layer summarizes what became newly active at a given traversal depth.
    In set semantics, a reaction is considered enabled when all of its
    reactant species are already reachable. In multiset semantics, a reaction
    is enabled when the current marking contains sufficient multiplicity for
    each reactant.

    :param depth:
        Layer index, starting at ``1`` for the first propagation step.
    :type depth: int
    :param newly_enabled_reactions:
        Reaction identifiers that became enabled at this layer.
    :type newly_enabled_reactions: List[str]
    :param newly_reachable_species:
        Species that became reachable for the first time at this layer.
    :type newly_reachable_species: List[str]
    :param all_reachable_species:
        Complete set of reachable species after this layer is applied.
    :type all_reachable_species: List[str]

    Example
    -------
    .. code-block:: python

        layer = ReachabilityLayer(
            depth=2,
            newly_enabled_reactions=["r3", "r4"],
            newly_reachable_species=["E", "F"],
            all_reachable_species=["A", "B", "C", "D", "E", "F"],
        )
    """

    depth: int
    newly_enabled_reactions: List[str]
    newly_reachable_species: List[str]
    all_reachable_species: List[str]


@dataclass
class ReachabilityResult:
    """
    Container for forward reachability results.

    This object stores the initial support, the layered traversal trace, and
    the first depth at which each species or reaction became reachable or
    enabled.

    :param initial_species:
        Initial reachable species support used to seed the traversal.
    :type initial_species: List[str]
    :param layers:
        Ordered list of reachability layers.
    :type layers: List[ReachabilityLayer]
    :param species_first_depth:
        Mapping from species identifier to the first layer depth at which that
        species became reachable. Initial species are assigned depth ``0``.
    :type species_first_depth: Dict[str, int]
    :param reaction_first_depth:
        Mapping from reaction identifier to the first layer depth at which that
        reaction became enabled.
    :type reaction_first_depth: Dict[str, int]

    Example
    -------
    .. code-block:: python

        result = ReachabilityResult(
            initial_species=["A", "B"],
            layers=[],
            species_first_depth={"A": 0, "B": 0},
            reaction_first_depth={},
        )
    """

    initial_species: List[str]
    layers: List[ReachabilityLayer]
    species_first_depth: Dict[str, int]
    reaction_first_depth: Dict[str, int]


class PathwayReachability:
    """
    Forward reachability utilities for SynCRN pathway analysis.

    This class provides layered forward propagation over a reaction hypergraph.
    It supports two related semantics:

    - **Set semantics**:
      a reaction is enabled once all reactant species are present in the
      reachable set; stoichiometric multiplicities are ignored for enabling.
    - **Multiset semantics**:
      a reaction is enabled only when the current marking contains enough
      copies of each reactant species to satisfy stoichiometric coefficients.

    The class can be loaded either from tokenized ``vertices`` / ``edges``
    data or directly from a SynCRN-like object via :meth:`load_syncrn`.

    Internally, reactions are stored as a mapping

    ``reaction_id -> (tail_multiset, head_multiset)``

    where the tail is the reactant multiset and the head is the product
    multiset.

    :param config:
        Optional reachability configuration. If omitted, a default
        :class:`ReachabilityConfig` is used.
    :type config: Optional[ReachabilityConfig]

    Example
    -------
    .. code-block:: python

        rr = PathwayReachability()
        rr.load_hypergraph(
            vertices=["A", "B", "C", "D"],
            edges={
                "r1": ({"A": 1, "B": 1}, {"C": 1}),
                "r2": ({"C": 1}, {"D": 1}),
            },
        )

        result = rr.compute_layers_set(initial_species=["A", "B"])
        print(result.species_first_depth)
        # {'A': 0, 'B': 0, 'C': 1, 'D': 2}
    """

    def __init__(self, config: Optional[ReachabilityConfig] = None) -> None:
        """
        Initialize an empty reachability engine.

        :param config:
            Optional reachability configuration.
        :type config: Optional[ReachabilityConfig]
        """
        self.vertices: Set[str] = set()
        self.edges: Dict[str, Tuple[Multiset, Multiset]] = {}
        self.species_token_to_id: Dict[str, str] = {}
        self.reaction_token_to_id: Dict[str, str] = {}
        self.species_id_to_token: Dict[str, str] = {}
        self.reaction_id_to_token: Dict[str, str] = {}
        self._config = config or ReachabilityConfig()

    # ------------------------------------------------------------------
    # data loading
    # ------------------------------------------------------------------

    def load_hypergraph(
        self,
        vertices: Iterable[str],
        edges: Mapping[str, Tuple[Mapping[str, int], Mapping[str, int]]],
    ) -> "PathwayReachability":
        """
        Load a tokenized reaction hypergraph directly.

        Each edge must map a reaction identifier to a pair
        ``(tail_multiset, head_multiset)``, where both multisets map species
        identifiers to strictly positive stoichiometric coefficients.

        Non-positive coefficients are discarded during normalization.

        :param vertices:
            Iterable of species identifiers.
        :type vertices: Iterable[str]
        :param edges:
            Mapping from reaction identifier to
            ``(reactant_multiset, product_multiset)``.
        :type edges: Mapping[str, Tuple[Mapping[str, int], Mapping[str, int]]]
        :returns:
            The current instance, to allow fluent chaining.
        :rtype: PathwayReachability

        Example
        -------
        .. code-block:: python

            rr = PathwayReachability().load_hypergraph(
                vertices=["A", "B", "C"],
                edges={
                    "r1": ({"A": 1}, {"B": 1}),
                    "r2": ({"B": 1}, {"C": 1}),
                },
            )
        """
        self.vertices = set(str(v) for v in vertices)
        self.edges = {
            str(eid): (
                {str(s): int(v) for s, v in tail.items() if int(v) > 0},
                {str(s): int(v) for s, v in head.items() if int(v) > 0},
            )
            for eid, (tail, head) in edges.items()
        }
        self.species_token_to_id = {v: v for v in self.vertices}
        self.reaction_token_to_id = {eid: eid for eid in self.edges}
        self.species_id_to_token = {v: v for v in self.vertices}
        self.reaction_id_to_token = {eid: eid for eid in self.edges}
        return self

    def load_syncrn(
        self,
        crn: object,
        *,
        species: str = "label",
        reaction: str = "id",
    ) -> "PathwayReachability":
        """
        Load reachability data from a SynCRN-like object.

        This method delegates tokenization to
        :func:`tokenize_syncrn_incidence`, then stores both the tokenized
        hypergraph and the forward/backward token-id lookup tables.

        :param crn:
            SynCRN-like object to tokenize.
        :type crn: object
        :param species:
            Species attribute used during tokenization, such as ``"label"`` or
            another species node attribute supported by the adapter.
        :type species: str
        :param reaction:
            Reaction attribute used during tokenization, such as ``"id"``.
        :type reaction: str
        :returns:
            The current instance, to allow fluent chaining.
        :rtype: PathwayReachability

        Example
        -------
        .. code-block:: python

            rr = PathwayReachability().load_syncrn(
                syncrn_object,
                species="label",
                reaction="id",
            )
        """
        vertices, edges, species_token_to_id, reaction_token_to_id, _ = (
            tokenize_syncrn_incidence(
                crn,
                species=species,
                reaction=reaction,
            )
        )
        self.load_hypergraph(vertices, edges)
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
    # small helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_marking(marking: Mapping[str, int]) -> Dict[str, int]:
        """
        Normalize a marking into a clean positive multiset.

        Species with non-positive counts are removed.

        :param marking:
            Input species-count mapping.
        :type marking: Mapping[str, int]
        :returns:
            Normalized marking containing only strictly positive counts.
        :rtype: Dict[str, int]
        """
        return {str(s): int(v) for s, v in marking.items() if int(v) > 0}

    @staticmethod
    def _support(marking: Mapping[str, int]) -> Set[str]:
        """
        Return the support of a marking.

        The support is the set of species whose count is strictly positive.

        :param marking:
            Input species-count mapping.
        :type marking: Mapping[str, int]
        :returns:
            Set of species present with positive multiplicity.
        :rtype: Set[str]
        """
        return {s for s, v in marking.items() if int(v) > 0}

    def _enabled_reactions_for_set(self, reachable: Set[str]) -> List[str]:
        """
        Return reactions enabled under set semantics.

        A reaction is enabled if all reactant species are already members of
        the reachable set. Stoichiometric coefficients are ignored for the
        enabling condition.

        :param reachable:
            Current reachable species set.
        :type reachable: Set[str]
        :returns:
            List of enabled reaction identifiers.
        :rtype: List[str]
        """
        enabled: List[str] = []
        for eid, (tail, _) in self.edges.items():
            if set(tail.keys()).issubset(reachable):
                enabled.append(eid)
        return enabled

    def _enabled_reactions_for_marking(self, marking: Mapping[str, int]) -> List[str]:
        """
        Return reactions enabled under multiset semantics.

        A reaction is enabled if the marking contains at least the required
        stoichiometric coefficient for every reactant species.

        :param marking:
            Current species marking.
        :type marking: Mapping[str, int]
        :returns:
            List of enabled reaction identifiers.
        :rtype: List[str]
        """
        enabled: List[str] = []
        for eid, (tail, _) in self.edges.items():
            if all(int(marking.get(s, 0)) >= int(coeff) for s, coeff in tail.items()):
                enabled.append(eid)
        return enabled

    def _fire_reactions_once_in_batch(
        self,
        marking: Mapping[str, int],
        reaction_ids: Iterable[str],
    ) -> Dict[str, int]:
        """
        Fire a batch of reactions once each from the given marking.

        Reactants are consumed and products are produced exactly once for each
        reaction in ``reaction_ids``. The resulting marking is normalized so
        that non-positive counts are removed.

        :param marking:
            Input marking before firing.
        :type marking: Mapping[str, int]
        :param reaction_ids:
            Iterable of reaction identifiers to fire once.
        :type reaction_ids: Iterable[str]
        :returns:
            Updated normalized marking after the batch firing.
        :rtype: Dict[str, int]
        """
        nxt = dict(marking)
        for eid in reaction_ids:
            tail, head = self.edges[eid]
            for s, coeff in tail.items():
                nxt[s] = nxt.get(s, 0) - int(coeff)
            for s, coeff in head.items():
                nxt[s] = nxt.get(s, 0) + int(coeff)
        return self._normalize_marking(nxt)

    @staticmethod
    def _update_first_depth_maps(
        *,
        depth: int,
        enabled_now: Iterable[str],
        new_species: Set[str],
        seen_reactions: Set[str],
        reaction_first_depth: Dict[str, int],
        species_first_depth: Dict[str, int],
    ) -> None:
        """
        Update first-seen depth maps for reactions and species.

        Reactions are assigned a first depth only once, when first observed as
        enabled. Species are assigned a first depth when they first appear in
        the newly reached set.

        :param depth:
            Current traversal depth.
        :type depth: int
        :param enabled_now:
            Reactions enabled at the current depth.
        :type enabled_now: Iterable[str]
        :param new_species:
            Species that became newly reachable at the current depth.
        :type new_species: Set[str]
        :param seen_reactions:
            Mutable set of reactions already assigned a first depth.
        :type seen_reactions: Set[str]
        :param reaction_first_depth:
            Output mapping from reaction identifier to first enabling depth.
        :type reaction_first_depth: Dict[str, int]
        :param species_first_depth:
            Output mapping from species identifier to first reachability depth.
        :type species_first_depth: Dict[str, int]
        """
        for eid in enabled_now:
            if eid not in seen_reactions:
                reaction_first_depth[eid] = depth
                seen_reactions.add(eid)
        for s in sorted(new_species):
            species_first_depth[s] = depth

    # ------------------------------------------------------------------
    # main computations
    # ------------------------------------------------------------------

    def compute_layers_set(
        self,
        initial_species: Iterable[str],
        max_layers: Optional[int] = None,
    ) -> ReachabilityResult:
        """
        Compute layered forward reachability in set semantics.

        In this mode, only species presence matters. Stoichiometric
        multiplicities are ignored when deciding whether a reaction is enabled.

        The algorithm proceeds layer by layer:
        1. find all reactions whose reactants are already reachable,
        2. keep only reactions not previously seen as enabled,
        3. collect all products they produce,
        4. add any newly produced species to the reachable set.

        :param initial_species:
            Initial set of reachable species.
        :type initial_species: Iterable[str]
        :param max_layers:
            Optional layer cap overriding the instance configuration.
        :type max_layers: Optional[int]
        :returns:
            Reachability result containing the full layered traversal trace.
        :rtype: ReachabilityResult

        Example
        -------
        .. code-block:: python

            rr = PathwayReachability().load_hypergraph(
                vertices=["A", "B", "C", "D"],
                edges={
                    "r1": ({"A": 1, "B": 1}, {"C": 1}),
                    "r2": ({"C": 1}, {"D": 1}),
                },
            )

            result = rr.compute_layers_set(initial_species=["A", "B"])

            for layer in result.layers:
                print(layer.depth, layer.newly_reachable_species)

            # 1 ['C']
            # 2 ['D']
        """
        max_layers = max_layers if max_layers is not None else self._config.max_layers
        reachable: Set[str] = {str(x) for x in initial_species}
        seen_reactions: Set[str] = set()

        species_first_depth: Dict[str, int] = {s: 0 for s in reachable}
        reaction_first_depth: Dict[str, int] = {}
        layers: List[ReachabilityLayer] = []

        for depth in range(1, max_layers + 1):
            newly_enabled = [
                eid
                for eid in self._enabled_reactions_for_set(reachable)
                if eid not in seen_reactions
            ]
            produced_this_round: Set[str] = set()
            for eid in newly_enabled:
                _, head = self.edges[eid]
                produced_this_round.update(head.keys())
            new_species = produced_this_round - reachable

            if not newly_enabled and not new_species:
                break

            self._update_first_depth_maps(
                depth=depth,
                enabled_now=newly_enabled,
                new_species=new_species,
                seen_reactions=seen_reactions,
                reaction_first_depth=reaction_first_depth,
                species_first_depth=species_first_depth,
            )
            reachable.update(new_species)

            layers.append(
                ReachabilityLayer(
                    depth=depth,
                    newly_enabled_reactions=sorted(newly_enabled),
                    newly_reachable_species=sorted(new_species),
                    all_reachable_species=sorted(reachable),
                )
            )

            if self._config.stop_when_no_new_species and not new_species:
                break

        return ReachabilityResult(
            initial_species=sorted(set(str(x) for x in initial_species)),
            layers=layers,
            species_first_depth=species_first_depth,
            reaction_first_depth=reaction_first_depth,
        )

    def compute_layers_multiset(
        self,
        initial_marking: Mapping[str, int],
        max_layers: Optional[int] = None,
    ) -> ReachabilityResult:
        """
        Compute layered forward reachability in multiset semantics.

        In this mode, species counts matter. A reaction is enabled only if the
        current marking contains enough multiplicity for every reactant. At
        each layer, all currently enabled reactions are fired once in batch.

        This procedure is useful when approximate token-flow behavior is
        desired, but it should not be confused with exhaustive Petri-net
        reachability analysis over all possible firing sequences.

        :param initial_marking:
            Initial species marking.
        :type initial_marking: Mapping[str, int]
        :param max_layers:
            Optional layer cap overriding the instance configuration.
        :type max_layers: Optional[int]
        :returns:
            Reachability result containing the full layered traversal trace.
        :rtype: ReachabilityResult

        Example
        -------
        .. code-block:: python

            rr = PathwayReachability().load_hypergraph(
                vertices=["A", "B", "C"],
                edges={
                    "r1": ({"A": 2}, {"B": 1}),
                    "r2": ({"B": 1}, {"C": 1}),
                },
            )

            result = rr.compute_layers_multiset(initial_marking={"A": 2})

            print(result.species_first_depth)
            # {'A': 0, 'B': 1, 'C': 2}
        """
        max_layers = max_layers if max_layers is not None else self._config.max_layers
        marking = self._normalize_marking(initial_marking)
        support = self._support(marking)

        species_first_depth: Dict[str, int] = {s: 0 for s in support}
        reaction_first_depth: Dict[str, int] = {}
        layers: List[ReachabilityLayer] = []
        seen_reactions: Set[str] = set()

        for depth in range(1, max_layers + 1):
            enabled_now = self._enabled_reactions_for_marking(marking)
            if not enabled_now:
                break

            next_marking = self._fire_reactions_once_in_batch(marking, enabled_now)
            next_support = self._support(next_marking)
            new_species = next_support - support

            self._update_first_depth_maps(
                depth=depth,
                enabled_now=enabled_now,
                new_species=new_species,
                seen_reactions=seen_reactions,
                reaction_first_depth=reaction_first_depth,
                species_first_depth=species_first_depth,
            )

            layers.append(
                ReachabilityLayer(
                    depth=depth,
                    newly_enabled_reactions=sorted(enabled_now),
                    newly_reachable_species=sorted(new_species),
                    all_reachable_species=sorted(next_support),
                )
            )

            marking = next_marking
            support = next_support

            if self._config.stop_when_no_new_species and not new_species:
                break

        return ReachabilityResult(
            initial_species=sorted(
                self._support(self._normalize_marking(initial_marking))
            ),
            layers=layers,
            species_first_depth=species_first_depth,
            reaction_first_depth=reaction_first_depth,
        )

    # ------------------------------------------------------------------
    # export helpers
    # ------------------------------------------------------------------

    def export_layers_json(
        self, result: ReachabilityResult, fn: str
    ) -> "PathwayReachability":
        """
        Export a reachability result to a JSON file.

        The output contains the initial species set, first-depth maps, and the
        full list of layered traversal records.

        :param result:
            Reachability result to serialize.
        :type result: ReachabilityResult
        :param fn:
            Output JSON filename.
        :type fn: str
        :returns:
            The current instance, to allow fluent chaining.
        :rtype: PathwayReachability
        :raises OSError:
            Raised if the target file cannot be written.

        Example
        -------
        .. code-block:: python

            rr.export_layers_json(result, "reachability_layers.json")
        """
        data = {
            "initial_species": result.initial_species,
            "species_first_depth": result.species_first_depth,
            "reaction_first_depth": result.reaction_first_depth,
            "layers": [
                {
                    "depth": layer.depth,
                    "newly_enabled_reactions": layer.newly_enabled_reactions,
                    "newly_reachable_species": layer.newly_reachable_species,
                    "all_reachable_species": layer.all_reachable_species,
                }
                for layer in result.layers
            ],
        }
        with open(fn, "w") as fh:
            json.dump(data, fh, indent=2)
        return self

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        """
        Return a compact developer-friendly representation.

        :returns:
            String representation including the number of vertices and edges.
        :rtype: str
        """
        return f"<PathwayReachability vertices={len(self.vertices)} edges={len(self.edges)}>"


def syncrn_to_reachability_inputs(
    crn: object,
    *,
    species: str = "label",
    reaction: str = "id",
) -> Tuple[List[str], Dict[str, Tuple[Dict[str, int], Dict[str, int]]]]:
    """
    Convert a SynCRN-like object into tokenized reachability inputs.

    This is a lightweight adapter returning only the tokenized species list and
    reaction hyperedges needed by :class:`PathwayReachability`.

    :param crn:
        SynCRN-like object to tokenize.
    :type crn: object
    :param species:
        Species attribute used during tokenization.
    :type species: str
    :param reaction:
        Reaction attribute used during tokenization.
    :type reaction: str
    :returns:
        Pair ``(vertices, edges)`` where ``vertices`` is the species token list
        and ``edges`` maps reaction token to ``(tail_multiset, head_multiset)``.
    :rtype: Tuple[List[str], Dict[str, Tuple[Dict[str, int], Dict[str, int]]]]

    Example
    -------
    .. code-block:: python

        vertices, edges = syncrn_to_reachability_inputs(
            syncrn_object,
            species="label",
            reaction="id",
        )
    """
    vertices, edges, _, _, _ = tokenize_syncrn_incidence(
        crn, species=species, reaction=reaction
    )
    return vertices, edges


def run_reachability_from_syncrn(
    crn: object,
    initial_species: Iterable[str],
    *,
    species: str = "label",
    reaction: str = "id",
    verbose: bool = True,
) -> Tuple[PathwayReachability, ReachabilityResult]:
    """
    Run layered qualitative reachability directly from a SynCRN-like object.

    This convenience function constructs a :class:`PathwayReachability`
    instance, loads the tokenized SynCRN representation, computes set-based
    layered reachability, and optionally prints a human-readable traversal
    summary.

    :param crn:
        SynCRN-like object to analyze.
    :type crn: object
    :param initial_species:
        Initial reachable species set.
    :type initial_species: Iterable[str]
    :param species:
        Species attribute used during tokenization.
    :type species: str
    :param reaction:
        Reaction attribute used during tokenization.
    :type reaction: str
    :param verbose:
        If ``True``, print the tokenized edges and each traversal layer.
    :type verbose: bool
    :returns:
        Pair ``(reachability_engine, result)``.
    :rtype: Tuple[PathwayReachability, ReachabilityResult]

    Example
    -------
    .. code-block:: python

        rr, result = run_reachability_from_syncrn(
            syncrn_object,
            initial_species=["A", "B"],
            species="label",
            reaction="id",
            verbose=True,
        )
    """
    rr = PathwayReachability().load_syncrn(crn, species=species, reaction=reaction)
    result = rr.compute_layers_set(initial_species=initial_species)

    if verbose:
        print("Edges:")
        for eid, (t, h) in rr.edges.items():
            print(f"  {eid}: {t} >> {h}")
        print("Initial species:", sorted(set(str(x) for x in initial_species)))
        for layer in result.layers:
            print(f"Layer {layer.depth}:")
            print("  Newly enabled reactions:", layer.newly_enabled_reactions)
            print("  Newly reachable species:", layer.newly_reachable_species)
            print("  All reachable species:", layer.all_reachable_species)

    return rr, result
