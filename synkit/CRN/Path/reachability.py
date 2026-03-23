from __future__ import annotations

"""
Forward reachability utilities for chemical reaction networks.

This module implements layer-by-layer forward reachability on CRN
hypergraphs. In contrast to pathway realizability, which asks whether a
prescribed reaction flow admits a feasible firing order, reachability
starts from an initial set or multiset of species and asks:

1. which reactions are enabled,
2. which new species become reachable,
3. at what layer each reaction/species is first reached.

High-level entry point: :class:`PathwayReachability`.

Typical workflow
----------------

1. Start from a :class:`CRNHyperGraph`::

       from synkit.CRN.Hypergraph.hypergraph import CRNHyperGraph

       hg = CRNHyperGraph()
       hg.parse_rxns(["A+B>>C", "C+D>>E"])

2. Convert the hypergraph into the simple tuple format used here via
   :func:`hypergraph_to_reachability_inputs`.

3. Load the data into :class:`PathwayReachability` and call one of:

   * :meth:`compute_layers_set` for qualitative/set reachability,
   * :meth:`compute_layers_multiset` for stoichiometric/multiset reachability.

Notes
-----
Two semantics are provided:

* **set reachability**:
  a species is either available or not; reactions may fire once they are
  enabled by presence of all reactants.

* **multiset reachability**:
  species counts matter; firing a reaction consumes reactants and produces
  products, allowing explicit state evolution over markings.

The set semantics is usually what is meant by:
"what is reachable in the first, second, third, ... step?"

References
----------
- Classical CRN / Petri-net reachability terminology.
- Murata (1989), Proc. IEEE — Petri nets: Properties, analysis and applications.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

import json

from synkit.CRN.Hypergraph.hypergraph import CRNHyperGraph

# shared type alias
Multiset = Dict[str, int]


# ---------------------------------------------------------------------------
# Reachability config + result containers
# ---------------------------------------------------------------------------


@dataclass
class ReachabilityConfig:
    """
    Configuration container for reachability traversal.

    :param max_layers: Maximum number of forward-expansion layers.
    :type max_layers: int
    :param stop_when_no_new_species: If ``True``, stop once no new species are
        discovered in the current layer.
    :type stop_when_no_new_species: bool
    """

    max_layers: int = 10_000
    stop_when_no_new_species: bool = True


@dataclass
class ReachabilityLayer:
    """
    One forward-expansion layer.

    :param depth: Layer index (1-based).
    :type depth: int
    :param newly_enabled_reactions: Reactions first enabled at this layer.
    :type newly_enabled_reactions: List[str]
    :param newly_reachable_species: Species first reached at this layer.
    :type newly_reachable_species: List[str]
    :param all_reachable_species: Cumulative reachable species after this layer.
    :type all_reachable_species: List[str]
    """

    depth: int
    newly_enabled_reactions: List[str]
    newly_reachable_species: List[str]
    all_reachable_species: List[str]


@dataclass
class ReachabilityResult:
    """
    Container for forward reachability results.

    :param initial_species: Initial available species.
    :type initial_species: List[str]
    :param layers: Layer-by-layer reachability result.
    :type layers: List[ReachabilityLayer]
    :param species_first_depth: Mapping species -> first depth reached
        (depth 0 means initially present).
    :type species_first_depth: Dict[str, int]
    :param reaction_first_depth: Mapping reaction id -> first depth enabled.
    :type reaction_first_depth: Dict[str, int]
    """

    initial_species: List[str]
    layers: List[ReachabilityLayer]
    species_first_depth: Dict[str, int]
    reaction_first_depth: Dict[str, int]


# ---------------------------------------------------------------------------
# PathwayReachability core
# ---------------------------------------------------------------------------


class PathwayReachability:
    """
    High-level forward reachability utilities for CRNs.

    Use case
    --------
    1. Construct an instance (optionally with a :class:`ReachabilityConfig`).
    2. Load a hypergraph via :meth:`load_hypergraph`.
    3. Call one of:

       * :meth:`compute_layers_set`
       * :meth:`compute_layers_multiset`

    Hypergraph format
    -----------------
    The internal representation mirrors that used in pathway realizability:

    * ``vertices`` — iterable of species identifiers (strings).
    * ``edges`` — mapping ``edge_id -> (tail_multiset, head_multiset)``.

    :param config: Optional configuration for forward traversal.
    :type config: ReachabilityConfig or None

    Examples
    --------
    .. code-block:: python

        from synkit.CRN.Hypergraph.hypergraph import CRNHyperGraph
        from synkit.CRN.Path.reachability import (
            hypergraph_to_reachability_inputs, PathwayReachability
        )

        hg = CRNHyperGraph()
        hg.parse_rxns(["A+B>>C", "C+D>>E", "E+F>>G+D"])

        vertices, edges = hypergraph_to_reachability_inputs(hg)

        rr = PathwayReachability()
        rr.load_hypergraph(vertices, edges)

        out = rr.compute_layers_set(initial_species={"A", "B", "D", "F"})
        for layer in out.layers:
            print(layer)
    """

    def __init__(self, config: Optional[ReachabilityConfig] = None) -> None:
        self.vertices: Set[str] = set()
        self.edges: Dict[str, Tuple[Multiset, Multiset]] = {}
        self._config = config or ReachabilityConfig()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_hypergraph(
        self,
        vertices: Iterable[str],
        edges: Mapping[str, Tuple[Mapping[str, int], Mapping[str, int]]],
    ) -> "PathwayReachability":
        """
        Load a hypergraph into the object.

        :param vertices: Species identifiers.
        :type vertices: Iterable[str]
        :param edges: Mapping ``edge_id -> (tail_multiset, head_multiset)``.
        :type edges: Mapping[str, Tuple[Mapping[str, int], Mapping[str, int]]]
        :returns: ``self`` (for fluent chaining).
        :rtype: PathwayReachability
        """
        self.vertices = set(vertices)
        self.edges = {
            eid: (dict(tail), dict(head)) for eid, (tail, head) in edges.items()
        }
        return self

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _normalize_marking(self, marking: Mapping[str, int]) -> Dict[str, int]:
        """
        Normalize a marking by keeping only positive counts.

        :param marking: Input species-count mapping.
        :type marking: Mapping[str, int]
        :returns: Normalized marking with positive integer counts only.
        :rtype: Dict[str, int]
        """
        return {str(s): int(v) for s, v in marking.items() if int(v) > 0}

    @staticmethod
    def _support(marking: Mapping[str, int]) -> Set[str]:
        """
        Return the support of a marking, i.e. species with positive count.

        :param marking: Species-count mapping.
        :type marking: Mapping[str, int]
        :returns: Species with strictly positive count.
        :rtype: Set[str]
        """
        return {s for s, v in marking.items() if int(v) > 0}

    def _enabled_reactions_for_set(self, reachable: Set[str]) -> List[str]:
        """
        Return reactions enabled under qualitative/set semantics.

        A reaction is enabled if all reactant species are present in the
        currently reachable set.

        :param reachable: Currently reachable species.
        :type reachable: Set[str]
        :returns: Enabled reaction identifiers.
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

        A reaction is enabled if each reactant is available with at least its
        required multiplicity in the current marking.

        :param marking: Current species-count mapping.
        :type marking: Mapping[str, int]
        :returns: Enabled reaction identifiers.
        :rtype: List[str]
        """
        enabled: List[str] = []
        for eid, (tail, _) in self.edges.items():
            if all(marking.get(s, 0) >= int(coeff) for s, coeff in tail.items()):
                enabled.append(eid)
        return enabled

    def _fire_reactions_once_in_batch(
        self,
        marking: Mapping[str, int],
        reaction_ids: Iterable[str],
    ) -> Dict[str, int]:
        """
        Fire each listed reaction once using synchronous/batched semantics.

        Consumption and production are accumulated against a copy of the input
        marking, then cleaned to retain only positive counts.

        :param marking: Input marking.
        :type marking: Mapping[str, int]
        :param reaction_ids: Reactions to fire once each.
        :type reaction_ids: Iterable[str]
        :returns: Updated marking after batch firing.
        :rtype: Dict[str, int]
        """
        next_marking = dict(marking)

        for eid in reaction_ids:
            tail, head = self.edges[eid]

            for s, coeff in tail.items():
                next_marking[s] = next_marking.get(s, 0) - int(coeff)

            for s, coeff in head.items():
                next_marking[s] = next_marking.get(s, 0) + int(coeff)

        return {s: v for s, v in next_marking.items() if v > 0}

    @staticmethod
    def _update_first_depth_maps(
        depth: int,
        enabled_now: Iterable[str],
        new_species: Set[str],
        seen_reactions: Set[str],
        reaction_first_depth: Dict[str, int],
        species_first_depth: Dict[str, int],
    ) -> None:
        """
        Update first-depth bookkeeping for reactions and species.

        :param depth: Current traversal depth.
        :type depth: int
        :param enabled_now: Reactions enabled or fired at this depth.
        :type enabled_now: Iterable[str]
        :param new_species: Species newly reached at this depth.
        :type new_species: Set[str]
        :param seen_reactions: Reactions already assigned a first depth.
        :type seen_reactions: Set[str]
        :param reaction_first_depth: Output mapping reaction -> first depth.
        :type reaction_first_depth: Dict[str, int]
        :param species_first_depth: Output mapping species -> first depth.
        :type species_first_depth: Dict[str, int]
        """
        for eid in enabled_now:
            if eid not in seen_reactions:
                reaction_first_depth[eid] = depth
                seen_reactions.add(eid)

        for s in sorted(new_species):
            species_first_depth[s] = depth

    # ------------------------------------------------------------------
    # Set / qualitative reachability
    # ------------------------------------------------------------------

    def compute_layers_set(
        self,
        initial_species: Iterable[str],
        max_layers: Optional[int] = None,
    ) -> ReachabilityResult:
        """
        Compute qualitative forward reachability by layers.

        A reaction becomes enabled when all of its reactant species are
        present in the currently reachable set. Once enabled, its products
        are added to the reachable set.

        This is the natural semantics for questions like:
        "what is reachable in the first, second, third, ... step?"

        :param initial_species: Species available at depth 0.
        :type initial_species: Iterable[str]
        :param max_layers: Optional override for maximum number of layers.
            Defaults to :attr:`ReachabilityConfig.max_layers`.
        :type max_layers: int or None
        :returns: Layered reachability result.
        :rtype: ReachabilityResult
        """
        max_layers = max_layers if max_layers is not None else self._config.max_layers

        reachable: Set[str] = set(initial_species)
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
            initial_species=sorted(set(initial_species)),
            layers=layers,
            species_first_depth=species_first_depth,
            reaction_first_depth=reaction_first_depth,
        )

    # ------------------------------------------------------------------
    # Multiset / stoichiometric reachability
    # ------------------------------------------------------------------

    def compute_layers_multiset(
        self,
        initial_marking: Mapping[str, int],
        max_layers: Optional[int] = None,
    ) -> ReachabilityResult:
        """
        Compute one possible layer-by-layer stoichiometric evolution.

        At each layer, all currently enabled reactions are fired once in
        batch. Reactant multiplicities are respected.

        This is not the full Petri-net reachability problem; rather, it is a
        deterministic forward simulation under synchronous/batched firing.

        :param initial_marking: Initial species counts.
        :type initial_marking: Mapping[str, int]
        :param max_layers: Optional override for maximum number of layers.
            Defaults to :attr:`ReachabilityConfig.max_layers`.
        :type max_layers: int or None
        :returns: Layered reachability result based on cumulative support
            of the marking.
        :rtype: ReachabilityResult
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
    # Export
    # ------------------------------------------------------------------

    def export_layers_json(
        self,
        result: ReachabilityResult,
        fn: str,
    ) -> "PathwayReachability":
        """
        Export a reachability result to JSON.

        :param result: Reachability result produced by this class.
        :type result: ReachabilityResult
        :param fn: Output filename.
        :type fn: str
        :returns: ``self``.
        :rtype: PathwayReachability
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
        return (
            f"<PathwayReachability vertices={len(self.vertices)} "
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
        out: Dict[str, int] = {}
        for x in side:  # type: ignore[arg-type]
            sx = str(x)
            out[sx] = out.get(sx, 0) + 1
        return out
    except Exception:
        return {}


def hypergraph_to_reachability_inputs(
    hg: CRNHyperGraph,
) -> Tuple[List[str], Dict[str, Tuple[Dict[str, int], Dict[str, int]]]]:
    """
    Convert :class:`CRNHyperGraph` into the tuple format used by
    :class:`PathwayReachability`.

    Outputs
    -------
    * ``vertices``: list of species names (strings).
    * ``edges_map``: ``{edge_id: (tail_dict, head_dict)}``.

    :param hg: Hypergraph describing the CRN.
    :type hg: CRNHyperGraph
    :returns: Tuple ``(vertices, edges_map)``.
    :rtype: Tuple[List[str], Dict[str, Tuple[Dict[str, int], Dict[str, int]]]]

    Examples
    --------
    .. code-block:: python

        vertices, edges = hypergraph_to_reachability_inputs(hg)
        rr = PathwayReachability().load_hypergraph(vertices, edges)
    """
    vertices = list(hg.species_list())
    edges: Dict[str, Tuple[Dict[str, int], Dict[str, int]]] = {}

    for e in hg.edge_list():
        eid = getattr(e, "id", None) or getattr(e, "edge_id", None) or str(e)
        r_side = _side_to_dict(getattr(e, "reactants", getattr(e, "lhs", None)))
        p_side = _side_to_dict(getattr(e, "products", getattr(e, "rhs", None)))
        edges[eid] = (r_side, p_side)

    return vertices, edges


def run_reachability_from_rxn_strings(
    rxns: Iterable[str],
    initial_species: Iterable[str],
    verbose: bool = True,
) -> Tuple[PathwayReachability, ReachabilityResult]:
    """
    Convenience harness:

    * Build a :class:`CRNHyperGraph` from reaction strings.
    * Convert to PathwayReachability inputs.
    * Run qualitative layered reachability.

    :param rxns: Iterable of reaction strings (``"A + B >> C"`` etc.).
    :type rxns: Iterable[str]
    :param initial_species: Species available at depth 0.
    :type initial_species: Iterable[str]
    :param verbose: If ``True``, print a small summary to stdout.
    :type verbose: bool
    :returns: Tuple ``(rr, result)`` where ``rr`` is the configured
        :class:`PathwayReachability` instance and ``result`` is the
        layered reachability output.
    :rtype: Tuple[PathwayReachability, ReachabilityResult]

    Examples
    --------
    .. code-block:: python

        rxns = ["A+B>>C", "C+D>>E", "E+F>>G+D"]
        rr, result = run_reachability_from_rxn_strings(
            rxns,
            initial_species={"A", "B", "D", "F"},
        )
        print(result.species_first_depth)
    """
    hg = CRNHyperGraph()
    hg.parse_rxns(list(rxns))

    vertices, edges = hypergraph_to_reachability_inputs(hg)
    rr = PathwayReachability()
    rr.load_hypergraph(vertices=vertices, edges=edges)

    result = rr.compute_layers_set(initial_species=initial_species)

    if verbose:
        print("Edges:")
        for eid, (t, h) in edges.items():
            print(f"  {eid}: {t} >> {h}")

        print("Initial species:", sorted(set(initial_species)))
        for layer in result.layers:
            print(f"Layer {layer.depth}:")
            print("  Newly enabled reactions:", layer.newly_enabled_reactions)
            print("  Newly reachable species:", layer.newly_reachable_species)
            print("  All reachable species:", layer.all_reachable_species)

    return rr, result
