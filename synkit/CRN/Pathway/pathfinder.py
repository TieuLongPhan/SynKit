from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

from ._adapter import tokenize_syncrn_incidence
from .realizability import PathwayRealizability

Multiset = Dict[str, int]


@dataclass
class PathFinderConfig:
    """
    Configuration for qualitative pathway search.

    This configuration controls the breadth-first search over qualitative
    set-semantics states.

    :param max_depth:
        Maximum number of reaction firings allowed in a candidate pathway.
    :type max_depth: int
    :param max_paths:
        Maximum number of pathway candidates to return.
    :type max_paths: int
    :param stop_after_first:
        Whether to stop the search immediately after the first accepted
        candidate is found.
    :type stop_after_first: bool
    :param deduplicate_by_flow:
        Whether to deduplicate candidates by their reaction-count flow vector
        rather than retaining multiple sequences with the same aggregate flow.
    :type deduplicate_by_flow: bool
    """

    max_depth: int = 12
    max_paths: int = 20
    stop_after_first: bool = False
    deduplicate_by_flow: bool = True


@dataclass
class PathwayCandidate:
    """
    One qualitative pathway candidate returned by :class:`PathwayFinder`.

    A candidate stores both the ordered reaction sequence used during
    qualitative search and the corresponding aggregated reaction-count flow.

    :param reactions: Ordered list of reaction ids in the candidate sequence.
    :type reactions: List[str]
    :param flow:
        Aggregated reaction firing counts derived from ``reactions``.
    :type flow: Dict[str, int]
    :param reached_species:
        Sorted list of species reachable after firing the candidate sequence in
        qualitative set semantics.
    :type reached_species: List[str]
    :param depth: Length of the reaction sequence.
    :type depth: int
    :param realizable:
        Optional exact realizability verdict computed afterwards from Petri-net
        semantics.
    :type realizable: Optional[bool]
    :param certificate:
        Optional firing certificate returned by exact realizability checking.
    :type certificate: Optional[List[str]]
    """

    reactions: List[str]
    flow: Dict[str, int]
    reached_species: List[str]
    depth: int
    realizable: Optional[bool] = None
    certificate: Optional[List[str]] = None


class PathwayFinder:
    """
    Qualitative source-to-target pathway search for SynCRN-like inputs.

    This class searches in set semantics: a reaction is usable once all its
    reactant species are present in the current available set, and firing it
    adds its product species to that set.

    Species are not consumed during the qualitative search. Exact
    stoichiometric and token-based validation can be applied afterwards using
    :meth:`validate_candidates`.
    """

    def __init__(self, config: Optional[PathFinderConfig] = None) -> None:
        """
        Initialize an empty pathway finder.

        :param config:
            Optional search configuration. If omitted, default configuration is
            used.
        :type config: Optional[PathFinderConfig]
        """
        self.vertices: Set[str] = set()
        self.edges: Dict[str, Tuple[Multiset, Multiset]] = {}
        self.species_token_to_id: Dict[str, str] = {}
        self.reaction_token_to_id: Dict[str, str] = {}
        self.species_id_to_token: Dict[str, str] = {}
        self.reaction_id_to_token: Dict[str, str] = {}
        self._config = config or PathFinderConfig()

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------

    def load_hypergraph(
        self,
        vertices: Iterable[str],
        edges: Mapping[str, Tuple[Mapping[str, int], Mapping[str, int]]],
    ) -> "PathwayFinder":
        """
        Load a qualitative reaction hypergraph directly.

        The hypergraph is represented by a set of species vertices and a mapping
        from reaction ids to ``(tail, head)`` multisets, where ``tail`` is the
        reactant side and ``head`` is the product side.

        :param vertices: Species identifiers in the hypergraph.
        :type vertices: Iterable[str]
        :param edges:
            Mapping from reaction id to a pair ``(tail, head)`` of multisets.
            Coefficients less than or equal to zero are ignored.
        :type edges: Mapping[str, Tuple[Mapping[str, int], Mapping[str, int]]]
        :returns: The current finder instance.
        :rtype: PathwayFinder
        """
        self.vertices = {str(v) for v in vertices}
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
    ) -> "PathwayFinder":
        """
        Load a SynCRN-like object through its incidence representation.

        The CRN is tokenized into species and reaction identifiers suitable for
        qualitative search.

        :param crn: SynCRN-like object to load.
        :type crn: object
        :param species:
            Species naming mode passed to the tokenizer, for example
            ``"label"`` or ``"id"``.
        :type species: str
        :param reaction:
            Reaction naming mode passed to the tokenizer, for example
            ``"id"`` or another supported tokenization key.
        :type reaction: str
        :returns: The current finder instance.
        :rtype: PathwayFinder
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
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_species(items: Iterable[str]) -> Set[str]:
        """
        Normalize a species iterable into a set of string identifiers.

        :param items: Input species collection.
        :type items: Iterable[str]
        :returns: Normalized species-id set.
        :rtype: Set[str]
        """
        return {str(x) for x in items}

    def _enabled_reactions(self, available: Set[str]) -> List[str]:
        """
        Return reactions enabled under qualitative set semantics.

        A reaction is enabled when all species appearing on its reactant side
        are already present in ``available``.

        :param available: Currently available species set.
        :type available: Set[str]
        :returns: Sorted list of enabled reaction ids.
        :rtype: List[str]
        """
        enabled: List[str] = []
        for eid, (tail, _) in self.edges.items():
            if set(tail.keys()).issubset(available):
                enabled.append(eid)
        return sorted(enabled)

    @staticmethod
    def _flow_from_sequence(seq: List[str]) -> Dict[str, int]:
        """
        Convert a reaction sequence into an aggregated flow vector.

        :param seq: Ordered reaction sequence.
        :type seq: List[str]
        :returns: Mapping from reaction id to firing count.
        :rtype: Dict[str, int]
        """
        flow: Dict[str, int] = {}
        for tid in seq:
            flow[tid] = flow.get(tid, 0) + 1
        return flow

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def find_paths_set(
        self,
        *,
        source_species: Iterable[str],
        target_species: Iterable[str],
        max_depth: Optional[int] = None,
        max_paths: Optional[int] = None,
        stop_after_first: Optional[bool] = None,
        deduplicate_by_flow: Optional[bool] = None,
    ) -> List[PathwayCandidate]:
        """
        Find qualitative source-to-target pathways using set semantics.

        The search is performed by breadth-first expansion over reachable
        species sets. At each step, enabled reactions add their product species
        to the current set without consuming reactants.

        :param source_species:
            Initial source species assumed to be present.
        :type source_species: Iterable[str]
        :param target_species:
            Target species that must all be contained in the reached set.
        :type target_species: Iterable[str]
        :param max_depth:
            Optional override for maximum search depth. If ``None``, the value
            from the configuration is used.
        :type max_depth: Optional[int]
        :param max_paths:
            Optional override for maximum number of returned candidates. If
            ``None``, the value from the configuration is used.
        :type max_paths: Optional[int]
        :param stop_after_first:
            Optional override controlling whether to stop after the first
            accepted candidate.
        :type stop_after_first: Optional[bool]
        :param deduplicate_by_flow:
            Optional override controlling whether candidates are deduplicated by
            aggregate flow.
        :type deduplicate_by_flow: Optional[bool]
        :returns:
            List of qualitative pathway candidates satisfying the target
            condition.
        :rtype: List[PathwayCandidate]
        :raises RuntimeError:
            If no network has been loaded.
        :raises ValueError:
            If ``target_species`` is empty.
        """
        if not self.edges:
            raise RuntimeError(
                "No network loaded; call load_syncrn() or load_hypergraph()."
            )

        max_depth = self._config.max_depth if max_depth is None else max_depth
        max_paths = self._config.max_paths if max_paths is None else max_paths
        stop_after_first = (
            self._config.stop_after_first
            if stop_after_first is None
            else stop_after_first
        )
        deduplicate_by_flow = (
            self._config.deduplicate_by_flow
            if deduplicate_by_flow is None
            else deduplicate_by_flow
        )

        source = self._clean_species(source_species)
        target = self._clean_species(target_species)
        if not target:
            raise ValueError("target_species must not be empty")

        results: List[PathwayCandidate] = []
        seen_flow_keys: Set[Tuple[Tuple[str, int], ...]] = set()
        best_depth_by_state: Dict[Tuple[str, ...], int] = {tuple(sorted(source)): 0}

        q: deque[Tuple[Set[str], List[str]]] = deque([(set(source), [])])

        while q and len(results) < max_paths:
            available, seq = q.popleft()
            depth = len(seq)

            if target.issubset(available):
                flow = self._flow_from_sequence(seq)
                flow_key = tuple(sorted(flow.items()))
                if (not deduplicate_by_flow) or (flow_key not in seen_flow_keys):
                    results.append(
                        PathwayCandidate(
                            reactions=list(seq),
                            flow=flow,
                            reached_species=sorted(available),
                            depth=depth,
                        )
                    )
                    seen_flow_keys.add(flow_key)
                    if stop_after_first:
                        break
                continue

            if depth >= max_depth:
                continue

            for eid in self._enabled_reactions(available):
                _, head = self.edges[eid]
                produced = set(head.keys())
                if not produced - available:
                    continue
                new_available = set(available) | produced
                state_key = tuple(sorted(new_available))
                new_depth = depth + 1
                old_best = best_depth_by_state.get(state_key)
                if old_best is not None and new_depth > old_best:
                    continue
                best_depth_by_state[state_key] = new_depth
                q.append((new_available, seq + [eid]))

        return results

    def validate_candidates(
        self,
        crn: object,
        candidates: List[PathwayCandidate],
        *,
        initial_marking: Mapping[str, int],
        species: str = "label",
        reaction: str = "id",
    ) -> List[PathwayCandidate]:
        """
        Validate qualitative candidates by exact Petri-net realizability.

        For each candidate, the aggregated reaction-count flow is checked using
        :class:`PathwayRealizability`. The returned candidates preserve the
        original qualitative information and add realizability metadata.

        :param crn: SynCRN-like object used for exact validation.
        :type crn: object
        :param candidates: Qualitative candidates to validate.
        :type candidates: List[PathwayCandidate]
        :param initial_marking:
            Initial place marking used for exact realizability checking.
        :type initial_marking: Mapping[str, int]
        :param species:
            Species naming mode passed through to
            :meth:`PathwayRealizability.load_syncrn_and_flow`.
        :type species: str
        :param reaction:
            Reaction naming mode passed through to
            :meth:`PathwayRealizability.load_syncrn_and_flow`.
        :type reaction: str
        :returns:
            New list of candidates augmented with realizability verdicts and
            certificates.
        :rtype: List[PathwayCandidate]
        """
        out: List[PathwayCandidate] = []
        for cand in candidates:
            pr = PathwayRealizability().load_syncrn_and_flow(
                crn,
                flow=cand.flow,
                initial_marking=initial_marking,
                species=species,
                reaction=reaction,
            )
            pr.build_petri_net_from_flow()
            ok, cert = pr.is_realizable()
            out.append(
                PathwayCandidate(
                    reactions=list(cand.reactions),
                    flow=dict(cand.flow),
                    reached_species=list(cand.reached_species),
                    depth=cand.depth,
                    realizable=ok,
                    certificate=None if cert is None else list(cert),
                )
            )
        return out


def run_pathfinder_from_syncrn(
    crn: object,
    *,
    source_species: Iterable[str],
    target_species: Iterable[str],
    initial_marking: Optional[Mapping[str, int]] = None,
    species: str = "label",
    reaction: str = "id",
    max_depth: int = 12,
    max_paths: int = 20,
    validate: bool = False,
    verbose: bool = True,
) -> List[PathwayCandidate]:
    """
    Convenience harness for qualitative pathway search on a SynCRN object.

    This helper constructs a :class:`PathwayFinder`, loads the CRN, runs
    qualitative search, and optionally validates the returned candidates by
    exact realizability.

    :param crn: SynCRN-like object to analyze.
    :type crn: object
    :param source_species:
        Initial source species assumed to be present.
    :type source_species: Iterable[str]
    :param target_species:
        Target species that must all be reachable.
    :type target_species: Iterable[str]
    :param initial_marking:
        Optional initial marking used only when ``validate=True``.
    :type initial_marking: Optional[Mapping[str, int]]
    :param species:
        Species naming mode used when tokenizing the CRN.
    :type species: str
    :param reaction:
        Reaction naming mode used when tokenizing the CRN.
    :type reaction: str
    :param max_depth: Maximum qualitative search depth.
    :type max_depth: int
    :param max_paths: Maximum number of pathway candidates to return.
    :type max_paths: int
    :param validate:
        Whether to validate candidates by exact Petri-net realizability.
    :type validate: bool
    :param verbose:
        Whether to print a simple textual summary of the returned candidates.
    :type verbose: bool
    :returns:
        List of qualitative pathway candidates, optionally enriched with exact
        realizability results.
    :rtype: List[PathwayCandidate]
    :raises ValueError:
        If ``validate=True`` but ``initial_marking`` is not provided.
    """
    finder = PathwayFinder(PathFinderConfig(max_depth=max_depth, max_paths=max_paths))
    finder.load_syncrn(crn, species=species, reaction=reaction)
    candidates = finder.find_paths_set(
        source_species=source_species,
        target_species=target_species,
        max_depth=max_depth,
        max_paths=max_paths,
    )
    if validate:
        if initial_marking is None:
            raise ValueError("initial_marking is required when validate=True")
        candidates = finder.validate_candidates(
            crn,
            candidates,
            initial_marking=initial_marking,
            species=species,
            reaction=reaction,
        )

    if verbose:
        for i, cand in enumerate(candidates, start=1):
            print(
                f"Path {i}: depth={cand.depth}, reactions={cand.reactions}, flow={cand.flow}"
            )
            if cand.realizable is not None:
                print(f"  realizable={cand.realizable}, certificate={cand.certificate}")

    return candidates
