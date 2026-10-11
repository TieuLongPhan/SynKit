"""Select mapping methods through one interface and report their search scope."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .exact.distance_search.exhaustive import ExactMapper
from .exact.distance_search.records import normalize_distance_target
from .exact.distance_search.search import enumerate_distance_mappings
from .exact.hybrid import enumerate_hybrid_distance_mappings
from .exact.kernel_solvers.enumerate import enumerate_kernel_optima
from .exact.kernel_solvers.kernel import apply_kernel_solution, extract_kernel
from .exact.kernel_solvers.milp import solve_kernel_milp
from .exact.search import enumerate_pabs_mappings
from .graph.labeled_graph import LabeledGraph
from .reaction_search import ReactionMappingResult, _map_reaction_with_search
from .slap.lap import _adjacency_and_elements, chemical_distance, recover_mapping
from .slap.sequential import GraphMatcher

_SCOPES = {
    "slap": "heuristic",
    "kernel": "reduced_kernel",
    "milp": "reduced_kernel",
    "distance": "full_space",
    "propagate": "full_space",
    "native": "full_space",
    "edits": "full_space",
    "exhaustive": "full_space",
}
_ENUMERATORS = {"distance", "propagate", "native", "edits"}
_REACTION_OPTIONS = {
    "hydrogen",
    "objective",
    "balance",
    "max_hydrogen_plans",
    "classification_max_search_nodes",
}
_DEFAULTS = {
    "CD": "minimal",
    "binary": False,
    "hydrogen": "heavy",
    "objective": "bond",
    "balance": "strict",
    "max_hydrogen_plans": 10_000,
    "classification_max_search_nodes": 100_000,
    "library_path": None,
    "search_config": None,
    "max_edit_support_pairs": 50_000,
}
_OPTION_NAMES = _DEFAULTS.keys() | {
    "time_limit_seconds",
    "max_mappings",
    "max_bijections",
}


@dataclass(frozen=True)
class MappingResult:
    """Common result for graph pairs and reaction SMILES.

    ``mappings[i][j]`` is the product index assigned to reactant atom ``j``.
    SMILES results retain symmetry class representatives. ``minimum_cost`` and
    ``optimality_proven`` apply within ``search_scope`` and the chosen objective.
    ``complete`` means enumeration and any chemical classification completed in
    that scope. It is always false for heuristic or single-optimum methods.
    A proven minimum can coexist with incomplete enumeration.

    For compressed hydrogen, the minimum is for heavy bonds; returned distances
    include conditional hydrogen lifts. See ``reaction_result`` for that scope.
    ``search_result`` preserves the underlying solver's records and statistics.
    """

    method: str
    search_scope: str
    mappings: tuple[tuple[int, ...], ...]
    distances: tuple[float, ...]
    minimum_cost: float | None
    optimality_proven: bool
    complete: bool
    incomplete_reason: str | None
    elapsed_seconds: float
    search_result: object
    reaction_result: ReactionMappingResult | None = None

    @property
    def mapped_reactions(self) -> tuple[str, ...]:
        """Mapped reaction SMILES, or an empty tuple for graph input."""
        return (
            ()
            if self.reaction_result is None
            else self.reaction_result.mapped_reactions
        )

    def as_dict(self) -> dict[str, object]:
        """Export common metadata and chemistry records without solver internals."""
        return {
            "method": self.method,
            "search_scope": self.search_scope,
            "mappings": [list(mapping) for mapping in self.mappings],
            "distances": list(self.distances),
            "minimum_cost": self.minimum_cost,
            "optimality_proven": self.optimality_proven,
            "complete": self.complete,
            "incomplete_reason": self.incomplete_reason,
            "elapsed_seconds": self.elapsed_seconds,
            "reaction_result": (
                None if self.reaction_result is None else self.reaction_result.as_dict()
            ),
        }


@dataclass
class _SearchOutcome:
    """Search results consumed by the reaction-SMILES frontend."""

    mappings: list[list[int]]
    distances: list[float]
    minimum_cost: float | None
    optimality_proven: bool
    complete: bool
    truncation_reason: str | None
    backend: str
    selected_mapping_count: int
    raw_result: object


def _prepare_graphs(data):
    if (
        not isinstance(data, (list, tuple))
        or len(data) != 2
        or not all(isinstance(graph, LabeledGraph) for graph in data)
    ):
        raise TypeError(
            "data must be reaction SMILES or a pair of LabeledGraph objects"
        )
    graphs = []
    inventories = []
    label_codes = {}
    for source in data:
        if any(source.graph[i].get(i, 0) != 0 for i in range(len(source.labels))):
            raise ValueError("Mapper requires loop-free graphs")
        adjacency, elements = _adjacency_and_elements(source, False)
        if len(elements) != len(source.labels):
            raise ValueError("Atom labels must have one entry per node")
        if not np.isfinite(adjacency).all() or (adjacency < 0).any():
            raise ValueError("Bond weights must be finite and non-negative")
        if not np.array_equal(adjacency, adjacency.T):
            raise ValueError("Mapper requires symmetric undirected graphs")
        graph = LabeledGraph(
            {
                i: {int(j): float(row[j]) for j in np.flatnonzero(row)}
                for i, row in enumerate(adjacency)
            },
            [label_codes.setdefault(element, len(label_codes)) for element in elements],
        )
        graph.props = dict(source.props)
        graph.props["atomic numbers"] = elements
        graphs.append(graph)
        inventories.append(Counter(elements))
    if inventories[0] != inventories[1]:
        raise ValueError("Endpoints must have equal atom-type inventories")
    return graphs


def _check_method_options(method, config, requested):
    """Reject queries and limits unsupported by the selected method."""
    if method not in _ENUMERATORS:
        if config["CD"] != "minimal":
            raise ValueError(f"method='{method}' only supports CD='minimal'")
        for name in ("time_limit_seconds", "max_mappings", "max_bijections"):
            if config[name] is not None:
                raise ValueError(f"method='{method}' does not support {name}")
    if method != "native" and config["library_path"] is not None:
        raise ValueError("library_path requires method='native'")
    if method == "native" and config["library_path"] is None:
        raise ValueError("method='native' requires a library_path from build_native")
    if method != "propagate" and config["search_config"] is not None:
        raise ValueError("search_config requires method='propagate'")
    if method != "edits" and "max_edit_support_pairs" in requested:
        raise ValueError("max_edit_support_pairs requires method='edits'")
    if method == "edits":
        if config["CD"] == "minimal" or not config["binary"]:
            raise ValueError("method='edits' requires numeric CD and binary=True")
        if config["max_bijections"] is not None:
            raise ValueError("method='edits' does not support max_bijections")


class Mapper:
    """Select a solver by a concise name and map SMILES or labeled graph pairs.

    ``Mapper(method='propagate')`` defaults to full-space exact Python search.
    :meth:`methods` lists the eight accepted names. ``slap`` is heuristic;
    ``kernel`` enumerates optima in a SLAP-derived restricted problem; ``milp``
    solves that problem for one optimum using optional PuLP/CBC. ``distance``
    and ``propagate`` enumerate full-space minimum or numeric CD shells.
    ``native`` requires a ``library_path`` from ``build_native``. ``edits`` requires
    numeric ``CD`` and ``binary=True``. ``exhaustive`` returns one global optimum
    and is intended for small graphs.

    Constructor options become defaults; options passed to :meth:`map` override
    them for that call. All methods use weighted bonds and heavy atoms by
    default. Graph input must be symmetric and loop-free, with equal atom-type
    inventories. Inputs are copied before matching.

    Enumeration methods default to a 10-second shared deadline and 100,000
    mappings. Assignment enumerators also cap the assignment space at 1,000,000
    bijections. SLAP, kernel, MILP and exhaustive methods have no deadline or
    output-cap support; supplying these limits raises an error. Existing solver
    APIs remain available for their specialized options.
    """

    def __init__(self, method="propagate", **options):
        if not isinstance(method, str) or method not in _SCOPES:
            raise ValueError(
                f"Unknown mapping method {method!r}; choose from {', '.join(_SCOPES)}"
            )
        self._check_option_names(options)
        self.method = method
        self._options = dict(options)

    @staticmethod
    def methods() -> tuple[str, ...]:
        """Return the supported mapping method names."""
        return tuple(_SCOPES)

    @staticmethod
    def _check_option_names(options):
        unknown = options.keys() - _OPTION_NAMES
        if unknown:
            raise TypeError(f"Unknown Mapper option(s): {', '.join(sorted(unknown))}")

    def map(self, data, **options) -> MappingResult:
        """Map reaction SMILES or ``(reactant_graph, product_graph)``.

        Common options are ``CD='minimal'`` and ``binary=False``. Enumeration
        methods accept ``time_limit_seconds`` and ``max_mappings``. Assignment
        enumerators accept ``max_bijections``. Reaction inputs also accept
        ``hydrogen``, ``objective``, ``balance``, ``max_hydrogen_plans``, and
        ``classification_max_search_nodes``, as documented by ``map_reaction``.
        ``search_config`` is propagation-only; ``library_path`` is native-only;
        ``max_edit_support_pairs`` limits the edit-support method.
        """
        self._check_option_names(options)
        requested = self._options | options
        method = self.method
        if method not in _SCOPES:
            raise ValueError(f"Unknown mapping method {method!r}")
        enumerating = method in _ENUMERATORS
        config = (
            _DEFAULTS
            | {
                "time_limit_seconds": 10.0 if enumerating else None,
                "max_mappings": 100_000 if enumerating else None,
                "max_bijections": (
                    1_000_000 if method in {"distance", "propagate", "native"} else None
                ),
            }
            | requested
        )
        config["CD"] = normalize_distance_target(config["CD"])
        if not isinstance(config["binary"], bool):
            raise TypeError("binary must be boolean")
        _check_method_options(method, config, requested)
        reaction_input = isinstance(data, str)
        if not reaction_input and requested.keys() & _REACTION_OPTIONS:
            raise ValueError(
                "Hydrogen, objective, balance and classification options require reaction SMILES"
            )
        started = time.perf_counter()
        if reaction_input:

            def search(graphs, **query):
                return self._solve(
                    graphs,
                    config | {"time_limit_seconds": query["time_limit_seconds"]},
                    mapping_callback=query["mapping_callback"],
                    collect_mappings=False,
                )

            reaction = _map_reaction_with_search(
                data,
                search_function=search,
                **{name: config[name] for name in _REACTION_OPTIONS},
                **{
                    name: config[name]
                    for name in (
                        "CD",
                        "binary",
                        "time_limit_seconds",
                        "max_mappings",
                        "max_bijections",
                        "search_config",
                        "library_path",
                    )
                },
            )
            outcome = reaction.search_result
            return MappingResult(
                method,
                _SCOPES[method],
                tuple(item.mapping for item in reaction.reactions),
                tuple(item.distance for item in reaction.reactions),
                outcome.minimum_cost,
                outcome.optimality_proven,
                reaction.complete,
                reaction.incomplete_reason,
                time.perf_counter() - started,
                outcome.raw_result,
                reaction,
            )
        outcome = self._solve(data, config)
        return MappingResult(
            method,
            _SCOPES[method],
            tuple(tuple(mapping) for mapping in outcome.mappings),
            tuple(outcome.distances),
            outcome.minimum_cost,
            outcome.optimality_proven,
            outcome.complete,
            outcome.truncation_reason,
            time.perf_counter() - started,
            outcome.raw_result,
        )

    def _enumerate_graphs(self, graphs, options, mapping_callback, collect_mappings):
        """Adapt distance-shell enumerators to the common result fields."""
        method = self.method
        query = {
            name: options[name]
            for name in (
                "CD",
                "binary",
                "time_limit_seconds",
                "max_mappings",
                "max_bijections",
            )
        }
        query.update(
            collect_mappings=collect_mappings,
            mapping_callback=mapping_callback,
            compute_minimum_cost=(options["CD"] == "minimal"),
        )
        if method == "distance":
            result = enumerate_distance_mappings(graphs, **query)
        elif method == "edits":
            result = enumerate_hybrid_distance_mappings(
                graphs,
                backend="edit_support",
                max_edit_support_pairs=options["max_edit_support_pairs"],
                **query,
            )
        else:
            if options["search_config"] is not None:
                query["config"] = options["search_config"]
            result = enumerate_pabs_mappings(
                graphs,
                backend="cpp" if method == "native" else "python",
                library_path=options["library_path"],
                **query,
            )
        return _SearchOutcome(
            result.mappings,
            result.distances,
            result.minimum_cost,
            result.minimum_cost is not None,
            result.complete,
            result.truncation_reason,
            result.backend,
            result.selected_mapping_count,
            result,
        )

    def _solve(self, data, options, *, mapping_callback=None, collect_mappings=True):
        graphs = _prepare_graphs(data)
        method, binary = self.method, options["binary"]
        if method in _ENUMERATORS:
            return self._enumerate_graphs(
                graphs, options, mapping_callback, collect_mappings
            )

        if method == "exhaustive":
            if binary:
                for graph in graphs:
                    graph.binarize_graph()
            raw = ExactMapper().solve(*graphs)
            mappings, proven, complete, reason = (
                [raw.mapping],
                True,
                False,
                "optimum_only",
            )
        else:
            matcher = GraphMatcher(binary=binary)
            matcher.get_maps(
                graphs, break_sym_targets=list(range(len(graphs[0].labels)))
            )
            if not matcher.results:
                raise RuntimeError("SLAP did not produce a mapping")
            if method == "slap":
                raw = matcher
                mappings = [recover_mapping(item["lgp"]) for item in matcher.results]
                proven, complete, reason = False, False, "heuristic"
            else:
                kernel = extract_kernel(matcher.results, graphs, binary=binary)
                if method == "kernel":
                    raw = enumerate_kernel_optima(kernel, unique=False)
                    mappings = raw.mappings
                    proven = bool(raw.proven_optimal and mappings)
                    complete = bool(raw.enumeration_complete and proven)
                    reason = None if complete else "kernel_enumeration_incomplete"
                else:
                    raw = solve_kernel_milp(kernel)
                    mappings = [
                        apply_kernel_solution(kernel, sub) for sub in raw.sub_mappings
                    ]
                    proven = bool(raw.proven_optimal and mappings)
                    complete, reason = False, (
                        "optimum_only" if proven else "optimum_unproven"
                    )
        distances = []
        for mapping in mappings:
            if sorted(mapping) != list(range(len(graphs[0].labels))) or any(
                graphs[0].labels[i] != graphs[1].labels[j]
                for i, j in enumerate(mapping)
            ):
                raise RuntimeError(
                    f"method='{method}' returned an invalid atom bijection"
                )
            cost = chemical_distance(graphs, mapping, binary)
            distances.append(cost)
            if mapping_callback is not None:
                mapping_callback(mapping, cost)
        minimum = min(distances) if proven and distances else None
        return _SearchOutcome(
            mappings if collect_mappings else [],
            distances if collect_mappings else [],
            minimum,
            proven,
            complete,
            reason,
            method,
            len(mappings),
            raw,
        )


__all__ = ["Mapper", "MappingResult"]
