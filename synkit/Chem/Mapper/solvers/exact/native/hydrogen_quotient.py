"""Experimental exact terminal-H quotient for minimum Lewis mapping classes.

For a fixed heavy-atom bijection f, neutral monovalent terminal hydrogens have
minimum bond cost sum_i |h_R[i] - h_P[f(i)]|. This remains true with unequal H
inventories when isolated zero-state H placeholders form the missing-side
reservoir. Adding this term before heavy-map optimization gives a global
explicit-H Lewis minimum, unlike optimizing heavy bonds and lifting afterward.

Only minimum queries use the quotient. Numeric queries, nonordinary H atoms,
and annotated stereochemistry fall back to the full explicit-H native solver.
Every optimal integer H transport is lifted, then the common chemical frontend
performs exact full-ITS classification. This is an experimental reduction, not
a replacement for the public default or a polynomial all-class enumerator.
"""

from __future__ import annotations

from dataclasses import replace
import time

from ....graph.labeled_graph import LabeledGraph
from ....reaction_search import _map_reaction_with_search
from ...slap.lap import chemical_distance
from ..hydrogen import enumerate_minimal_hydrogen_transfers
from .candidates import NativeEnumerationStop
from .search import enumerate_cpp_mappings


def _quotient_graph(graph):
    """Return a cost-preserving reduced graph and original-index H groups."""
    hydrogen = [i for i, label in enumerate(graph.labels) if label[0] == 1]
    if not hydrogen:
        return None, "no_hydrogen_reduction"
    if any(graph.labels[i] != (1, 0) for i in hydrogen):
        return None, "isotopic_hydrogen"
    if any(graph.props["stereo"]) or any(graph.props["_stereo_anchor"]):
        return None, "annotated_stereochemistry"
    hydrogen_set = set(hydrogen)
    heavy = [i for i in range(len(graph.labels)) if i not in hydrogen_set]
    positions = {atom: i for i, atom in enumerate(heavy)}
    groups = [[] for _ in heavy]
    reservoir = []
    for atom in hydrogen:
        if graph.props["dummy"][atom]:
            if graph.graph[atom]:
                return None, "bonded_hydrogen_placeholder"
            reservoir.append(atom)
            continue
        if graph.props["charge"][atom] or graph.props["radical"][atom]:
            return None, "nonneutral_or_radical_hydrogen"
        neighbors = graph.graph[atom]
        if len(neighbors) != 1:
            return None, "nonterminal_hydrogen"
        parent, weight = next(iter(neighbors.items()))
        if weight != 1 or parent not in positions or graph.labels[parent][0] == 0:
            return None, "nonordinary_hydrogen_bond"
        groups[positions[parent]].append(atom)
    labels = [graph.labels[i] for i in heavy]
    h_anchor = len(labels)
    labels.append((0, -2))
    adjacency = {
        positions[i]: {positions[j]: weight for j, weight in graph.graph[i].items()
                       if j in positions}
        for i in heavy
    }
    adjacency[h_anchor] = {}
    for i, children in enumerate(groups):
        if children:
            adjacency[i][h_anchor] = adjacency[h_anchor][i] = len(children)
    levels = {0} | {value for neighbors in adjacency.values() for value in neighbors.values()}
    if len(levels) > 16:
        return None, "quotient_bond_level_limit"
    reduced = LabeledGraph(adjacency, labels)
    for name in ("charge", "radical", "hcount", "stereo", "dummy", "_stereo_anchor"):
        values = graph.props[name]
        reduced.props[name] = [values[i] for i in heavy] + [
            True if name == "dummy" else "" if name == "stereo" else 0
        ]
    return (reduced, heavy, groups, reservoir), None


def map_reaction_hydrogen_quotient(
    reaction,
    *,
    library_path,
    CD="minimal",
    CD_mode="exact",
    balance="strict",
    time_limit_seconds=10.0,
    max_mappings=100_000,
    max_bijections=None,
    max_hydrogen_plans=10_000,
    classification_max_search_nodes=100_000,
):
    """Map all minimum Lewis classes through a checked H-leaf quotient.

    Full graph electron-anchor weights are retained verbatim, so heavy-atom
    nonbonding electrons are never re-estimated after hydrogens are removed.
    Unsupported inputs and numeric CD queries use the full explicit-H engine.
    Assignment-space cardinality and its optional admission cap concern the
    reduced problem; full labeled multiplicities remain explicitly unknown.
    """

    # Some RDKit stereo classes have no CIP label. Preserve the full search
    # contract by declining reduction for every explicit chiral annotation.
    raw_stereo = False
    if isinstance(reaction, str) and ("@" in reaction or "|" in reaction):
        raw_stereo = True

    def search(graphs, **options):
        elapsed_start = time.perf_counter()
        seconds = options["time_limit_seconds"]
        deadline = None if seconds is None else elapsed_start + seconds
        options.pop("backend")
        if options["CD"] != "minimal":
            result = enumerate_cpp_mappings(graphs, **options)
            statistics = dict(result.backend_statistics or {})
            statistics["hydrogen_quotient_fallback"] = "numeric_distance_query"
            return replace(result, backend_statistics=statistics)
        prepared = [_quotient_graph(graph) for graph in graphs]
        failure = next((reason for _, reason in prepared if reason), None)
        if raw_stereo:
            failure = "annotated_stereochemistry"
        if failure:
            options["time_limit_seconds"] = (
                None if deadline is None else max(0, deadline - time.perf_counter())
            )
            result = enumerate_cpp_mappings(graphs, **options)
            statistics = dict(result.backend_statistics or {})
            statistics["hydrogen_quotient_fallback"] = failure
            return replace(result, backend_statistics=statistics)
        left, right = [value for value, _ in prepared]
        reduced = left[0], right[0]
        callback = options["mapping_callback"]
        observed = plans_observed = 0

        def remaining():
            return None if deadline is None else max(0, deadline - time.perf_counter())

        def lift(mapping, cost):
            nonlocal observed, plans_observed
            original_mapping = [-1] * len(graphs[0].labels)
            left_heavy, right_heavy = left[1], right[1]
            for i, atom in enumerate(left_heavy):
                original_mapping[atom] = right_heavy[mapping[i]]
            reactant_groups = [list(group) for group in left[2]] + [list(left[3])]
            product_groups = [list(right[2][mapping[i]]) for i in range(len(left_heavy))]
            product_groups.append(list(right[3]))
            rcounts = [len(group) for group in reactant_groups]
            pcounts = [len(group) for group in product_groups]
            plans = enumerate_minimal_hydrogen_transfers(
                rcounts, pcounts, range(len(rcounts)), max_plans=max_hydrogen_plans,
                time_limit_seconds=remaining(),
            )
            for plan in plans.plans:
                if deadline is not None and time.perf_counter() >= deadline:
                    raise NativeEnumerationStop("hydrogen_quotient:time_limit")
                source = [list(group) for group in reactant_groups]
                target = [list(group) for group in product_groups]
                lifted = list(original_mapping)

                def transfer(i, j, count):
                    for atom, image in zip(source[i][:count], target[j][:count]):
                        lifted[atom] = image
                    del source[i][:count]
                    del target[j][:count]

                for i, count in enumerate(plan.preserved):
                    transfer(i, i, count)
                for i, j, count in plan.transfers:
                    transfer(i, j, count)
                if sorted(lifted) != list(range(len(lifted))):
                    raise RuntimeError("H quotient did not lift a compatible bijection")
                if chemical_distance(graphs, lifted, False) != cost:
                    raise RuntimeError("H quotient cost differs from full Lewis cost")
                observed += 1
                callback(lifted, cost)
                if options["max_mappings"] is not None and observed >= options["max_mappings"]:
                    raise NativeEnumerationStop("mapping_limit")
            plans_observed += plans.observed_plan_count
            if not plans.complete:
                raise NativeEnumerationStop("hydrogen_quotient:" + plans.truncation_reason)

        query = dict(options)
        query.update(
            mapping_callback=lift, collect_mappings=False, time_limit_seconds=remaining()
        )
        result = enumerate_cpp_mappings(reduced, **query)
        statistics = dict(result.backend_statistics or {})
        statistics.update(
            hydrogen_quotient=True, full_nodes=len(graphs[0].labels),
            quotient_nodes=len(reduced[0].labels), hydrogen_transport_plans=plans_observed,
            mapping_count_scope="lifted_hydrogen_transport_candidates",
        )
        return replace(
            result, selected_mapping_count=observed,
            selected_labeled_mapping_count=None,
            scope="full_lewis_minimum_via_terminal_hydrogen_quotient",
            backend_statistics=statistics,
        )

    return _map_reaction_with_search(
        reaction, search_function=search, backend="cpp", library_path=library_path,
        CD=CD, CD_mode=CD_mode, objective="lewis", hydrogen="explicit", balance=balance,
        time_limit_seconds=time_limit_seconds, max_mappings=max_mappings,
        max_bijections=max_bijections, max_hydrogen_plans=max_hydrogen_plans,
        classification_max_search_nodes=classification_max_search_nodes,
    )


__all__ = ["map_reaction_hydrogen_quotient"]
