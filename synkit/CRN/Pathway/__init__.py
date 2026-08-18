"""Pathway-level analysis over a reaction network.

Three complementary questions are answered here:

- **Reachability** — which species can be produced at all, and in how many
  synthesis layers.
- **Path finding** — which concrete routes connect a source set to a target.
- **Realizability** — whether a requested reaction flow can actually be fired
  in some order without ever going negative.

.. rubric:: Example

.. code-block:: python

    from synkit.CRN.Structure import SynCRN
    from synkit.CRN.Pathway import PathwayReachability

    crn = SynCRN.from_reaction_strings(["A>>B", "B>>C"])
    result = PathwayReachability().load_syncrn(crn).compute_layers_set({"A"})
    print(result)
"""

from __future__ import annotations

from .reachability import (
    PathwayReachability,
    ReachabilityConfig,
    ReachabilityLayer,
    ReachabilityResult,
    run_reachability_from_syncrn,
    syncrn_to_reachability_inputs,
)
from .realizability import (
    PathwayRealizability,
    RealizabilityConfig,
    RealizabilitySummary,
    run_realizability_from_syncrn,
    syncrn_to_pr_inputs,
)
from .pathfinder import (
    PathFinderConfig,
    PathwayCandidate,
    PathwayFinder,
    run_pathfinder_from_syncrn,
)

__all__ = [
    # reachability
    "PathwayReachability",
    "ReachabilityConfig",
    "ReachabilityLayer",
    "ReachabilityResult",
    "run_reachability_from_syncrn",
    "syncrn_to_reachability_inputs",
    # realizability
    "PathwayRealizability",
    "RealizabilityConfig",
    "RealizabilitySummary",
    "run_realizability_from_syncrn",
    "syncrn_to_pr_inputs",
    # path finding
    "PathFinderConfig",
    "PathwayCandidate",
    "PathwayFinder",
    "run_pathfinder_from_syncrn",
]
