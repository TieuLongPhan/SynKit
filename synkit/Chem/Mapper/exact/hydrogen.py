"""Exact second-pass hydrogen transfer plans for a fixed heavy-atom AAM."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class HydrogenTransferPlan:
    """One parent-index-distinct minimum-edit hydrogen flow."""

    transfers: tuple[tuple[int, int, int], ...]
    preserved: tuple[int, ...]
    distance: int
    labeled_mapping_count: int


@dataclass(frozen=True)
class HydrogenEnumerationResult:
    """Compressed exact hydrogen lifts for one fixed heavy-atom mapping."""

    plans: tuple[HydrogenTransferPlan, ...]
    minimum_distance: int
    labeled_mapping_count: int
    complete: bool
    observed_plan_count: int


@dataclass(frozen=True)
class HydrogenLiftSummary:
    """Closed-form optimum over all explicit-H bijections for one heavy map."""

    minimum_distance: int
    transferred_hydrogen_count: int
    labeled_mapping_count: int


def _transport_hydrogen_counts(
    reactant_hcounts,
    product_hcounts,
    heavy_mapping,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate counts and transport product parents into reactant order."""
    reactant = tuple(int(value) for value in reactant_hcounts)
    product = tuple(int(value) for value in product_hcounts)
    mapping = tuple(int(image) for image in heavy_mapping)
    size = len(reactant)
    if len(product) != size or sorted(mapping) != list(range(size)):
        raise ValueError("H-count arrays and heavy_mapping must have equal size")
    if any(value < 0 for value in reactant + product):
        raise ValueError("hydrogen counts must be non-negative")
    mapped_product = tuple(product[image] for image in mapping)
    if sum(reactant) != sum(mapped_product):
        raise ValueError("reactant and product hydrogen inventories differ")
    return reactant, mapped_product


def summarize_minimal_hydrogen_lifts(
    reactant_hcounts,
    product_hcounts,
    heavy_mapping,
) -> HydrogenLiftSummary:
    """Return the exact best H distance and multiplicity without flow search.

    Hydrogens are assumed to be indistinguishable in the objective, attached
    by one unit-weight bond to a single heavy parent, and balanced between the
    endpoints.  For transported product counts ``q_i``, the minimum distance
    is ``sum_i |r_i-q_i|``.  If ``T`` hydrogens change parent, the number of
    labeled explicit-H bijections attaining that distance is

    ``T! * prod_i(max(r_i, q_i)! / abs(r_i-q_i)!)``.

    The calculation is linear in the number of heavy atoms apart from integer
    arithmetic; it does not enumerate donor--acceptor flow matrices.
    """
    reactant, mapped_product = _transport_hydrogen_counts(
        reactant_hcounts,
        product_hcounts,
        heavy_mapping,
    )
    transferred = sum(
        max(left - right, 0) for left, right in zip(reactant, mapped_product)
    )
    labeled_count = math.factorial(transferred)
    for left, right in zip(reactant, mapped_product):
        labeled_count *= math.factorial(max(left, right)) // math.factorial(
            abs(left - right)
        )
    return HydrogenLiftSummary(
        minimum_distance=2 * transferred,
        transferred_hydrogen_count=transferred,
        labeled_mapping_count=labeled_count,
    )


def _labeled_count(reactant_counts, product_counts, cells):
    numerator = math.prod(math.factorial(value) for value in reactant_counts)
    numerator *= math.prod(math.factorial(value) for value in product_counts)
    denominator = math.prod(math.factorial(value) for value in cells)
    return numerator // denominator


def enumerate_minimal_hydrogen_transfers(  # noqa: C901
    reactant_hcounts,
    product_hcounts,
    heavy_mapping,
    *,
    max_plans=None,
    collect_plans=True,
) -> HydrogenEnumerationResult:
    """Enumerate minimum-edit implicit-H lifts of a heavy-atom mapping.

    The product H counts are first transported into reactant order by
    ``heavy_mapping``.  The maximum possible number of H--parent bonds is kept;
    remaining donor hydrogens are distributed over deficit parents.  Plans are
    parent-flow matrices, while the number of labeled explicit-H bijections for
    each matrix is computed exactly as

    ``prod(row_count!) * prod(column_count!) / prod(cell_count!)``.
    """
    reactant, mapped_product = _transport_hydrogen_counts(
        reactant_hcounts,
        product_hcounts,
        heavy_mapping,
    )
    if isinstance(max_plans, bool) or (
        max_plans is not None and (not isinstance(max_plans, int) or max_plans < 1)
    ):
        raise ValueError("max_plans must be a positive integer or None")
    if not isinstance(collect_plans, bool):
        raise TypeError("collect_plans must be boolean")

    preserved = tuple(min(left, right) for left, right in zip(reactant, mapped_product))
    donors = [left - kept for left, kept in zip(reactant, preserved)]
    acceptors = [right - kept for right, kept in zip(mapped_product, preserved)]
    donor_atoms = [index for index, count in enumerate(donors) if count]
    acceptor_atoms = [index for index, count in enumerate(acceptors) if count]
    minimum_distance = sum(donors) + sum(acceptors)
    plans = []
    observed = 0
    labeled_total = 0
    complete = True
    flows = {}

    def emit():
        nonlocal observed, labeled_total, complete
        if max_plans is not None and observed >= max_plans:
            complete = False
            return False
        transfers = tuple(
            (source, target, count)
            for (source, target), count in sorted(flows.items())
            if count
        )
        cells = list(preserved) + [count for _, _, count in transfers]
        multiplicity = _labeled_count(reactant, mapped_product, cells)
        observed += 1
        labeled_total += multiplicity
        if collect_plans:
            plans.append(
                HydrogenTransferPlan(
                    transfers=transfers,
                    preserved=preserved,
                    distance=minimum_distance,
                    labeled_mapping_count=multiplicity,
                )
            )
        return True

    def assign_acceptors(row, column, remaining):
        if column == len(acceptor_atoms):
            if remaining == 0:
                yield None
            return
        target = acceptor_atoms[column]
        maximum = min(remaining, acceptors[target])
        for count in range(maximum + 1):
            flows[(donor_atoms[row], target)] = count
            acceptors[target] -= count
            yield from assign_acceptors(row, column + 1, remaining - count)
            acceptors[target] += count
        flows.pop((donor_atoms[row], target), None)

    def assign_donors(row):
        if not complete:
            return
        if row == len(donor_atoms):
            if not any(acceptors):
                emit()
            return
        source = donor_atoms[row]
        for _ in assign_acceptors(row, 0, donors[source]):
            assign_donors(row + 1)
            if not complete:
                return

    if donor_atoms:
        assign_donors(0)
    else:
        emit()
    return HydrogenEnumerationResult(
        plans=tuple(plans),
        minimum_distance=minimum_distance,
        labeled_mapping_count=labeled_total,
        complete=complete,
        observed_plan_count=observed,
    )


def enumerate_lgp_hydrogen_transfers(
    lgp,
    heavy_mapping,
    **kwargs,
) -> HydrogenEnumerationResult:
    """Read ``hcounts`` metadata from a Mapper graph pair and enumerate lifts."""
    counts = []
    for graph in lgp:
        values = graph.props.get("hcounts")
        if values is None or len(values) != len(graph.labels):
            raise ValueError("both endpoint graphs require per-atom hcounts metadata")
        counts.append(values)
    return enumerate_minimal_hydrogen_transfers(
        counts[0],
        counts[1],
        heavy_mapping,
        **kwargs,
    )


__all__ = [
    "HydrogenEnumerationResult",
    "HydrogenLiftSummary",
    "HydrogenTransferPlan",
    "enumerate_lgp_hydrogen_transfers",
    "enumerate_minimal_hydrogen_transfers",
    "summarize_minimal_hydrogen_lifts",
]
