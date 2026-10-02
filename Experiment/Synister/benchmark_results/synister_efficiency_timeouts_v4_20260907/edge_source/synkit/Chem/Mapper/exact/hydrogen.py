"""Exact second-pass hydrogen transfer plans for a fixed heavy-atom AAM."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class HydrogenTransferPlan:
    """One chemically distinct minimum-edit parent-to-parent hydrogen flow."""

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
    reactant = tuple(int(value) for value in reactant_hcounts)
    product = tuple(int(value) for value in product_hcounts)
    mapping = tuple(int(image) for image in heavy_mapping)
    size = len(reactant)
    if len(product) != size or sorted(mapping) != list(range(size)):
        raise ValueError("H-count arrays and heavy_mapping must have equal size")
    if any(value < 0 for value in reactant + product):
        raise ValueError("hydrogen counts must be non-negative")
    if isinstance(max_plans, bool) or (
        max_plans is not None and (not isinstance(max_plans, int) or max_plans < 1)
    ):
        raise ValueError("max_plans must be a positive integer or None")
    if not isinstance(collect_plans, bool):
        raise TypeError("collect_plans must be boolean")

    mapped_product = tuple(product[image] for image in mapping)
    if sum(reactant) != sum(mapped_product):
        raise ValueError("reactant and product hydrogen inventories differ")
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
    "HydrogenTransferPlan",
    "enumerate_lgp_hydrogen_transfers",
    "enumerate_minimal_hydrogen_transfers",
]
