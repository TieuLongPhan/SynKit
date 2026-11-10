"""Exact composition of one persistent nonnegative integer resource.

This algebra retains intermediate availability. It is independent of atom
alignment, graph gluing, inferred populations and physical conservation.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceUpdate:
    """Partial translation ``x -> x - demand + supply`` for ``x >= demand``.

    Demand and supply may both be positive: their common part is a guard,
    not redundant bookkeeping. Coordinates of different atoms or resources
    must be aligned explicitly by the caller before using this operation.
    """

    demand: int = 0
    supply: int = 0

    def __post_init__(self):
        if any(type(value) is not int or value < 0
               for value in (self.demand, self.supply)):
            raise ValueError("Resource demand and supply require nonnegative integers.")

    @property
    def delta(self):
        return self.supply - self.demand

    def apply(self, population: int) -> int:
        """Apply to an available population; refuse invalid or exhausted input."""
        if type(population) is not int or population < 0:
            raise ValueError("A resource population requires a nonnegative integer.")
        if population < self.demand:
            raise ValueError("The resource population does not satisfy the demand.")
        return population + self.delta

    def then(self, after: "ResourceUpdate") -> "ResourceUpdate":
        """Return the exact sequential partial map, including its domain guard."""
        if not isinstance(after, ResourceUpdate):
            raise TypeError("Resource composition requires another ResourceUpdate.")
        return ResourceUpdate(
            self.demand + max(0, after.demand - self.supply),
            after.supply + max(0, self.supply - after.demand),
        )

    def reversed(self) -> "ResourceUpdate":
        """Return the partial inverse, defined on outputs of this update."""
        return ResourceUpdate(self.supply, self.demand)
