from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class DerivationRecord:
    """
    Abstract direct derivation record.

    This stores the reaction event as multisets of standardized species without
    changing the public NetworkX CRN representation.
    """

    event_id: int
    label: str
    step: int
    rule_index: int
    reactants: Tuple[str, ...]
    products: Tuple[str, ...]


@dataclass
class DerivationLog:
    records: List[DerivationRecord] = field(default_factory=list)

    def clear(self) -> None:
        self.records.clear()

    def append(
        self,
        *,
        event_id: int,
        label: str,
        step: int,
        rule_index: int,
        reactants: Tuple[str, ...],
        products: Tuple[str, ...],
    ) -> None:
        self.records.append(
            DerivationRecord(
                event_id=int(event_id),
                label=str(label),
                step=int(step),
                rule_index=int(rule_index),
                reactants=tuple(reactants),
                products=tuple(products),
            )
        )

    def as_dicts(self) -> List[Dict[str, object]]:
        return [
            {
                "event_id": r.event_id,
                "label": r.label,
                "step": r.step,
                "rule_index": r.rule_index,
                "reactants": list(r.reactants),
                "products": list(r.products),
            }
            for r in self.records
        ]
