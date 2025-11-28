# crn/pathway.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple
from collections import Counter

from .utils import inflow_outflow, format_state
from .network import ReactionNetwork


@dataclass
class Pathway:
    """
    Ordered sequence of reaction applications.

    :param reaction_ids: Reaction ids in forward chronological order.
    :param states: Canonical states (multisets) aligned with reaction steps.
    """

    reaction_ids: List[int] = field(default_factory=list)
    states: List[Counter] = field(default_factory=list)

    # ---- Fluent mutators ----
    def append(self, rid: int, state_after: Counter) -> "Pathway":
        """
        Append a step to the pathway.

        :param rid: Reaction id.
        :param state_after: State after applying the reaction.
        :returns: self
        """
        self.reaction_ids.append(rid)
        self.states.append(Counter(state_after))
        return self

    def extend(self, rids: List[int], states_after: List[Counter]) -> "Pathway":
        """
        Extend pathway by multiple steps.

        :param rids: Reaction ids.
        :param states_after: List of states after each reaction.
        :returns: self
        """
        self.reaction_ids.extend(list(rids))
        self.states.extend([Counter(s) for s in states_after])
        return self

    # ---- Properties ----
    @property
    def steps(self) -> int:
        """Number of steps in the pathway."""
        return len(self.reaction_ids)

    @property
    def start(self) -> Counter:
        """Initial state (empty Counter if none)."""
        return self.states[0] if self.states else Counter()

    @property
    def end(self) -> Counter:
        """Final state (empty Counter if none)."""
        return self.states[-1] if self.states else Counter()

    # ---- Accessors for original / canonical RSMI strings ----
    def as_original_rsmi_list(self, network: ReactionNetwork) -> List[str]:
        """
        Return the original (atom-mapped) reaction SMILES sequence for this pathway.

        :param network: ReactionNetwork instance that contains the Reaction objects.
        :returns: list of original_raw strings in forward order.
        """
        return [network.reactions[rid].original_raw for rid in self.reaction_ids]

    def as_canonical_rsmi_list(self, network: ReactionNetwork) -> List[str]:
        """
        Return the canonical (standardized) reaction SMILES sequence for this pathway.

        :param network: ReactionNetwork instance that contains the Reaction objects.
        :returns: list of canonical_raw strings in forward order.
        """
        return [network.reactions[rid].canonical_raw for rid in self.reaction_ids]

    # ---- Analytics ----
    def compute_flow(self) -> Tuple[Counter, Counter]:
        """
        Compute inflow/outflow for the full pathway w.r.t. its initial state.

        :returns: (inflow_counter, outflow_counter)
        """
        if not self.states:
            return Counter(), Counter()
        return inflow_outflow(self.start, self.end)

    def summary(self) -> str:
        """
        Human-readable summary (steps + inflow/outflow).
        """
        inflow, outflow = self.compute_flow()
        return f"Pathway(steps={self.steps}) inflow={format_state(inflow)} outflow={format_state(outflow)}"

    # ---- (De)serialization ----
    def to_dict(self) -> dict:
        """Serialize pathway to dict."""
        return {
            "reaction_ids": list(self.reaction_ids),
            "states": [dict(s) for s in self.states],
        }

    @staticmethod
    def from_dict(d: dict) -> "Pathway":
        """Deserialize from dict."""
        return Pathway(
            reaction_ids=list(d.get("reaction_ids", [])),
            states=[Counter(s) for s in d.get("states", [])],
        )

    def __repr__(self) -> str:
        return f"Pathway(steps={self.steps})"
