from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Set, Any

from .reaction import Reaction
from .exceptions import CRNError


class ReactionNetwork:
    """
    Container for :class:`crn.reaction.Reaction` records with a fluent API.

    Builder pattern
    ---------------
    - Construct with an iterable of reactions or :py:meth:`from_raw_list`.
    - Use fluent mutators (e.g., :py:meth:`keep_reactions`, :py:meth:`keep_molecules`)
      to refine the in-memory selection; each returns ``self`` for chaining.
    - Access results via properties (:pyattr:`reactions`, :pyattr:`n_reactions`).

    :param reactions: Initial reaction iterable.
    """

    def __init__(self, reactions: Iterable[Reaction] = ()) -> None:
        """
        Initialize a ReactionNetwork.

        :param reactions: Iterable of Reaction objects to populate the network.
        :raises CRNError: If duplicate reaction ids are encountered.
        """
        self._reactions: Dict[int, Reaction] = {}
        for r in reactions:
            if r.id in self._reactions:
                raise CRNError(
                    f"Duplicate reaction id {r.id} in ReactionNetwork construction"
                )
            self._reactions[r.id] = r
        self._view_ids: Optional[Set[int]] = None  # None => full view

    # ---- Construction ----
    @classmethod
    def from_raw_list(
        cls,
        raw_list: List[str],
        standardizer: Optional[object] = None,
        remove_aam: bool = True,
    ) -> "ReactionNetwork":
        """
        Construct a ReactionNetwork from a list of reaction SMILES.

        Always calls Reaction.standardize(...) so each Reaction.canonical_raw is set.
        If no standardizer is provided, Reaction.standardize() will set
        canonical_raw == original_raw (safe fallback).
        """
        reactions: List[Reaction] = []
        for i, raw in enumerate(raw_list):
            r = Reaction(id=i, original_raw=raw)
            # Always call standardize — Reaction.standardize must handle standardizer=None
            r.standardize(standardizer, remove_aam=remove_aam)
            r.build()
            reactions.append(r)
        return cls(reactions)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ReactionNetwork":
        """
        Create a network from dictionary produced by :py:meth:`to_dict`.

        :param d: Dictionary with a "reactions" list.
        :returns: ReactionNetwork instance.
        :raises CRNError: If the dictionary is malformed or Reaction construction fails.
        """
        try:
            raw_rxns = d.get("reactions", [])
            rxns = [Reaction.from_dict(x) for x in raw_rxns]
        except Exception as exc:
            raise CRNError(
                "Malformed network dictionary: failed to construct Reaction objects"
            ) from exc
        return ReactionNetwork(rxns)

    # ---- Serialization ----
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the full network (not the view) to a dict.

        :returns: Dict suitable for JSON serialization.
        """
        return {
            "reactions": [
                r.to_dict()
                for r in sorted(self._reactions.values(), key=lambda x: x.id)
            ]
        }

    def to_json(self, path: str) -> "ReactionNetwork":
        """
        Save full network JSON to ``path`` and return ``self``.

        :param path: Output file path.
        :returns: self
        :raises CRNError: On file write errors.
        """
        apath = os.path.abspath(path)
        try:
            with open(apath, "w", encoding="utf-8") as fh:
                json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        except OSError as exc:
            raise CRNError(f"Failed to write ReactionNetwork JSON to {apath}") from exc
        return self

    @staticmethod
    def from_json(path: str) -> "ReactionNetwork":
        """
        Load network from JSON file.

        :param path: Input file path.
        :returns: New ReactionNetwork instance.
        :raises CRNError: If file I/O or JSON parsing fails, or content is invalid.
        """
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError as exc:
            raise CRNError(f"JSON file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise CRNError(f"Invalid JSON in {path}") from exc
        except OSError as exc:
            raise CRNError(f"Failed to read JSON from {path}") from exc

        try:
            return ReactionNetwork.from_dict(data)
        except CRNError:
            raise
        except Exception as exc:
            raise CRNError(
                f"Failed to parse ReactionNetwork from JSON content in {path}"
            ) from exc

    # ---- Fluent view mutators ----
    def reset_view(self) -> "ReactionNetwork":
        """
        Reset any active view filters and return ``self``.

        :returns: self
        """
        self._view_ids = None
        return self

    def keep_reactions(self, ids: Iterable[int]) -> "ReactionNetwork":
        """
        Restrict the active view to a set of reaction ids.

        :param ids: Reaction ids to keep (iterable of ints).
        :returns: self
        :raises CRNError: If ids is not an iterable of integers.
        """
        try:
            ids_set = set(int(x) for x in ids)
        except Exception as exc:
            raise CRNError("keep_reactions expects an iterable of integers") from exc
        valid = ids_set & set(self._reactions.keys())
        self._view_ids = valid if self._view_ids is None else (self._view_ids & valid)
        return self

    def keep_molecules(self, tokens: Iterable[str]) -> "ReactionNetwork":
        """
        Restrict view to reactions touching at least one token.

        :param tokens: Species tokens to match (iterable of strings).
        :returns: self
        :raises CRNError: If tokens is not an iterable of strings.
        """
        try:
            want = set(str(t) for t in tokens)
        except Exception as exc:
            raise CRNError("keep_molecules expects an iterable of strings") from exc

        keep: Set[int] = set()
        for rid, rx in self._reactions.items():
            if want & set(rx.reactants_can.keys()) or want & set(
                rx.products_can.keys()
            ):
                keep.add(rid)
        self._view_ids = keep if self._view_ids is None else (self._view_ids & keep)
        return self

    # ---- Accessors / helpers ----
    @property
    def reactions(self) -> Dict[int, Reaction]:
        """
        Dictionary of reactions in the current view.

        :returns: Mapping reaction id -> Reaction
        """
        if self._view_ids is None:
            return dict(self._reactions)
        return {rid: self._reactions[rid] for rid in sorted(self._view_ids)}

    @property
    def n_reactions(self) -> int:
        """
        Number of reactions in the current view.

        :returns: int
        """
        return len(self.reactions)

    def get_reaction(self, rid: int) -> Reaction:
        """
        Return the Reaction with the given id.

        :param rid: Reaction id.
        :returns: Reaction instance.
        :raises CRNError: If reaction id not found.
        """
        try:
            return self._reactions[rid]
        except KeyError:
            raise CRNError(f"Reaction id {rid} not found in network")

    def add_reaction(self, reaction: Reaction) -> "ReactionNetwork":
        """
        Add a Reaction to the network.

        :param reaction: Reaction object to add.
        :returns: self
        :raises CRNError: If a reaction with the same id already exists.
        """
        if reaction.id in self._reactions:
            raise CRNError(f"Reaction id {reaction.id} already exists in network")
        self._reactions[reaction.id] = reaction
        return self

    def remove_reaction(self, rid: int) -> "ReactionNetwork":
        """
        Remove a Reaction by id.

        :param rid: Reaction id to remove.
        :returns: self
        :raises CRNError: If reaction id not found.
        """
        try:
            del self._reactions[rid]
            # If view was restricting to this id, update view
            if self._view_ids is not None:
                self._view_ids.discard(rid)
            return self
        except KeyError:
            raise CRNError(f"Reaction id {rid} not found; cannot remove")

    # ---- Misc / dunder ----
    def __len__(self) -> int:
        """
        Length equals number of reactions in current view.

        :returns: int
        """
        return self.n_reactions

    def __iter__(self):
        """
        Iterate over Reaction objects in the current view.
        """
        return iter(self.reactions.values())

    def __repr__(self) -> str:
        """
        Compact representation for debugging.
        """
        return f"<ReactionNetwork n={self.n_reactions}>"
