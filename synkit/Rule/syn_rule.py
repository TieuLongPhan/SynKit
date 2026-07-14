"""syn_rule.py
================
Immutable description of a reaction template (SynRule) with canonical forms
and optional implicit‐hydrogen stripping.

Key features
------------
* **Fragment decomposition** – splits the ITS graph into rc, left, and right.
* **Implicit H‐handling** – converts explicit H nodes into hcount + h_pairs.
* **Canonicalisation** – wraps rc/left/right in SynGraph for stable signatures.
* **Value‑object semantics** – `__eq__`/`__hash__` use fragment signatures.

Quick start
-----------
>>> from synkit.Graph.syn_rule import SynRule
>>> rule = SynRule.from_smart("[CH3:1]C>>[CH2:1]C")
>>> rule.left.signature, rule.right.signature
('abc123...', 'def456...')

"""

from __future__ import annotations
from typing import Any, Mapping, Optional, Tuple

import networkx as nx

from synkit.Graph.syn_graph import SynGraph
from synkit.Graph.canon_graph import GraphCanonicaliser
from synkit.Graph.ITS.its_decompose import its_decompose
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Graph.ITS.its_reverter import ITSReverter
from synkit.Graph.Hyrogen._misc import normalize_h_pair_graph
from synkit.Graph.Stereo import StereoOutcome
from synkit.IO.chem_converter import (
    ITSFormat,
    detect_its_format,
    rsmi_to_its,
    gml_to_its,
)

__all__ = ["SynRule"]


class SynRule:
    """
    Immutable reaction template: rc, left, and right fragments as SynGraph Object.

    Parameters
    ----------
    rc_graph : nx.Graph
        Raw reaction-centre (RC) graph.
    name : str, default ``"rule"``
        Identifier for the rule.
    canonicaliser : Optional[GraphCanonicaliser]
        Custom canonicaliser; if *None* a default is created.
    canon : bool, default ``True``
        If *True*, build canonical forms and SHA-256 signatures.
    implicit_h : bool, default ``True``
        Convert explicit hydrogens in the **rc/left/right** fragments to an
        integer ``hcount`` attribute and record cross-fragment hydrogen pairs
        in a ``h_pairs`` attribute.
    stereo_outcomes : mapping, optional
        Explicit product-branch declarations keyed by descriptor target. A
        ``RACEMIC`` outcome uses the stored product descriptor as a reference
        orientation and makes application emit it and its inverse.

    Attributes
    ----------
    rc : SynGraph
        Wrapped reaction‐centre graph.
    left : SynGraph
        Wrapped left fragment.
    right : SynGraph
        Wrapped right fragment.
    canonical_smiles : Optional[Tuple[str,str]]
        Pair of left/right fragment SHA‐256 signatures (or None if canon=False).
    """

    # ------------------------------------------------------------------ #
    # Alternate constructors                                             #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_smart(
        cls,
        smart: str,
        name: str = "rule",
        canonicaliser: Optional[GraphCanonicaliser] = None,
        *,
        canon: bool = True,
        implicit_h: bool = True,
        format: ITSFormat = "typesGH",
        stereo_outcomes: Optional[
            Mapping[str, StereoOutcome | str | Mapping[str, Any]]
        ] = None,
    ) -> "SynRule":
        """Instantiate from a SMARTS string."""
        return cls(
            rsmi_to_its(smart, format=format),
            name=name,
            canonicaliser=canonicaliser,
            canon=canon,
            implicit_h=implicit_h,
            format=format,
            stereo_outcomes=stereo_outcomes,
        )

    @classmethod
    def from_gml(
        cls,
        gml: str,
        name: str = "rule",
        canonicaliser: Optional[GraphCanonicaliser] = None,
        *,
        canon: bool = True,
        implicit_h: bool = True,
    ) -> "SynRule":
        """Instantiate from a GML string."""
        return cls(
            gml_to_its(gml),
            name=name,
            canonicaliser=canonicaliser,
            canon=canon,
            implicit_h=implicit_h,
        )

    # ------------------------------------------------------------------ #
    # Initialiser                                                        #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        rc: nx.Graph,
        name: str = "rule",
        canonicaliser: Optional[GraphCanonicaliser] = None,
        *,
        canon: bool = True,
        implicit_h: bool = True,
        format: Optional[ITSFormat] = None,
        stereo_outcomes: Optional[
            Mapping[str, StereoOutcome | str | Mapping[str, Any]]
        ] = None,
    ) -> None:
        self._name = name
        self._canon_enabled = canon
        self._implicit_h = implicit_h
        self._canonicaliser = canonicaliser or GraphCanonicaliser()

        # Fragment decomposition
        rc_graph = rc.copy()
        self._format = format or detect_its_format(rc_graph)
        if self._implicit_h:
            rc_graph = normalize_h_pair_graph(rc_graph)

        left_graph, right_graph = self._decompose(rc_graph, self._format)

        # Optional H-stripping
        if self._implicit_h and self._format == "typesGH":
            self._strip_explicit_h(rc_graph, left_graph, right_graph)
            # Update typesGH tuples with new hcount.
            for node, att in rc_graph.nodes(data=True):
                t0, t1 = att["typesGH"]
                new_t0 = (
                    t0[0],
                    t0[1],
                    left_graph.nodes[node]["hcount"] + t0[2],
                    t0[3],
                    t0[4],
                )
                new_t1 = (
                    t1[0],
                    t1[1],
                    right_graph.nodes[node]["hcount"] + t1[2],
                    t1[3],
                    t1[4],
                )
                att["typesGH"] = (new_t0, new_t1)
            left_graph, right_graph = self._decompose(rc_graph, self._format)
        elif self._implicit_h and self._format == "tuple":
            self._strip_explicit_h_tuple(rc_graph, left_graph, right_graph)
            left_graph, right_graph = self._decompose(rc_graph, self._format)
        stereo_sides = rc_graph.graph.get("stereo_descriptors", {})
        self.stereo_guards = dict(stereo_sides.get("reactant", {}))
        self.stereo_effects = dict(rc_graph.graph.get("stereo_changes", {}))
        raw_outcomes = (
            stereo_outcomes
            if stereo_outcomes is not None
            else rc_graph.graph.get("stereo_outcomes", {})
        )
        self.stereo_outcomes = {
            key: StereoOutcome.from_value(value) for key, value in raw_outcomes.items()
        }
        self.stereo_query_policies = dict(
            rc_graph.graph.get("stereo_query_policies", {})
        )
        invalid_query_policies = {
            value
            for value in self.stereo_query_policies.values()
            if value not in {"exact", "wildcard", "either"}
        }
        if invalid_query_policies:
            raise ValueError(
                "Stereo query policies must be 'exact', 'wildcard', or "
                f"'either'; received {sorted(invalid_query_policies)!r}."
            )
        self._reverse_stereo_outcomes = {
            key: StereoOutcome.from_value(value)
            for key, value in rc_graph.graph.get("stereo_reverse_outcomes", {}).items()
        }
        self._reverse_stereo_query_policies = dict(
            rc_graph.graph.get("stereo_reverse_query_policies", {})
        )
        self._validate_stereo_outcomes()
        rc_graph.graph["stereo_outcomes"] = {
            key: outcome.to_dict() for key, outcome in self.stereo_outcomes.items()
        }
        rc_graph.graph["stereo_query_policies"] = dict(self.stereo_query_policies)
        if self._reverse_stereo_outcomes:
            rc_graph.graph["stereo_reverse_outcomes"] = {
                key: outcome.to_dict()
                for key, outcome in self._reverse_stereo_outcomes.items()
            }
            rc_graph.graph["stereo_reverse_query_policies"] = dict(
                self._reverse_stereo_query_policies
            )

        # ---------- wrap graphs ---------------------------------------- #
        self.rc = SynGraph(rc_graph, self._canonicaliser, canon=canon)
        self.left = SynGraph(left_graph, self._canonicaliser, canon=canon)
        self.right = SynGraph(right_graph, self._canonicaliser, canon=canon)

        self.stereo_mode = "ignore"

        self.canonical_smiles: Optional[Tuple[str, str]] = (
            (self.left.signature, self.right.signature) if canon else None
        )

    # ================================================================== #
    # Private utilities                                                  #
    # ================================================================== #
    def _validate_stereo_outcomes(self) -> None:
        """Require explicit branching declarations to match stored effects."""
        for key, outcome in self.stereo_outcomes.items():
            if key not in self.stereo_effects:
                raise ValueError(f"Stereo outcome target {key!r} has no rule effect.")
            change = self.stereo_effects[key]
            if outcome.kind == "SINGLE":
                continue
            if (
                change.change != "FORMED"
                or change.after is None
                or change.after.parity not in (-1, 1)
            ):
                raise ValueError(
                    f"{outcome.kind} outcome is only valid for a specified, "
                    "newly formed chiral descriptor."
                )

    @staticmethod
    def _decompose(rc: nx.Graph, format: ITSFormat) -> tuple[nx.Graph, nx.Graph]:
        """Return left/right fragments for either supported ITS representation."""
        if format == "tuple":
            reverter = ITSReverter(rc)
            return reverter.to_reactant_graph(), reverter.to_product_graph()
        return its_decompose(rc)

    @staticmethod
    def _strip_explicit_h(
        rc: nx.Graph,
        left: nx.Graph,
        right: nx.Graph,
    ) -> None:
        """Remove explicit hydrogens from rc, left, right—but only when *both*
        left & right agree the H should be implicit.

        Otherwise an H remains explicit in all three graphs.
        """

        def _removable_on(graph: nx.Graph, h: str) -> bool:
            # H+ (no neighbors) ⇒ not removable
            nbrs = list(graph.neighbors(h))
            if not nbrs:
                return False
            # H–H only ⇒ not removable
            if all(graph.nodes[n].get("element") == "H" for n in nbrs):
                return False
            # otherwise bonded to ≥1 heavy ⇒ removable
            return True

        def _fully_removable(h: str) -> bool:
            # only remove if BOTH left and right say removable
            return _removable_on(left, h) and _removable_on(right, h)

        # 1) initialize hcount & h_pairs
        for g in (rc, left, right):
            for n, data in g.nodes(data=True):
                data["hcount"] = 0
                if data.get("element") != "H":
                    data.setdefault("h_pairs", [])

        # 2) shared H: only those removable on both sides
        shared = sorted(
            n
            for n, d in left.nodes(data=True)
            if d.get("element") == "H" and right.has_node(n) and _fully_removable(n)
        )

        pair_id = 1
        for h in shared:
            for g in (left, right, rc):
                if not g.has_node(h):
                    continue
                for nbr in list(g.neighbors(h)):
                    if g.nodes[nbr].get("element") != "H":
                        g.nodes[nbr]["hcount"] += 1
                        # only shared H get pair-IDs
                        g.nodes[nbr].setdefault("h_pairs", []).append(pair_id)
                g.remove_node(h)
            pair_id += 1

        # 3) remaining explicit H in any graph: strip only if fully_removable
        for g in (rc, left, right):
            for h in [n for n, d in g.nodes(data=True) if d.get("element") == "H"]:
                if not _fully_removable(h):
                    # at least one side wants to keep it explicit → skip
                    continue
                # else both agree → convert to implicit
                for nbr in list(g.neighbors(h)):
                    if g.nodes[nbr].get("element") != "H":
                        g.nodes[nbr]["hcount"] += 1
                g.remove_node(h)

    @staticmethod
    def _strip_explicit_h_tuple(
        rc: nx.Graph,
        left: nx.Graph,
        right: nx.Graph,
    ) -> None:
        """Tuple-style equivalent of legacy explicit-H stripping."""

        def _removable_on(graph: nx.Graph, h: int) -> bool:
            if not graph.has_node(h):
                return False
            nbrs = list(graph.neighbors(h))
            if not nbrs:
                return False
            return not all(graph.nodes[n].get("element") == "H" for n in nbrs)

        def _fully_removable(h: int) -> bool:
            return _removable_on(left, h) and _removable_on(right, h)

        for graph in (left, right):
            for _, data in graph.nodes(data=True):
                if data.get("element") != "H":
                    data.setdefault("h_pairs", [])
                    data.setdefault("h_pairs_left", [])
                    data.setdefault("h_pairs_right", [])
                    data.setdefault("h_pair_atom_maps", {})

        for _, data in rc.nodes(data=True):
            element = data.get("element")
            is_h = (
                isinstance(element, tuple)
                and len(element) == 2
                and all(value == "H" for value in element)
            )
            if not is_h:
                data.setdefault("h_pairs", [])
                data.setdefault("h_pairs_left", [])
                data.setdefault("h_pairs_right", [])
                data.setdefault("h_pair_atom_maps", {})

        removable = sorted(
            node
            for node, attrs in left.nodes(data=True)
            if attrs.get("element") == "H"
            and right.has_node(node)
            and _fully_removable(node)
        )

        for pair_id, h in enumerate(removable, start=1):
            atom_map = left.nodes[h].get("atom_map", h)
            for side, graph in (("left", left), ("right", right)):
                for nbr in list(graph.neighbors(h)):
                    if graph.nodes[nbr].get("element") != "H":
                        graph.nodes[nbr]["hcount"] += 1
                        graph.nodes[nbr].setdefault("h_pairs", []).append(pair_id)
                        graph.nodes[nbr].setdefault(f"h_pairs_{side}", []).append(
                            pair_id
                        )
                        graph.nodes[nbr].setdefault("h_pair_atom_maps", {})[
                            pair_id
                        ] = atom_map
                graph.remove_node(h)
            if rc.has_node(h):
                rc.remove_node(h)

        for node, attrs in rc.nodes(data=True):
            if node not in left or node not in right:
                continue
            if attrs.get("element") == ("H", "H"):
                continue
            left_h = left.nodes[node].get("hcount", 0)
            right_h = right.nodes[node].get("hcount", 0)
            attrs["hcount"] = (left_h, right_h)
            attrs["h_pairs"] = sorted(
                set(left.nodes[node].get("h_pairs", []))
                | set(right.nodes[node].get("h_pairs", []))
            )
            attrs["h_pairs_left"] = sorted(left.nodes[node].get("h_pairs_left", []))
            attrs["h_pairs_right"] = sorted(right.nodes[node].get("h_pairs_right", []))
            attrs["h_pair_atom_maps"] = {
                **left.nodes[node].get("h_pair_atom_maps", {}),
                **right.nodes[node].get("h_pair_atom_maps", {}),
            }
            typesgh = attrs.get("typesGH")
            if typesgh and len(typesgh) == 2:
                react_attr, prod_attr = typesgh
                attrs["typesGH"] = (
                    tuple(list(react_attr[:2]) + [left_h] + list(react_attr[3:])),
                    tuple(list(prod_attr[:2]) + [right_h] + list(prod_attr[3:])),
                )

    # ================================================================== #
    # Dunder methods                                                     #
    # ================================================================== #
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SynRule)
            and self.canonical_smiles == other.canonical_smiles
            and self._stereo_signature() == other._stereo_signature()
        )

    def __hash__(self) -> int:
        return hash((self.canonical_smiles, self._stereo_signature()))

    def _stereo_signature(self) -> tuple:
        guards = tuple(
            sorted(
                (key, descriptor.canonical_form())
                for key, descriptor in self.stereo_guards.items()
            )
        )
        effects = tuple(
            sorted(
                (
                    key,
                    change.change,
                    change.before.canonical_form() if change.before else None,
                    change.after.canonical_form() if change.after else None,
                    change.transition.canonical_form() if change.transition else None,
                )
                for key, change in self.stereo_effects.items()
            )
        )
        outcomes = tuple(
            sorted(
                (key, outcome.signature())
                for key, outcome in self.stereo_outcomes.items()
            )
        )
        query_policies = tuple(sorted(self.stereo_query_policies.items()))
        reverse_outcomes = tuple(
            sorted(
                (key, outcome.signature())
                for key, outcome in self._reverse_stereo_outcomes.items()
            )
        )
        return guards, effects, outcomes, query_policies, reverse_outcomes

    def __str__(self) -> str:
        if self._canon_enabled and self.canonical_smiles:
            ls, rs = self.canonical_smiles
            return f"<SynRule {self._name!r} left={ls[:8]}… right={rs[:8]}…>"
        return f"<SynRule {self._name!r} (raw only)>"

    def __repr__(self) -> str:
        try:
            v_rc, e_rc = self.rc.raw.number_of_nodes(), self.rc.raw.number_of_edges()
            v_l, e_l = self.left.raw.number_of_nodes(), self.left.raw.number_of_edges()
            v_r, e_r = (
                self.right.raw.number_of_nodes(),
                self.right.raw.number_of_edges(),
            )
        except Exception:
            v_rc = e_rc = v_l = e_l = v_r = e_r = 0
        return (
            f"SynRule(name={self._name!r}, "
            f"rc=(|V|={v_rc},|E|={e_rc}), "
            f"left=(|V|={v_l},|E|={e_l}), "
            f"right=(|V|={v_r},|E|={e_r}))"
        )

    # ================================================================== #
    # Public API                                                         #
    # ================================================================== #
    def reversed(self, *, balance_its: bool = False) -> "SynRule":
        """Return a rule with reactant/product stereo semantics reversed.

        A forward two-enantiomer product outcome becomes an ``either``
        reactant guard in reverse, because the reverse rule consumes either
        enantiomer but does not create two achiral products. Reversing again
        restores the original product outcome and query policies.
        """
        left_graph, right_graph = self._decompose(self.rc.raw, self._format)
        if self._format == "tuple":
            reversed_graph = ITSConstruction.construct(
                right_graph,
                left_graph,
                balance_its=balance_its,
            )
        else:
            reversed_graph = ITSConstruction.ITSGraph(
                right_graph,
                left_graph,
                balance_its=balance_its,
            )

        reversed_graph.graph["stereo_changes"] = {
            key: change.reverse() for key, change in self.stereo_effects.items()
        }
        sides = self.rc.raw.graph.get("stereo_descriptors", {})
        reversed_graph.graph["stereo_descriptors"] = {
            "reactant": dict(sides.get("product", {})),
            "product": dict(sides.get("reactant", {})),
        }
        if "transition" in sides:
            reversed_graph.graph["stereo_descriptors"]["transition"] = dict(
                sides["transition"]
            )

        if self._reverse_stereo_outcomes:
            # This rule is already the reverse view of a branching rule.
            reversed_graph.graph["stereo_outcomes"] = {
                key: outcome.to_dict()
                for key, outcome in self._reverse_stereo_outcomes.items()
            }
            reversed_graph.graph["stereo_query_policies"] = dict(
                self._reverse_stereo_query_policies
            )
        else:
            reversed_graph.graph["stereo_outcomes"] = {}
            reverse_policies = dict(self.stereo_query_policies)
            for key, outcome in self.stereo_outcomes.items():
                if outcome.kind != "SINGLE":
                    reverse_policies[key] = "either"
            reversed_graph.graph["stereo_query_policies"] = reverse_policies
            if self.stereo_outcomes:
                reversed_graph.graph["stereo_reverse_outcomes"] = {
                    key: outcome.to_dict()
                    for key, outcome in self.stereo_outcomes.items()
                }
                reversed_graph.graph["stereo_reverse_query_policies"] = dict(
                    self.stereo_query_policies
                )

        return SynRule(
            reversed_graph,
            name=self._name,
            canonicaliser=self._canonicaliser,
            canon=self._canon_enabled,
            implicit_h=self._implicit_h,
            format=self._format,
        )

    def help(self) -> None:
        """Pretty-print raw / canonical contents for quick inspection."""
        print(f"SynRule name={self._name!r}")
        print("→ Full (raw) rc_graph edges:")
        for u, v, d in self.rc.raw.edges(data=True):
            print(f"   ({u}, {v}): {d}")

        if not self._canon_enabled:
            print("→ Canonicalisation disabled.")
            return

        print("\n→ Full canonical_graph edges:")
        for u, v, d in self.rc.canonical.edges(data=True):  # type: ignore[attr-defined]
            print(f"   ({u}, {v}): {d}")

        print("\n→ Left fragment:")
        self.left.help()
        print("\n→ Right fragment:")
        self.right.help()
        print("\n→ Fragment signatures:", self.canonical_smiles)
