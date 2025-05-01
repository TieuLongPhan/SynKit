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
from typing import Optional, Tuple

import networkx as nx

from synkit.Graph.syn_graph import SynGraph
from synkit.Graph.canon_graph import GraphCanonicaliser
from synkit.Graph.ITS.its_decompose import its_decompose

from synkit.IO.chem_converter import rsmi_to_its, gml_to_its

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
    ) -> "SynRule":
        """Instantiate from a SMARTS string."""
        return cls(
            rsmi_to_its(smart),
            name=name,
            canonicaliser=canonicaliser,
            canon=canon,
            implicit_h=implicit_h,
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
    ) -> None:
        self._name = name
        self._canon_enabled = canon
        self._implicit_h = implicit_h
        self._canonicaliser = canonicaliser or GraphCanonicaliser()

        # Fragment decomposition
        rc_graph = rc.copy()
        left_graph, right_graph = its_decompose(rc_graph)

        # Optional H-stripping
        if self._implicit_h:
            self._strip_explicit_h(rc_graph, left_graph, right_graph)

        # Update typesGH tuples with new hcount
        for node, att in rc_graph.nodes(data=True):
            # unpack the old tuples
            t0, t1 = att["typesGH"]

            # build new versions with the updated hcount at position 2
            new_t0 = (t0[0], t0[1], left_graph.nodes[node]["hcount"], t0[3], t0[4])
            new_t1 = (t1[0], t1[1], right_graph.nodes[node]["hcount"], t1[3], t1[4])

            # reassign the attribute to a fresh tuple-of-tuples
            att["typesGH"] = (new_t0, new_t1)

        # ---------- wrap graphs ---------------------------------------- #
        self.rc = SynGraph(rc_graph, self._canonicaliser, canon=canon)
        self.left = SynGraph(left_graph, self._canonicaliser, canon=canon)
        self.right = SynGraph(right_graph, self._canonicaliser, canon=canon)

        self.canonical_smiles: Optional[Tuple[str, str]] = (
            (self.left.signature, self.right.signature) if canon else None
        )

    # ================================================================== #
    # Private utilities                                                  #
    # ================================================================== #
    @staticmethod
    def _strip_explicit_h(
        rc: nx.Graph,
        left: nx.Graph,
        right: nx.Graph,
    ) -> None:
        """
        Remove explicit hydrogens from rc, left, right—but only when *both*
        left & right agree the H should be implicit.  Otherwise an H remains
        explicit in all three graphs.
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

    # ================================================================== #
    # Dunder methods                                                     #
    # ================================================================== #
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SynRule)
            and self.canonical_smiles == other.canonical_smiles
        )

    def __hash__(self) -> int:
        return hash(self.canonical_smiles)

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
