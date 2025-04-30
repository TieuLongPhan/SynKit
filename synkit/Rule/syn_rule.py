import networkx as nx
from typing import Optional, Tuple
from synkit.IO.chem_converter import rsmi_to_its, gml_to_its
from synkit.Graph.ITS.its_decompose import its_decompose
from synkit.Graph.syn_graph import SynGraph
from synkit.Graph.canon_graph import GraphCanonicaliser


class SynRule:
    """
    Immutable description of a reaction template.

    Parameters
    ----------
    rc_graph : nx.Graph
        Raw reaction-centre (ITS) graph.
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

    Notes
    -----
    •  **rc**, **left**, and **right** are always returned as
       :class:`~synkit.Graph.syn_graph.SynGraph` wrappers.
    •  ``canonical_smiles`` is a pair of hex digests *(left, right)*
       when *canon=True*; otherwise ``None``.
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
        """Build from an ITS-compatible reaction **SMILES/SMARTS** string."""
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
        """Build from a **GML** string."""
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

        # ---------- split into fragments BEFORE any wrapping ------------ #
        rc_graph = rc.copy()
        left_graph, right_graph = its_decompose(rc_graph)

        if self._implicit_h:
            self._strip_explicit_h(rc_graph, left_graph, right_graph)

        # rc_graph = ITSConstruction().ITSGraph(left_graph, right_graph)
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
        Remove explicit hydrogens from *rc*, *left*, *right* and:

        1. Ensure every heavy atom has ``hcount`` (initialised to 0).
        2. For each shared hydrogen *H* appearing in **both** *left* and *right*:
           • increment ``hcount`` on its neighbours in each fragment
           • assign an integer pair-ID to the neighbours’ ``h_pairs`` list
        3. Delete *H* from all three graphs (and associated edges).

        The pair-ID (1, 2, …) is assigned in sorted **H-node** order, providing
        reproducible numbering across runs.
        """
        # 1) ensure hcount exists
        for g in (rc, left, right):
            for _, data in g.nodes(data=True):
                data["hcount"] = 0

        # 2) shared hydrogens (present in both left & right)
        shared_H = sorted(
            n
            for n, d in left.nodes(data=True)
            if d.get("element") == "H" and right.has_node(n)
        )

        pair_id = 1
        for h in shared_H:
            # tag neighbours in *all* graphs
            for g in (left, right, rc):
                if g.has_node(h):
                    for nbr in g.neighbors(h):
                        g.nodes[nbr]["hcount"] += 1
                        g.nodes[nbr].setdefault("h_pairs", []).append(pair_id)
                    g.remove_node(h)
            pair_id += 1

        # 3) any remaining explicit H unique to one graph
        for g in (rc, left, right):
            lone_H = [n for n, d in g.nodes(data=True) if d.get("element") == "H"]
            for h in lone_H:
                for nbr in g.neighbors(h):
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
    # Convenience                                                        #
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
