import unittest
import io
import contextlib

import networkx as nx

from synkit.CRN.Hypergraph.hypergraph import CRNHyperGraph
from synkit.CRN.Hypergraph.hyperedge import HyperEdge
from synkit.CRN.Hypergraph.conversion import (
    hypergraph_to_bipartite,
    bipartite_to_hypergraph,
    hypergraph_to_species_graph,
    species_graph_to_hypergraph,
    rxns_to_hypergraph,
    hypergraph_to_rxn_strings,
    print_species_summary,
    print_edge_list,
    print_graph_attrs,
)


class TestConversionHypergraph(unittest.TestCase):
    def _example_hypergraph(self) -> CRNHyperGraph:
        """Helper to build the example hypergraph."""
        rxns = [
            "2A>>B+3C",
            "2B>>D",
            "D+C>>E",
            "D+3C>>F",
            "E+2C>>F",
            "3B>>G",
            "G+3C>>H",
            "B+C>>I",
            "I+C>>J",
            "E+I>>K",
            "K+C>>H",
        ]
        H = rxns_to_hypergraph(rxns)
        mol_map = {
            "A": "CH4",
            "B": "C2H2",
            "C": "H2",
            "D": "C4H4",
            "E": "C4H6",
            "F": "C4H10",
            "G": "C6H6",
            "H": "C6H12",
            "I": "C2H4",
            "J": "C2H6",
            "K": "C6H10",
        }
        H.set_mol_map(mapping=mol_map, strict=False)
        return H

    # ------------------------------------------------------------------
    # Pair 1: Hypergraph <-> Bipartite
    # ------------------------------------------------------------------

    def test_hypergraph_to_bipartite_basic_structure_and_attrs(self):
        H = self._example_hypergraph()
        G = hypergraph_to_bipartite(
            H,
            integer_ids=False,
            include_edge_id_attr=True,
            include_mol=True,
        )

        # Graph type
        self.assertIsInstance(G, nx.DiGraph)

        # Node kinds and bipartite markers
        kinds = set(nx.get_node_attributes(G, "kind").values())
        self.assertIn("species", kinds)
        self.assertIn("reaction", kinds)

        bip_vals = set(nx.get_node_attributes(G, "bipartite").values())
        self.assertEqual(bip_vals, {0, 1})

        # Species nodes exist and carry mol attribute
        labels = nx.get_node_attributes(G, "label")
        mols = nx.get_node_attributes(G, "mol")
        for s in H.species:
            sp_nodes = [n for n, lab in labels.items() if lab == s]
            self.assertTrue(sp_nodes, msg=f"No node found for species {s}")
            node = sp_nodes[0]
            self.assertEqual(G.nodes[node]["kind"], "species")
            if s in H.species_to_mol:
                self.assertEqual(mols[node], H.species_to_mol[s])

        # Reaction nodes exist for every hyperedge and have edge_id attr
        edge_ids = {e.id for e in H.edge_list()}
        reaction_nodes = [
            n for n, k in nx.get_node_attributes(G, "kind").items() if k == "reaction"
        ]
        self.assertGreaterEqual(len(reaction_nodes), len(edge_ids))

        rxn_edge_ids = set()
        for n in reaction_nodes:
            rxn_edge_ids.add(G.nodes[n].get("edge_id"))
        self.assertTrue(edge_ids.issubset(rxn_edge_ids))

        # Check stoichiometry + role on a specific edge:
        # choose 2A >> B + 3C
        first = None
        for e in H.edge_list():
            if set(e.reactants.keys()) == {"A"} and set(e.products.keys()) == {
                "B",
                "C",
            }:
                first = e
                break
        self.assertIsNotNone(first, "Example reaction 2A>>B+3C not found")

        # locate its reaction node in G
        rxn_node = None
        for n in reaction_nodes:
            if G.nodes[n].get("edge_id") == first.id:
                rxn_node = n
                break
        self.assertIsNotNone(rxn_node, "Reaction node for 2A>>B+3C not found")

        # find the species node for A
        sp_node_A = None
        for n, lab in labels.items():
            if lab == "A":
                sp_node_A = n
                break
        self.assertIsNotNone(sp_node_A, "Species node for A not found")

        # A -> reaction: stoich=2, role=reactant
        self.assertTrue(G.has_edge(sp_node_A, rxn_node))
        ed = G[sp_node_A][rxn_node]
        self.assertEqual(ed.get("stoich"), 2)
        self.assertEqual(ed.get("role"), "reactant")

        # reaction -> C: stoich=3, role=product
        sp_node_C = None
        for n, lab in labels.items():
            if lab == "C":
                sp_node_C = n
                break
        self.assertIsNotNone(sp_node_C, "Species node for C not found")

        self.assertTrue(G.has_edge(rxn_node, sp_node_C))
        ed2 = G[rxn_node][sp_node_C]
        self.assertEqual(ed2.get("stoich"), 3)
        self.assertEqual(ed2.get("role"), "product")

    def test_hypergraph_to_bipartite_excludes_isolated_species(self):
        H = self._example_hypergraph()
        # Add an isolated species (no edges)
        H.species.add("X")  # manually isolated

        G = hypergraph_to_bipartite(
            H,
            include_isolated_species=False,
            integer_ids=False,
        )
        labels = nx.get_node_attributes(G, "label")
        self.assertNotIn("X", labels.values())

    def test_bipartite_to_hypergraph_roundtrip_with_edge_ids_and_mol(self):
        H = self._example_hypergraph()
        G = hypergraph_to_bipartite(
            H,
            integer_ids=False,
            include_edge_id_attr=True,
            include_mol=True,
        )
        H2 = bipartite_to_hypergraph(G)

        # Species preserved
        self.assertEqual(sorted(H.species), sorted(H2.species))

        # Each edge should be present with same id and stoichiometry
        self.assertEqual(set(H.edges.keys()), set(H2.edges.keys()))
        for eid, e in H.edges.items():
            e2 = H2.edges[eid]
            self.assertEqual(e.reactants.to_dict(), e2.reactants.to_dict())
            self.assertEqual(e.products.to_dict(), e2.products.to_dict())
            self.assertEqual(e.rule, e2.rule)

        # Molecule mapping preserved
        self.assertEqual(H.species_to_mol, H2.species_to_mol)

    def test_bipartite_to_hypergraph_roundtrip_without_edge_ids(self):
        H = self._example_hypergraph()
        G = hypergraph_to_bipartite(
            H,
            integer_ids=False,
            include_edge_id_attr=False,
            include_mol=False,
        )
        H2 = bipartite_to_hypergraph(G)

        # Species preserved
        self.assertEqual(sorted(H.species), sorted(H2.species))
        self.assertEqual(len(H.edges), len(H2.edges))

        # Check that for every original edge there's a matching one in H2
        original = [
            (e.reactants.to_dict(), e.products.to_dict(), e.rule) for e in H.edge_list()
        ]
        reconstructed = [
            (e.reactants.to_dict(), e.products.to_dict(), e.rule)
            for e in H2.edge_list()
        ]
        for tpl in original:
            self.assertIn(tpl, reconstructed)

    # ------------------------------------------------------------------
    # Pair 2: Hypergraph <-> Species graph
    # ------------------------------------------------------------------

    def test_hypergraph_to_species_graph_basic_and_mol(self):
        H = self._example_hypergraph()
        S = hypergraph_to_species_graph(H, include_mol=True)

        # All species nodes should be present and labelled
        self.assertEqual(set(S.nodes), set(H.species))
        for s in H.species:
            self.assertEqual(S.nodes[s]["kind"], "species")
            self.assertEqual(S.nodes[s]["label"], s)
            if s in H.species_to_mol:
                self.assertEqual(S.nodes[s]["mol"], H.species_to_mol[s])

        # Check that A -> C exists with correct attrs (from 2A>>B+3C)
        first = None
        for e in H.edge_list():
            if set(e.reactants.keys()) == {"A"} and set(e.products.keys()) == {
                "B",
                "C",
            }:
                first = e
                break
        self.assertIsNotNone(first, "Example reaction 2A>>B+3C not found")

        self.assertTrue(S.has_edge("A", "C"))
        ed = S["A"]["C"]
        self.assertIn(first.id, ed["via"])
        self.assertIn(first.rule, ed["rules"])
        self.assertEqual(ed["stoich_r"], 2)
        self.assertEqual(ed["stoich_p"], 3)

    def test_species_graph_to_hypergraph_roundtrip_with_via_and_mol(self):
        H = self._example_hypergraph()
        S = hypergraph_to_species_graph(H, include_mol=True)
        H2 = species_graph_to_hypergraph(S, default_rule="r", mol_attr="mol")

        # Species preserved
        self.assertEqual(sorted(H.species), sorted(H2.species))
        # Edge ids should be the via ids (original ids)
        self.assertEqual(set(H.edges.keys()), set(H2.edges.keys()))

        for eid, e in H.edges.items():
            e2 = H2.edges[eid]
            self.assertEqual(e.reactants.to_dict(), e2.reactants.to_dict())
            self.assertEqual(e.products.to_dict(), e2.products.to_dict())
            self.assertEqual(e.rule, e2.rule)

        # Molecule mapping preserved
        self.assertEqual(H.species_to_mol, H2.species_to_mol)

    def test_species_graph_to_hypergraph_without_via_creates_edges(self):
        # Manually build a species graph without 'via'
        G = nx.DiGraph()
        G.add_node("A", label="A")
        G.add_node("B", label="B")
        G.add_edge("A", "B", stoich_r=2, stoich_p=1)
        H = species_graph_to_hypergraph(G, default_rule="r", mol_attr=None)

        self.assertEqual(sorted(H.species), ["A", "B"])
        self.assertEqual(len(H.edges), 1)
        e = next(iter(H.edge_list()))
        self.assertEqual(e.reactants.to_dict(), {"A": 2})
        self.assertEqual(e.products.to_dict(), {"B": 1})

    # ------------------------------------------------------------------
    # Pair 3: Reaction strings <-> Hypergraph
    # ------------------------------------------------------------------

    def test_rxns_to_hypergraph_and_hypergraph_to_rxn_strings_roundtrip(self):
        rxns = ["A+B>>C | rule=R1", "2A>>D", "C>>A"]
        H = rxns_to_hypergraph(rxns, default_rule="R0")
        self.assertIsInstance(H, CRNHyperGraph)
        self.assertEqual(sorted(H.species), sorted({"A", "B", "C", "D"}))

        lines = hypergraph_to_rxn_strings(
            H,
            include_rule_suffix=True,
            include_edge_id=True,
            sort=True,
        )
        self.assertTrue(all(">>" in ln for ln in lines))
        self.assertTrue(any("rule=" in ln for ln in lines))
        self.assertTrue(any("id=" in ln for ln in lines))

        joined = "\n".join(lines)
        self.assertIn("2A", joined)

        # turn back into a hypergraph
        H2 = rxns_to_hypergraph(lines, parse_rule_from_suffix=True)

        # Species preserved
        self.assertEqual(sorted(H2.species), sorted(H.species))

        # Compare reactions by canonical representation
        def canonical_reactions(Hg: CRNHyperGraph):
            reps = []
            for e in Hg.edge_list():
                r = tuple(sorted(e.reactants.to_dict().items()))
                p = tuple(sorted(e.products.to_dict().items()))
                reps.append((r, p, e.rule))
            return sorted(reps)

        orig = canonical_reactions(H)
        new = canonical_reactions(H2)
        self.assertEqual(orig, new)

    def test_hypergraph_to_rxn_strings_without_suffix_or_ids(self):
        H = self._example_hypergraph()
        lines = hypergraph_to_rxn_strings(
            H,
            include_rule_suffix=False,
            include_edge_id=False,
            sort=True,
        )
        self.assertTrue(all(">>" in ln for ln in lines))
        self.assertFalse(any("rule=" in ln for ln in lines))
        self.assertFalse(any("id=" in ln for ln in lines))

    # ------------------------------------------------------------------
    # Pretty-print helpers
    # ------------------------------------------------------------------

    def test_print_species_summary_outputs_content(self):
        H = self._example_hypergraph()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_species_summary(H)
        out = buf.getvalue()
        self.assertIn("Species", out)
        self.assertIn("A", out)
        self.assertIn("In-edges", out)
        self.assertIn("Out-edges", out)

    def test_print_edge_list_outputs_reactions(self):
        H = self._example_hypergraph()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_edge_list(H, show_stoich=True)
        out = buf.getvalue()
        self.assertIn("Edge id", out)
        self.assertIn("Reactants >> Products", out)
        # We know 2A>>B+3C exists; its textual form should appear
        self.assertIn("2A", out)
        self.assertIn("B", out)
        self.assertIn("C", out)

    def test_print_graph_attrs_on_species_graph(self):
        H = self._example_hypergraph()
        G = hypergraph_to_species_graph(H, include_mol=True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_graph_attrs(G, include_nodes=True, include_edges=True, max_rows=5)
        out = buf.getvalue()
        self.assertIn("Nodes:", out)
        self.assertIn("Edges:", out)
        # At least some species labels should appear
        self.assertTrue(
            any(
                s in out
                for s in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
            )
        )


if __name__ == "__main__":
    unittest.main()
