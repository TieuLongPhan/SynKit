import unittest
import networkx as nx

from synkit.CRN.Hypergraph import CRNHyperGraph
from synkit.CRN.Hypergraph.conversion import (
    rxns_to_hypergraph,
    hypergraph_to_bipartite,
    bipartite_to_hypergraph,
    hypergraph_to_species_graph,
    species_graph_to_hypergraph,
    hypergraph_to_rxn_strings,
)


def edge_signature(e):
    reactants = frozenset(sorted(tuple(sorted(e.reactants.to_dict().items()))))
    products = frozenset(sorted(tuple(sorted(e.products.to_dict().items()))))
    return (reactants, products, getattr(e, "rule", None))


class TestHypergraphConversion(unittest.TestCase):
    def test_rxns_to_hypergraph_basic(self):
        rxns = ["A + B >> C", "C >> A", "2 A >> D"]
        H = rxns_to_hypergraph(rxns)
        self.assertIsInstance(H, CRNHyperGraph)
        self.assertEqual(set(H.species), {"A", "B", "C", "D"})
        self.assertEqual(len(H.edges), 3)

    def test_hypergraph_to_bipartite_and_back_integer_ids_false(self):
        rxns = ["A + B >> C", "C >> A", "2 A >> D"]
        H = rxns_to_hypergraph(rxns)
        B = hypergraph_to_bipartite(H, integer_ids=False, include_edge_id_attr=True)
        for s in H.species:
            self.assertIn(f"S:{s}", B.nodes)
        rxn_nodes = [n for n, d in B.nodes(data=True) if d.get("kind") == "reaction"]
        self.assertTrue(len(rxn_nodes) > 0)
        H2 = bipartite_to_hypergraph(B, integer_ids=False)
        self.assertEqual(set(H2.species), set(H.species))
        sigs_H = {edge_signature(e) for e in H.edge_list()}
        sigs_H2 = {edge_signature(e) for e in H2.edge_list()}
        self.assertEqual(sigs_H, sigs_H2)

    def test_bipartite_integer_ids_true_structure(self):
        rxns = ["A + B >> C", "C >> A", "2 A >> D"]
        H = rxns_to_hypergraph(rxns)
        B = hypergraph_to_bipartite(H, integer_ids=True)
        bip_attrs = nx.get_node_attributes(B, "bipartite")
        self.assertEqual(set(bip_attrs.values()), {0, 1})
        sto_attrs = [d.get("stoich") for u, v, d in B.edges(data=True)]
        self.assertTrue(all((isinstance(s, int) for s in sto_attrs if s is not None)))

    def test_hypergraph_to_species_graph_roundtrip(self):
        rxns = ["A + B >> C", "C >> A", "2 A >> D"]
        H = rxns_to_hypergraph(rxns)
        S = hypergraph_to_species_graph(H)
        self.assertTrue(S.number_of_nodes() >= 1)
        H_rec = species_graph_to_hypergraph(S)
        self.assertEqual(set(H_rec.species), set(H.species))
        for e in H_rec.edge_list():
            for r in e.reactants.keys():
                self.assertIn(r, S.nodes)
            for p in e.products.keys():
                self.assertIn(p, S.nodes)

    def test_hypergraph_to_rxn_strings_and_suffix(self):
        rxns = ["X >> Y | rule=Rf", "Y >> X | rule=Rb"]
        H = rxns_to_hypergraph(rxns)
        lines = hypergraph_to_rxn_strings(
            H, include_rule_suffix=True, include_edge_id=True
        )
        self.assertTrue(all(">>" in ln for ln in lines))
        self.assertTrue(any("rule=" in ln for ln in lines))
        self.assertTrue(any("id=" in ln for ln in lines))

    def test_parse_rule_suffix_preserved(self):
        rxns = ["P + Q >> R | rule=MYRULE"]
        H = rxns_to_hypergraph(rxns, parse_rule_from_suffix=True)
        self.assertEqual(len(H.edges), 1)
        e = next(iter(H.edge_list()))
        self.assertEqual(e.rule, "MYRULE")

    def test_productless_and_empty_side_handling(self):
        rxns = ["A >> ∅", "∅ >> B", "C >> D"]
        H = rxns_to_hypergraph(rxns)
        self.assertEqual(set(H.species), {"A", "B", "C", "D"})
        self.assertEqual(len(H.edges), 3)
        B = hypergraph_to_bipartite(H, integer_ids=False, include_isolated_species=True)
        for s in ["A", "B", "C", "D"]:
            self.assertIn(f"S:{s}", B.nodes)

    def test_autocatalytic_roundtrip(self):
        rxns = ["A + X >> 2 X"]
        H = rxns_to_hypergraph(rxns)
        e = next(iter(H.edge_list()))
        self.assertEqual(e.products.to_dict().get("X", 0), 2)
        B = hypergraph_to_bipartite(H, integer_ids=False)
        H2 = bipartite_to_hypergraph(B, integer_ids=False)
        self.assertEqual(
            {edge_signature(e) for e in H.edge_list()},
            {edge_signature(e) for e in H2.edge_list()},
        )

    def test_roundtrip_id_independence(self):
        rxns = ["A + B >> C", "C >> A", "2 A >> D"]
        H = rxns_to_hypergraph(rxns)
        B = hypergraph_to_bipartite(H, integer_ids=False, include_edge_id_attr=True)
        H2 = bipartite_to_hypergraph(B, integer_ids=False)
        s0 = {
            (
                frozenset(sorted(tuple(sorted(e.reactants.to_dict().items())))),
                frozenset(sorted(tuple(sorted(e.products.to_dict().items())))),
                e.rule,
            )
            for e in H.edge_list()
        }
        s1 = {
            (
                frozenset(sorted(tuple(sorted(e.reactants.to_dict().items())))),
                frozenset(sorted(tuple(sorted(e.products.to_dict().items())))),
                e.rule,
            )
            for e in H2.edge_list()
        }
        self.assertEqual(s0, s1)


if __name__ == "__main__":
    unittest.main()
