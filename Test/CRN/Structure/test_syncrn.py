import unittest

import networkx as nx

# Adjust this import to your actual module path
# Example:
from synkit.CRN.Structure.syncrn import SynCRN

# from your_module import SynCRN


RXNS = [
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


class TestSynCRNFromReactionStrings(unittest.TestCase):
    def setUp(self):
        self.syn = SynCRN.from_reaction_strings(RXNS)

    def test_basic_sizes(self):
        self.assertEqual(self.syn.n_species, 11)
        self.assertEqual(self.syn.n_reactions, 11)
        self.assertEqual(self.syn.n_rules, 0)

    def test_species_ids(self):
        self.assertEqual(
            self.syn.species_ids,
            ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
        )

    def test_reaction_ids(self):
        self.assertEqual(
            self.syn.reaction_ids,
            ["12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22"],
        )

    def test_rule_ids_empty_without_pairwise_rules(self):
        self.assertEqual(self.syn.rule_ids, [])

    def test_repr(self):
        self.assertEqual(
            repr(self.syn),
            "SynCRN(n_species=11, n_reactions=11, n_rules=0)",
        )

    def test_species_labels_follow_first_appearance(self):
        labels = [self.syn.species[sid].label for sid in self.syn.species_ids]
        self.assertEqual(
            labels, ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
        )

    def test_species_token_modes(self):
        self.assertEqual(self.syn._species_token("1", "id"), "1")
        self.assertEqual(self.syn._species_token("1", "label"), "A")
        self.assertEqual(self.syn._species_token("1", "smiles"), "A")
        self.assertEqual(self.syn._species_token("1", "source"), "1")

    def test_species_token_invalid_mode(self):
        with self.assertRaises(ValueError):
            self.syn._species_token("1", "bad_mode")

    def test_format_reaction_default(self):
        self.assertEqual(self.syn.format_reaction("12"), "12: 2A >> B + 3C")
        self.assertEqual(self.syn.format_reaction("13"), "13: 2B >> D")
        self.assertEqual(self.syn.format_reaction("22"), "22: K + C >> H")

    def test_format_reaction_without_id(self):
        self.assertEqual(
            self.syn.format_reaction("12", include_id=False),
            "2A >> B + 3C",
        )

    def test_to_equations(self):
        eqs = self.syn.to_equations()
        self.assertEqual(len(eqs), 11)
        self.assertEqual(eqs[0], "12: 2A >> B + 3C")
        self.assertEqual(eqs[1], "13: 2B >> D")
        self.assertEqual(eqs[-1], "22: K + C >> H")

    def test_describe_without_species(self):
        expected = "\n".join(
            [
                "SynCRN: 11 species, 11 reactions",
                "  12: 2A >> B + 3C",
                "  13: 2B >> D",
                "  14: C + D >> E",
                "  15: 3C + D >> F",
                "  16: 2C + E >> F",
                "  17: 3B >> G",
                "  18: 3C + G >> H",
                "  19: B + C >> I",
                "  20: C + I >> J",
                "  21: E + I >> K",
                "  22: K + C >> H",
            ]
        )
        self.assertEqual(self.syn.describe(), expected)

    def test_describe_with_species(self):
        text = self.syn.describe(include_species=True, species="label")
        self.assertIn("SynCRN: 11 species, 11 reactions", text)
        self.assertIn("Species: A, B, C, D, E, F, G, H, I, J, K", text)

    def test_str(self):
        text = str(self.syn)
        self.assertIn("SynCRN: 11 species, 11 reactions", text)
        self.assertIn("Species: A, B, C, D, E, F, G, H, I, J, K", text)

    def test_to_dict_basic(self):
        d = self.syn.to_dict()
        self.assertEqual(d["metadata"]["source"], "reaction_strings")
        self.assertEqual(d["metadata"]["n_input_reactions"], 11)
        self.assertFalse(d["metadata"]["has_pairwise_rules"])
        self.assertEqual(len(d["species"]), 11)
        self.assertEqual(len(d["reactions"]), 11)
        self.assertEqual(len(d["rules"]), 0)

    def test_species_table_contents(self):
        sp1 = self.syn.species["1"]
        self.assertEqual(sp1.id, "1")
        self.assertEqual(sp1.label, "A")
        self.assertEqual(sp1.source_node_id, 1)
        self.assertEqual(sp1.source_attrs["kind"], "species")
        self.assertEqual(sp1.source_attrs["label"], "A")

        sp11 = self.syn.species["11"]
        self.assertEqual(sp11.label, "K")

    def test_reaction_table_contents_first_reaction(self):
        rxn = self.syn.reactions["12"]
        self.assertEqual(rxn.id, "12")
        self.assertEqual(rxn.source_node_id, 12)
        self.assertEqual(rxn.source_kind, "rule")
        self.assertEqual(rxn.lhs.to_dict(), {"1": 2})
        self.assertEqual(rxn.rhs.to_dict(), {"2": 1, "3": 3})
        self.assertIsNone(rxn.rule_id)
        self.assertIsNone(rxn.rule_index)
        self.assertIsNone(rxn.rule_repr)
        self.assertEqual(rxn.source_attrs["kind"], "rule")
        self.assertEqual(rxn.source_attrs["rxn_repr"], "2A>>B+3C")

    def test_reaction_table_contents_last_reaction(self):
        rxn = self.syn.reactions["22"]
        self.assertEqual(rxn.lhs.to_dict(), {"3": 1, "11": 1})
        self.assertEqual(rxn.rhs.to_dict(), {"8": 1})

    def test_reactant_and_product_edge_attrs(self):
        rxn = self.syn.reactions["12"]
        self.assertEqual(
            rxn.reactant_edge_attrs,
            {"1": {"role": "reactant", "stoich": 2}},
        )
        self.assertEqual(
            rxn.product_edge_attrs,
            {
                "2": {"role": "product", "stoich": 1},
                "3": {"role": "product", "stoich": 3},
            },
        )

    def test_to_stoichiometric_matrices_shapes_and_orders(self):
        mats = self.syn.to_stoichiometric_matrices()
        self.assertEqual(mats["species_order"], self.syn.species_ids)
        self.assertEqual(mats["reaction_order"], self.syn.reaction_ids)
        self.assertEqual(len(mats["S_minus"]), 11)
        self.assertEqual(len(mats["S_plus"]), 11)
        self.assertEqual(len(mats["S"]), 11)
        self.assertTrue(all(len(row) == 11 for row in mats["S_minus"]))
        self.assertTrue(all(len(row) == 11 for row in mats["S_plus"]))
        self.assertTrue(all(len(row) == 11 for row in mats["S"]))

    def test_to_stoichiometric_matrices_first_reaction_column(self):
        mats = self.syn.to_stoichiometric_matrices()
        j = mats["reaction_order"].index("12")
        s_minus_col = [row[j] for row in mats["S_minus"]]
        s_plus_col = [row[j] for row in mats["S_plus"]]
        s_col = [row[j] for row in mats["S"]]

        self.assertEqual(s_minus_col, [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(s_plus_col, [0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(s_col, [-2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_to_stoichiometric_matrices_last_reaction_column(self):
        mats = self.syn.to_stoichiometric_matrices()
        j = mats["reaction_order"].index("22")
        s_minus_col = [row[j] for row in mats["S_minus"]]
        s_plus_col = [row[j] for row in mats["S_plus"]]
        s_col = [row[j] for row in mats["S"]]

        self.assertEqual(s_minus_col, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1])
        self.assertEqual(s_plus_col, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        self.assertEqual(s_col, [0, 0, -1, 0, 0, 0, 0, 1, 0, 0, -1])

    def test_to_petrinet(self):
        pn = self.syn.to_petrinet()
        self.assertEqual(pn["places"], self.syn.species_ids)
        self.assertEqual(pn["transitions"], self.syn.reaction_ids)

        self.assertEqual(pn["pre"]["1"], {"12": 2})
        self.assertEqual(pn["post"]["2"], {"12": 1})
        self.assertEqual(pn["post"]["3"], {"12": 3})

        self.assertEqual(pn["pre"]["11"], {"22": 1})
        self.assertEqual(pn["pre"]["8"], {})
        self.assertEqual(pn["post"]["8"]["18"], 1)
        self.assertEqual(pn["post"]["8"]["22"], 1)

    def test_to_digraph_source_ids(self):
        g = self.syn.to_digraph()
        self.assertIsInstance(g, nx.DiGraph)

        self.assertIn(1, g.nodes)
        self.assertIn(12, g.nodes)
        self.assertIn(22, g.nodes)

        self.assertEqual(g.nodes[1]["kind"], "species")
        self.assertEqual(g.nodes[1]["label"], "A")
        self.assertEqual(g.nodes[12]["kind"], "rule")
        self.assertEqual(g.nodes[12]["label"], "12")

        self.assertEqual(g[1][12]["role"], "reactant")
        self.assertEqual(g[1][12]["stoich"], 2)
        self.assertEqual(g[12][2]["role"], "product")
        self.assertEqual(g[12][2]["stoich"], 1)
        self.assertEqual(g[12][3]["stoich"], 3)

    def test_to_digraph_internal_ids(self):
        g = self.syn.to_digraph(node_ids="internal", reaction_kind="reaction")
        self.assertIn("1", g.nodes)
        self.assertIn("12", g.nodes)

        self.assertEqual(g.nodes["1"]["kind"], "species")
        self.assertEqual(g.nodes["1"]["syncrn_id"], "1")
        self.assertEqual(g.nodes["1"]["source_node_id"], 1)

        self.assertEqual(g.nodes["12"]["kind"], "reaction")
        self.assertEqual(g.nodes["12"]["syncrn_id"], "12")
        self.assertEqual(g.nodes["12"]["source_node_id"], 12)

        self.assertEqual(g["1"]["12"]["role"], "reactant")
        self.assertEqual(g["1"]["12"]["stoich"], 2)
        self.assertEqual(g["12"]["2"]["role"], "product")

    def test_to_digraph_invalid_node_ids(self):
        with self.assertRaises(ValueError):
            self.syn.to_digraph(node_ids="bad")

    def test_from_reaction_strings_empty(self):
        syn = SynCRN.from_reaction_strings([])
        self.assertEqual(syn.n_species, 0)
        self.assertEqual(syn.n_reactions, 0)
        self.assertEqual(syn.n_rules, 0)
        self.assertEqual(syn.metadata["source"], "reaction_strings")
        self.assertEqual(syn.metadata["n_input_reactions"], 0)
        self.assertFalse(syn.metadata["has_pairwise_rules"])

    def test_from_reaction_strings_invalid_rxns_type(self):
        with self.assertRaises(TypeError):
            SynCRN.from_reaction_strings("A>>B")  # type: ignore[arg-type]

    def test_from_reaction_strings_invalid_rxns_entries(self):
        with self.assertRaises(TypeError):
            SynCRN.from_reaction_strings(["A>>B", 123])  # type: ignore[list-item]

    def test_from_reaction_strings_invalid_rules_length(self):
        with self.assertRaises(ValueError):
            SynCRN.from_reaction_strings(
                ["A>>B", "B>>C"],
                rules=["rule1"],
            )

    def test_from_reaction_strings_invalid_rules_entry_type(self):
        with self.assertRaises(TypeError):
            SynCRN.from_reaction_strings(
                ["A>>B"],
                rules=[123],  # type: ignore[list-item]
            )

    def test_from_reaction_strings_with_pairwise_rules(self):
        rules = [f"rule_{i}" for i in range(len(RXNS))]
        syn = SynCRN.from_reaction_strings(RXNS, rules=rules)

        self.assertEqual(syn.n_species, 11)
        self.assertEqual(syn.n_reactions, 11)
        self.assertEqual(syn.n_rules, 11)
        self.assertTrue(syn.metadata["has_pairwise_rules"])

        self.assertEqual(syn.rule_ids[0], "1")
        self.assertEqual(syn.rules["1"].rule_index, 0)
        self.assertEqual(syn.rules["1"].rule_repr, "rule_0")
        self.assertEqual(syn.rules["1"].label, "r0")

        self.assertEqual(syn.reactions["12"].rule_id, "1")
        self.assertEqual(syn.reactions["12"].rule_index, 0)
        self.assertEqual(syn.reactions["12"].rule_repr, "rule_0")

    def test_from_reaction_strings_malformed_missing_arrow(self):
        with self.assertRaises(ValueError):
            SynCRN.from_reaction_strings(["A+B"])

    def test_from_reaction_strings_empty_lhs_strict(self):
        with self.assertRaises(ValueError):
            SynCRN.from_reaction_strings([">>B"], strict=True)

    def test_from_reaction_strings_empty_rhs_strict(self):
        with self.assertRaises(ValueError):
            SynCRN.from_reaction_strings(["A>>"], strict=True)

    def test_from_reaction_strings_empty_side_nonstrict(self):
        syn = SynCRN.from_reaction_strings([">>B"], strict=False)
        self.assertEqual(syn.n_species, 1)
        self.assertEqual(syn.n_reactions, 1)
        self.assertEqual(syn.reactions["2"].lhs.to_dict(), {})
        self.assertEqual(syn.reactions["2"].rhs.to_dict(), {"1": 1})

    def test_from_digraph_roundtrip_source_ids(self):
        g = self.syn.to_digraph()
        syn2 = SynCRN.from_digraph(g)

        self.assertEqual(syn2.n_species, self.syn.n_species)
        self.assertEqual(syn2.n_reactions, self.syn.n_reactions)

        expected = [
            "2A >> B + 3C",
            "2B >> D",
            "C + D >> E",
            "3C + D >> F",
            "2C + E >> F",
            "3B >> G",
            "3C + G >> H",
            "B + C >> I",
            "I + C >> J",
            "I + E >> K",
            "K + C >> H",
        ]
        self.assertEqual(
            syn2.to_equations(species="label", include_id=False),
            expected,
        )

    def test_from_digraph_requires_digraph(self):
        with self.assertRaises(TypeError):
            SynCRN.from_digraph(nx.Graph())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
