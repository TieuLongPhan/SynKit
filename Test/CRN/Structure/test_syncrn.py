import unittest

import networkx as nx
import numpy as np

from synkit.CRN.Structure.syncrn import SynCRN

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
            [f"s_{i}" for i in range(1, 12)],
        )

    def test_reaction_ids(self):
        self.assertEqual(
            self.syn.reaction_ids,
            [f"r_{i}" for i in range(1, 12)],
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
        self.assertEqual(self.syn._species_token("s_1", "id"), "s_1")
        self.assertEqual(self.syn._species_token("s_1", "label"), "A")
        self.assertEqual(self.syn._species_token("s_1", "smiles"), "A")
        self.assertEqual(self.syn._species_token("s_1", "source"), "s_1")

    def test_species_token_invalid_mode(self):
        with self.assertRaises(ValueError):
            self.syn._species_token("s_1", "bad_mode")

    def test_format_reaction_default(self):
        self.assertEqual(self.syn.format_reaction("r_1"), "r_1: 2A >> B + 3C")
        self.assertEqual(self.syn.format_reaction("r_2"), "r_2: 2B >> D")
        self.assertEqual(self.syn.format_reaction("r_11"), "r_11: K + C >> H")

    def test_format_reaction_without_id(self):
        self.assertEqual(
            self.syn.format_reaction("r_1", include_id=False),
            "2A >> B + 3C",
        )

    def test_to_equations(self):
        eqs = self.syn.to_equations()
        self.assertEqual(len(eqs), 11)
        self.assertEqual(eqs[0], "r_1: 2A >> B + 3C")
        self.assertEqual(eqs[1], "r_2: 2B >> D")
        self.assertEqual(eqs[-1], "r_11: K + C >> H")

    def test_describe_without_species(self):
        expected = "\n".join(
            [
                "SynCRN: 11 species, 11 reactions",
                "  r_1: 2A >> B + 3C",
                "  r_2: 2B >> D",
                "  r_3: C + D >> E",
                "  r_4: 3C + D >> F",
                "  r_5: 2C + E >> F",
                "  r_6: 3B >> G",
                "  r_7: 3C + G >> H",
                "  r_8: B + C >> I",
                "  r_9: C + I >> J",
                "  r_10: E + I >> K",
                "  r_11: K + C >> H",
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
        sp1 = self.syn.species["s_1"]
        self.assertEqual(sp1.id, "s_1")
        self.assertEqual(sp1.label, "A")
        self.assertEqual(sp1.source_node_id, "s_1")
        self.assertEqual(sp1.source_attrs["kind"], "species")
        self.assertEqual(sp1.source_attrs["label"], "A")

        sp11 = self.syn.species["s_11"]
        self.assertEqual(sp11.label, "K")

    def test_reaction_table_contents_first_reaction(self):
        rxn = self.syn.reactions["r_1"]
        self.assertEqual(rxn.id, "r_1")
        self.assertEqual(rxn.source_node_id, "r_1")
        self.assertEqual(rxn.source_kind, "rule")
        self.assertEqual(rxn.lhs.to_dict(), {"s_1": 2})
        self.assertEqual(rxn.rhs.to_dict(), {"s_2": 1, "s_3": 3})
        self.assertIsNone(rxn.rule_id)
        self.assertIsNone(rxn.rule_index)
        self.assertIsNone(rxn.rule_repr)
        self.assertEqual(rxn.source_attrs["kind"], "rule")
        self.assertEqual(rxn.source_attrs["rxn_repr"], "2A>>B+3C")

    def test_reaction_table_contents_last_reaction(self):
        rxn = self.syn.reactions["r_11"]
        self.assertEqual(rxn.lhs.to_dict(), {"s_3": 1, "s_11": 1})
        self.assertEqual(rxn.rhs.to_dict(), {"s_8": 1})

    def test_reactant_and_product_edge_attrs(self):
        rxn = self.syn.reactions["r_1"]
        self.assertEqual(
            rxn.reactant_edge_attrs,
            {"s_1": {"role": "reactant", "stoich": 2}},
        )
        self.assertEqual(
            rxn.product_edge_attrs,
            {
                "s_2": {"role": "product", "stoich": 1},
                "s_3": {"role": "product", "stoich": 3},
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
        j = mats["reaction_order"].index("r_1")
        s_minus_col = [row[j] for row in mats["S_minus"]]
        s_plus_col = [row[j] for row in mats["S_plus"]]
        s_col = [row[j] for row in mats["S"]]

        self.assertEqual(s_minus_col, [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(s_plus_col, [0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(s_col, [-2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_to_stoichiometric_matrices_last_reaction_column(self):
        mats = self.syn.to_stoichiometric_matrices()
        j = mats["reaction_order"].index("r_11")
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

        self.assertEqual(pn["pre"]["s_1"], {"r_1": 2})
        self.assertEqual(pn["post"]["s_2"], {"r_1": 1})
        self.assertEqual(pn["post"]["s_3"], {"r_1": 3})

        self.assertEqual(pn["pre"]["s_11"], {"r_11": 1})
        self.assertEqual(pn["pre"]["s_8"], {})
        self.assertEqual(pn["post"]["s_8"]["r_7"], 1)
        self.assertEqual(pn["post"]["s_8"]["r_11"], 1)

    def test_to_digraph_source_ids(self):
        g = self.syn.to_digraph()
        self.assertIsInstance(g, nx.DiGraph)

        self.assertIn("s_1", g.nodes)
        self.assertIn("r_1", g.nodes)
        self.assertIn("r_11", g.nodes)

        self.assertEqual(g.nodes["s_1"]["kind"], "species")
        self.assertEqual(g.nodes["s_1"]["label"], "A")
        self.assertEqual(g.nodes["r_1"]["kind"], "rule")
        self.assertEqual(g.nodes["r_1"]["label"], "r_1")

        self.assertEqual(g["s_1"]["r_1"]["role"], "reactant")
        self.assertEqual(g["s_1"]["r_1"]["stoich"], 2)
        self.assertEqual(g["r_1"]["s_2"]["role"], "product")
        self.assertEqual(g["r_1"]["s_2"]["stoich"], 1)
        self.assertEqual(g["r_1"]["s_3"]["stoich"], 3)

    def test_to_digraph_internal_ids(self):
        g = self.syn.to_digraph(node_ids="internal", reaction_kind="reaction")
        self.assertIn("s_1", g.nodes)
        self.assertIn("r_1", g.nodes)

        self.assertEqual(g.nodes["s_1"]["kind"], "species")
        self.assertEqual(g.nodes["s_1"]["syncrn_id"], "s_1")
        self.assertEqual(g.nodes["s_1"]["source_node_id"], "s_1")

        self.assertEqual(g.nodes["r_1"]["kind"], "reaction")
        self.assertEqual(g.nodes["r_1"]["syncrn_id"], "r_1")
        self.assertEqual(g.nodes["r_1"]["source_node_id"], "r_1")

        self.assertEqual(g["s_1"]["r_1"]["role"], "reactant")
        self.assertEqual(g["s_1"]["r_1"]["stoich"], 2)
        self.assertEqual(g["r_1"]["s_2"]["role"], "product")

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

        self.assertEqual(syn.rule_ids[0], "rule_1")
        self.assertEqual(syn.rules["rule_1"].rule_index, 0)
        self.assertEqual(syn.rules["rule_1"].rule_repr, "rule_0")
        self.assertEqual(syn.rules["rule_1"].label, "r0")

        self.assertEqual(syn.reactions["r_1"].rule_id, "rule_1")
        self.assertEqual(syn.reactions["r_1"].rule_index, 0)
        self.assertEqual(syn.reactions["r_1"].rule_repr, "rule_0")

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
        self.assertEqual(syn.reactions["r_1"].lhs.to_dict(), {})
        self.assertEqual(syn.reactions["r_1"].rhs.to_dict(), {"s_1": 1})

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
            "C + I >> J",
            "E + I >> K",
            "K + C >> H",
        ]
        self.assertEqual(
            syn2.to_equations(species="label", include_id=False),
            expected,
        )
        # Round-tripping through a graph must not permute the network.
        self.assertEqual(
            syn2.to_equations(species="label", include_id=False),
            self.syn.to_equations(species="label", include_id=False),
        )

    def test_from_digraph_requires_digraph(self):
        with self.assertRaises(TypeError):
            SynCRN.from_digraph(nx.Graph())  # type: ignore[arg-type]


class TestSynCRNIdPolicy(unittest.TestCase):
    """Both constructors must mint ids under the same policy (F9)."""

    RXNS = ["2A>>B+3C", "B+C>>D"]

    def test_default_is_prefixed_and_matches_from_digraph(self):
        syn = SynCRN.from_reaction_strings(self.RXNS)
        self.assertEqual(syn.species_ids, ["s_1", "s_2", "s_3", "s_4"])
        self.assertEqual(syn.reaction_ids, ["r_1", "r_2"])

        rebuilt = SynCRN.from_digraph(syn.to_digraph())
        self.assertEqual(rebuilt.species_ids, syn.species_ids)
        self.assertEqual(rebuilt.reaction_ids, syn.reaction_ids)

    def test_species_and_reaction_namespaces_are_disjoint(self):
        syn = SynCRN.from_reaction_strings(self.RXNS)
        self.assertEqual(set(syn.species_ids) & set(syn.reaction_ids), set())

    def test_numeric_style_reproduces_legacy_scheme(self):
        syn = SynCRN.from_reaction_strings(self.RXNS, id_style="numeric")
        self.assertEqual(syn.species_ids, ["1", "2", "3", "4"])
        self.assertEqual(syn.reaction_ids, ["5", "6"])
        self.assertEqual(syn.species["1"].source_node_id, 1)
        self.assertEqual(syn.reactions["5"].source_node_id, 5)

    def test_numeric_style_on_from_digraph(self):
        syn = SynCRN.from_reaction_strings(self.RXNS)
        rebuilt = SynCRN.from_digraph(syn.to_digraph(), id_style="numeric")
        self.assertEqual(rebuilt.species_ids, ["1", "2", "3", "4"])
        self.assertEqual(rebuilt.reaction_ids, ["5", "6"])

    def test_custom_prefixes(self):
        syn = SynCRN.from_reaction_strings(
            self.RXNS,
            rules=["ra", "rb"],
            species_prefix="sp",
            reaction_prefix="rx",
            rule_prefix="ru",
        )
        self.assertEqual(syn.species_ids[0], "sp1")
        self.assertEqual(syn.reaction_ids[0], "rx1")
        self.assertEqual(syn.rule_ids[0], "ru1")

    def test_unknown_id_style_raises(self):
        with self.assertRaises(ValueError):
            SynCRN.from_reaction_strings(self.RXNS, id_style="roman")
        with self.assertRaises(ValueError):
            SynCRN.from_digraph(nx.DiGraph(), id_style="roman")

    def test_id_style_recorded_in_metadata(self):
        self.assertEqual(
            SynCRN.from_reaction_strings(self.RXNS).metadata["id_style"],
            "prefixed",
        )
        self.assertEqual(
            SynCRN.from_reaction_strings([]).metadata["id_style"],
            "prefixed",
        )

    def test_two_digit_ids_keep_numeric_order_on_roundtrip(self):
        rxns = [f"A{i}>>A{i + 1}" for i in range(1, 13)]
        syn = SynCRN.from_reaction_strings(rxns)
        rebuilt = SynCRN.from_digraph(syn.to_digraph())
        self.assertEqual(rebuilt.reaction_ids, syn.reaction_ids)
        self.assertEqual(
            rebuilt.to_equations(species="label", include_id=False),
            syn.to_equations(species="label", include_id=False),
        )


class TestSynCRNStoichiometricMatrices(unittest.TestCase):
    """Matrix views must be numpy-native and sparse-capable (F13)."""

    def setUp(self):
        self.syn = SynCRN.from_reaction_strings(["2A>>B+3C", "B+C>>D"])

    def test_dense_is_numpy_integer_array(self):
        mats = self.syn.to_stoichiometric_matrices()
        for key in ("S_minus", "S_plus", "S"):
            self.assertIsInstance(mats[key], np.ndarray)
            self.assertEqual(mats[key].shape, (4, 2))
            self.assertEqual(mats[key].dtype, np.int64)

    def test_dense_values(self):
        mats = self.syn.to_stoichiometric_matrices()
        np.testing.assert_array_equal(
            mats["S"],
            np.array([[-2, 0], [1, -1], [3, -1], [0, 1]]),
        )

    def test_dtype_override(self):
        mats = self.syn.to_stoichiometric_matrices(dtype=np.float64)
        self.assertEqual(mats["S"].dtype, np.float64)

    def test_sparse_matches_dense(self):
        dense = self.syn.to_stoichiometric_matrices()
        sparse = self.syn.to_stoichiometric_matrices(sparse=True)
        for key in ("S_minus", "S_plus", "S"):
            self.assertTrue(hasattr(sparse[key], "nnz"))
            np.testing.assert_array_equal(sparse[key].toarray(), dense[key])

    def test_sparse_stores_only_nonzeros(self):
        rxns = [f"A{i}>>A{i + 1}" for i in range(200)]
        syn = SynCRN.from_reaction_strings(rxns)
        sparse = syn.to_stoichiometric_matrices(sparse=True)
        self.assertEqual(sparse["S_minus"].shape, (201, 200))
        self.assertEqual(sparse["S_minus"].nnz, 200)
        self.assertEqual(sparse["S"].nnz, 400)

    def test_empty_network(self):
        mats = SynCRN.from_reaction_strings([]).to_stoichiometric_matrices()
        self.assertEqual(mats["S"].shape, (0, 0))


if __name__ == "__main__":
    unittest.main()
