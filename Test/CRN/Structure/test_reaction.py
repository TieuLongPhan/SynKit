import unittest

from synkit.CRN.Structure.reaction import _clean_counts, RXNSide, Reaction


class TestCleanCounts(unittest.TestCase):
    def test_none_returns_empty_dict(self):
        self.assertEqual(_clean_counts(None), {})

    def test_empty_returns_empty_dict(self):
        self.assertEqual(_clean_counts({}), {})

    def test_removes_zero_and_sorts_keys(self):
        counts = {"b": 2, "a": 1, "c": 0}
        self.assertEqual(_clean_counts(counts), {"a": 1, "b": 2})

    def test_keeps_only_positive_integers(self):
        counts = {"x": 3, "y": 1, "z": 0}
        self.assertEqual(_clean_counts(counts), {"x": 3, "y": 1})

    def test_raises_for_non_string_species_id(self):
        with self.assertRaises(TypeError):
            _clean_counts({1: 2})  # type: ignore[arg-type]

    def test_raises_for_non_int_coefficient(self):
        with self.assertRaises(TypeError):
            _clean_counts({"A": 1.5})  # type: ignore[arg-type]

    def test_raises_for_negative_coefficient(self):
        with self.assertRaises(ValueError):
            _clean_counts({"A": -1})


class TestRXNSide(unittest.TestCase):
    def test_post_init_cleans_counts(self):
        side = RXNSide({"b": 2, "a": 1, "c": 0})
        self.assertEqual(side.counts, {"a": 1, "b": 2})

    def test_bool_false_for_empty(self):
        self.assertFalse(RXNSide({}))
        self.assertFalse(RXNSide({"A": 0}))

    def test_bool_true_for_nonempty(self):
        self.assertTrue(RXNSide({"A": 1}))

    def test_len_returns_number_of_species(self):
        side = RXNSide({"A": 2, "B": 1})
        self.assertEqual(len(side), 2)

    def test_iter_returns_items(self):
        side = RXNSide({"A": 2, "B": 1})
        self.assertEqual(list(side), [("A", 2), ("B", 1)])

    def test_items_returns_list(self):
        side = RXNSide({"A": 2, "B": 1})
        self.assertEqual(side.items(), [("A", 2), ("B", 1)])

    def test_get_existing_species(self):
        side = RXNSide({"A": 2})
        self.assertEqual(side.get("A"), 2)

    def test_get_missing_species_default_zero(self):
        side = RXNSide({"A": 2})
        self.assertEqual(side.get("B"), 0)

    def test_get_missing_species_custom_default(self):
        side = RXNSide({"A": 2})
        self.assertEqual(side.get("B", 99), 99)

    def test_to_dict_returns_copy_like_mapping(self):
        side = RXNSide({"A": 2})
        self.assertEqual(side.to_dict(), {"A": 2})
        self.assertIsNot(side.to_dict(), side.counts)


class TestReaction(unittest.TestCase):
    def setUp(self):
        self.reaction = Reaction(
            id="r_1",
            source_node_id="node_1",
            source_kind="rule",
            lhs=RXNSide({"s1": 2, "s2": 1}),
            rhs=RXNSide({"s3": 1}),
            label="example",
            step=3,
            rule_index=7,
            app_index=11,
            rule_repr="[A]>>[B]",
            rule_id="rule_7",
            source_attrs={"kind": "rule", "foo": "bar"},
            metadata={"score": 0.95},
            reactant_edge_attrs={"s1": {"weight": 2}},
            product_edge_attrs={"s3": {"weight": 1}},
        )
        self.species_token = lambda sid: {
            "s1": "A",
            "s2": "B",
            "s3": "C",
        }[sid]

    def test_format_side_empty(self):
        rxn = Reaction(
            id="r_0",
            source_node_id="node_0",
            source_kind="rule",
            lhs=RXNSide({}),
            rhs=RXNSide({}),
        )
        self.assertEqual(rxn.format_side(RXNSide({}), str), "∅")

    def test_format_side_nonempty(self):
        text = self.reaction.format_side(self.reaction.lhs, self.species_token)
        self.assertEqual(text, "2A + B")

    def test_format_default(self):
        text = self.reaction.format(self.species_token)
        self.assertEqual(text, "r_1: 2A + B >> C")

    def test_format_without_id(self):
        text = self.reaction.format(self.species_token, include_id=False)
        self.assertEqual(text, "2A + B >> C")

    def test_format_with_custom_arrow(self):
        text = self.reaction.format(self.species_token, arrow="->")
        self.assertEqual(text, "r_1: 2A + B -> C")

    def test_format_include_step(self):
        text = self.reaction.format(self.species_token, include_step=True)
        self.assertEqual(text, "r_1: 2A + B >> C  (step=3)")

    def test_format_include_rule_prefers_rule_id(self):
        text = self.reaction.format(self.species_token, include_rule=True)
        self.assertEqual(
            text,
            "r_1: 2A + B >> C  (rule_id=rule_7, rule_repr=[A]>>[B])",
        )

    def test_format_include_step_and_rule(self):
        text = self.reaction.format(
            self.species_token,
            include_step=True,
            include_rule=True,
        )
        self.assertEqual(
            text,
            "r_1: 2A + B >> C  (step=3, rule_id=rule_7, rule_repr=[A]>>[B])",
        )

    def test_format_include_rule_falls_back_to_rule_index(self):
        rxn = Reaction(
            id="r_2",
            source_node_id="node_2",
            source_kind="rule",
            lhs=RXNSide({"s1": 1}),
            rhs=RXNSide({"s2": 1}),
            rule_index=5,
            rule_repr="X>>Y",
        )

        def token(sid: str) -> str:
            return {"s1": "X", "s2": "Y"}[sid]

        text = rxn.format(token, include_rule=True)
        self.assertEqual(text, "r_2: X >> Y  (rule_index=5, rule_repr=X>>Y)")

    def test_to_dict(self):
        d = self.reaction.to_dict()
        self.assertEqual(
            d,
            {
                "id": "r_1",
                "source_node_id": "node_1",
                "source_kind": "rule",
                "label": "example",
                "lhs": {"s1": 2, "s2": 1},
                "rhs": {"s3": 1},
                "step": 3,
                "rule_index": 7,
                "app_index": 11,
                "rule_repr": "[A]>>[B]",
                "rule_id": "rule_7",
                "source_attrs": {"kind": "rule", "foo": "bar"},
                "metadata": {"score": 0.95},
                "reactant_edge_attrs": {"s1": {"weight": 2}},
                "product_edge_attrs": {"s3": {"weight": 1}},
            },
        )

    def test_to_dict_returns_copied_nested_dicts(self):
        d = self.reaction.to_dict()
        self.assertIsNot(d["source_attrs"], self.reaction.source_attrs)
        self.assertIsNot(d["metadata"], self.reaction.metadata)
        self.assertIsNot(
            d["reactant_edge_attrs"]["s1"],
            self.reaction.reactant_edge_attrs["s1"],
        )
        self.assertIsNot(
            d["product_edge_attrs"]["s3"],
            self.reaction.product_edge_attrs["s3"],
        )


if __name__ == "__main__":
    unittest.main()
