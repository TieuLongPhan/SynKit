import unittest

from synkit.CRN.Structure.rule import Rule


class TestRule(unittest.TestCase):
    def test_init_minimal(self):
        rule = Rule(id="rule_1")
        self.assertEqual(rule.id, "rule_1")
        self.assertIsNone(rule.rule_index)
        self.assertIsNone(rule.rule_repr)
        self.assertIsNone(rule.label)
        self.assertEqual(rule.metadata, {})

    def test_init_full(self):
        rule = Rule(
            id="rule_2",
            rule_index=7,
            rule_repr="[C:1]>>[C:1][O]",
            label="hydroxylation",
            metadata={"source": "template_db", "score": 0.9},
        )
        self.assertEqual(rule.id, "rule_2")
        self.assertEqual(rule.rule_index, 7)
        self.assertEqual(rule.rule_repr, "[C:1]>>[C:1][O]")
        self.assertEqual(rule.label, "hydroxylation")
        self.assertEqual(rule.metadata, {"source": "template_db", "score": 0.9})

    def test_signature_with_both_fields(self):
        rule = Rule(
            id="rule_1",
            rule_index=3,
            rule_repr="A>>B",
        )
        self.assertEqual(rule.signature, (3, "A>>B"))

    def test_signature_with_none_fields(self):
        rule = Rule(id="rule_1")
        self.assertEqual(rule.signature, (None, None))

    def test_signature_with_partial_fields(self):
        rule = Rule(id="rule_1", rule_index=5)
        self.assertEqual(rule.signature, (5, None))

        rule = Rule(id="rule_2", rule_repr="X>>Y")
        self.assertEqual(rule.signature, (None, "X>>Y"))

    def test_to_dict_minimal(self):
        rule = Rule(id="rule_1")
        self.assertEqual(
            rule.to_dict(),
            {
                "id": "rule_1",
                "rule_index": None,
                "rule_repr": None,
                "label": None,
                "metadata": {},
            },
        )

    def test_to_dict_full(self):
        rule = Rule(
            id="rule_3",
            rule_index=10,
            rule_repr="A+B>>C",
            label="addition",
            metadata={"origin": "manual", "version": 1},
        )
        self.assertEqual(
            rule.to_dict(),
            {
                "id": "rule_3",
                "rule_index": 10,
                "rule_repr": "A+B>>C",
                "label": "addition",
                "metadata": {"origin": "manual", "version": 1},
            },
        )

    def test_to_dict_returns_metadata_copy(self):
        rule = Rule(
            id="rule_4",
            metadata={"a": 1},
        )
        d = rule.to_dict()
        self.assertEqual(d["metadata"], {"a": 1})
        self.assertIsNot(d["metadata"], rule.metadata)

    def test_mutable_default_metadata_is_not_shared(self):
        r1 = Rule(id="rule_1")
        r2 = Rule(id="rule_2")

        r1.metadata["x"] = 123

        self.assertEqual(r1.metadata, {"x": 123})
        self.assertEqual(r2.metadata, {})


if __name__ == "__main__":
    unittest.main()
