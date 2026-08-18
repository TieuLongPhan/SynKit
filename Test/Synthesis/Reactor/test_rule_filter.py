import unittest

from synkit.IO.chem_converter import rsmi_to_graph, rsmi_to_its, smiles_to_graph
from synkit.Synthesis.Reactor import RuleFilter


class TestRuleFilter(unittest.TestCase):
    def test_tuple_rule_uses_tuple_decomposition(self):
        host, _ = rsmi_to_graph("[CH3:1][Cl:2]>>[CH3:1][Cl:2]")
        rule = rsmi_to_its(
            "[CH3:1][Cl:2]>>[CH3:1].[Cl:2]",
            core=True,
            format="tuple",
        )

        filtered = RuleFilter(host, [rule], engine="nx")

        self.assertEqual(filtered.new_rules, [rule])

    def test_wildcard_context_is_removed_before_prefiltering(self):
        host = smiles_to_graph("CC")
        rule = rsmi_to_its("[*:1][C:2]>>[*:1][O:2]", core=False)

        for engine in ("turbo", "sing", "nx"):
            with self.subTest(engine=engine):
                filtered = RuleFilter(host, [rule], engine=engine)
                self.assertEqual(filtered.new_rules, [rule])

    def test_empty_pattern_matches_every_host(self):
        host = smiles_to_graph("C")
        rule = rsmi_to_its("[*:1]>>[*:1]", core=True)

        for engine in ("turbo", "sing", "nx"):
            with self.subTest(engine=engine):
                filtered = RuleFilter(host, [rule], engine=engine)
                self.assertEqual(filtered.new_rules, [rule])


if __name__ == "__main__":
    unittest.main()
