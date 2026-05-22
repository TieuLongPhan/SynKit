import unittest

from synkit.IO.chem_converter import rsmi_to_graph, rsmi_to_its
from synkit.Synthesis.Reactor.rule_filter import RuleFilter


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


if __name__ == "__main__":
    unittest.main()
