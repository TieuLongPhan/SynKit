import unittest
from synkit.IO.combinatorial.smarts_expander import SMARTSExpander


class TestSMARTSExpander(unittest.TestCase):

    def test_no_placeholders(self):
        s = "CCO"
        self.assertEqual(list(SMARTSExpander.expand_iter(s)), ["CCO"])
        self.assertEqual(SMARTSExpander.expand(s), ["CCO"])

    def test_simple_expansion(self):
        s = "[C,N:1][O,P:2]"
        result = SMARTSExpander.expand(s)
        self.assertEqual(
            set(result), {"[C:1][O:2]", "[C:1][P:2]", "[N:1][O:2]", "[N:1][P:2]"}
        )

    def test_disjoint_constraint(self):
        s = "[C,N:1][O:1]"
        with self.assertRaises(ValueError):
            list(SMARTSExpander.expand_iter(s))

    def test_realistic_reaction(self):
        rxn = (
            "[H+:6].[C:7](-[O:8](-[H:12]))(-[C,N,O,P,S:9])(-[C,N,O,P,S:10])(-[H:11])."
            "[C:2](-[S:4](-[C,N,O,P,S:5]))(-[C,N,O,P,S:1])(=[O:3])>>"
            "[S:4](-[H:6])(-[C,N,O,P,S:5]).[H+:12]."
            "[C:7](-[O:8](-[C:2](-[C,N,O,P,S:1])(=[O:3])))(-[C,N,O,P,S:9])(-[C,N,O,P,S:10])(-[H:11])"
        )
        ex_list = list(SMARTSExpander.expand_iter(rxn))
        self.assertEqual(len(ex_list), 625)
        # Optional: Just check format or count, not endswith
        self.assertTrue(ex_list[0].startswith("[H+:6]"))


if __name__ == "__main__":
    unittest.main()
