import unittest
from synkit.Synthesis.Reactor.strategy import Strategy


class TestStrategy(unittest.TestCase):
    def test_from_string_with_enum(self):
        # Passing through an existing Strategy returns it unchanged
        self.assertIs(Strategy.from_string(Strategy.ALL), Strategy.ALL)
        self.assertIs(Strategy.from_string(Strategy.COMPONENT), Strategy.COMPONENT)
        self.assertIs(Strategy.from_string(Strategy.BACKTRACK), Strategy.BACKTRACK)

    def test_from_string_valid_codes(self):
        # Lower-case codes
        self.assertIs(Strategy.from_string("all"), Strategy.ALL)
        self.assertIs(Strategy.from_string("comp"), Strategy.COMPONENT)
        self.assertIs(Strategy.from_string("bt"), Strategy.BACKTRACK)

    def test_from_string_case_insensitive(self):
        # Mixed-case codes should still work
        self.assertIs(Strategy.from_string("All"), Strategy.ALL)
        self.assertIs(Strategy.from_string("COMP"), Strategy.COMPONENT)
        self.assertIs(Strategy.from_string("Bt"), Strategy.BACKTRACK)

    def test_from_string_invalid_raises_value_error(self):
        with self.assertRaises(ValueError) as cm:
            Strategy.from_string("unknown")
        msg = str(cm.exception)
        self.assertIn("Unknown strategy: 'unknown'", msg)

    def test_str_returns_value(self):
        self.assertEqual(str(Strategy.ALL), "all")
        self.assertEqual(str(Strategy.COMPONENT), "comp")
        self.assertEqual(str(Strategy.BACKTRACK), "bt")

    def test_repr_returns_enum_style(self):
        self.assertEqual(repr(Strategy.ALL), "Strategy.ALL")
        self.assertEqual(repr(Strategy.COMPONENT), "Strategy.COMPONENT")
        self.assertEqual(repr(Strategy.BACKTRACK), "Strategy.BACKTRACK")


if __name__ == "__main__":
    unittest.main()
