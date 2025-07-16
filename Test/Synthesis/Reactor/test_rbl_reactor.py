import unittest
from synkit.IO import its_to_rsmi, rsmi_to_its
from synkit.Synthesis.Reactor.rbl_engine import RBLEngine


class TestRBLEngine(unittest.TestCase):
    def test_example1(self):
        # Example 1
        rsmi = "CCC(=O)(O)>>CCC(=O)OC"
        raw_template = (
            "[CH3:1][C:2](=[O:3])[OH:4]."
            "[CH3:5][O:6][H:7]>>"
            "[CH3:1][C:2](=[O:3])[O:6][CH3:5]."
            "[H:7][OH:4]"
        )
        its = rsmi_to_its(raw_template, core=True)
        template = its_to_rsmi(its)

        engine = RBLEngine(rsmi, template)
        result = engine.fit()

        expected = [
            (
                "[CH3:1][CH2:2][C:3](=[O:4])[OH:7]."
                "[OH:5][CH3:6]>>"
                "[CH3:1][CH2:2][C:3](=[O:4])[O:5][CH3:6]."
                "[OH2:7]"
            )
        ]
        self.assertEqual(result, expected)

    def test_example2(self):
        # Example 2
        rsmi = "CCC(=O)OC>>CCC(=O)OCC"
        template = "[C:1][O:2].[O:3][H:4]>>[C:1][O:3].[O:2][H:4]"

        engine = RBLEngine(rsmi, template)
        result = engine.fit()

        expected = [
            (
                "[CH3:1][CH2:2][C:3](=[O:4])[O:8][CH3:9]."
                "[OH:5][CH2:6][CH3:7]>>"
                "[CH3:1][CH2:2][C:3](=[O:4])[O:5][CH2:6][CH3:7]."
                "[OH:8][CH3:9]"
            )
        ]
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
