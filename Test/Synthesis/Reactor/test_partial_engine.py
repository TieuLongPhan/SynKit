import unittest
from synkit.Synthesis.Reactor.partial_engine import PartialEngine


class TestPartialEngine(unittest.TestCase):
    def test_forward_direction_example(self):
        """
        Example 1:
        PartialEngine(smi='CCC(=O)OC',
                      template='[C:1][O:2].[O:3][H:4]>>[C:1][O:3].[O:2][H:4]')
        .fit(invert=False)
        should return the two forward wildcarded SMARTS.
        """
        smi = "CCC(=O)OC"
        template = "[C:1][O:2].[O:3][H:4]>>[C:1][O:3].[O:2][H:4]"
        engine = PartialEngine(smi, template)
        result = engine.fit(invert=False)
        expected = [
            "[CH3:1][CH2:2][C:3](=[O:4])[O:5][CH3:6].[OH:7][*:8]>>"
            "[CH3:1][CH2:2][C:3](=[O:4])[O:7][*:8].[OH:5][CH3:6]",
            "[CH3:1][CH2:2][C:3](=[O:4])[O:5][CH3:6].[OH:7][*:8]>>"
            "[CH3:1][CH2:2][C:3](=[O:4])[OH:5].[CH3:6][O:7][*:8]",
        ]
        self.assertEqual(result, expected)

    def test_backward_direction_example(self):
        """
        Example 2:
        PartialEngine(smi='CCC(=O)OCC',
                      template='[C:1][O:2].[O:3][H:4]>>[C:1][O:3].[O:2][H:4]')
        .fit(invert=True)
        should return the two backward wildcarded SMARTS.
        """
        smi = "CCC(=O)OCC"
        template = "[C:1][O:2].[O:3][H:4]>>[C:1][O:3].[O:2][H:4]"
        engine = PartialEngine(smi, template)
        result = engine.fit(invert=True)
        expected = [
            "[CH3:1][CH2:2][C:3](=[O:4])[O:8][*:9].[OH:5][CH2:6][CH3:7]>>"
            "[CH3:1][CH2:2][C:3](=[O:4])[O:5][CH2:6][CH3:7].[OH:8][*:9]",
            "[CH2:6]([CH3:7])[O:8][*:9].[CH3:1][CH2:2][C:3](=[O:4])[OH:5]>>"
            "[CH3:1][CH2:2][C:3](=[O:4])[O:5][CH2:6][CH3:7].[OH:8][*:9]",
        ]
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
