import unittest
from synkit.IO.chem_converter import rsmi_to_its
from synkit.Graph.ITS.its_decompose import get_rc
from synkit.Graph.MTG.mtg import MTG


class TestMTG(unittest.TestCase):

    def setUp(self) -> None:
        test_1 = [
            "[CH:4]([H:7])([H:8])[CH:5]=[O:6]>>[CH:4]([H:8])=[CH:5][O:6]([H:7])",
            "[CH3:1][CH:2]=[O:3].[CH:4]([H:8])=[CH:5][O:6]([H:7])>>[CH3:1][CH:2]([O:3][H:7])[CH:4]([H:8])[CH:5]=[O:6]",
        ]
        self.test_graph_1 = [get_rc(rsmi_to_its(var)) for var in test_1]

    def test_MTG_1(self):
        mtg = MTG(self.test_graph_1[0:2], mcs_mol=True)
        self.assertEqual(mtg._graph.number_of_nodes(), 6)
        self.assertEqual(mtg._graph.number_of_edges(), 7)


if __name__ == "__main__":
    unittest.main()
