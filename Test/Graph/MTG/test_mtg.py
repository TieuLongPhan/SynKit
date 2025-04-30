import unittest
from synkit.IO.chem_converter import rsmi_to_its
from synkit.Graph.ITS.its_decompose import get_rc
from synkit.Graph.MTG.group_comp import GroupComp
from synkit.Graph.MTG.mtg import MTG


class TestMTG(unittest.TestCase):

    def setUp(self) -> None:
        test_1 = [
            "[CH:4]([H:7])([H:8])[CH:5]=[O:6]>>[CH:4]([H:8])=[CH:5][O:6]([H:7])",
            "[CH3:1][CH:2]=[O:3].[CH:4]([H:8])=[CH:5][O:6]([H:7])>>[CH3:1][CH:2]([O:3][H:7])[CH:4]([H:8])[CH:5]=[O:6]",
        ]
        self.test_graph_1 = [get_rc(rsmi_to_its(var)) for var in test_1]
        test_2 = [
            "[CH2:1]=[CH:2]-[CH2+:3]>>[CH2+:1]-[CH:2]=[CH2:3]",
            "[H:1]-[CH2:2]-[CH2+:3]>>[CH2:2]=[CH2:3].[H+:1]",
        ]
        self.test_graph_2 = [get_rc(rsmi_to_its(var)) for var in test_2]

    def test_MTG_1(self):
        grp = GroupComp(self.test_graph_1[0], self.test_graph_1[1])
        candidates = grp.get_mapping()
        print(candidates)
        mtg = MTG(self.test_graph_1[0], self.test_graph_1[1], candidates[0])
        self.assertEqual(len(mtg.get_nodes()), 6)
        self.assertEqual(len(mtg.get_edges()), 7)

    def test_MTG_2(self):
        grp = GroupComp(self.test_graph_2[0], self.test_graph_2[1])
        candidates = grp.get_mapping()
        mtg = MTG(self.test_graph_2[0], self.test_graph_2[1], candidates[0])
        self.assertEqual(len(mtg.get_nodes()), 5)
        self.assertEqual(len(mtg.get_edges()), 4)


if __name__ == "__main__":
    unittest.main()
