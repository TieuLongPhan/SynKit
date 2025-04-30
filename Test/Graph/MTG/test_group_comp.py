import unittest
from synkit.IO.chem_converter import rsmi_to_its
from synkit.Graph.MTG.groupoid import node_constraint
from synkit.Graph.MTG.group_comp import GroupComp


class TestGroupComp(unittest.TestCase):

    def setUp(self) -> None:
        test_1 = [
            "[CH:4]([H:7])([H:8])[CH:5]=[O:6]>>[CH:4]([H:8])=[CH:5][O:6]([H:7])",
            "[CH3:1][CH:2]=[O:3].[CH:4]([H:8])=[CH:5][O:6]([H:7])>>[CH3:1][CH:2]([O:3][H:7])[CH:4]([H:8])[CH:5]=[O:6]",
        ]
        self.test_graph_1 = [rsmi_to_its(var) for var in test_1]
        test_2 = [
            "[CH2:1]=[CH:2]-[CH2+:3]>>[CH2+:1]-[CH:2]=[CH2:3]",
            "[H:1]-[CH2:2]-[CH2+:3]>>[CH2:2]=[CH2:3].[H+:1]",
        ]
        self.test_graph_2 = [rsmi_to_its(var) for var in test_2]

    def test_get_mapping(self):
        g = GroupComp(self.test_graph_1[0], self.test_graph_1[1])
        m = g.get_mapping(include_singleton=False)
        self.assertEqual(len(m), 4)

    def test_get_mapping_singleton(self):
        g = GroupComp(self.test_graph_1[0], self.test_graph_1[1])
        m = g.get_mapping(include_singleton=True)
        self.assertEqual(len(m), 10)

    def test_get_mapping_from_nodes(self):
        m0 = node_constraint(
            self.test_graph_2[0].nodes(data=True), self.test_graph_2[1].nodes(data=True)
        )
        g = GroupComp(self.test_graph_2[0], self.test_graph_2[1])
        m = g.get_mapping_from_nodes(
            m0,
            self.test_graph_2[0].edges(data=True),
            self.test_graph_2[1].edges(data=True),
        )
        self.assertEqual(len(m), 1)

    def test_get_mapping_fallback(self):
        g = GroupComp(self.test_graph_2[0], self.test_graph_2[1])
        m = g.get_mapping(
            include_singleton=False
        )  # even False if cannot find candidate will fall back
        self.assertEqual(len(m), 1)


if __name__ == "__main__":
    unittest.main()
