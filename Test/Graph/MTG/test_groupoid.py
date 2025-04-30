import unittest
from synkit.IO.chem_converter import rsmi_to_its
from synkit.Graph.MTG.groupoid import charge_tuple, node_constraint, edge_constraint


class TestGroupoid(unittest.TestCase):

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

    def test_direct_charges(self):
        attrs = {"charges": (0, 1)}
        self.assertEqual(charge_tuple(attrs), (0, 1))

    def test_both_fields_prioritize_charges(self):
        attrs = {
            "charges": (1, 2),
            "typesGH": ((None, None, None, 3, None), (None, None, None, 4, None)),
        }
        # 'charges' should take precedence over 'typesGH'
        self.assertEqual(charge_tuple(attrs), (1, 2))

    def test_charges_not_tuple(self):
        attrs = {
            "charges": [0, 1],  # not a tuple
            "typesGH": ((None, None, None, 2, None), (None, None, None, 3, None)),
        }
        # Non-tuple 'charges' should be ignored in favor of typesGH
        self.assertEqual(charge_tuple(attrs), (2, 3))

    def test_charges_wrong_length(self):
        attrs = {
            "charges": (0,),  # length != 2
            "typesGH": ((None, None, None, 5, None), (None, None, None, 6, None)),
        }
        # Invalid 'charges' length => fallback to typesGH
        self.assertEqual(charge_tuple(attrs), (5, 6))

    def test_typesGH_valid(self):
        attrs = {"typesGH": ((None, None, None, 7, None), (None, None, None, 8, None))}
        self.assertEqual(charge_tuple(attrs), (7, 8))

    def test_typesGH_too_short(self):
        attrs = {"typesGH": ((None, None, None, 9, None),)}  # only one tuple
        # Not enough entries => (None, None)
        self.assertEqual(charge_tuple(attrs), (None, None))

    def test_typesGH_inner_exception(self):
        attrs = {"typesGH": ((None, None), (None,))}  # inner tuples too short
        # Should catch exception and return (None, None)
        self.assertEqual(charge_tuple(attrs), (None, None))

    def test_no_fields(self):
        # No relevant keys => (None, None)
        self.assertEqual(charge_tuple({}), (None, None))

    def test_node_constraint(self):
        m1 = node_constraint(
            self.test_graph_1[0].nodes(data=True), self.test_graph_1[1].nodes(data=True)
        )
        self.assertEqual(len(m1.keys()), 5)

        m2 = node_constraint(
            self.test_graph_2[0].nodes(data=True), self.test_graph_2[1].nodes(data=True)
        )
        self.assertEqual(len(m2.keys()), 3)

    def test_edge_constraint_no_map(self):
        m1 = edge_constraint(
            self.test_graph_1[0].edges(data=True),
            self.test_graph_1[1].edges(data=True),
            algorithm="bt",
        )
        self.assertEqual(len(m1), 46)  # backtracking

        self.assertEqual(
            len(
                edge_constraint(
                    self.test_graph_1[0].edges(data=True),
                    self.test_graph_1[1].edges(data=True),
                    algorithm="vf2",
                )
            ),
            30,
        )  # vf2

        self.assertEqual(
            len(
                edge_constraint(
                    self.test_graph_1[0].edges(data=True),
                    self.test_graph_1[1].edges(data=True),
                    algorithm="vf3",
                )
            ),
            30,
        )  # vf3

        m2 = edge_constraint(
            self.test_graph_2[0].edges(data=True), self.test_graph_2[1].edges(data=True)
        )
        self.assertEqual(len(m2), 2)  # backtracking

        self.assertEqual(
            len(
                edge_constraint(
                    self.test_graph_2[0].edges(data=True),
                    self.test_graph_2[1].edges(data=True),
                    algorithm="vf2",
                )
            ),
            2,
        )

        self.assertEqual(
            len(
                edge_constraint(
                    self.test_graph_2[0].edges(data=True),
                    self.test_graph_2[1].edges(data=True),
                    algorithm="vf3",
                )
            ),
            2,
        )

    def test_edge_constraint_map(self):
        m0 = node_constraint(
            self.test_graph_1[0].nodes(data=True), self.test_graph_1[1].nodes(data=True)
        )
        m1 = edge_constraint(
            self.test_graph_1[0].edges(data=True),
            self.test_graph_1[1].edges(data=True),
            m0,
        )
        self.assertEqual(len(m1), 4)

        self.assertEqual(
            len(
                edge_constraint(
                    self.test_graph_1[0].edges(data=True),
                    self.test_graph_1[1].edges(data=True),
                    m0,
                    algorithm="vf2",
                )
            ),
            1,
        )

        self.assertEqual(
            len(
                edge_constraint(
                    self.test_graph_1[0].edges(data=True),
                    self.test_graph_1[1].edges(data=True),
                    m0,
                    algorithm="vf3",
                )
            ),
            1,
        )

        m0 = node_constraint(
            self.test_graph_2[0].nodes(data=True), self.test_graph_2[1].nodes(data=True)
        )
        m2 = edge_constraint(
            self.test_graph_2[0].edges(data=True),
            self.test_graph_2[1].edges(data=True),
            m0,
        )
        self.assertEqual(len(m2), 0)


if __name__ == "__main__":
    unittest.main()
