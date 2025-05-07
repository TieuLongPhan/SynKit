import unittest
from synkit.IO.data_io import load_from_pickle
from synkit.IO.chem_converter import rsmi_to_its
from synkit.Graph.ITS.its_decompose import get_rc
from synkit.Graph.Matcher.graph_morphism import (
    graph_isomorphism,
    subgraph_isomorphism,
    maximum_connected_common_subgraph,
    heuristics_MCCS,
)


class TestGraphMorphism(unittest.TestCase):

    def setUp(self):
        self.graphs = load_from_pickle("Data/Testcase/graph.pkl.gz")
        rsmi = (
            "[F:1][C:2]([F:3])([F:4])[c:5]1[cH:6][n:8][c:9]([Cl:10])"
            + "[c:11]([Br:12])[cH:7]1.[O:13]([CH2:14][Na:16])[H:15]"
            + ">>[Cl:10][Na:16].[F:1][C:2]([F:3])([F:4])[c:5]1[cH:6]"
            + "[n:8][c:9]([O:13][CH2:14][H:15])[c:11]([Br:12])[cH:7]1"
        )
        self.its = rsmi_to_its(rsmi)
        self.rc = get_rc(self.its)

    def test_graph_isomorphism_true(self):
        result = graph_isomorphism(
            self.graphs[0]["RC"], self.graphs[3]["RC"], use_defaults=True
        )
        self.assertTrue(result)

    def test_graph_isomorphism_false(self):
        result = graph_isomorphism(
            self.graphs[0]["RC"], self.graphs[1]["RC"], use_defaults=True
        )
        self.assertFalse(result)

    def test_graph_subgraph_morphism_true(self):
        result = subgraph_isomorphism(self.graphs[0]["RC"], self.graphs[0]["ITS"])
        self.assertTrue(result)

    def test_graph_subgraph_morphism_false(self):
        result = subgraph_isomorphism(self.graphs[0]["RC"], self.graphs[1]["ITS"])
        self.assertFalse(result)

    def test_subgraph_monomorphism(self):
        # Is monomorphims
        result = subgraph_isomorphism(self.rc, self.its, check_type="mono")
        self.assertTrue(result)
        # not induce subgraph
        result = subgraph_isomorphism(self.rc, self.its, check_type="induced")
        self.assertFalse(result)

    def test_maximum_connected_common_subgraph(self):
        mcs = maximum_connected_common_subgraph(
            self.graphs[0]["RC"], self.graphs[1]["RC"]
        )
        self.assertEqual(mcs.number_of_nodes(), 3)
        self.assertGreater(
            self.graphs[0]["RC"].number_of_nodes(), mcs.number_of_nodes()
        )

    def test_heuristics_MCCS(self):
        graphs = [value["RC"] for value in self.graphs]
        mcs = heuristics_MCCS(graphs[:3])
        self.assertEqual(mcs.number_of_nodes(), 1)
        self.assertGreater(graphs[0].number_of_nodes(), mcs.number_of_nodes())


if __name__ == "__main__":
    unittest.main()
