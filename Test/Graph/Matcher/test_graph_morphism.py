import unittest
import networkx as nx
from synkit.IO.data_io import load_from_pickle
from synkit.IO.chem_converter import rsmi_to_its
from synkit.Graph.ITS.its_decompose import get_rc
from synkit.Graph.Matcher.graph_morphism import (
    find_graph_isomorphism,
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

    def test_filter_does_not_assume_shared_node_identifiers(self):
        host = nx.cycle_graph(3)
        pattern = nx.Graph()
        pattern.add_edge(10, 11, order=1)
        nx.set_node_attributes(host, "C", "element")
        nx.set_node_attributes(host, 0, "charge")
        nx.set_edge_attributes(host, 1, "order")
        nx.set_node_attributes(pattern, "C", "element")
        nx.set_node_attributes(pattern, 0, "charge")

        self.assertTrue(
            subgraph_isomorphism(
                pattern,
                host,
                use_filter=True,
                check_type="monomorphism",
            )
        )

    def test_find_graph_isomorphism_dispatches_directed_graphs(self):
        graph = nx.DiGraph([(0, 1)])
        relabelled = nx.relabel_nodes(graph, {0: 10, 1: 11})

        mapping = find_graph_isomorphism(graph, relabelled, use_defaults=False)

        self.assertIsNotNone(mapping)
        self.assertEqual(set(mapping), set(graph))

    def test_directed_subgraph_preserves_orientation(self):
        host = nx.DiGraph()
        host.add_edge(0, 1)
        host.nodes[0].update(element="C", charge=0)
        host.nodes[1].update(element="O", charge=0)
        pattern = nx.DiGraph()
        pattern.add_edge(10, 11)
        pattern.nodes[10].update(element="O", charge=0)
        pattern.nodes[11].update(element="C", charge=0)

        self.assertFalse(
            subgraph_isomorphism(
                pattern,
                host,
                check_type="monomorphism",
            )
        )

    def test_find_graph_isomorphism_matches_multiedge_attributes(self):
        graph = nx.MultiDiGraph()
        graph.add_edge(0, 1, order=1)
        graph.add_edge(0, 1, order=2)
        relabelled = nx.relabel_nodes(graph, {0: 10, 1: 11})

        self.assertIsNotNone(find_graph_isomorphism(graph, relabelled))

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
