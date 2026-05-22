import unittest

import networkx as nx

from synkit.Graph.ITS.its_reverter import ITSReverter
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Graph.Mech.electron_accounting import refresh_electron_fields
from synkit.Graph.MTG.mtg import MTG
from synkit.IO import load_database, rsmi_to_its


class TestTupleMTG(unittest.TestCase):
    def setUp(self):
        g0 = nx.Graph()
        g0.add_node(
            1,
            element="N",
            aromatic=False,
            hcount=2,
            charge=0,
            lone_pairs=1,
            radical=0,
            valence_electrons=5,
        )
        g0.add_node(
            2,
            element="C",
            aromatic=False,
            hcount=3,
            charge=0,
            lone_pairs=0,
            radical=0,
            valence_electrons=4,
        )
        g0.add_edge(
            1,
            2,
            order=1.0,
            kekule_order=1.0,
            sigma_order=1.0,
            pi_order=0.0,
        )

        g1 = g0.copy()
        g1.nodes[1]["hcount"] = 1
        g1.nodes[1]["radical"] = 1
        g1.edges[1, 2].update(
            order=2.0,
            kekule_order=2.0,
            sigma_order=1.0,
            pi_order=1.0,
        )

        g2 = g0.copy()
        g2.nodes[1]["charge"] = 1
        g2.nodes[1]["lone_pairs"] = 0

        self.step_1 = ITSConstruction.construct(g0, g1)
        self.step_2 = ITSConstruction.construct(g1, g2)

    def test_tuple_its_detection_preserves_electron_fields(self):
        mtg = MTG([self.step_1, self.step_2], mappings=[{1: 1, 2: 2}])

        self.assertTrue(mtg._tuple_its)
        self.assertEqual(mtg._graphs[0].nodes[1]["hcount"], (2, 1))
        self.assertEqual(mtg._graphs[0].nodes[1]["radical"], (0, 1))
        self.assertEqual(mtg._graphs[1].nodes[1]["lone_pairs"], (1, 0))

    def test_tuple_composed_its_keeps_tuple_node_state(self):
        mtg = MTG([self.step_1, self.step_2], mappings=[{1: 1, 2: 2}])

        composed = mtg.get_compose_its()

        self.assertEqual(composed.nodes[1]["hcount"], (2, 2))
        self.assertEqual(composed.nodes[1]["lone_pairs"], (1, 0))
        self.assertEqual(composed.edges[1, 2]["order"], (1.0, 1.0))
        self.assertEqual(composed.edges[1, 2]["kekule_order"], (1.0, 1.0))
        self.assertEqual(composed.edges[1, 2]["sigma_order"], (1.0, 1.0))
        self.assertEqual(composed.edges[1, 2]["pi_order"], (0.0, 0.0))

    def test_tuple_mtg_round_trips_ordered_its_steps(self):
        mtg = MTG([self.step_1, self.step_2])

        rebuilt = mtg.get_its_steps()

        self.assertEqual(len(rebuilt), 2)
        self.assertEqual(rebuilt[0].nodes[1]["hcount"], (2, 1))
        self.assertEqual(rebuilt[0].edges[1, 2]["pi_order"], (0.0, 1.0))
        self.assertEqual(rebuilt[1].nodes[1]["lone_pairs"], (1, 0))
        self.assertEqual(rebuilt[1].edges[1, 2]["pi_order"], (1.0, 0.0))

    def test_tuple_mtg_keeps_node_timelines(self):
        mtg = MTG([self.step_1, self.step_2], mappings=[{1: 1, 2: 2}])

        graph = mtg.get_mtg()

        self.assertEqual(graph.nodes[1]["element"], "N")
        self.assertEqual(graph.nodes[1]["valence_electrons"], 5)
        self.assertEqual(graph.nodes[1]["hcount"], (2, 1, 2))
        self.assertEqual(graph.nodes[1]["radical"], (0, 1, 0))
        self.assertEqual(graph.nodes[1]["lone_pairs"], (1, 1, 0))
        self.assertNotIn("typesGH", graph.nodes[1])
        self.assertNotIn("neighbors", graph.nodes[1])
        self.assertNotIn("hcount_history", graph.nodes[1])

    def test_tuple_mtg_keeps_electron_authoritative_edge_timelines(self):
        mtg = MTG([self.step_1, self.step_2], mappings=[{1: 1, 2: 2}])

        edge = mtg.get_mtg().edges[1, 2]

        self.assertEqual(edge["kekule_order"], (1.0, 2.0, 1.0))
        self.assertEqual(edge["sigma_order"], (1.0, 1.0, 1.0))
        self.assertEqual(edge["pi_order"], (0.0, 1.0, 0.0))
        self.assertNotIn("pi_order_history", edge)
        self.assertNotIn("pi_order_step_history", edge)


class TestCuratedTupleMTGMechanisms(unittest.TestCase):
    @staticmethod
    def _atom(
        element,
        *,
        hcount=0,
        charge=0,
        lone_pairs=0,
        radical=0,
        valence_electrons=None,
    ):
        valence = {
            "H": 1,
            "C": 4,
            "N": 5,
            "O": 6,
            "Cl": 7,
        }
        return {
            "element": element,
            "aromatic": False,
            "hcount": hcount,
            "charge": charge,
            "lone_pairs": lone_pairs,
            "radical": radical,
            "valence_electrons": valence_electrons or valence[element],
        }

    @staticmethod
    def _add_bond(graph, u, v, sigma=1.0, pi=0.0):
        graph.add_edge(
            u,
            v,
            order=sigma + pi,
            kekule_order=sigma + pi,
            sigma_order=sigma,
            pi_order=pi,
        )

    def _graph(self, nodes, edges):
        graph = nx.Graph()
        for node, attrs in nodes.items():
            graph.add_node(node, **attrs)
        for edge in edges:
            self._add_bond(graph, *edge)
        return refresh_electron_fields(graph)

    def test_lone_pair_donation_history_recomputes_charge_path(self):
        g0 = self._graph(
            {
                1: self._atom("N", hcount=3, lone_pairs=1),
                2: self._atom("C", hcount=3),
                3: self._atom("Cl", lone_pairs=3),
            },
            [(2, 3, 1.0, 0.0)],
        )
        g1 = self._graph(
            {
                1: self._atom("N", hcount=3, charge=1, lone_pairs=0),
                2: self._atom("C", hcount=3),
                3: self._atom("Cl", charge=-1, lone_pairs=4),
            },
            [(1, 2, 1.0, 0.0)],
        )
        g2 = self._graph(
            {
                1: self._atom("N", hcount=2, lone_pairs=1),
                2: self._atom("C", hcount=3),
                3: self._atom("Cl", hcount=1, lone_pairs=3),
            },
            [(1, 2, 1.0, 0.0), (3, 1, 1.0, 0.0)],
        )

        mtg = MTG(
            [ITSConstruction.construct(g0, g1), ITSConstruction.construct(g1, g2)],
            mappings=[{1: 1, 2: 2, 3: 3}],
        )

        self.assertEqual(mtg.get_mtg().nodes[1]["lone_pairs"], (1, 0, 1))
        self.assertEqual(mtg.get_mtg().nodes[1]["charge"], (0, 1, 0))
        self.assertEqual(
            mtg.get_mtg().edges[1, 2]["sigma_order"],
            (0.0, 1.0, 1.0),
        )
        self.assertEqual(mtg.get_compose_its().edges[1, 2]["sigma_order"], (0.0, 1.0))

    def test_radical_progression_keeps_unpaired_electron_timeline(self):
        g0 = self._graph(
            {
                1: self._atom("C", hcount=3),
                2: self._atom("Cl", lone_pairs=3),
            },
            [(1, 2, 1.0, 0.0)],
        )
        g1 = self._graph(
            {
                1: self._atom("C", hcount=3, radical=1),
                2: self._atom("Cl", radical=1, lone_pairs=3),
            },
            [],
        )
        g2 = self._graph(
            {
                1: self._atom("C", hcount=3),
                2: self._atom("Cl", lone_pairs=3),
            },
            [(1, 2, 1.0, 0.0)],
        )

        mtg = MTG(
            [ITSConstruction.construct(g0, g1), ITSConstruction.construct(g1, g2)],
            mappings=[{1: 1, 2: 2}],
        )

        self.assertEqual(mtg.get_mtg().nodes[1]["radical"], (0, 1, 0))
        self.assertEqual(
            mtg.get_mtg().edges[1, 2]["sigma_order"],
            (1.0, 0.0, 1.0),
        )
        self.assertEqual(mtg.get_compose_its().edges[1, 2]["sigma_order"], (1.0, 1.0))

    def test_rsmi_tuple_mtg_composes_back_to_outer_states(self):
        step_1 = rsmi_to_its(
            "[CH2:1]=[CH2:2].[H:3][H:4]>>[CH3:1][CH2:2][H:4]",
            format="tuple",
        )
        step_2 = rsmi_to_its(
            "[CH3:1][CH2:2][H:4]>>[CH3:1][CH3:2]",
            format="tuple",
        )

        mtg = MTG([step_1, step_2], mappings=[{1: 1, 2: 2, 4: 4}])
        composed = mtg.get_compose_its()

        left = ITSReverter(composed).to_reactant_graph()
        right = ITSReverter(composed).to_product_graph()

        self.assertTrue(left.has_edge(1, 2))
        self.assertTrue(right.has_edge(1, 2))
        self.assertEqual(composed.edges[1, 2]["pi_order"], (1.0, 0.0))
        self.assertEqual(composed.nodes[2]["hcount"], (2, 3))

    def test_mech_fixture_round_trips_ordered_tuple_rsmi_steps(self):
        data = load_database("./Data/Testcase/mech.json.gz")
        mech = data[0]["mechanisms"][1]
        steps = [step["smart_string"] for step in mech["steps"]]
        its_steps = [rsmi_to_its(step, format="tuple", core=False) for step in steps]
        mtg = MTG(its_steps)
        rebuilt = mtg.get_its_steps()

        self.assertEqual(len(rebuilt), len(its_steps))
        for original, recovered in zip(its_steps, rebuilt):
            self.assertEqual(set(original.nodes()), set(recovered.nodes()))
            self.assertEqual(
                {tuple(sorted(edge)) for edge in original.edges()},
                {tuple(sorted(edge)) for edge in recovered.edges()},
            )

            for node in original.nodes():
                for key in (
                    "element",
                    "atom_map",
                    "hcount",
                    "charge",
                    "radical",
                    "lone_pairs",
                    "valence_electrons",
                ):
                    self.assertEqual(
                        recovered.nodes[node].get(key),
                        original.nodes[node].get(key),
                    )
            for u, v in original.edges():
                edge = tuple(sorted((u, v)))
                for key in ("order", "kekule_order", "sigma_order", "pi_order"):
                    self.assertEqual(
                        recovered.edges[edge].get(key),
                        original.edges[edge].get(key),
                    )

        exported = mtg.get_rsmi_steps()
        self.assertEqual(len(exported), len(steps))
        self.assertTrue(all(">>" in step for step in exported))

    def test_mech_fixture_tuple_mtg_automatic_mapping_matches_identity_mapping(self):
        data = load_database("./Data/Testcase/mech.json.gz")
        mech = data[0]["mechanisms"][1]
        its_steps = [
            rsmi_to_its(step["smart_string"], format="tuple", core=False)
            for step in mech["steps"]
        ]
        identity_mappings = [
            {n: n for n in sorted(set(left.nodes()) & set(right.nodes()))}
            for left, right in zip(its_steps, its_steps[1:])
        ]

        auto = MTG(its_steps)
        explicit = MTG(its_steps, mappings=identity_mappings)

        self.assertEqual(auto.node_mapping, explicit.node_mapping)
        self.assertEqual(
            set(auto.get_mtg().edges()),
            set(explicit.get_mtg().edges()),
        )

    def test_all_mech_fixture_mechanisms_round_trip_ordered_tuple_its(self):
        data = load_database("./Data/Testcase/mech.json.gz")

        for mech in data[0]["mechanisms"]:
            with self.subTest(mech=mech["mech_name"]):
                its_steps = [
                    rsmi_to_its(step["smart_string"], format="tuple", core=False)
                    for step in mech["steps"]
                ]
                rebuilt = MTG(its_steps).get_its_steps()

                self.assertEqual(len(rebuilt), len(its_steps))
                for original, recovered in zip(its_steps, rebuilt):
                    self.assertEqual(set(original.nodes()), set(recovered.nodes()))
                    self.assertEqual(
                        {tuple(sorted(edge)) for edge in original.edges()},
                        {tuple(sorted(edge)) for edge in recovered.edges()},
                    )


if __name__ == "__main__":
    unittest.main()
