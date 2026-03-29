from __future__ import annotations

import unittest

import networkx as nx

from synkit.CRN.Structure import SynCRN
from synkit.CRN.Petrinet import (
    PetriNet,
    SynCRNIncidence,
    Transition,
    _accumulate_incidence_from_edges,
    _build_pre_post_from_graph,
    _build_reaction_index,
    _build_species_index,
    _coerce_stoich,
    _edge_side_from_graph,
    _extract_from_syncrn_digraph,
    _extract_from_syncrn_object,
    _graph_node_kind,
    _naturalish_key,
    _partition_syncrn_nodes,
    _safe_int,
    extract_syncrn_incidence,
)


class TestPetriNetHelpers(unittest.TestCase):
    """Unit tests for low-level helper functions."""

    def test_safe_int(self) -> None:
        self.assertEqual(_safe_int(3), 3)
        self.assertEqual(_safe_int("4"), 4)
        self.assertEqual(_safe_int("abc", default=7), 7)

    def test_coerce_stoich_accepts_positive(self) -> None:
        self.assertEqual(_coerce_stoich(1), 1)
        self.assertEqual(_coerce_stoich("2"), 2)

    def test_coerce_stoich_rejects_nonpositive(self) -> None:
        with self.assertRaises(ValueError):
            _coerce_stoich(0)
        with self.assertRaises(ValueError):
            _coerce_stoich(-1)

    def test_naturalish_key(self) -> None:
        self.assertLess(_naturalish_key("2"), _naturalish_key("10"))
        self.assertIsInstance(_naturalish_key("A"), tuple)

    def test_graph_node_kind(self) -> None:
        self.assertEqual(_graph_node_kind({"kind": "species"}), "species")
        self.assertEqual(_graph_node_kind({"kind": " Rule "}), "rule")
        self.assertEqual(_graph_node_kind({}), "")

    def test_edge_side_from_graph_prefers_role(self) -> None:
        self.assertEqual(
            _edge_side_from_graph("A", "r1", {"role": "reactant"}, "r1"),
            "lhs",
        )
        self.assertEqual(
            _edge_side_from_graph("r1", "B", {"role": "product"}, "r1"),
            "rhs",
        )

    def test_edge_side_from_graph_falls_back_to_direction(self) -> None:
        self.assertEqual(
            _edge_side_from_graph("A", "r1", {}, "r1"),
            "lhs",
        )
        self.assertEqual(
            _edge_side_from_graph("r1", "B", {}, "r1"),
            "rhs",
        )


class TestSynCRNIncidenceExtraction(unittest.TestCase):
    """Unit tests for extracting canonical SynCRN incidence."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rxns = [
            "2A>>B+3C",
            "2B>>D",
            "C+D>>E",
            "3C+D>>F",
            "2C+E>>F",
            "3B>>G",
            "3C+G>>H",
            "B+C>>I",
            "C+I>>J",
            "E+I>>K",
            "C+K>>H",
        ]
        cls.syn = SynCRN.from_reaction_strings(cls.rxns)
        cls.g = cls.syn.to_digraph()

    def test_extract_from_syncrn_object(self) -> None:
        inc = _extract_from_syncrn_object(self.syn)

        self.assertIsInstance(inc, SynCRNIncidence)
        self.assertEqual(set(inc.species_order), set(map(str, self.syn.species.keys())))
        self.assertEqual(
            set(inc.reaction_order), set(map(str, self.syn.reactions.keys()))
        )
        self.assertEqual(len(inc.pre), len(self.syn.reactions))
        self.assertEqual(len(inc.post), len(self.syn.reactions))

    def test_extract_from_syncrn_digraph(self) -> None:
        inc = _extract_from_syncrn_digraph(self.g)

        self.assertIsInstance(inc, SynCRNIncidence)
        self.assertGreater(len(inc.species_order), 0)
        self.assertGreater(len(inc.reaction_order), 0)
        self.assertEqual(inc.metadata["source_graph_type"], "DiGraph")

    def test_extract_dispatch_from_object(self) -> None:
        inc = extract_syncrn_incidence(self.syn)
        self.assertIsInstance(inc, SynCRNIncidence)

    def test_extract_dispatch_from_graph(self) -> None:
        inc = extract_syncrn_incidence(self.g)
        self.assertIsInstance(inc, SynCRNIncidence)

    def test_extract_dispatch_from_to_digraph_object(self) -> None:
        class Wrapper:
            def __init__(self, syn):
                self._syn = syn

            def to_digraph(self):
                return self._syn.to_digraph()

        inc = extract_syncrn_incidence(Wrapper(self.syn))
        self.assertIsInstance(inc, SynCRNIncidence)

    def test_extract_invalid_input_raises(self) -> None:
        with self.assertRaises(TypeError):
            extract_syncrn_incidence(object())

    def test_extract_from_non_digraph_raises(self) -> None:
        with self.assertRaises(TypeError):
            _extract_from_syncrn_digraph(nx.Graph())

    def test_partition_syncrn_nodes(self) -> None:
        species_nodes, reaction_nodes = _partition_syncrn_nodes(self.g)

        self.assertGreater(len(species_nodes), 0)
        self.assertGreater(len(reaction_nodes), 0)

        for n in species_nodes:
            self.assertEqual(_graph_node_kind(self.g.nodes[n]), "species")
        for n in reaction_nodes:
            self.assertIn(_graph_node_kind(self.g.nodes[n]), {"reaction", "rule"})

    def test_build_species_index(self) -> None:
        species_nodes, _ = _partition_syncrn_nodes(self.g)
        species_order, species_labels, species_source_node_ids, species_node_to_id = (
            _build_species_index(self.g, species_nodes)
        )

        self.assertEqual(len(species_order), len(species_nodes))
        self.assertEqual(set(species_order), set(species_labels))
        self.assertEqual(set(species_order), set(species_source_node_ids))
        self.assertEqual(set(species_nodes), set(species_node_to_id))

    def test_build_reaction_index(self) -> None:
        _, reaction_nodes = _partition_syncrn_nodes(self.g)
        (
            reaction_order,
            reaction_labels,
            reaction_source_node_ids,
            reaction_node_to_id,
        ) = _build_reaction_index(self.g, reaction_nodes)

        self.assertEqual(len(reaction_order), len(reaction_nodes))
        self.assertEqual(set(reaction_order), set(reaction_labels))
        self.assertEqual(set(reaction_order), set(reaction_source_node_ids))
        self.assertEqual(set(reaction_nodes), set(reaction_node_to_id))

    def test_accumulate_incidence_from_edges(self) -> None:
        species_nodes, reaction_nodes = _partition_syncrn_nodes(self.g)
        _, _, _, species_node_to_id = _build_species_index(self.g, species_nodes)

        pre, post = _accumulate_incidence_from_edges(
            self.g,
            reaction_nodes[0],
            species_node_to_id,
        )

        self.assertIsInstance(pre, dict)
        self.assertIsInstance(post, dict)

    def test_build_pre_post_from_graph(self) -> None:
        species_nodes, reaction_nodes = _partition_syncrn_nodes(self.g)
        _, _, _, species_node_to_id = _build_species_index(self.g, species_nodes)
        _, _, _, reaction_node_to_id = _build_reaction_index(self.g, reaction_nodes)

        pre, post = _build_pre_post_from_graph(
            crn=self.g,
            reaction_nodes=reaction_nodes,
            reaction_node_to_id=reaction_node_to_id,
            species_node_to_id=species_node_to_id,
        )

        self.assertEqual(set(pre), set(reaction_node_to_id.values()))
        self.assertEqual(set(post), set(reaction_node_to_id.values()))

    def test_graph_and_object_extraction_have_same_ids(self) -> None:
        inc_obj = _extract_from_syncrn_object(self.syn)
        inc_g = _extract_from_syncrn_digraph(self.g)

        self.assertEqual(set(inc_obj.species_order), set(inc_g.species_order))
        self.assertEqual(set(inc_obj.reaction_order), set(inc_g.reaction_order))

    def test_rule_kind_is_accepted(self) -> None:
        g = nx.DiGraph()
        g.add_node("sA", kind="species", syncrn_id="A", label="A")
        g.add_node("r1", kind="rule", syncrn_id="1", label="A>>B")
        g.add_node("sB", kind="species", syncrn_id="B", label="B")
        g.add_edge("sA", "r1", stoich=1, role="reactant")
        g.add_edge("r1", "sB", stoich=1, role="product")

        inc = _extract_from_syncrn_digraph(g)

        self.assertEqual(inc.species_order, ["A", "B"])
        self.assertEqual(inc.reaction_order, ["1"])
        self.assertEqual(inc.pre["1"], {"A": 1})
        self.assertEqual(inc.post["1"], {"B": 1})


class TestTransition(unittest.TestCase):
    """Unit tests for Transition dataclass behavior."""

    def test_transition_repr(self) -> None:
        t = Transition(
            tid="r1",
            pre={"A": 1},
            post={"B": 1},
            label="A>>B",
        )
        s = repr(t)
        self.assertIn("Transition", s)
        self.assertIn("r1", s)
        self.assertIn("A>>B", s)


class TestPetriNet(unittest.TestCase):
    """Unit tests for PetriNet container and marking semantics."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rxns = [
            "2A>>B+3C",
            "2B>>D",
            "C+D>>E",
        ]
        cls.syn = SynCRN.from_reaction_strings(cls.rxns)
        cls.g = cls.syn.to_digraph()

    def test_init_empty(self) -> None:
        net = PetriNet()

        self.assertEqual(net.places, set())
        self.assertEqual(net.transitions, {})
        self.assertEqual(net.place_order, [])
        self.assertEqual(net.transition_order, [])

    def test_add_place(self) -> None:
        net = PetriNet()
        net.add_place("A", label="Species A", source_node_id="node_A")

        self.assertIn("A", net.places)
        self.assertEqual(net.place_order, ["A"])
        self.assertEqual(net.place_labels["A"], "Species A")
        self.assertEqual(net.place_source_node_ids["A"], "node_A")

    def test_add_transition_creates_places(self) -> None:
        net = PetriNet()
        net.add_transition("r1", pre={"A": 1}, post={"B": 2}, label="A>>2B")

        self.assertIn("A", net.places)
        self.assertIn("B", net.places)
        self.assertIn("r1", net.transitions)
        self.assertEqual(net.transition_order, ["r1"])
        self.assertEqual(net.transition_labels["r1"], "A>>2B")
        self.assertEqual(net.transitions["r1"].pre, {"A": 1})
        self.assertEqual(net.transitions["r1"].post, {"B": 2})

    def test_add_transition_filters_nonpositive_weights(self) -> None:
        net = PetriNet()
        net.add_transition("r1", pre={"A": 1, "B": 0}, post={"C": -1, "D": 2})

        self.assertEqual(net.transitions["r1"].pre, {"A": 1})
        self.assertEqual(net.transitions["r1"].post, {"D": 2})

    def test_enabled_true(self) -> None:
        net = PetriNet()
        net.add_transition("r1", pre={"A": 2, "B": 1}, post={"C": 1})

        self.assertTrue(net.enabled({"A": 2, "B": 1}, "r1"))
        self.assertTrue(net.enabled({"A": 3, "B": 1, "X": 10}, "r1"))

    def test_enabled_false(self) -> None:
        net = PetriNet()
        net.add_transition("r1", pre={"A": 2, "B": 1}, post={"C": 1})

        self.assertFalse(net.enabled({"A": 1, "B": 1}, "r1"))
        self.assertFalse(net.enabled({"A": 2}, "r1"))

    def test_fire(self) -> None:
        net = PetriNet()
        net.add_transition("r1", pre={"A": 2, "B": 1}, post={"C": 3})

        nxt = net.fire({"A": 3, "B": 1}, "r1")

        self.assertEqual(nxt["A"], 1)
        self.assertEqual(nxt["B"], 0)
        self.assertEqual(nxt["C"], 3)

    def test_fire_can_produce_negative_if_not_enabled(self) -> None:
        net = PetriNet()
        net.add_transition("r1", pre={"A": 2}, post={"B": 1})

        nxt = net.fire({"A": 1}, "r1")

        self.assertEqual(nxt["A"], -1)
        self.assertEqual(nxt["B"], 1)

    def test_marking_to_tuple_and_back(self) -> None:
        net = PetriNet()
        net.add_place("A")
        net.add_place("B")
        net.add_place("C")

        tup = net.marking_to_tuple({"A": 2, "C": 1})
        self.assertEqual(tup, (2, 0, 1))

        m = net.tuple_to_marking((2, 0, 1))
        self.assertEqual(m, {"A": 2, "C": 1})

    def test_place_name_and_transition_name(self) -> None:
        net = PetriNet()
        net.add_place("A", label="Species A")
        net.add_transition("r1", pre={"A": 1}, post={"B": 1}, label="A>>B")

        self.assertEqual(net.place_name("A"), "Species A")
        self.assertEqual(net.place_name("B"), "B")
        self.assertEqual(net.transition_name("r1"), "A>>B")
        self.assertEqual(net.transition_name("r2"), "r2")

    def test_to_pre_post(self) -> None:
        net = PetriNet()
        net.add_transition("r1", pre={"A": 2, "B": 1}, post={"C": 1}, label="2A+B>>C")

        data = net.to_pre_post()

        self.assertEqual(data["places"], net.place_order)
        self.assertEqual(data["transitions"], net.transition_order)
        self.assertEqual(data["pre"]["A"]["r1"], 2)
        self.assertEqual(data["pre"]["B"]["r1"], 1)
        self.assertEqual(data["post"]["C"]["r1"], 1)
        self.assertIn("transition_labels", data)

    def test_repr(self) -> None:
        net = PetriNet()
        net.add_transition("r1", pre={"A": 1}, post={"B": 1})
        s = repr(net)

        self.assertIn("PetriNet", s)
        self.assertIn("n_places=2", s)
        self.assertIn("n_transitions=1", s)

    def test_from_syncrn_object(self) -> None:
        net = PetriNet.from_syncrn(self.syn)

        self.assertGreater(len(net.places), 0)
        self.assertEqual(len(net.transitions), len(self.syn.reactions))
        self.assertEqual(net.metadata["source"], "syncrn")

    def test_from_syncrn_digraph(self) -> None:
        net = PetriNet.from_syncrn(self.g)

        self.assertGreater(len(net.places), 0)
        self.assertEqual(len(net.transitions), len(self.syn.reactions))
        self.assertEqual(net.metadata["source"], "syncrn")

    def test_from_syncrn_preserves_orders(self) -> None:
        inc = extract_syncrn_incidence(self.syn)
        net = PetriNet.from_syncrn(self.syn)

        self.assertEqual(net.place_order, inc.species_order)
        self.assertEqual(net.transition_order, inc.reaction_order)

    def test_from_syncrn_transition_content_matches_incidence(self) -> None:
        inc = extract_syncrn_incidence(self.syn)
        net = PetriNet.from_syncrn(self.syn)

        for tid in inc.reaction_order:
            self.assertEqual(net.transitions[tid].pre, inc.pre[tid])
            self.assertEqual(net.transitions[tid].post, inc.post[tid])

    def test_from_syncrn_simple_stoichiometry(self) -> None:
        syn = SynCRN.from_reaction_strings(["2A>>B+3C"])
        net = PetriNet.from_syncrn(syn)

        tid = net.transition_order[0]

        pre_labeled = {
            net.place_name(p): w for p, w in net.transitions[tid].pre.items()
        }
        post_labeled = {
            net.place_name(p): w for p, w in net.transitions[tid].post.items()
        }

        self.assertEqual(pre_labeled, {"A": 2})
        self.assertEqual(post_labeled, {"B": 1, "C": 3})

    def test_from_syncrn_roundtrip_export(self) -> None:
        net = PetriNet.from_syncrn(self.syn)
        data = net.to_pre_post()

        self.assertEqual(data["places"], net.place_order)
        self.assertEqual(data["transitions"], net.transition_order)
        for tid in net.transition_order:
            t = net.transitions[tid]
            for p, w in t.pre.items():
                self.assertEqual(data["pre"][p][tid], w)
            for p, w in t.post.items():
                self.assertEqual(data["post"][p][tid], w)


if __name__ == "__main__":
    unittest.main()
