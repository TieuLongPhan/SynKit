from __future__ import annotations

import unittest

from synkit.CRN.Structure import SynCRN
from synkit.CRN.Petrinet.net import PetriNet
from synkit.CRN.Petrinet.structure import (
    _as_petri,
    _is_siphon,
    _is_trap,
    _minimal_sets,
    _render_place_sets,
    find_siphons,
    find_traps,
    species_transition_neighborhoods,
)


class TestSiphonTrapHelpers(unittest.TestCase):
    """Unit tests for low-level siphon/trap helper functions."""

    def test_as_petri_returns_same_object_for_petrinet(self) -> None:
        net = PetriNet()
        self.assertIs(_as_petri(net), net)

    def test_as_petri_converts_syncrn(self) -> None:
        syn = SynCRN.from_reaction_strings(["A>>B"])
        net = _as_petri(syn)

        self.assertIsInstance(net, PetriNet)
        self.assertEqual(len(net.transitions), 1)

    def test_minimal_sets(self) -> None:
        candidates = [
            {"A", "B"},
            {"A"},
            {"B", "C"},
            {"A", "B", "C"},
            {"C"},
        ]
        out = _minimal_sets(candidates)

        self.assertEqual(
            {frozenset(x) for x in out}, {frozenset({"A"}), frozenset({"C"})}
        )

    def test_render_place_sets_id(self) -> None:
        net = PetriNet()
        net.add_place("1", label="A")
        net.add_place("2", label="B")

        out = _render_place_sets(net, [{"1", "2"}], names="id")
        self.assertEqual(out, [{"1", "2"}])

    def test_render_place_sets_label(self) -> None:
        net = PetriNet()
        net.add_place("1", label="A")
        net.add_place("2", label="B")

        out = _render_place_sets(net, [{"1", "2"}], names="label")
        self.assertEqual(out, [{"A", "B"}])

    def test_render_place_sets_invalid(self) -> None:
        net = PetriNet()
        net.add_place("1", label="A")

        with self.assertRaises(ValueError):
            _render_place_sets(net, [{"1"}], names="bad")


class TestSiphonsAndTrapsOnHandBuiltPetriNet(unittest.TestCase):
    """Unit tests using small hand-built Petri nets with known structure."""

    def test_is_siphon_false_for_empty(self) -> None:
        net = PetriNet()
        net.add_place("A")
        self.assertFalse(_is_siphon(net, set()))

    def test_is_trap_false_for_empty(self) -> None:
        net = PetriNet()
        net.add_place("A")
        self.assertFalse(_is_trap(net, set()))

    def test_single_place_sink_is_siphon(self) -> None:
        """
        Net: A -> B
        {B} is a siphon because any transition producing into B consumes from B? No.
        Actually for A->B, {B} is NOT a siphon.
        But {A} is a siphon because no transition produces into A.
        """
        net = PetriNet()
        net.add_transition("t1", pre={"A": 1}, post={"B": 1})

        self.assertTrue(_is_siphon(net, {"A"}))
        self.assertFalse(_is_siphon(net, {"B"}))

    def test_single_place_source_is_not_trap(self) -> None:
        """
        Net: A -> B
        {A} is not a trap because t1 consumes from A but does not produce back to A.
        {B} is a trap because no transition consumes from B.
        """
        net = PetriNet()
        net.add_transition("t1", pre={"A": 1}, post={"B": 1})

        self.assertFalse(_is_trap(net, {"A"}))
        self.assertTrue(_is_trap(net, {"B"}))

    def test_cycle_set_is_both_siphon_and_trap(self) -> None:
        """
        Net: A -> B, B -> A
        {A,B} is both a siphon and a trap.
        """
        net = PetriNet()
        net.add_transition("t1", pre={"A": 1}, post={"B": 1})
        net.add_transition("t2", pre={"B": 1}, post={"A": 1})

        self.assertTrue(_is_siphon(net, {"A", "B"}))
        self.assertTrue(_is_trap(net, {"A", "B"}))

    def test_find_siphons_simple_chain(self) -> None:
        """
        Net: A -> B
        Minimal siphons: {A}
        """
        net = PetriNet()
        net.add_transition("t1", pre={"A": 1}, post={"B": 1})

        siphons_id = find_siphons(net, names="id")
        self.assertEqual({frozenset(x) for x in siphons_id}, {frozenset({"A"})})

    def test_find_traps_simple_chain(self) -> None:
        """
        Net: A -> B
        Minimal traps: {B}
        """
        net = PetriNet()
        net.add_transition("t1", pre={"A": 1}, post={"B": 1})

        traps_id = find_traps(net, names="id")
        self.assertEqual({frozenset(x) for x in traps_id}, {frozenset({"B"})})

    def test_find_siphons_cycle(self) -> None:
        """
        Net: A <-> B
        Minimal siphon: {A,B}
        """
        net = PetriNet()
        net.add_transition("t1", pre={"A": 1}, post={"B": 1})
        net.add_transition("t2", pre={"B": 1}, post={"A": 1})

        siphons_id = find_siphons(net, names="id")
        self.assertEqual({frozenset(x) for x in siphons_id}, {frozenset({"A", "B"})})

    def test_find_traps_cycle(self) -> None:
        """
        Net: A <-> B
        Minimal trap: {A,B}
        """
        net = PetriNet()
        net.add_transition("t1", pre={"A": 1}, post={"B": 1})
        net.add_transition("t2", pre={"B": 1}, post={"A": 1})

        traps_id = find_traps(net, names="id")
        self.assertEqual({frozenset(x) for x in traps_id}, {frozenset({"A", "B"})})

    def test_max_size_can_prevent_larger_cycle_solution(self) -> None:
        """
        Net: A <-> B
        With max_size=1, the minimal {A,B} solution cannot be found.
        """
        net = PetriNet()
        net.add_transition("t1", pre={"A": 1}, post={"B": 1})
        net.add_transition("t2", pre={"B": 1}, post={"A": 1})

        self.assertEqual(find_siphons(net, max_size=1, names="id"), [])
        self.assertEqual(find_traps(net, max_size=1, names="id"), [])

    def test_label_rendering(self) -> None:
        net = PetriNet()
        net.add_place("1", label="A")
        net.add_place("2", label="B")
        net.add_transition("t1", pre={"1": 1}, post={"2": 1})

        siphons_label = find_siphons(net, names="label")
        traps_label = find_traps(net, names="label")

        self.assertEqual({frozenset(x) for x in siphons_label}, {frozenset({"A"})})
        self.assertEqual({frozenset(x) for x in traps_label}, {frozenset({"B"})})


class TestSpeciesTransitionNeighborhoods(unittest.TestCase):
    """Unit tests for producer/consumer neighborhood summaries."""

    def test_species_transition_neighborhoods_handbuilt(self) -> None:
        net = PetriNet()
        net.add_place("1", label="A")
        net.add_place("2", label="B")
        net.add_transition("t1", pre={"1": 1}, post={"2": 1}, label="A>>B")
        net.add_transition("t2", pre={"2": 1}, post={"1": 1}, label="B>>A")

        out = species_transition_neighborhoods(net)

        self.assertEqual(set(out), {"1", "2"})
        self.assertEqual(out["1"]["label"], "A")
        self.assertEqual(out["2"]["label"], "B")
        self.assertEqual(out["1"]["producer_transitions"], ["t2"])
        self.assertEqual(out["1"]["consumer_transitions"], ["t1"])
        self.assertEqual(out["2"]["producer_transitions"], ["t1"])
        self.assertEqual(out["2"]["consumer_transitions"], ["t2"])

    def test_species_transition_neighborhoods_syncrn_smoke(self) -> None:
        syn = SynCRN.from_reaction_strings(
            [
                "A>>B",
                "B>>C",
            ]
        )
        out = species_transition_neighborhoods(syn)

        self.assertTrue(out)
        for pid, info in out.items():
            self.assertIn("label", info)
            self.assertIn("producer_transitions", info)
            self.assertIn("consumer_transitions", info)
            self.assertIsInstance(info["producer_transitions"], list)
            self.assertIsInstance(info["consumer_transitions"], list)


class TestSiphonsAndTrapsOnSynCRN(unittest.TestCase):
    """Integration tests on SynCRN input."""

    def test_find_siphons_and_traps_syncrn_smoke(self) -> None:
        syn = SynCRN.from_reaction_strings(["A>>B"])
        siphons = find_siphons(syn, names="label")
        traps = find_traps(syn, names="label")

        self.assertEqual({frozenset(x) for x in siphons}, {frozenset({"A"})})
        self.assertEqual({frozenset(x) for x in traps}, {frozenset({"B"})})

    def test_find_siphons_and_traps_accept_petrinet(self) -> None:
        net = PetriNet()
        net.add_transition("t1", pre={"A": 1}, post={"B": 1})

        siphons = find_siphons(net, names="id")
        traps = find_traps(net, names="id")

        self.assertEqual({frozenset(x) for x in siphons}, {frozenset({"A"})})
        self.assertEqual({frozenset(x) for x in traps}, {frozenset({"B"})})


if __name__ == "__main__":
    unittest.main()
