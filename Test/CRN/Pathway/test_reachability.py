from __future__ import annotations

import unittest

from synkit.CRN.Pathway.reachability import PathwayReachability
from synkit.CRN.Structure import SynCRN


class TestPathwayReachabilityFromSynCRN(unittest.TestCase):
    """Unit tests for reachability computed from a SynCRN instance."""

    @staticmethod
    def _build_syn() -> SynCRN:
        """
        Build the standard SynCRN test network.

        :returns: Parsed SynCRN instance.
        :rtype: SynCRN
        """
        rxns = [
            "2A>>B+3C",
            "2B>>D",
            "D+C>>E",
            "D+3C>>F",
            "E+2C>>F",
            "3B>>G",
            "G+3C>>H",
            "B+C>>I",
            "I+C>>J",
            "E+I>>K",
            "K+C>>H",
        ]
        return SynCRN.from_reaction_strings(rxns)

    def _build_rr(self) -> PathwayReachability:
        """
        Build a reachability engine from the SynCRN test network.

        :returns: Loaded reachability engine.
        :rtype: PathwayReachability
        """
        syn = self._build_syn()
        rr = PathwayReachability()
        rr.load_syncrn(syn, species="label", reaction="id")
        return rr

    def test_load_syncrn_populates_graph_and_maps(self) -> None:
        """
        Loading from SynCRN should populate species, reactions, and reversible
        token/id lookup maps.

        Species token -> id is not assumed to be identity. Instead we test
        round-trip consistency:
            token -> id -> token
        """
        rr = self._build_rr()

        self.assertEqual(
            rr.vertices,
            {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"},
        )
        self.assertEqual(len(rr.edges), 11)

        for tok in rr.vertices:
            sid = rr.species_token_to_id[tok]
            self.assertIn(sid, rr.species_id_to_token)
            self.assertEqual(rr.species_id_to_token[sid], tok)

        self.assertEqual(len(rr.reaction_token_to_id), 11)
        self.assertEqual(len(rr.reaction_id_to_token), 11)
        for tok, rid in rr.reaction_token_to_id.items():
            self.assertEqual(rr.reaction_id_to_token[rid], tok)

    def test_set_reachability_from_A_reaches_expected_depths(self) -> None:
        """
        Under set semantics, starting from A reaches the whole network.

        Important implementation detail:
        all newly enabled reactions in a layer contribute their products in the
        same batch. Therefore F and H appear earlier than in a purely
        step-by-step hand simulation.

        Expected layers:
            depth 1: B, C
            depth 2: D, G, I
            depth 3: E, F, H, J
            depth 4: K
            depth 5: no new species, but K+C>>H becomes newly enabled
        """
        rr = self._build_rr()
        result = rr.compute_layers_set(initial_species={"A"})

        self.assertEqual(result.initial_species, ["A"])

        self.assertEqual(result.species_first_depth["A"], 0)
        self.assertEqual(result.species_first_depth["B"], 1)
        self.assertEqual(result.species_first_depth["C"], 1)
        self.assertEqual(result.species_first_depth["D"], 2)
        self.assertEqual(result.species_first_depth["G"], 2)
        self.assertEqual(result.species_first_depth["I"], 2)
        self.assertEqual(result.species_first_depth["E"], 3)
        self.assertEqual(result.species_first_depth["F"], 3)
        self.assertEqual(result.species_first_depth["H"], 3)
        self.assertEqual(result.species_first_depth["J"], 3)
        self.assertEqual(result.species_first_depth["K"], 4)

        self.assertEqual(
            set(result.species_first_depth),
            {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"},
        )

        self.assertEqual(len(result.layers), 5)
        self.assertEqual(result.layers[0].newly_reachable_species, ["B", "C"])
        self.assertEqual(result.layers[1].newly_reachable_species, ["D", "G", "I"])
        self.assertEqual(result.layers[2].newly_reachable_species, ["E", "F", "H", "J"])
        self.assertEqual(result.layers[3].newly_reachable_species, ["K"])
        self.assertEqual(result.layers[4].newly_reachable_species, [])

        self.assertEqual(len(result.reaction_first_depth), 11)

    def test_set_reachability_from_BC_reaches_expected_downstream_species(self) -> None:
        """
        Starting from B and C under set semantics reaches the downstream part
        of the network, but never A.

        Expected layers:
            depth 1: D, G, I
            depth 2: E, F, H, J
            depth 3: K
            depth 4: no new species, but K+C>>H becomes newly enabled
        """
        rr = self._build_rr()
        result = rr.compute_layers_set(initial_species={"B", "C"})

        self.assertEqual(result.species_first_depth["B"], 0)
        self.assertEqual(result.species_first_depth["C"], 0)

        self.assertNotIn("A", result.species_first_depth)

        self.assertEqual(result.species_first_depth["D"], 1)
        self.assertEqual(result.species_first_depth["G"], 1)
        self.assertEqual(result.species_first_depth["I"], 1)
        self.assertEqual(result.species_first_depth["E"], 2)
        self.assertEqual(result.species_first_depth["F"], 2)
        self.assertEqual(result.species_first_depth["H"], 2)
        self.assertEqual(result.species_first_depth["J"], 2)
        self.assertEqual(result.species_first_depth["K"], 3)

        self.assertEqual(
            set(result.species_first_depth),
            {"B", "C", "D", "E", "F", "G", "H", "I", "J", "K"},
        )

        self.assertEqual(len(result.layers), 4)
        self.assertEqual(result.layers[0].newly_reachable_species, ["D", "G", "I"])
        self.assertEqual(result.layers[1].newly_reachable_species, ["E", "F", "H", "J"])
        self.assertEqual(result.layers[2].newly_reachable_species, ["K"])
        self.assertEqual(result.layers[3].newly_reachable_species, [])

    def test_set_reachability_from_A_reaches_H_at_depth_3(self) -> None:
        """
        Starting from A alone reaches C at depth 1 and H at depth 3 under the
        current batched set-semantics implementation.
        """
        rr = self._build_rr()
        result = rr.compute_layers_set(initial_species={"A"})

        self.assertIn("C", result.species_first_depth)
        self.assertIn("H", result.species_first_depth)
        self.assertEqual(result.species_first_depth["C"], 1)
        self.assertEqual(result.species_first_depth["H"], 3)

    def test_multiset_reachability_from_two_A_progresses_to_J(self) -> None:
        """
        Under multiset semantics, starting from A:2 gives:

            depth 1: 2A>>B+3C   -> B, C
            depth 2: B+C>>I     -> I
            depth 3: I+C>>J     -> J

        No other species are reached.
        """
        rr = self._build_rr()
        result = rr.compute_layers_multiset(initial_marking={"A": 2})

        self.assertEqual(result.initial_species, ["A"])

        self.assertEqual(result.species_first_depth["A"], 0)
        self.assertEqual(result.species_first_depth["B"], 1)
        self.assertEqual(result.species_first_depth["C"], 1)
        self.assertEqual(result.species_first_depth["I"], 2)
        self.assertEqual(result.species_first_depth["J"], 3)

        for s in ["D", "E", "F", "G", "H", "K"]:
            self.assertNotIn(s, result.species_first_depth)

        self.assertEqual(len(result.layers), 3)
        self.assertEqual(result.layers[0].newly_reachable_species, ["B", "C"])
        self.assertEqual(result.layers[1].newly_reachable_species, ["I"])
        self.assertEqual(result.layers[2].newly_reachable_species, ["J"])

    def test_multiset_reachability_from_insufficient_A_is_blocked(self) -> None:
        """
        Under multiset semantics, A:1 is insufficient to enable 2A>>B+3C,
        so no progress is possible.
        """
        rr = self._build_rr()
        result = rr.compute_layers_multiset(initial_marking={"A": 1})

        self.assertEqual(result.initial_species, ["A"])
        self.assertEqual(result.species_first_depth["A"], 0)

        for s in ["B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]:
            self.assertNotIn(s, result.species_first_depth)

        self.assertEqual(len(result.layers), 0)
        self.assertEqual(result.reaction_first_depth, {})

    def test_multiset_reachability_from_six_A_repeats_first_reaction_but_still_only_reaches_IJ(
        self,
    ) -> None:
        """
        With six copies of A, the first reaction can fire repeatedly across
        layers. However, because the traversal fires each currently enabled
        reaction once per layer, the B+C>>I branch keeps consuming B and the
        network still only reaches I and J, not D or G.

        This is a regression test for the current batch-firing semantics.
        """
        rr = self._build_rr()
        result = rr.compute_layers_multiset(initial_marking={"A": 6})

        self.assertEqual(result.species_first_depth["A"], 0)
        self.assertEqual(result.species_first_depth["B"], 1)
        self.assertEqual(result.species_first_depth["C"], 1)
        self.assertEqual(result.species_first_depth["I"], 2)
        self.assertEqual(result.species_first_depth["J"], 3)

        self.assertNotIn("D", result.species_first_depth)
        self.assertNotIn("E", result.species_first_depth)
        self.assertNotIn("F", result.species_first_depth)
        self.assertNotIn("G", result.species_first_depth)
        self.assertNotIn("H", result.species_first_depth)
        self.assertNotIn("K", result.species_first_depth)

        self.assertEqual(len(result.layers), 4)
        self.assertEqual(result.layers[0].newly_reachable_species, ["B", "C"])
        self.assertEqual(result.layers[1].newly_reachable_species, ["I"])
        self.assertEqual(result.layers[2].newly_reachable_species, ["J"])
        self.assertEqual(result.layers[3].newly_reachable_species, [])

    def test_repr(self) -> None:
        """The repr should expose the number of vertices and edges."""
        rr = self._build_rr()
        text = repr(rr)
        self.assertIn("PathwayReachability", text)
        self.assertIn("vertices=11", text)
        self.assertIn("edges=11", text)


if __name__ == "__main__":
    unittest.main()
