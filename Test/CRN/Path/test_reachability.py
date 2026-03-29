from __future__ import annotations

import unittest

from synkit.CRN.Hypergraph.hypergraph import CRNHyperGraph
from synkit.CRN.Path.reachability import (
    PathwayReachability,
    hypergraph_to_reachability_inputs,
)


class TestPathwayReachability(unittest.TestCase):
    def _build_rr(self, rxns: list[str]) -> PathwayReachability:
        hg = CRNHyperGraph()
        hg.parse_rxns(rxns)

        vertices, edges = hypergraph_to_reachability_inputs(hg)
        rr = PathwayReachability()
        rr.load_hypergraph(vertices, edges)
        return rr

    def test_layered_set_reachability_cascade(self) -> None:
        """
        Network:

            A+B>>C
            C+D>>E
            E+F>>G+D

        Initial species:
            A, B, D, F

        Expected qualitative layers:
            depth 0: A, B, D, F
            depth 1: C
            depth 2: E
            depth 3: G

        D is re-produced in the third step but is not new at that point.
        """
        rr = self._build_rr(
            [
                "A+B>>C",
                "C+D>>E",
                "E+F>>G+D",
            ]
        )

        result = rr.compute_layers_set(initial_species={"A", "B", "D", "F"})

        self.assertEqual(result.species_first_depth["A"], 0)
        self.assertEqual(result.species_first_depth["B"], 0)
        self.assertEqual(result.species_first_depth["D"], 0)
        self.assertEqual(result.species_first_depth["F"], 0)

        self.assertEqual(result.species_first_depth["C"], 1)
        self.assertEqual(result.species_first_depth["E"], 2)
        self.assertEqual(result.species_first_depth["G"], 3)

        self.assertEqual(len(result.layers), 3)
        self.assertEqual(result.layers[0].newly_reachable_species, ["C"])
        self.assertEqual(result.layers[1].newly_reachable_species, ["E"])
        self.assertEqual(result.layers[2].newly_reachable_species, ["G"])

        # One newly enabled reaction per layer in this simple cascade.
        self.assertEqual(len(result.layers[0].newly_enabled_reactions), 1)
        self.assertEqual(len(result.layers[1].newly_enabled_reactions), 1)
        self.assertEqual(len(result.layers[2].newly_enabled_reactions), 1)

    def test_blocked_cascade_stops_when_required_species_missing(self) -> None:
        """
        Same network as above, but omit D from the initial species:

            A+B>>C
            C+D>>E
            E+F>>G+D

        With initial species A, B, F only:
        - first layer can still produce C,
        - second layer cannot proceed because D is missing,
        - therefore E and G are never reached.
        """
        rr = self._build_rr(
            [
                "A+B>>C",
                "C+D>>E",
                "E+F>>G+D",
            ]
        )

        result = rr.compute_layers_set(initial_species={"A", "B", "F"})

        self.assertEqual(result.species_first_depth["A"], 0)
        self.assertEqual(result.species_first_depth["B"], 0)
        self.assertEqual(result.species_first_depth["F"], 0)
        self.assertEqual(result.species_first_depth["C"], 1)

        self.assertNotIn("E", result.species_first_depth)
        self.assertNotIn("G", result.species_first_depth)

        self.assertEqual(len(result.layers), 1)
        self.assertEqual(result.layers[0].newly_reachable_species, ["C"])
        self.assertEqual(
            set(result.layers[0].all_reachable_species),
            {"A", "B", "C", "F"},
        )

    def test_multiset_reachability_respects_stoichiometric_availability(self) -> None:
        """
        Multiset semantics should respect counts.

        Network:
            2A>>B
            B>>C

        Initial marking:
            A: 2

        Then:
        - layer 1 can fire 2A>>B, reaching B,
        - layer 2 can fire B>>C, reaching C.
        """
        rr = self._build_rr(
            [
                "2A>>B",
                "B>>C",
            ]
        )

        result = rr.compute_layers_multiset(initial_marking={"A": 2})

        self.assertEqual(result.species_first_depth["A"], 0)
        self.assertEqual(result.species_first_depth["B"], 1)
        self.assertEqual(result.species_first_depth["C"], 2)

        self.assertEqual(len(result.layers), 2)
        self.assertEqual(result.layers[0].newly_reachable_species, ["B"])
        self.assertEqual(result.layers[1].newly_reachable_species, ["C"])

    def test_multiset_reachability_blocks_when_counts_are_insufficient(self) -> None:
        """
        Same network:
            2A>>B
            B>>C

        But with only one A initially, 2A>>B is not enabled at all.
        Therefore no new species should be reached.
        """
        rr = self._build_rr(
            [
                "2A>>B",
                "B>>C",
            ]
        )

        result = rr.compute_layers_multiset(initial_marking={"A": 1})

        self.assertEqual(result.species_first_depth["A"], 0)
        self.assertNotIn("B", result.species_first_depth)
        self.assertNotIn("C", result.species_first_depth)
        self.assertEqual(len(result.layers), 0)


if __name__ == "__main__":
    unittest.main()
