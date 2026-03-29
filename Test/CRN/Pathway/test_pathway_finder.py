from __future__ import annotations

import unittest

from synkit.CRN.Pathway.pathfinder import (
    PathFinderConfig,
    PathwayCandidate,
    PathwayFinder,
    run_pathfinder_from_syncrn,
)
from synkit.CRN.Structure import SynCRN


class TestPathwayFinderFromSynCRN(unittest.TestCase):
    """Unit tests for qualitative pathway search on SynCRN inputs."""

    @staticmethod
    def _build_syn() -> SynCRN:
        """
        Build the standard SynCRN testcase.

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

    def _build_finder(
        self,
        config: PathFinderConfig | None = None,
    ) -> PathwayFinder:
        """
        Build a path finder loaded from the SynCRN testcase.

        :param config: Optional finder configuration.
        :type config: PathFinderConfig | None
        :returns: Loaded path finder.
        :rtype: PathwayFinder
        """
        syn = self._build_syn()
        finder = PathwayFinder(config=config)
        finder.load_syncrn(syn, species="label", reaction="id")
        return finder

    @staticmethod
    def _find_reaction_token(
        finder: PathwayFinder,
        tail: dict[str, int],
        head: dict[str, int],
    ) -> str:
        """
        Find the reaction token whose tail/head multisets match exactly.

        :param finder: Loaded path finder.
        :type finder: PathwayFinder
        :param tail: Reactant multiset.
        :type tail: dict[str, int]
        :param head: Product multiset.
        :type head: dict[str, int]
        :returns: Matching reaction token.
        :rtype: str
        :raises AssertionError: If no unique matching reaction is found.
        """
        matches = [
            eid
            for eid, (t, h) in finder.edges.items()
            if dict(t) == dict(tail) and dict(h) == dict(head)
        ]
        if len(matches) != 1:
            raise AssertionError(
                f"Expected exactly one reaction for {tail} >> {head}, got {matches}"
            )
        return matches[0]

    def _reaction_tokens(self) -> dict[str, str]:
        """
        Resolve semantic shortcut names onto concrete reaction tokens.

        :returns: Mapping from semantic shortcut to reaction token.
        :rtype: dict[str, str]
        """
        finder = self._build_finder()
        return {
            "r1": self._find_reaction_token(finder, {"A": 2}, {"B": 1, "C": 3}),
            "r2": self._find_reaction_token(finder, {"B": 2}, {"D": 1}),
            "r3": self._find_reaction_token(finder, {"D": 1, "C": 1}, {"E": 1}),
            "r4": self._find_reaction_token(finder, {"D": 1, "C": 3}, {"F": 1}),
            "r5": self._find_reaction_token(finder, {"E": 1, "C": 2}, {"F": 1}),
            "r6": self._find_reaction_token(finder, {"B": 3}, {"G": 1}),
            "r7": self._find_reaction_token(finder, {"G": 1, "C": 3}, {"H": 1}),
            "r8": self._find_reaction_token(finder, {"B": 1, "C": 1}, {"I": 1}),
            "r9": self._find_reaction_token(finder, {"I": 1, "C": 1}, {"J": 1}),
            "r10": self._find_reaction_token(finder, {"E": 1, "I": 1}, {"K": 1}),
            "r11": self._find_reaction_token(finder, {"K": 1, "C": 1}, {"H": 1}),
        }

    @staticmethod
    def _flow_signature(flow: dict[str, int]) -> tuple[tuple[str, int], ...]:
        """
        Convert a flow mapping into a stable comparable signature.

        :param flow: Reaction-count flow.
        :type flow: dict[str, int]
        :returns: Sorted tuple representation.
        :rtype: tuple[tuple[str, int], ...]
        """
        return tuple(sorted(flow.items()))

    def test_load_syncrn_populates_graph_and_roundtrip_maps(self) -> None:
        """
        Loading from SynCRN should populate species, reactions, and reversible
        token/id lookup maps.
        """
        finder = self._build_finder()

        self.assertEqual(
            finder.vertices,
            {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"},
        )
        self.assertEqual(len(finder.edges), 11)

        for tok in finder.vertices:
            sid = finder.species_token_to_id[tok]
            self.assertIn(sid, finder.species_id_to_token)
            self.assertEqual(finder.species_id_to_token[sid], tok)

        self.assertEqual(len(finder.reaction_token_to_id), 11)
        self.assertEqual(len(finder.reaction_id_to_token), 11)
        for tok, rid in finder.reaction_token_to_id.items():
            self.assertEqual(finder.reaction_id_to_token[rid], tok)

    def test_find_paths_set_raises_without_loaded_network(self) -> None:
        """
        Searching without a loaded network should raise a runtime error.
        """
        finder = PathwayFinder()
        with self.assertRaises(RuntimeError):
            finder.find_paths_set(source_species={"A"}, target_species={"J"})

    def test_find_paths_set_raises_on_empty_target(self) -> None:
        """
        The target species collection must not be empty.
        """
        finder = self._build_finder()
        with self.assertRaises(ValueError):
            finder.find_paths_set(source_species={"A"}, target_species=set())

    def test_find_paths_set_from_A_to_J_returns_shortest_candidate(self) -> None:
        """
        From source A to target J, the shortest qualitative pathway should be:

            2A>>B+3C
            B+C>>I
            I+C>>J

        Under set semantics this has depth 3.
        """
        finder = self._build_finder()
        rxn = self._reaction_tokens()

        candidates = finder.find_paths_set(
            source_species={"A"},
            target_species={"J"},
            max_depth=6,
            max_paths=1,
            stop_after_first=True,
        )

        self.assertEqual(len(candidates), 1)

        cand = candidates[0]
        self.assertEqual(cand.depth, 3)
        self.assertEqual(cand.reactions, [rxn["r1"], rxn["r8"], rxn["r9"]])
        self.assertEqual(
            cand.flow,
            {
                rxn["r1"]: 1,
                rxn["r8"]: 1,
                rxn["r9"]: 1,
            },
        )
        self.assertIn("J", cand.reached_species)
        self.assertEqual(cand.depth, len(cand.reactions))
        self.assertEqual(cand.realizable, None)
        self.assertEqual(cand.certificate, None)

    def test_find_paths_set_from_A_to_H_returns_shortest_qualitative_candidate(
        self,
    ) -> None:
        """
        From source A to target H, the shortest qualitative pathway should be:

            2A>>B+3C
            3B>>G
            G+3C>>H

        This is valid in qualitative set semantics because only presence, not
        multiplicity, is checked during search.
        """
        finder = self._build_finder()
        rxn = self._reaction_tokens()

        candidates = finder.find_paths_set(
            source_species={"A"},
            target_species={"H"},
            max_depth=6,
            max_paths=1,
            stop_after_first=True,
        )

        self.assertEqual(len(candidates), 1)

        cand = candidates[0]
        self.assertEqual(cand.depth, 3)
        self.assertEqual(cand.reactions, [rxn["r1"], rxn["r6"], rxn["r7"]])
        self.assertEqual(
            cand.flow,
            {
                rxn["r1"]: 1,
                rxn["r6"]: 1,
                rxn["r7"]: 1,
            },
        )
        self.assertIn("H", cand.reached_species)

    def test_find_paths_set_from_A_to_F_contains_two_distinct_flows(self) -> None:
        """
        Target F should be reachable through at least two distinct qualitative
        flows:

            path 1: r1, r2, r4
            path 2: r1, r2, r3, r5

        This test checks for the presence of both aggregate flows rather than
        relying on an exact total candidate count.
        """
        finder = self._build_finder()
        rxn = self._reaction_tokens()

        candidates = finder.find_paths_set(
            source_species={"A"},
            target_species={"F"},
            max_depth=6,
            max_paths=20,
            stop_after_first=False,
            deduplicate_by_flow=True,
        )

        self.assertGreaterEqual(len(candidates), 2)

        got_flows = {self._flow_signature(c.flow) for c in candidates}
        expected_1 = self._flow_signature(
            {
                rxn["r1"]: 1,
                rxn["r2"]: 1,
                rxn["r4"]: 1,
            }
        )
        expected_2 = self._flow_signature(
            {
                rxn["r1"]: 1,
                rxn["r2"]: 1,
                rxn["r3"]: 1,
                rxn["r5"]: 1,
            }
        )

        self.assertIn(expected_1, got_flows)
        self.assertIn(expected_2, got_flows)

        for cand in candidates:
            self.assertIsInstance(cand, PathwayCandidate)
            self.assertEqual(cand.depth, len(cand.reactions))
            self.assertIn("F", cand.reached_species)

    def test_validate_candidates_marks_J_candidate_realizable_from_A2(self) -> None:
        """
        The qualitative candidate for J should be exactly realizable from
        initial marking A:2.
        """
        finder = self._build_finder()
        rxn = self._reaction_tokens()
        syn = self._build_syn()

        candidates = finder.find_paths_set(
            source_species={"A"},
            target_species={"J"},
            max_depth=6,
            max_paths=1,
            stop_after_first=True,
        )
        validated = finder.validate_candidates(
            syn,
            candidates,
            initial_marking={"A": 2},
            species="label",
            reaction="id",
        )

        self.assertEqual(len(validated), 1)
        cand = validated[0]

        self.assertTrue(cand.realizable)
        self.assertEqual(cand.certificate, [rxn["r1"], rxn["r8"], rxn["r9"]])

    def test_validate_candidates_marks_H_candidate_unrealizable_from_A2(self) -> None:
        """
        The shortest qualitative candidate for H is not exactly realizable from
        A:2 because the step 3B>>G requires three copies of B, while one firing
        of 2A>>B+3C produces only one B.
        """
        finder = self._build_finder()
        syn = self._build_syn()

        candidates = finder.find_paths_set(
            source_species={"A"},
            target_species={"H"},
            max_depth=6,
            max_paths=1,
            stop_after_first=True,
        )
        validated = finder.validate_candidates(
            syn,
            candidates,
            initial_marking={"A": 2},
            species="label",
            reaction="id",
        )

        self.assertEqual(len(validated), 1)
        cand = validated[0]

        self.assertFalse(cand.realizable)
        self.assertIsNone(cand.certificate)

    def test_run_pathfinder_from_syncrn_wrapper_with_validation(self) -> None:
        """
        The convenience wrapper should return validated candidates when
        ``validate=True``.
        """
        syn = self._build_syn()
        rxn = self._reaction_tokens()

        candidates = run_pathfinder_from_syncrn(
            syn,
            source_species={"A"},
            target_species={"J"},
            initial_marking={"A": 2},
            species="label",
            reaction="id",
            max_depth=6,
            max_paths=1,
            validate=True,
            verbose=False,
        )

        self.assertEqual(len(candidates), 1)
        cand = candidates[0]

        self.assertEqual(cand.reactions, [rxn["r1"], rxn["r8"], rxn["r9"]])
        self.assertTrue(cand.realizable)
        self.assertEqual(cand.certificate, [rxn["r1"], rxn["r8"], rxn["r9"]])

    def test_run_pathfinder_from_syncrn_wrapper_requires_initial_marking_when_validate_true(
        self,
    ) -> None:
        """
        Validation requires an exact initial marking.
        """
        syn = self._build_syn()

        with self.assertRaises(ValueError):
            run_pathfinder_from_syncrn(
                syn,
                source_species={"A"},
                target_species={"J"},
                species="label",
                reaction="id",
                max_depth=6,
                max_paths=1,
                validate=True,
                verbose=False,
            )


if __name__ == "__main__":
    unittest.main()
