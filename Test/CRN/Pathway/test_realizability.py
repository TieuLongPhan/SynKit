from __future__ import annotations

import unittest

from synkit.CRN.Pathway.realizability import (
    PathwayRealizability,
    run_realizability_from_syncrn,
    syncrn_to_pr_inputs,
)
from synkit.CRN.Structure import SynCRN


class TestPathwayRealizabilityFromSynCRN(unittest.TestCase):
    """Unit tests for pathway realizability on SynCRN inputs."""

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

    def _build_pr(
        self,
        flow: dict[str, int] | None = None,
        initial_marking: dict[str, int] | None = None,
    ) -> PathwayRealizability:
        """
        Build a realizability instance from the SynCRN testcase.

        :param flow: Requested reaction flow keyed by reaction token.
        :type flow: dict[str, int] | None
        :param initial_marking: Initial marking keyed by species label.
        :type initial_marking: dict[str, int] | None
        :returns: Loaded realizability instance.
        :rtype: PathwayRealizability
        """
        syn = self._build_syn()
        pr = PathwayRealizability()
        pr.load_syncrn_and_flow(
            syn,
            flow={} if flow is None else flow,
            initial_marking={} if initial_marking is None else initial_marking,
            species="label",
            reaction="id",
        )
        return pr

    @staticmethod
    def _find_reaction_token(
        pr: PathwayRealizability,
        tail: dict[str, int],
        head: dict[str, int],
    ) -> str:
        """
        Find the reaction token whose tail/head multisets match exactly.

        :param pr: Loaded realizability instance.
        :type pr: PathwayRealizability
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
            for eid, (t, h) in pr.edges.items()
            if dict(t) == dict(tail) and dict(h) == dict(head)
        ]
        if len(matches) != 1:
            raise AssertionError(
                f"Expected exactly one reaction for {tail} >> {head}, got {matches}"
            )
        return matches[0]

    def _reaction_tokens(self) -> dict[str, str]:
        """
        Resolve a small semantic name -> reaction token map for the testcase.

        :returns: Mapping of convenient names to concrete reaction tokens.
        :rtype: dict[str, str]
        """
        pr = self._build_pr()

        return {
            "r1": self._find_reaction_token(pr, {"A": 2}, {"B": 1, "C": 3}),
            "r2": self._find_reaction_token(pr, {"B": 2}, {"D": 1}),
            "r3": self._find_reaction_token(pr, {"D": 1, "C": 1}, {"E": 1}),
            "r5": self._find_reaction_token(pr, {"E": 1, "C": 2}, {"F": 1}),
            "r8": self._find_reaction_token(pr, {"B": 1, "C": 1}, {"I": 1}),
            "r9": self._find_reaction_token(pr, {"I": 1, "C": 1}, {"J": 1}),
        }

    def test_load_syncrn_and_flow_populates_graph_and_roundtrip_maps(self) -> None:
        """
        Loading from SynCRN should populate species, reactions, and reversible
        token/id lookup maps.
        """
        pr = self._build_pr()

        self.assertEqual(
            pr.vertices,
            {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"},
        )
        self.assertEqual(len(pr.edges), 11)

        for tok in pr.vertices:
            sid = pr.species_token_to_id[tok]
            self.assertIn(sid, pr.species_id_to_token)
            self.assertEqual(pr.species_id_to_token[sid], tok)

        self.assertEqual(len(pr.reaction_token_to_id), 11)
        self.assertEqual(len(pr.reaction_id_to_token), 11)
        for tok, rid in pr.reaction_token_to_id.items():
            self.assertEqual(pr.reaction_id_to_token[rid], tok)

    def test_syncrn_to_pr_inputs_resolves_flow_and_marking(self) -> None:
        """
        The convenience adapter should return tokenized edges plus resolved flow
        and initial marking maps.
        """
        syn = self._build_syn()
        tmp = self._build_pr()
        rxn = self._reaction_tokens()

        vertices, edges, flow_map, marking_map = syncrn_to_pr_inputs(
            syn,
            flow={
                rxn["r1"]: 1,
                rxn["r8"]: 1,
            },
            initial_marking={"A": 2},
            species="label",
            reaction="id",
        )

        self.assertEqual(
            set(vertices),
            {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"},
        )
        self.assertEqual(len(edges), 11)
        self.assertEqual(marking_map, {"A": 2})

        self.assertEqual(flow_map[rxn["r1"]], 1)
        self.assertEqual(flow_map[rxn["r8"]], 1)
        self.assertEqual(sum(flow_map.values()), 2)

        # Sanity: same token set as the loaded instance.
        self.assertEqual(set(edges), set(tmp.edges))

    def test_build_petri_net_from_flow_creates_aux_places_and_exact_goals(self) -> None:
        """
        Building the augmented Petri net should create the expected external
        supply and target places for each active reaction.
        """
        rxn = self._reaction_tokens()
        flow = {
            rxn["r1"]: 1,
            rxn["r8"]: 1,
            rxn["r9"]: 1,
        }
        pr = self._build_pr(flow=flow, initial_marking={"A": 2})
        pr.build_petri_net_from_flow()

        self.assertEqual(pr.initial_marking["A"], 2)
        self.assertEqual(len(pr.goal_atleast), 0)

        for eid in flow:
            ext = f"__ext__{eid}"
            tgt = f"__target__{eid}"

            self.assertIn(ext, pr.petri.place_order)
            self.assertIn(tgt, pr.petri.place_order)
            self.assertEqual(pr.initial_marking[ext], 1)
            self.assertEqual(pr.initial_marking[tgt], 0)
            self.assertEqual(pr.goal_exact[ext], 0)
            self.assertEqual(pr.goal_exact[tgt], 1)

        self.assertEqual(set(pr.petri.transition_order), set(flow))

    def test_is_realizable_simple_chain_returns_expected_certificate(self) -> None:
        """
        The flow

            2A>>B+3C
            B+C>>I
            I+C>>J

        should be realizable from initial marking A:2 with the unique firing
        sequence r1, r8, r9.
        """
        rxn = self._reaction_tokens()
        flow = {
            rxn["r1"]: 1,
            rxn["r8"]: 1,
            rxn["r9"]: 1,
        }
        pr = self._build_pr(flow=flow, initial_marking={"A": 2})
        pr.build_petri_net_from_flow()

        ok, cert = pr.is_realizable()

        self.assertTrue(ok)
        self.assertEqual(cert, [rxn["r1"], rxn["r8"], rxn["r9"]])
        self.assertEqual(pr.certificate, cert)

    def test_is_realizable_repeated_firings_returns_expected_certificate(self) -> None:
        """
        The longer flow

            2*(2A>>B+3C), 2B>>D, D+C>>E, E+2C>>F

        should be realizable from A:4 with the unique certificate

            r1, r1, r2, r3, r5
        """
        rxn = self._reaction_tokens()
        flow = {
            rxn["r1"]: 2,
            rxn["r2"]: 1,
            rxn["r3"]: 1,
            rxn["r5"]: 1,
        }
        pr = self._build_pr(flow=flow, initial_marking={"A": 4})
        pr.build_petri_net_from_flow()

        ok, cert = pr.is_realizable()

        self.assertTrue(ok)
        self.assertEqual(
            cert,
            [rxn["r1"], rxn["r1"], rxn["r2"], rxn["r3"], rxn["r5"]],
        )

    def test_konig_can_be_true_while_bfs_is_false(self) -> None:
        """
        The König-style test is only sufficient, not necessary.

        Flow:
            2A>>B+3C
            2B>>D

        Initial marking:
            A:2

        The dependency graph is acyclic, so the König test returns True.
        However, exact realizability fails because one firing of 2A>>B+3C
        produces only one B, which is insufficient for 2B>>D.
        """
        rxn = self._reaction_tokens()
        flow = {
            rxn["r1"]: 1,
            rxn["r2"]: 1,
        }
        pr = self._build_pr(flow=flow, initial_marking={"A": 2})
        pr.build_petri_net_from_flow()

        self.assertTrue(pr.is_realizable_via_konig())

        ok, cert = pr.is_realizable()
        self.assertFalse(ok)
        self.assertIsNone(cert)
        self.assertIsNone(pr.certificate)

    def test_scaled_realizable_returns_false_for_infeasible_flow(self) -> None:
        """
        Scaling an infeasible flow should still fail here.

        The base flow r1+r2 is already impossible from A:2, and multiplying the
        requested reaction counts does not fix the shortage of B.
        """
        rxn = self._reaction_tokens()
        flow = {
            rxn["r1"]: 1,
            rxn["r2"]: 1,
        }
        pr = self._build_pr(flow=flow, initial_marking={"A": 2})
        pr.build_petri_net_from_flow()

        ok, k = pr.is_scaled_realizable(k_max=3)

        self.assertFalse(ok)
        self.assertIsNone(k)

    def test_borrow_realizable_returns_false_for_infeasible_flow(self) -> None:
        """
        Borrow realizability should also fail for the infeasible flow r1+r2 from
        A:2 on this acyclic testcase.
        """
        rxn = self._reaction_tokens()
        flow = {
            rxn["r1"]: 1,
            rxn["r2"]: 1,
        }
        pr = self._build_pr(flow=flow, initial_marking={"A": 2})
        pr.build_petri_net_from_flow()

        ok, borrow = pr.is_borrow_realizable(max_borrow_each=1)

        self.assertFalse(ok)
        self.assertIsNone(borrow)

    def test_summary_reports_active_flow_initial_marking_and_goals(self) -> None:
        """
        The summary should report the loaded network size, active flow, initial
        marking, and exact auxiliary goals after Petri-net construction.
        """
        rxn = self._reaction_tokens()
        flow = {
            rxn["r1"]: 2,
            rxn["r2"]: 1,
            rxn["r3"]: 1,
            rxn["r5"]: 1,
        }
        pr = self._build_pr(flow=flow, initial_marking={"A": 4})
        pr.build_petri_net_from_flow()

        summary = pr.summary()

        self.assertEqual(summary.n_species, 11)
        self.assertEqual(summary.n_reactions, 11)
        self.assertEqual(summary.active_flow, flow)
        self.assertEqual(summary.initial_marking, {"A": 4})
        self.assertEqual(len(summary.goal_exact), 2 * len(flow))
        self.assertEqual(summary.goal_atleast, {})

    def test_run_realizability_from_syncrn_wrapper(self) -> None:
        """
        The convenience wrapper should run both the König-style sufficient test
        and the exact BFS search, returning a consistent info dictionary.
        """
        syn = self._build_syn()
        rxn = self._reaction_tokens()
        flow = {
            rxn["r1"]: 1,
            rxn["r8"]: 1,
            rxn["r9"]: 1,
        }

        pr, info = run_realizability_from_syncrn(
            syn,
            flow=flow,
            initial_marking={"A": 2},
            species="label",
            reaction="id",
            verbose=False,
        )

        self.assertTrue(info["konig"])
        self.assertTrue(info["bfs"])
        self.assertEqual(info["certificate"], [rxn["r1"], rxn["r8"], rxn["r9"]])

        summary = info["summary"]
        self.assertEqual(summary.n_species, 11)
        self.assertEqual(summary.n_reactions, 11)
        self.assertEqual(summary.active_flow, flow)

        self.assertIsInstance(pr, PathwayRealizability)

    def test_repr(self) -> None:
        """The repr should expose the number of species and reactions."""
        pr = self._build_pr()
        text = repr(pr)
        self.assertIn("PathwayRealizability", text)
        self.assertIn("species=11", text)
        self.assertIn("reactions=11", text)


if __name__ == "__main__":
    unittest.main()
