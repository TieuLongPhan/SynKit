from __future__ import annotations

import unittest

from synkit.CRN.Hypergraph.hypergraph import CRNHyperGraph
from synkit.CRN.Path.realizability import (
    PathwayRealizability,
    hypergraph_to_pr_inputs,
)


class TestPathwayRealizability(unittest.TestCase):
    def _build_pr(
        self,
        rxns: list[str],
        flow: dict[str, int] | None = None,
    ) -> PathwayRealizability:
        hg = CRNHyperGraph()
        hg.parse_rxns(rxns)

        vertices, edges, flow_map = hypergraph_to_pr_inputs(hg, flow=flow)
        pr = PathwayRealizability()
        pr.load_hypergraph_and_flow(vertices, edges, flow_map)
        pr.build_petri_net_from_flow()
        return pr

    def test_linear_pathway_is_realizable_and_returns_certificate(self) -> None:
        """
        Linear acyclic pathway:

            >>A
            A>>B
            B>>

        With unit flow on every reaction, the pathway should be realizable.
        We also expect the König sufficient test to pass for this DAG-like
        structure.
        """
        pr = self._build_pr(
            [
                ">>A",
                "A>>B",
                "B>>",
            ]
        )

        konig_ok = pr.is_realizable_via_konig()
        bfs_ok, cert = pr.is_realizable()

        self.assertTrue(konig_ok, "König test should pass on an acyclic pathway.")
        self.assertTrue(bfs_ok, "Linear pathway should be realizable.")
        self.assertIsNotNone(cert, "A realizable pathway should return a certificate.")
        self.assertEqual(len(cert), 3, "All three reactions should fire exactly once.")
        self.assertEqual(pr.certificate, cert)

    def test_autocatalytic_cycle_is_not_realizable_without_seed(self) -> None:
        """
        Cyclic / seed-dependent pathway:

            A>>2A

        Interpreted as:
        - one firing requires one A token as input,
        - but the Petri encoding starts species places at 0,
        - so without borrow/seed there is no way to start.

        Hence the flow using this reaction once should not be realizable.
        Also the König test should fail because the flow-induced representation
        contains a cycle through species A and the reaction node.
        """
        pr = self._build_pr(["A>>2A"])

        konig_ok = pr.is_realizable_via_konig()
        bfs_ok, cert = pr.is_realizable()

        self.assertFalse(
            konig_ok,
            "König sufficient test should fail on this cyclic self-dependent case.",
        )
        self.assertFalse(
            bfs_ok,
            "Autocatalytic reaction without initial seed should not be realizable.",
        )
        self.assertIsNone(cert)
        self.assertIsNone(pr.certificate)

    def test_borrow_realizable_recovers_seeded_cycle(self) -> None:
        """
        Seeded 2-step cycle:

            A>>B
            B>>A

        With zero initial species tokens, the pathway is not realizable because
        neither reaction can start.

        If we allow borrowing one token of A, we can fire:

            A>>B
            B>>A

        and return to the borrowed state at the end, so the pathway becomes
        borrow-realizable.
        """
        pr = self._build_pr(
            [
                "A>>B",
                "B>>A",
            ]
        )

        # Without borrowing: not realizable
        ok0, cert0 = pr.is_realizable()
        self.assertFalse(ok0)
        self.assertIsNone(cert0)

        # With borrowing: realizable
        ok, borrow = pr.is_borrow_realizable(max_borrow_each=1)

        self.assertTrue(ok, "Borrow-realizability should rescue the seeded cycle.")
        self.assertIsNotNone(borrow)
        self.assertTrue(ok, "Borrow-realizability should rescue the seeded cycle.")
        self.assertIsNotNone(borrow)
        self.assertGreaterEqual(sum(borrow.values()), 1)
        self.assertTrue(
            borrow.get("A", 0) >= 1 or borrow.get("B", 0) >= 1,
            "Expected the borrow vector to seed either A or B.",
        )

    def test_scaled_realizable_returns_factor_one_for_linear_case(self) -> None:
        """
        A realizable pathway should also be scaled-realizable, and the smallest
        working factor for a plainly realizable unit-flow linear chain should be 1.
        """
        pr = self._build_pr(
            [
                ">>A",
                "A>>B",
                "B>>",
            ]
        )

        ok, k = pr.is_scaled_realizable(k_max=3)

        self.assertTrue(ok)
        self.assertEqual(k, 1)


if __name__ == "__main__":
    unittest.main()
