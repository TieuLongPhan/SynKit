from __future__ import annotations

import itertools
import random
import unittest

from synkit.CRN.Petrinet import PetriNet
from synkit.CRN.Petrinet.siphon_search import (
    SiphonSearchLimit,
    max_siphon_within,
    max_trap_within,
    minimal_siphons,
    minimal_traps,
)
from synkit.CRN.Petrinet.structure import (
    _is_siphon,
    _is_trap,
    _minimal_sets,
    find_siphons,
    find_traps,
)
from synkit.CRN.Structure import SynCRN


def _net(rxns: list[str]) -> PetriNet:
    return PetriNet.from_syncrn(SynCRN.from_reaction_strings(rxns))


def _brute_force(net: PetriNet, predicate) -> list[list[str]]:
    """Reference implementation: test every subset, then keep minimal ones."""
    places = net.place_order
    candidates = []
    for k in range(1, len(places) + 1):
        for combo in itertools.combinations(places, k):
            if predicate(net, set(combo)):
                candidates.append(set(combo))
    return sorted(sorted(s) for s in _minimal_sets(candidates))


class TestClosureOperator(unittest.TestCase):
    """The maximal-siphon / maximal-trap closure."""

    def test_max_siphon_of_reversible_pair(self) -> None:
        """A <-> B is entirely a siphon."""
        net = _net(["A>>B", "B>>A"])

        self.assertEqual(
            max_siphon_within(net, set(net.place_order)),
            set(net.place_order),
        )

    def test_max_siphon_of_open_network(self) -> None:
        """
        For A >> B the whole set is still a siphon.

        The only transition produces into B but also consumes from A, so the
        condition holds on ``{A, B}``. Minimality is a separate question: the
        minimal siphon here is ``{A}``, since nothing produces into A.
        """
        net = _net(["A>>B"])

        self.assertEqual(
            max_siphon_within(net, set(net.place_order)),
            set(net.place_order),
        )
        self.assertEqual([len(s) for s in minimal_siphons(net)], [1])

    def test_max_siphon_drops_unproduced_place(self) -> None:
        """Restricting to {B} alone leaves nothing: B's producer is uncovered."""
        net = _net(["A>>B"])
        b = [p for p in net.place_order if net.place_name(p) == "B"][0]

        self.assertEqual(max_siphon_within(net, {b}), set())

    def test_closure_result_is_always_valid(self) -> None:
        """Whatever survives the closure must satisfy the defining condition."""
        net = _net(["A+B>>C", "C>>A", "B>>C"])

        siphon = max_siphon_within(net, set(net.place_order))
        trap = max_trap_within(net, set(net.place_order))

        if siphon:
            self.assertTrue(_is_siphon(net, siphon))
        if trap:
            self.assertTrue(_is_trap(net, trap))

    def test_empty_allowed_set(self) -> None:
        """Closing the empty set yields the empty set, not an error."""
        net = _net(["A>>B"])

        self.assertEqual(max_siphon_within(net, set()), set())


class TestMinimalSiphonsAgainstBruteForce(unittest.TestCase):
    """Branch-and-bound must agree exactly with exhaustive enumeration."""

    HANDWRITTEN = [
        ["A>>B"],
        ["A>>B", "B>>A"],
        ["A>>B", "B>>C", "C>>A"],
        ["A>>B", "C>>D"],
        ["A+B>>C", "C>>A+B"],
        ["2A>>B", "B>>2A"],
        ["A>>B", "B>>A", "B>>C", "C>>B"],
        ["A>>B+C", "B+C>>A"],
        ["A+B>>C+D", "C>>A", "D>>B"],
        ["A>>B", "B>>C", "C>>D", "D>>A", "A>>E"],
    ]

    def test_handwritten_networks(self) -> None:
        """Structured cases covering cycles, splits, joins and stoichiometry."""
        for rxns in self.HANDWRITTEN:
            with self.subTest(rxns=rxns):
                net = _net(rxns)
                self.assertEqual(
                    sorted(sorted(s) for s in minimal_siphons(net)),
                    _brute_force(net, _is_siphon),
                )
                self.assertEqual(
                    sorted(sorted(s) for s in minimal_traps(net)),
                    _brute_force(net, _is_trap),
                )

    def test_random_networks(self) -> None:
        """Randomized differential test against the reference implementation."""
        rng = random.Random(20260815)

        for trial in range(30):
            n = rng.randint(3, 6)
            species = [chr(ord("A") + i) for i in range(n)]
            rxns = []
            for _ in range(rng.randint(2, 5)):
                lhs = rng.sample(species, rng.randint(1, 2))
                rhs = rng.sample(species, rng.randint(1, 2))
                rxns.append("+".join(lhs) + ">>" + "+".join(rhs))

            with self.subTest(trial=trial, rxns=rxns):
                net = _net(rxns)
                self.assertEqual(
                    sorted(sorted(s) for s in minimal_siphons(net)),
                    _brute_force(net, _is_siphon),
                )
                self.assertEqual(
                    sorted(sorted(s) for s in minimal_traps(net)),
                    _brute_force(net, _is_trap),
                )


class TestSearchBehaviour(unittest.TestCase):
    """Size bounds, budgets and result validity."""

    def test_results_are_siphons_and_minimal(self) -> None:
        """Every result satisfies the definition and contains no other result."""
        net = _net(["A>>B", "B>>A", "C>>D", "D>>C", "B>>C"])
        found = minimal_siphons(net)

        for s in found:
            self.assertTrue(_is_siphon(net, s))
        for a in found:
            for b in found:
                if a is not b:
                    self.assertFalse(b < a)

    def test_max_size_filters_results(self) -> None:
        """A size bound removes larger siphons but keeps smaller ones."""
        net = _net(["A>>B", "B>>A", "B>>C", "C>>B"])
        unbounded = minimal_siphons(net)
        bounded = minimal_siphons(net, max_size=1)

        self.assertTrue(all(len(s) <= 1 for s in bounded))
        self.assertTrue(len(bounded) <= len(unbounded))

    def test_scales_past_brute_force_range(self) -> None:
        """
        A 60-species chain is far beyond 2**n enumeration but trivial here.
        """
        net = _net([f"S{i}>>S{i+1}" for i in range(59)])
        found = minimal_siphons(net)

        self.assertEqual(len(found), 1)
        self.assertTrue(_is_siphon(net, found[0]))

    def test_strict_limit_raises(self) -> None:
        """An exhausted budget is reported rather than silently truncating."""
        net = _net(["A>>B", "B>>A", "C>>D", "D>>C", "B>>C", "D>>A"])

        with self.assertRaises(SiphonSearchLimit):
            minimal_siphons(net, max_nodes=1, strict_limit=True)

    def test_lenient_limit_returns_partial(self) -> None:
        """Without strict_limit the search degrades instead of raising."""
        net = _net(["A>>B", "B>>A", "C>>D", "D>>C"])
        found = minimal_siphons(net, max_nodes=1, strict_limit=False)

        self.assertIsInstance(found, list)

    def test_public_wrappers_accept_syncrn(self) -> None:
        """find_siphons / find_traps still take SynCRN and label output."""
        crn = SynCRN.from_reaction_strings(["A>>B", "B>>A"])

        self.assertEqual(find_siphons(crn), [{"A", "B"}])
        self.assertEqual(find_traps(crn), [{"A", "B"}])


if __name__ == "__main__":
    unittest.main()
