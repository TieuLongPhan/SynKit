import unittest

import networkx as nx

from synkit.CRN.Props.deficiency import (
    Complex,
    CRNTSummary,
    _exact_rank,
    complex_graph,
    complexes,
    crnt_summary,
    deficiency,
    deficiency_one_verdict,
    deficiency_zero_verdict,
    is_deficiency_one_applicable,
    is_deficiency_zero_applicable,
    is_reversible,
    is_weakly_reversible,
    linkage_class_deficiencies,
    linkage_classes,
    strong_linkage_classes,
    terminal_strong_linkage_classes,
)
from synkit.CRN.Structure.syncrn import SynCRN


def crn(*rxns):
    return SynCRN.from_reaction_strings(list(rxns))


class TestComplex(unittest.TestCase):
    def test_str_renders_chemically(self):
        self.assertEqual(str(Complex((2, 1, 0), ("A", "B", "C"))), "2A + B")
        self.assertEqual(str(Complex((1, 0), ("A", "B"))), "A")
        self.assertEqual(str(Complex((0, 0), ("A", "B"))), "0")

    def test_zero_and_support(self):
        cx = Complex((0, 3), ("A", "B"))
        self.assertFalse(cx.is_zero)
        self.assertEqual(cx.support, ("B",))
        self.assertTrue(Complex((0, 0), ("A", "B")).is_zero)

    def test_to_dict_drops_zeros(self):
        self.assertEqual(Complex((2, 0, 1), ("A", "B", "C")).to_dict(), {"A": 2, "C": 1})

    def test_labels_do_not_affect_identity(self):
        self.assertEqual(Complex((1, 0), ("A", "B")), Complex((1, 0), ("X", "Y")))
        self.assertEqual(
            len({Complex((1, 0), ("A", "B")), Complex((1, 0), ("X", "Y"))}), 1
        )


class TestComplexExtraction(unittest.TestCase):
    def test_complexes_are_deduplicated(self):
        found = {str(c) for c in complexes(crn("A+B>>C", "C>>A+B", "C>>D"))}
        self.assertEqual(found, {"A + B", "C", "D"})

    def test_zero_complex_from_outflow(self):
        found = {str(c) for c in complexes(SynCRN.from_reaction_strings(
            ["A>>"], strict=False
        ))}
        self.assertEqual(found, {"A", "0"})

    def test_complex_graph_has_one_edge_per_reaction(self):
        graph = complex_graph(crn("A>>B", "B>>A", "B>>C"))
        self.assertIsInstance(graph, nx.DiGraph)
        self.assertEqual(graph.number_of_nodes(), 3)
        self.assertEqual(graph.number_of_edges(), 3)

    def test_complex_graph_nodes_carry_labels(self):
        graph = complex_graph(crn("2A>>B"))
        labels = {data["label"] for _, data in graph.nodes(data=True)}
        self.assertEqual(labels, {"2A", "B"})


class TestExactRank(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(_exact_rank([]), 0)
        self.assertEqual(_exact_rank([[]]), 0)

    def test_dependent_rows(self):
        self.assertEqual(_exact_rank([[1, 0], [2, 0]]), 1)

    def test_full_rank(self):
        self.assertEqual(_exact_rank([[1, 0], [0, 1]]), 2)

    def test_fractional_entries_are_exact(self):
        # A float rank routine can misjudge this; the exact one must not.
        self.assertEqual(_exact_rank([[0.1, 0.3], [0.2, 0.6]]), 1)


class TestLinkageClasses(unittest.TestCase):
    def test_disjoint_blocks(self):
        self.assertEqual(linkage_classes(crn("A>>B", "C>>D")), [[0, 1], [2, 3]])

    def test_strong_linkage_classes_split_irreversible_arrow(self):
        strong = strong_linkage_classes(crn("A>>B"))
        self.assertEqual(strong, [[0], [1]])

    def test_terminal_strong_linkage_classes(self):
        terminal = terminal_strong_linkage_classes(crn("A>>B", "B>>C", "C>>B"))
        self.assertEqual(terminal, [[1, 2]])

    def test_weak_reversibility(self):
        self.assertTrue(is_weakly_reversible(crn("A>>B", "B>>A")))
        self.assertFalse(is_weakly_reversible(crn("A>>B")))
        # A cycle is weakly reversible without any reaction being reversed.
        self.assertTrue(is_weakly_reversible(crn("A>>B", "B>>C", "C>>A")))

    def test_reversibility_is_stricter_than_weak_reversibility(self):
        cycle = crn("A>>B", "B>>C", "C>>A")
        self.assertTrue(is_weakly_reversible(cycle))
        self.assertFalse(is_reversible(cycle))
        self.assertTrue(is_reversible(crn("A>>B", "B>>A")))

    def test_empty_network_is_weakly_reversible(self):
        self.assertTrue(is_weakly_reversible(SynCRN.from_reaction_strings([])))


class TestDeficiency(unittest.TestCase):
    """Deficiencies checked against the published values for each network."""

    CASES = [
        # (name, reactions, expected deficiency, expected weak reversibility)
        ("reversible isomerization", ["A>>B", "B>>A"], 0, True),
        ("irreversible isomerization", ["A>>B"], 0, False),
        ("isomerization cycle", ["A>>B", "B>>C", "C>>A"], 0, True),
        (
            "Michaelis-Menten",
            ["E+S>>ES", "ES>>E+S", "ES>>E+P"],
            0,
            False,
        ),
        (
            "reversible Michaelis-Menten",
            ["E+S>>ES", "ES>>E+S", "ES>>E+P", "E+P>>ES"],
            0,
            True,
        ),
        ("Edelstein", ["A>>2A", "A+B>>C", "C>>A+B", "C>>B"], 1, False),
        (
            "Horn-Jackson",
            ["2A>>A+B", "A+B>>2A", "A+B>>2B", "2B>>A+B", "2B>>2A", "2A>>2B"],
            1,
            True,
        ),
        (
            "two disjoint reversible blocks",
            ["A>>B", "B>>A", "C>>D", "D>>C"],
            0,
            True,
        ),
    ]

    def test_published_deficiencies(self):
        for name, rxns, expected_delta, expected_wr in self.CASES:
            with self.subTest(network=name):
                network = crn(*rxns)
                self.assertEqual(deficiency(network), expected_delta)
                self.assertEqual(is_weakly_reversible(network), expected_wr)

    def test_deficiency_is_never_negative(self):
        for name, rxns, _, _ in self.CASES:
            with self.subTest(network=name):
                self.assertGreaterEqual(deficiency(crn(*rxns)), 0)

    def test_deficiency_equals_n_minus_l_minus_s(self):
        network = crn("A>>2A", "A+B>>C", "C>>A+B", "C>>B")
        report = crnt_summary(network)
        self.assertEqual(
            report.deficiency,
            report.n_complexes - report.n_linkage_classes - report.rank,
        )

    def test_linkage_class_deficiencies_sum_at_most_total(self):
        for name, rxns, _, _ in self.CASES:
            with self.subTest(network=name):
                network = crn(*rxns)
                self.assertLessEqual(
                    sum(linkage_class_deficiencies(network)), deficiency(network)
                )

    def test_linkage_class_deficiencies_disjoint_blocks(self):
        self.assertEqual(
            linkage_class_deficiencies(crn("A>>B", "B>>A", "C>>D", "D>>C")),
            [0, 0],
        )


class TestDeficiencyZeroTheorem(unittest.TestCase):
    def test_weakly_reversible_zero_deficiency(self):
        verdict = deficiency_zero_verdict(crn("A>>B", "B>>A"))
        self.assertTrue(verdict["applicable"])
        self.assertEqual(verdict["conclusion"], "unique_stable_equilibrium")
        self.assertIn("locally asymptotically stable", verdict["statement"])

    def test_not_weakly_reversible_zero_deficiency(self):
        verdict = deficiency_zero_verdict(crn("A>>B"))
        self.assertTrue(verdict["applicable"])
        self.assertEqual(verdict["conclusion"], "no_positive_equilibrium")

    def test_nonzero_deficiency_is_inconclusive(self):
        verdict = deficiency_zero_verdict(crn("A>>2A", "A+B>>C", "C>>A+B", "C>>B"))
        self.assertFalse(verdict["applicable"])
        self.assertEqual(verdict["conclusion"], "inconclusive")

    def test_applicability_helper_matches_deficiency(self):
        self.assertTrue(is_deficiency_zero_applicable(crn("A>>B", "B>>A")))
        self.assertFalse(
            is_deficiency_zero_applicable(crn("A>>2A", "A+B>>C", "C>>A+B", "C>>B"))
        )


class TestDeficiencyOneTheorem(unittest.TestCase):
    def test_horn_jackson_network(self):
        verdict = deficiency_one_verdict(
            crn("2A>>A+B", "A+B>>2A", "A+B>>2B", "2B>>A+B", "2B>>2A", "2A>>2B")
        )
        self.assertEqual(verdict["deficiency"], 1)
        self.assertTrue(verdict["applicable"])
        self.assertEqual(verdict["conclusion"], "exactly_one_equilibrium")

    def test_reports_which_hypothesis_failed(self):
        # Edelstein: two terminal strong linkage classes is fine, but the
        # linkage-class deficiencies (0, 0) do not sum to the network's 1.
        verdict = deficiency_one_verdict(crn("A>>2A", "A+B>>C", "C>>A+B", "C>>B"))
        self.assertFalse(verdict["applicable"])
        self.assertFalse(verdict["deficiencies_sum_to_total"])
        self.assertIn("deficiencies_sum_to_total", verdict["statement"])

    def test_zero_deficiency_network_satisfies_hypotheses(self):
        self.assertTrue(is_deficiency_one_applicable(crn("A>>B", "B>>A")))

    def test_not_weakly_reversible_gives_at_most_one(self):
        verdict = deficiency_one_verdict(crn("A>>B"))
        self.assertTrue(verdict["applicable"])
        self.assertEqual(verdict["conclusion"], "at_most_one_equilibrium")


class TestCRNTSummary(unittest.TestCase):
    def setUp(self):
        self.report = crnt_summary(crn("A>>B", "B>>A", "B>>C", "C>>B"))

    def test_counts(self):
        self.assertEqual(self.report.n_species, 3)
        self.assertEqual(self.report.n_reactions, 4)
        self.assertEqual(self.report.n_complexes, 3)
        self.assertEqual(self.report.n_linkage_classes, 1)
        self.assertEqual(self.report.rank, 2)
        self.assertEqual(self.report.deficiency, 0)

    def test_flags(self):
        self.assertTrue(self.report.is_weakly_reversible)
        self.assertTrue(self.report.is_reversible)

    def test_to_dict_roundtrips_fields(self):
        payload = self.report.to_dict()
        self.assertEqual(payload["deficiency"], 0)
        self.assertEqual(payload["complexes"], ["A", "B", "C"])
        self.assertIn("deficiency_zero", payload)

    def test_str_is_readable(self):
        text = str(self.report)
        self.assertIn("deficiency delta=0", text)
        self.assertIn("weakly reversible: True", text)

    def test_from_crn_classmethod(self):
        self.assertIsInstance(CRNTSummary.from_crn(crn("A>>B")), CRNTSummary)

    def test_accepts_a_raw_digraph(self):
        graph = crn("A>>B", "B>>A").to_digraph()
        self.assertEqual(deficiency(graph), 0)
        self.assertTrue(is_weakly_reversible(graph))

    def test_reaction_kind_graph_matches_rule_kind_graph(self):
        network = crn("A+B>>C", "C>>A+B", "C>>D")
        as_rule = crnt_summary(network.to_digraph(reaction_kind="rule"))
        as_reaction = crnt_summary(network.to_digraph(reaction_kind="reaction"))
        self.assertEqual(as_rule.to_dict(), as_reaction.to_dict())


if __name__ == "__main__":
    unittest.main()
