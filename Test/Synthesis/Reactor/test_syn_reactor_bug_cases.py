import unittest

import networkx as nx

from synkit.Chem.Reaction.standardize import Standardize
from synkit.IO.chem_converter import rsmi_to_its
from synkit.Rule import SynRule
from synkit.Synthesis.Reactor.syn_reactor import SynReactor

AROMATIC_N_OXIDATION = (
    "[CH3:1][c:2]1[n:3][c:4]([Br:5])[cH:6][cH:7][c:8]1[F:9]."
    "[O:10]=[C:11]([O:12][O:13][H:21])[c:14]1[cH:15][cH:16][cH:17]"
    "[c:18]([Cl:19])[cH:20]1>>"
    "[CH3:1][c:2]1[n+:3]([O-:13])[c:4]([Br:5])[cH:6][cH:7][c:8]1[F:9]."
    "[O:10]=[C:11]([O:12][H:21])[c:14]1[cH:15][cH:16][cH:17]"
    "[c:18]([Cl:19])[cH:20]1"
)

HYDROGEN_SENSITIVE_DEDUP = (
    "[O:1]([CH:2]([CH:3]([CH2:4][c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1)"
    "[N:11]([CH2:12][c:13]1[cH:14][cH:15][cH:16][cH:17][cH:18]1)"
    "[C:19](=[O:20])[OH:21])[H:23])[H:24].[O:22]>>"
    "[O:1]=[CH:2][CH:3]([CH2:4][c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1)"
    "[N:11]([CH2:12][c:13]1[cH:14][cH:15][cH:16][cH:17][cH:18]1)"
    "[C:19](=[O:20])[OH:21].[O:22]([H:23])[H:24]"
)

LEGACY_REVERSE_ROLE_SYMMETRY = (
    "[BH2:1][CH2:2][OH:5].[Br:3][CH2:4][NH2:6]>>"
    "[BH2:1][Br:3].[OH:5][CH2:2][CH2:4][NH2:6]"
)


def _legacy_core_rule(smart):
    graph = rsmi_to_its(
        smart,
        core=True,
        drop_non_aam=False,
        use_index_as_atom_map=True,
        node_attrs=[
            "element",
            "aromatic",
            "hcount",
            "charge",
            "neighbors",
            "atom_map",
        ],
        edge_attrs=["order"],
        format="typesGH",
    )
    return SynRule(graph, canon=False, implicit_h=True, format="typesGH")


def _standardized_reactor_outputs(smart):
    standardizer = Standardize()
    expected = standardizer.fit(smart, remove_aam=True, ignore_stereo=True)
    host = expected.split(">>", 1)[0]
    reactor = SynReactor(
        host,
        _legacy_core_rule(smart),
        template_format="typesGH",
        explicit_h=False,
        implicit_temp=False,
        automorphism=True,
    )
    outputs = [
        standardizer.fit(candidate, remove_aam=True, ignore_stereo=True)
        for candidate in reactor.smarts_list
    ]
    return expected, reactor, outputs


class TestSynReactorBugCases(unittest.TestCase):
    def test_truncated_aromatic_n_core_does_not_invent_zero_pi_role(self):
        expected, reactor, outputs = _standardized_reactor_outputs(AROMATIC_N_OXIDATION)

        self.assertEqual(reactor.mapping_count, 1)
        self.assertIn(expected, outputs)

    def test_typesgh_dedup_preserves_distinct_hydrogen_destinations(self):
        expected, reactor, outputs = _standardized_reactor_outputs(
            HYDROGEN_SENSITIVE_DEDUP
        )

        self.assertEqual(reactor.mapping_count, 3)
        self.assertEqual(len(reactor.its_list), 3)
        self.assertEqual(len(outputs), 1)
        self.assertIn(expected, outputs)

    def test_tuple_rewrite_preserves_host_kekule_phase(self):
        host = nx.Graph()
        for node in (0, 1):
            host.add_node(
                node,
                element="C",
                aromatic=True,
                hcount=0,
                charge=0,
                neighbors=[],
                lone_pairs=0,
                radical=0,
                valence_electrons=4,
                atom_map=node + 1,
            )
        host.add_edge(
            0,
            1,
            order=1.5,
            kekule_order=2.0,
            sigma_order=1.0,
            pi_order=1.0,
        )

        rc = nx.Graph()
        for node in (0, 1):
            rc.add_node(
                node,
                element=("C", "C"),
                aromatic=(True, False),
                hcount=(0, 0),
                charge=(0, 0),
                neighbors=([], []),
                lone_pairs=(0, 0),
                radical=(0, 0),
                valence_electrons=(4, 4),
                present=(True, True),
                typesGH=(
                    ("C", True, 0, 0, []),
                    ("C", False, 0, 0, []),
                ),
            )
        rc.add_edge(
            0,
            1,
            order=(1.5, 1.0),
            kekule_order=(1.0, 1.0),
            sigma_order=(1.0, 1.0),
            pi_order=(0.0, 0.0),
            standard_order=0.5,
        )

        rewritten = SynReactor._glue_graph(
            host,
            rc,
            {0: 0, 1: 1},
            refresh_electrons=False,
            electron_aware=True,
        )[0]

        self.assertEqual(rewritten.edges[0, 1]["order"], (1.5, 1.0))
        self.assertEqual(rewritten.edges[0, 1]["kekule_order"], (2.0, 1.0))
        self.assertEqual(rewritten.edges[0, 1]["pi_order"], (1.0, 0.0))

    def test_legacy_reverse_automorphism_keeps_distinct_rewrite_roles(self):
        standardizer = Standardize()
        expected = standardizer.fit(
            LEGACY_REVERSE_ROLE_SYMMETRY,
            remove_aam=True,
            ignore_stereo=True,
        )
        product = expected.split(">>", 1)[1]

        for automorphism in (False, True):
            with self.subTest(automorphism=automorphism):
                reactor = SynReactor(
                    product,
                    _legacy_core_rule(LEGACY_REVERSE_ROLE_SYMMETRY),
                    invert=True,
                    template_format="typesGH",
                    radical_policy="ignore",
                    explicit_h=False,
                    implicit_temp=False,
                    automorphism=automorphism,
                )
                outputs = [
                    standardizer.fit(
                        candidate,
                        remove_aam=True,
                        ignore_stereo=True,
                    )
                    for candidate in reactor.smarts_list
                ]

                self.assertEqual(reactor.mapping_count, 2)
                self.assertIn(expected, outputs)

    def test_diels_alder_diene_reversal_keeps_two_mappings(self):
        """Do not over-prune the two valid diene orientations."""
        smart = (
            "[CH2:1]=[CH:2][CH:3]=[CH2:4]."
            "[CH2:5]=[CH:6][CH:7]=[O:8]>>"
            "[CH2:1]1[CH:2]=[CH:3][CH2:4][CH2:5][CH:6]1[CH:7]=[O:8]"
        )

        for automorphism in (False, True):
            with self.subTest(automorphism=automorphism):
                reactor = SynReactor(
                    "C=CC=C.C=CC=O",
                    smart,
                    template_format="tuple",
                    explicit_h=False,
                    automorphism=automorphism,
                )

                self.assertEqual(reactor.mapping_count, 2)
                self.assertEqual(
                    {
                        tuple(mapping[index] for index in (1, 2, 3, 4))
                        for mapping in reactor.mappings
                    },
                    {(1, 2, 3, 4), (4, 3, 2, 1)},
                )
                self.assertEqual(len(reactor.smarts), 1)

    def test_tuple_backward_cross_coupling_keeps_aromatic_role_context(self):
        smart = (
            "[CH3:10][CH2:11][O:12][C:13](=[O:14])[c:15]1[cH:16][cH:18][cH:19]"
            "[c:20]([B:21]([OH:22])[OH:23])[cH:17]1."
            "[CH3:1][c:2]1[cH:3][cH:5][cH:6][c:7]([Br:8])[c:4]1[I:9]"
            ">>"
            "[CH3:1][c:2]1[cH:3][cH:5][cH:6][c:7]([Br:8])[c:4]1-"
            "[c:20]1[cH:17][c:15]([C:13]([O:12][CH2:11][CH3:10])=[O:14])"
            "[cH:16][cH:18][cH:19]1.[I:9][B:21]([OH:22])[OH:23]"
        )
        expected = Standardize().fit(smart)
        reactants, products = expected.split(">>")
        rc = rsmi_to_its(smart, core=True, format="tuple")

        forward = SynReactor(
            substrate=reactants,
            template=rc,
            implicit_temp=False,
            explicit_h=False,
        )
        backward = SynReactor(
            substrate=products,
            template=rc,
            implicit_temp=False,
            explicit_h=False,
            invert=True,
        )

        forward_smis = [
            Standardize().fit(candidate, remove_aam=True)
            for candidate in forward.smarts
        ]
        backward_smis = [
            Standardize().fit(candidate, remove_aam=True)
            for candidate in backward.smarts
        ]

        self.assertIn(expected, forward_smis)
        self.assertIn(expected, backward_smis)


if __name__ == "__main__":
    unittest.main()
