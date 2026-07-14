import unittest

import networkx as nx
from rdkit import Chem

from synkit.IO.chem_converter import detect_its_format, rsmi_to_its
from synkit.Synthesis.Reactor.syn_reactor import SynReactor

ETHANE_DEHYDROGENATION = "[CH2:1]([H:3])[CH2:2]([H:4])>>" "[CH2:1]=[CH2:2].[H:3][H:4]"
METHANOL_DEHYDROGENATION = "[O:1]([H:3])[CH2:2]([H:4])>>" "[O+:1]=[CH2:2].[H:3][H:4]"


class TestSynReactorRewriteModes(unittest.TestCase):
    def test_public_templates_conserve_mapped_atoms(self):
        parser = Chem.SmilesParserParams()
        parser.removeHs = False
        for reaction in (ETHANE_DEHYDROGENATION, METHANOL_DEHYDROGENATION):
            reactants, products = reaction.split(">>", 1)
            mapped_atoms = []
            for side in (reactants, products):
                molecule = Chem.MolFromSmiles(side, parser)
                self.assertIsNotNone(molecule)
                mapped_atoms.append(
                    {atom.GetAtomMapNum() for atom in molecule.GetAtoms()}
                )

            self.assertEqual(mapped_atoms[0], mapped_atoms[1])
            self.assertEqual(mapped_atoms[0], {1, 2, 3, 4})

    def test_detects_legacy_template(self):
        rc = nx.Graph()
        rc.add_edge(1, 2, order=(1.0, 2.0))

        self.assertFalse(SynReactor._is_electron_aware_template(rc))

    def test_detects_electron_aware_template(self):
        rc = nx.Graph()
        rc.add_edge(
            1,
            2,
            order=(1.0, 2.0),
            sigma_order=(1.0, 1.0),
            pi_order=(0.0, 1.0),
        )

        self.assertTrue(SynReactor._is_electron_aware_template(rc))

    def test_invert_tuple_template_preserves_tuple_representation(self):
        reactor = SynReactor(
            "C=C.[H][H]",
            ETHANE_DEHYDROGENATION,
            invert=True,
            explicit_h=False,
            template_format="tuple",
        )

        self.assertEqual(detect_its_format(reactor.rule.rc.raw), "tuple")
        self.assertEqual(reactor.rule.rc.raw.edges[1, 2]["pi_order"], (1.0, 0.0))

    def test_electron_aware_rewrite_refreshes_product_accounting(self):
        host = nx.Graph()
        host.add_node(
            1,
            element="C",
            charge=0,
            hcount=3,
            lone_pairs=0,
            radical=0,
            valence_electrons=4,
        )
        host.add_node(
            2,
            element="C",
            charge=0,
            hcount=3,
            lone_pairs=0,
            radical=0,
            valence_electrons=4,
        )
        host.add_edge(1, 2, order=1.0, sigma_order=1.0, pi_order=0.0)

        rc = nx.Graph()
        rc.add_node(10, typesGH=(("C", False, 3, 0, []), ("C", False, 2, 0, [])))
        rc.add_node(20, typesGH=(("C", False, 3, 0, []), ("C", False, 2, 0, [])))
        rc.add_edge(
            10,
            20,
            order=(1.0, 2.0),
            sigma_order=(1.0, 1.0),
            pi_order=(0.0, 1.0),
            standard_order=-1.0,
        )

        rewritten = SynReactor._glue_graph(host, rc, {10: 1, 20: 2})[0]

        self.assertEqual(rewritten.edges[1, 2]["sigma_order"], (1.0, 1.0))
        self.assertEqual(rewritten.edges[1, 2]["pi_order"], (0.0, 1.0))
        self.assertEqual(rewritten.edges[1, 2]["kekule_order"][1], 2.0)
        self.assertEqual(rewritten.nodes[1]["hcount"], (3, 2))
        self.assertEqual(rewritten.nodes[1]["recomputed_charge"][1], 0.0)

    def test_electron_aware_to_smarts_uses_kekule_product_reconstruction(self):
        its = nx.Graph()
        for node in range(6):
            its.add_node(
                node,
                element=("C", "C"),
                charge=(0, 0),
                hcount=(1, 1),
                lone_pairs=(0, 0),
                radical=(0, 0),
                valence_electrons=(4, 4),
                present=(True, True),
            )
        cycle_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
        for idx, edge in enumerate(cycle_edges):
            order = 1.0 if idx % 2 == 0 else 2.0
            its.add_edge(
                *edge,
                order=(1.5, 1.5),
                kekule_order=(order, order),
                sigma_order=(1.0, 1.0),
                pi_order=(order - 1.0, order - 1.0),
                standard_order=0.0,
            )
        its.graph["electron_aware_rewrite"] = True

        self.assertEqual(SynReactor._to_smarts(its), "c1ccccc1>>c1ccccc1")

    def test_legacy_output_does_not_switch_modes_from_host_sigma_pi(self):
        its = nx.Graph()
        its.add_node(
            1,
            element="C",
            charge=0,
            hcount=3,
            typesGH=(("C", False, 3, 0, []), ("C", False, 3, 0, [])),
        )
        its.add_node(
            2,
            element="C",
            charge=0,
            hcount=3,
            typesGH=(("C", False, 3, 0, []), ("C", False, 3, 0, [])),
        )
        its.add_edge(
            1,
            2,
            order=(1.0, 1.0),
            sigma_order=(1.0, 1.0),
            pi_order=(0.0, 0.0),
            standard_order=0.0,
        )
        its.graph["electron_aware_rewrite"] = False

        self.assertEqual(
            SynReactor._to_smarts(its),
            "[CH3:1][CH3:2]>>[CH3:1][CH3:2]",
        )

    def test_public_tuple_template_reaches_electron_aware_rewrite(self):
        reactor = SynReactor(
            "CC",
            ETHANE_DEHYDROGENATION,
            explicit_h=False,
            template_format="tuple",
        )

        self.assertEqual(detect_its_format(reactor.rule.rc.raw), "tuple")
        self.assertTrue(reactor.its_list)
        self.assertTrue(reactor.its_list[0].graph["electron_aware_rewrite"])
        self.assertEqual(len(reactor.smarts), 1)
        self.assertIn(".[H:", reactor.smarts[0].split(">>", 1)[1])

    def test_diagnostics_are_opt_in_and_do_not_change_products(self):
        baseline = SynReactor(
            "CC",
            ETHANE_DEHYDROGENATION,
            explicit_h=False,
            template_format="tuple",
        )
        diagnosed = SynReactor(
            "CC",
            ETHANE_DEHYDROGENATION,
            explicit_h=False,
            template_format="tuple",
            electron_diagnostics=True,
        )

        self.assertEqual(baseline.diagnostics, [])
        self.assertEqual(diagnosed.smarts, baseline.smarts)
        self.assertEqual(len(diagnosed.diagnostics), 1)
        self.assertTrue(diagnosed.diagnostics[0]["electron_aware_rewrite"])
        self.assertEqual(diagnosed.diagnostics[0]["mismatch_count"], 0)

    def test_diagnostics_report_public_nonzero_mismatch(self):
        template = rsmi_to_its(
            METHANOL_DEHYDROGENATION,
            format="tuple",
            drop_non_aam=False,
            use_index_as_atom_map=True,
        )
        template.nodes[1]["lone_pairs"] = (2, 0)
        reactor = SynReactor(
            "[OH:1][CH3:2]",
            template,
            explicit_h=False,
            electron_diagnostics=True,
        )

        self.assertEqual(len(reactor.smarts), 1)
        self.assertEqual(len(reactor.diagnostics), 1)
        self.assertEqual(reactor.diagnostics[0]["mismatch_count"], 1)
        self.assertEqual(
            reactor.diagnostics[0]["mismatches"][1],
            {"charge": 1, "recomputed_charge": 3.0},
        )


if __name__ == "__main__":
    unittest.main()
