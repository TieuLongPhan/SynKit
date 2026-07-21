import unittest
from unittest.mock import patch

import networkx as nx

from synkit.Chem.Reaction.aam_validator import AAMValidator
from synkit.Graph.ITS.its_expand import (
    ITSExpand,
    ITSExpansionError,
    smiles_to_graph as expansion_smiles_to_graph,
)
from synkit.Graph.ITS.its_builder import ITSBuilder


class TestPartialExpand(unittest.TestCase):

    def test_its_builder_does_not_mutate_source_graph(self):
        source = nx.Graph()
        source.add_node(
            1,
            element="C",
            aromatic=False,
            hcount=3,
            charge=0,
            neighbors=["C"],
            atom_map=1,
        )
        source.add_node(
            2,
            element="C",
            aromatic=False,
            hcount=3,
            charge=0,
            neighbors=["C"],
            atom_map=2,
        )
        source.add_edge(1, 2, order=1.0)
        before_nodes = {node: dict(attrs) for node, attrs in source.nodes(data=True)}
        before_edges = {(u, v): dict(attrs) for u, v, attrs in source.edges(data=True)}

        ITSBuilder.ITSGraph(source, nx.Graph())

        self.assertEqual(dict(source.nodes(data=True)), before_nodes)
        self.assertEqual(
            {(u, v): dict(attrs) for u, v, attrs in source.edges(data=True)},
            before_edges,
        )

    def test_expand_aam_with_its(self):
        input_rsmi = "CC[CH2:3][Cl:1].[NH2:2][H:4]>>CC[CH2:3][NH2:2].[Cl:1][H:4]"
        output_rsmi = ITSExpand.expand_aam_with_its(input_rsmi, use_G=True)
        expected_rsmi = (
            "[CH3:1][CH2:2][CH2:3][Cl:4].[NH2:5][H:6]"
            + ">>[CH3:1][CH2:2][CH2:3][NH2:5].[Cl:4][H:6]"
        )
        self.assertTrue(AAMValidator.smiles_check(output_rsmi, expected_rsmi, "ITS"))

    def test_expand_with_relabel(self):
        input_rsmi = "CC[CH2:3][Cl:1].[NH2:2][H:4]>>CC[CH2:3][NH2:2].[Cl:1][H:4]"
        output_rsmi = ITSExpand.expand_aam_with_its(input_rsmi, relabel=True)
        expected_rsmi = (
            "[CH3:1][CH2:2][CH2:3][Cl:4].[NH2:5][H:6]"
            + ">>[CH3:1][CH2:2][CH2:3][NH2:5].[Cl:4][H:6]"
        )
        self.assertTrue(AAMValidator.smiles_check(output_rsmi, expected_rsmi, "ITS"))

    def test_expand_rsmi_uses_minimal_fresh_map_path(self):
        input_rsmi = "CC[CH2:30][Cl:10].[NH2:20]>>CC[CH2:30][NH2:20].[Cl:10]"

        output_rsmi = ITSExpand.expand_rsmi(input_rsmi)

        self.assertTrue(ITSExpand.endpoint_constitutions_match(input_rsmi, output_rsmi))
        self.assertNotIn(":30]", output_rsmi)
        self.assertTrue(
            AAMValidator.smiles_check(
                output_rsmi,
                "[CH3:1][CH2:2][CH2:3][Cl:4].[NH2:5]>>"
                "[CH3:1][CH2:2][CH2:3][NH2:5].[Cl:4]",
                "ITS",
            )
        )

    def test_expand_rsmi_transports_radicals_without_guards(self):
        input_rsmi = (
            "CC(C)(C[O:11][N+:10]([O-])=O)C.[Ar]" ">>CC(C)(C[O:11])C.[O-][N+:10]=O.[Ar]"
        )

        output_rsmi = ITSExpand.expand_rsmi(
            input_rsmi,
            preserve_radical_state=True,
        )

        self.assertTrue(ITSExpand.endpoint_constitutions_match(input_rsmi, output_rsmi))
        self.assertNotIn(":11]", output_rsmi)

    def test_expand_preserve_sparse_atom_maps(self):
        input_rsmi = (
            "Br[C:64]1=[CH:63][CH:62]=[C:61]([S-:10])[CH:72]=[CH:73]1."
            "C[CH+:20][C:31]1=[C:32](C)[CH:33]=[C:34](C)[CH:35]=[C:36]1C"
            ">>"
            "C[CH:20]([S:10][C:61]1=[CH:62][CH:63]=[C:64](Br)[CH:73]=[CH:72]1)"
            "[C:31]1=[C:32](C)[CH:33]=[C:34](C)[CH:35]=[C:36]1C"
        )

        output_rsmi = ITSExpand.expand_aam_with_its(
            input_rsmi,
            relabel=False,
            preserve_older_map=True,
        )

        self.assertIn(":10]", output_rsmi)
        self.assertIn(":20]", output_rsmi)

    def test_core_expansion_converts_only_selected_full_endpoint(self):
        input_rsmi = "CC[CH2:3][Cl:1].[NH2:2]>>CC[CH2:3][NH2:2].[Cl:1]"

        with patch(
            "synkit.Graph.ITS.its_expand.smiles_to_graph",
            wraps=expansion_smiles_to_graph,
        ) as converter:
            ITSExpand.expand_aam_with_its(
                input_rsmi,
                use_G=True,
                preserve_older_map=True,
            )

        self.assertEqual(converter.call_count, 1)
        self.assertEqual(converter.call_args.args[0], input_rsmi.split(">>")[0])

    def test_constitutional_expansion_omits_stereo_registry(self):
        input_rsmi = "CC[CH2:3][Cl:1].[NH2:2]>>CC[CH2:3][NH2:2].[Cl:1]"

        with patch(
            "synkit.Graph.ITS.its_expand.smiles_to_graph",
            wraps=expansion_smiles_to_graph,
        ) as converter:
            output_rsmi = ITSExpand.expand_aam_with_its(
                input_rsmi,
                use_G=True,
                preserve_older_map=True,
                constitutional_only=True,
            )

        self.assertFalse(converter.call_args.kwargs["include_stereo_descriptors"])
        self.assertTrue(ITSExpand.endpoint_constitutions_match(input_rsmi, output_rsmi))

    def test_mapped_radical_transfer_does_not_use_graph_matcher(self):
        source = nx.Graph()
        source.add_node(
            1,
            element="O",
            hcount=0,
            charge=0,
            atom_map=10,
            radical=1,
        )
        target = nx.Graph()
        target.add_node(
            7,
            element="O",
            hcount=0,
            charge=0,
            atom_map=10,
            radical=0,
        )

        with patch(
            "synkit.Graph.ITS.its_expand.nx.algorithms.isomorphism.GraphMatcher"
        ) as matcher:
            ITSExpand._transfer_endpoint_radicals(source, target)

        matcher.assert_not_called()
        self.assertEqual(target.nodes[7]["radical"], 1)

    def test_unmapped_radical_uses_unique_mapped_neighbour_anchor(self):
        source = nx.Graph()
        source.add_node(
            1,
            element="O",
            aromatic=False,
            hcount=0,
            charge=0,
            atom_map=0,
            radical=1,
        )
        source.add_node(
            2,
            element="O",
            aromatic=False,
            hcount=0,
            charge=0,
            atom_map=10,
            radical=0,
        )
        source.add_edge(1, 2, order=1.0)
        target = nx.Graph()
        target.add_node(
            7,
            element="O",
            aromatic=False,
            hcount=0,
            charge=0,
            atom_map=8,
            radical=0,
        )
        target.add_node(
            8,
            element="O",
            aromatic=False,
            hcount=0,
            charge=0,
            atom_map=10,
            radical=0,
        )
        target.add_edge(7, 8, order=1.0)

        with patch(
            "synkit.Graph.ITS.its_expand.nx.algorithms.isomorphism.GraphMatcher"
        ) as matcher:
            ITSExpand._transfer_endpoint_radicals(source, target)

        matcher.assert_not_called()
        self.assertEqual(target.nodes[7]["radical"], 1)
        self.assertEqual(target.nodes[8]["radical"], 0)

    def test_guarded_expansion_handles_unmapped_peroxide_radical(self):
        input_rsmi = (
            "[H:20][C:21]1(C=CC=C[CH:22]1)O.[O:10][O]>>"
            "[H:20][O:10][O].C1=C[CH:22]=[C:21](C=C1)O"
        )

        report = ITSExpand.expand_aam_with_its_report(
            input_rsmi,
            preserve_older_map=True,
            fallback_to_other_side=True,
            require_constitution_preservation=True,
            fold_unmapped_explicit_hydrogens=True,
            ignore_stereochemistry=True,
            explicit_hydrogen=True,
            preserve_radical_state=True,
        )

        self.assertTrue(report.radical_state_preserved)
        self.assertTrue(report.constitution_guard_passed)
        self.assertTrue(ITSExpand.endpoint_constitutions_match(input_rsmi, report.rsmi))

    def test_unanchored_radical_transfer_ignores_reassigned_atom_maps(self):
        source = nx.Graph()
        source.add_node(
            1,
            element="C",
            aromatic=True,
            hcount=0,
            charge=0,
            atom_map=28,
            radical=1,
        )
        source.add_node(
            2,
            element="C",
            aromatic=True,
            hcount=1,
            charge=0,
            atom_map=0,
            radical=0,
        )
        source.add_edge(1, 2, order=1.5)

        target = nx.Graph()
        target.add_node(
            11,
            element="C",
            aromatic=True,
            hcount=0,
            charge=0,
            atom_map=1,
            radical=0,
        )
        target.add_node(
            12,
            element="C",
            aromatic=True,
            hcount=1,
            charge=0,
            atom_map=2,
            radical=0,
        )
        target.add_edge(11, 12, order=1.5)

        ITSExpand._transfer_endpoint_radicals(
            source,
            target,
            preserve_atom_maps=False,
        )

        self.assertEqual(target.nodes[11]["radical"], 1)
        self.assertEqual(target.nodes[12]["radical"], 0)

    def test_unanchored_general_expansion_preserves_localized_radical(self):
        input_rsmi = (
            "Fc1ccc([C:2](CC2CCN(CC3CC3)CC2)=[O:1])cc1."
            "[O:21]([H:30])[H:31]."
            "c1cc[c:28]([Cl:29])[c:23]([Mg:22])c1>>"
            "Fc1ccc([C:2](CC2CCN(CC3CC3)CC2)([O:1][H:30])"
            "[c:23]2cccc[c:28]2[H:31])cc1.[O:21]([Mg:22])[Cl:29]"
        )

        report = ITSExpand.expand_aam_with_its_report(
            input_rsmi,
            preserve_older_map=False,
            fallback_to_other_side=True,
            require_constitution_preservation=True,
            fold_unmapped_explicit_hydrogens=True,
            ignore_stereochemistry=True,
            explicit_hydrogen=True,
            preserve_radical_state=True,
            constitutional_only=True,
        )

        self.assertTrue(report.constitution_guard_passed)
        self.assertTrue(ITSExpand.endpoint_constitutions_match(input_rsmi, report.rsmi))

    def test_guarded_product_fallback_preserves_constitution(self):
        input_rsmi = "[CH2:10][C:20]1=[CH:21]C=CC=C1" ">>[CH2:10]=[C:20]1C=CC=C[CH:21]1"

        with self.assertRaisesRegex(
            ITSExpansionError, "reactant-side ITS expansion failed"
        ):
            ITSExpand.expand_aam_with_its(
                input_rsmi,
                preserve_older_map=True,
            )

        output_rsmi = ITSExpand.expand_aam_with_its(
            input_rsmi,
            preserve_older_map=True,
            fallback_to_other_side=True,
            require_constitution_preservation=True,
        )

        self.assertTrue(ITSExpand.endpoint_constitutions_match(input_rsmi, output_rsmi))
        for atom_map in (10, 20, 21):
            self.assertIn(f":{atom_map}]", output_rsmi)

        report = ITSExpand.expand_aam_with_its_report(
            input_rsmi,
            preserve_older_map=True,
            fallback_to_other_side=True,
            require_constitution_preservation=True,
        )
        self.assertEqual(report.preferred_side, "reactant")
        self.assertEqual(report.selected_side, "product")
        self.assertTrue(report.fallback_used)
        self.assertTrue(report.constitution_guard_passed)
        self.assertIn("serialization returned no result", report.fallback_reason)

    def test_guarded_fallback_rejects_changed_constitution(self):
        input_rsmi = (
            "[H][C:10]([H])[H].[H]/[C:20]([H])=[C:21]([H])/[H]"
            ">>[CH2:21][C:20]([H])([H])[C:10]([H])([H])[H]"
        )

        with self.assertRaisesRegex(
            ITSExpansionError, "changed an endpoint constitution"
        ):
            ITSExpand.expand_aam_with_its(
                input_rsmi,
                preserve_older_map=True,
                fallback_to_other_side=True,
                require_constitution_preservation=True,
            )

    def test_fold_unmapped_explicit_hydrogens_prevents_endpoint_loss(self):
        input_rsmi = (
            "CC(C)(C[O:20])C.CC(C)([CH:11]([H:10])[O:12])C>>"
            "CC(C)(C[O:20][H:10])C.CC(C)([C:11]([H])=[O:12])C"
        )

        report = ITSExpand.expand_aam_with_its_report(
            input_rsmi,
            preserve_older_map=True,
            fallback_to_other_side=True,
            require_constitution_preservation=True,
            fold_unmapped_explicit_hydrogens=True,
            explicit_hydrogen=True,
        )

        self.assertEqual(report.selected_side, "reactant")
        self.assertTrue(report.unmapped_explicit_hydrogens_folded)
        self.assertEqual(report.folded_unmapped_explicit_hydrogen_count, 1)
        self.assertTrue(report.explicit_hydrogen_serialization)
        self.assertTrue(ITSExpand.endpoint_constitutions_match(input_rsmi, report.rsmi))
        self.assertIn("[CH:11]=[O:12]", report.rsmi)

    def test_ignore_stereochemistry_recovers_constitutional_expansion(self):
        input_rsmi = (
            "[H:20][Br:21].CCC/[C:10]=C/Br>>" "[H:20]/[C:10](=C\\Br)/CCC.[Br:21]"
        )

        report = ITSExpand.expand_aam_with_its_report(
            input_rsmi,
            preserve_older_map=True,
            fallback_to_other_side=True,
            require_constitution_preservation=True,
            ignore_stereochemistry=True,
            preserve_radical_state=True,
        )

        self.assertTrue(report.stereochemistry_ignored_for_expansion)
        self.assertTrue(report.constitution_guard_passed)
        self.assertTrue(ITSExpand.endpoint_constitutions_match(input_rsmi, report.rsmi))
        self.assertIn("[H:20]", report.rsmi)


if __name__ == "__main__":
    unittest.main()
