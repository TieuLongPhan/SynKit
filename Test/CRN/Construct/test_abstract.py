from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from synkit.CRN.Construct.abstract import (
    AbstractReactionExtractor,
    AbstractReactionNetwork,
    _excel_label,
    _first_present,
    _normalize_abstract_side,
    _split_reaction_smiles,
    deduplicate_abstract_reactions,
)


class TestHelperFunctions(unittest.TestCase):
    def test_split_reaction_smiles_basic(self) -> None:
        reactants, products = _split_reaction_smiles("C=C.O>>CCO")
        self.assertEqual(reactants, ["C=C", "O"])
        self.assertEqual(products, ["CCO"])

    def test_split_reaction_smiles_with_spaces_and_empty_sides(self) -> None:
        reactants, products = _split_reaction_smiles("  A.B  >>  C.D  ")
        self.assertEqual(reactants, ["A", "B"])
        self.assertEqual(products, ["C", "D"])

        reactants2, products2 = _split_reaction_smiles(">>C")
        self.assertEqual(reactants2, [])
        self.assertEqual(products2, ["C"])

        reactants3, products3 = _split_reaction_smiles("A>>")
        self.assertEqual(reactants3, ["A"])
        self.assertEqual(products3, [])

    def test_split_reaction_smiles_invalid(self) -> None:
        with self.assertRaises(ValueError):
            _split_reaction_smiles("A.B>C")

    def test_excel_label(self) -> None:
        self.assertEqual(_excel_label(0), "A")
        self.assertEqual(_excel_label(25), "Z")
        self.assertEqual(_excel_label(26), "AA")
        self.assertEqual(_excel_label(27), "AB")
        self.assertEqual(_excel_label(51), "AZ")
        self.assertEqual(_excel_label(52), "BA")

    def test_excel_label_negative(self) -> None:
        with self.assertRaises(ValueError):
            _excel_label(-1)

    def test_normalize_abstract_side(self) -> None:
        self.assertEqual(_normalize_abstract_side("B+A+C"), ["A", "B", "C"])
        self.assertEqual(_normalize_abstract_side(" B +  A + "), ["A", "B"])

    def test_first_present(self) -> None:
        record = {"a": None, "b": "value_b", "c": "value_c"}
        self.assertEqual(_first_present(record, ["a", "b", "c"]), "value_b")
        self.assertEqual(_first_present(record, ["x", "c"]), "value_c")
        self.assertIsNone(_first_present(record, ["x", "y"]))

    def test_deduplicate_abstract_reactions(self) -> None:
        reactions = [
            "A+B>>C",
            "B+A>>C",
            "A>>A",
            "C>>D",
            "invalid",
            "C>>D",
        ]
        filtered = deduplicate_abstract_reactions(reactions)
        self.assertEqual(filtered, ["A+B>>C", "C>>D"])


class TestAbstractReactionNetwork(unittest.TestCase):
    def test_to_dict(self) -> None:
        network = AbstractReactionNetwork(
            molecule_pool=["CCO", "O", "C=C"],
            reactions=["A+B>>C"],
            templates={"R1": "[C:1]=[C:2].[H:3][OH:4]>>[C:1]([H:3])[C:2][OH:4]"},
            label_to_molecule={"A": "C=C", "B": "O", "C": "CCO"},
        )

        result = network.to_dict()
        self.assertEqual(result["molecule_pool"], ["CCO", "O", "C=C"])
        self.assertEqual(result["reactions"], ["A+B>>C"])
        self.assertEqual(
            result["templates"]["R1"],
            "[C:1]=[C:2].[H:3][OH:4]>>[C:1]([H:3])[C:2][OH:4]",
        )
        self.assertEqual(result["label_to_molecule"]["A"], "C=C")

    def test_to_json_payload(self) -> None:
        network = AbstractReactionNetwork(
            molecule_pool=["A_mol", "B_mol"],
            reactions=["A>>B"],
            templates={"R1": "rule1"},
            label_to_molecule={"A": "A_mol", "B": "B_mol"},
        )

        payload = network.to_json_payload(name="demo_network")
        self.assertEqual(payload["meta"]["name"], "demo_network")
        self.assertEqual(payload["meta"]["version"], 1)
        self.assertEqual(len(payload["examples"]), 1)
        self.assertEqual(payload["examples"][0]["reactions"], ["A>>B"])

    def test_save_json(self) -> None:
        network = AbstractReactionNetwork(
            molecule_pool=["mol1", "mol2"],
            reactions=["A>>B"],
            templates={"R1": "rule1"},
            label_to_molecule={"A": "mol1", "B": "mol2"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "network.json"
            network.save_json(output_path)

            self.assertTrue(output_path.exists())

            with output_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

            self.assertEqual(payload["meta"]["name"], "network")
            self.assertEqual(payload["examples"][0]["molecule_pool"], ["mol1", "mol2"])
            self.assertEqual(payload["examples"][0]["label_to_molecule"]["A"], "mol1")

    def test_save_json_with_custom_name(self) -> None:
        network = AbstractReactionNetwork(
            molecule_pool=["mol1"],
            reactions=["A>>"],
            templates={},
            label_to_molecule={"A": "mol1"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "x.json"
            network.save_json(output_path, name="custom_name")

            with output_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

            self.assertEqual(payload["meta"]["name"], "custom_name")


class TestAbstractReactionExtractor(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = AbstractReactionExtractor()

    def test_iter_reaction_records_module_like(self) -> None:
        data = {
            "reactions": [
                {"id": "R1", "smiles": "A.B>>C"},
                {"id": "R2", "smiles": "C>>D"},
                "not_a_mapping",
            ]
        }

        records = list(self.extractor.iter_reaction_records(data))
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0][0]["id"], "R1")
        self.assertEqual(records[0][1], "")
        self.assertEqual(records[1][0]["id"], "R2")
        self.assertEqual(records[1][1], "")

    def test_iter_reaction_records_pathway_like(self) -> None:
        data = {
            "by_module": {
                "M00001": {
                    "reactions": [
                        {"id": "R1", "smiles": "A>>B"},
                        {"id": "R2", "smiles": "B>>C"},
                    ]
                },
                "M00002": {
                    "reactions": [
                        {"id": "R3", "smiles": "C>>D"},
                        "bad_record",
                    ]
                },
                "M_bad": "not_a_mapping",
            }
        }

        records = list(self.extractor.iter_reaction_records(data))
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0][1], "M00001")
        self.assertEqual(records[2][1], "M00002")
        self.assertEqual(records[2][0]["id"], "R3")

    def test_extract_reactions_and_templates_from_direct_reactions(self) -> None:
        reactions, templates = self.extractor.extract_reactions_and_templates(
            reactions=["A>>B", "B>>C"],
            templates={"R1": "rule1"},
        )
        self.assertEqual(reactions, ["A>>B", "B>>C"])
        self.assertEqual(templates, {"R1": "rule1"})

    def test_extract_reactions_and_templates_empty_inputs(self) -> None:
        reactions, templates = self.extractor.extract_reactions_and_templates()
        self.assertEqual(reactions, [])
        self.assertEqual(templates, {})

    def test_extract_reactions_and_templates_default_keys(self) -> None:
        data = {
            "reactions": [
                {"id": "R1", "smiles": "A.B>>C", "rule": "rule1"},
                {"kegg_id": "R2", "reaction": "C>>D", "template": "rule2"},
                {"id": "R3", "rxn_smiles": "D>>E", "smirks": "rule3"},
            ]
        }

        reactions, templates = self.extractor.extract_reactions_and_templates(data=data)

        self.assertEqual(reactions, ["A.B>>C", "C>>D", "D>>E"])
        self.assertEqual(
            templates,
            {
                "R1": "rule1",
                "R2": "rule2",
                "R3": "rule3",
            },
        )

    def test_extract_reactions_and_templates_custom_keys(self) -> None:
        data = {
            "reactions": [
                {"rid": "X1", "reaction_smiles": "A>>B", "transform": "t1"},
                {"rid": "X2", "reaction_smiles": "B>>C", "transform": "t2"},
            ]
        }

        reactions, templates = self.extractor.extract_reactions_and_templates(
            data=data,
            reaction_id_keys=["rid"],
            reaction_smiles_keys=["reaction_smiles"],
            template_keys=["transform"],
        )

        self.assertEqual(reactions, ["A>>B", "B>>C"])
        self.assertEqual(templates, {"X1": "t1", "X2": "t2"})

    def test_extract_reactions_and_templates_drop_missing_true(self) -> None:
        data = {
            "reactions": [
                {"id": "R1", "smiles": "A>>B", "rule": "rule1"},
                {"id": "R2", "rule": "rule2"},
            ]
        }

        reactions, templates = self.extractor.extract_reactions_and_templates(
            data=data,
            drop_missing_smiles_reactions=True,
        )

        self.assertEqual(reactions, ["A>>B"])
        self.assertEqual(templates, {"R1": "rule1"})

    def test_extract_reactions_and_templates_drop_missing_false(self) -> None:
        data = {
            "reactions": [
                {"id": "R1", "smiles": "A>>B", "rule": "rule1"},
                {"id": "R2", "rule": "rule2"},
                {"id": "R3", "smiles": "B>>C"},
            ]
        }

        reactions, templates = self.extractor.extract_reactions_and_templates(
            data=data,
            drop_missing_smiles_reactions=False,
        )

        self.assertEqual(reactions, ["A>>B", "", "B>>C"])
        self.assertEqual(templates, {"R1": "rule1", "R2": "rule2"})

    def test_extract_reactions_and_templates_prefix_module_id(self) -> None:
        data = {
            "by_module": {
                "M1": {
                    "reactions": [
                        {"id": "R1", "smiles": "A>>B", "rule": "rule1"},
                    ]
                }
            }
        }

        reactions, templates = self.extractor.extract_reactions_and_templates(
            data=data,
            prefix_module_in_reaction_id=True,
        )

        self.assertEqual(reactions, ["A>>B"])
        self.assertEqual(templates, {"M1:R1": "rule1"})

    def test_extract_reactions_and_templates_no_prefix_module_id(self) -> None:
        data = {
            "by_module": {
                "M1": {
                    "reactions": [
                        {"id": "R1", "smiles": "A>>B", "rule": "rule1"},
                    ]
                }
            }
        }

        reactions, templates = self.extractor.extract_reactions_and_templates(
            data=data,
            prefix_module_in_reaction_id=False,
        )

        self.assertEqual(reactions, ["A>>B"])
        self.assertEqual(templates, {"R1": "rule1"})

    def test_build_molecule_pool_appearance(self) -> None:
        parsed = [
            (["B", "A"], ["C"]),
            (["C"], ["D", "A"]),
        ]
        molecule_pool = self.extractor.build_molecule_pool(parsed, order="appearance")
        self.assertEqual(molecule_pool, ["B", "A", "C", "D"])

    def test_build_molecule_pool_sorted(self) -> None:
        parsed = [
            (["B", "A"], ["C"]),
            (["C"], ["D", "A"]),
        ]
        molecule_pool = self.extractor.build_molecule_pool(parsed, order="sorted")
        self.assertEqual(molecule_pool, ["A", "B", "C", "D"])

    def test_build_molecule_pool_invalid_order(self) -> None:
        with self.assertRaises(ValueError):
            self.extractor.build_molecule_pool([(["A"], ["B"])], order="unknown")

    def test_build_from_reactions_basic(self) -> None:
        network = self.extractor.build(
            reactions=["C#C.[H][H]>>C=C", "C=C.O>>CCO"],
            order="appearance",
        )

        self.assertEqual(network.molecule_pool, ["C#C", "[H][H]", "C=C", "O", "CCO"])
        self.assertEqual(network.reactions, ["A+B>>C", "C+D>>E"])
        self.assertEqual(
            network.label_to_molecule,
            {
                "A": "C#C",
                "B": "[H][H]",
                "C": "C=C",
                "D": "O",
                "E": "CCO",
            },
        )
        self.assertEqual(network.templates, {})

    def test_build_from_reactions_sorted_order(self) -> None:
        network = self.extractor.build(
            reactions=["B.A>>C", "C>>D"],
            order="sorted",
        )

        self.assertEqual(network.molecule_pool, ["A", "B", "C", "D"])
        self.assertEqual(network.reactions, ["B+A>>C", "C>>D"])
        self.assertEqual(network.label_to_molecule["A"], "A")
        self.assertEqual(network.label_to_molecule["B"], "B")

    def test_build_with_deduplicate(self) -> None:
        network = self.extractor.build(
            reactions=["A.B>>C", "B.A>>C", "D>>D", "C>>E"],
            deduplicate=True,
            order="appearance",
        )

        self.assertEqual(network.reactions, ["A+B>>C", "C>>E"])

    def test_build_with_custom_join_tokens(self) -> None:
        network = self.extractor.build(
            reactions=["A.B>>C.D"],
            reactant_join="|",
            product_join=";",
        )
        self.assertEqual(network.reactions, ["A|B>>C;D"])

    def test_build_from_data_with_templates_and_save(self) -> None:
        data = {
            "reactions": [
                {"id": "R1", "smiles": "X.Y>>Z", "rule": "rule_xy"},
                {"id": "R2", "smiles": "Z>>W", "rule": "rule_z"},
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "network.json"
            network = self.extractor.build(
                data=data,
                save_as=output_path,
            )

            self.assertEqual(network.molecule_pool, ["X", "Y", "Z", "W"])
            self.assertEqual(network.reactions, ["A+B>>C", "C>>D"])
            self.assertEqual(network.templates, {"R1": "rule_xy", "R2": "rule_z"})
            self.assertTrue(output_path.exists())

            with output_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

            self.assertEqual(payload["meta"]["name"], "network")
            self.assertEqual(payload["examples"][0]["templates"]["R1"], "rule_xy")

    def test_build_from_data_with_custom_keys(self) -> None:
        data = {
            "reactions": [
                {"rid": "RX1", "reaction_smiles": "M.N>>P", "transform": "tr1"},
                {"rid": "RX2", "reaction_smiles": "P>>Q", "transform": "tr2"},
            ]
        }

        network = self.extractor.build(
            data=data,
            reaction_id_keys=["rid"],
            reaction_smiles_keys=["reaction_smiles"],
            template_keys=["transform"],
        )

        self.assertEqual(network.molecule_pool, ["M", "N", "P", "Q"])
        self.assertEqual(network.reactions, ["A+B>>C", "C>>D"])
        self.assertEqual(network.templates, {"RX1": "tr1", "RX2": "tr2"})

    def test_build_propagates_invalid_reaction_smiles(self) -> None:
        with self.assertRaises(ValueError):
            self.extractor.build(reactions=["A.B>C"])


if __name__ == "__main__":
    unittest.main()
