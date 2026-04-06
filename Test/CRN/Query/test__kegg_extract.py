from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rdkit import Chem

from synkit.CRN.Query.kegg_extract import KEGGExtractor
from synkit.CRN.Query.kegg_parse import get_compound_ids_from_equations


class FakeKEGGClient:
    def __init__(
        self, text_by_path: dict[str, str], optional_text_by_path: dict[str, str | None]
    ):
        self.text_by_path = text_by_path
        self.optional_text_by_path = optional_text_by_path

    def get_text(self, path: str) -> str:
        return self.text_by_path[path]

    def get_optional_text(self, path: str) -> str | None:
        return self.optional_text_by_path.get(path)


class SuccessfulMapper:
    def get_attention_guided_atom_maps(
        self, reactions: list[str]
    ) -> list[dict[str, str]]:
        return [{"mapped_rxn": f"mapped::{reactions[0]}"}]


class FailingMapper:
    def get_attention_guided_atom_maps(
        self, reactions: list[str]
    ) -> list[dict[str, str]]:
        raise RuntimeError("mapping failed")


class TestKEGGExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        water = Chem.MolFromSmiles("O")
        ethanol = Chem.MolFromSmiles("CCO")
        acetaldehyde = Chem.MolFromSmiles("CC=O")
        assert water is not None
        assert ethanol is not None
        assert acetaldehyde is not None

        cls.water_molblock = Chem.MolToMolBlock(water)
        cls.ethanol_molblock = Chem.MolToMolBlock(ethanol)
        cls.acetaldehyde_molblock = Chem.MolToMolBlock(acetaldehyde)

    def make_extractor(self, mapper_cls=SuccessfulMapper) -> KEGGExtractor:
        text_by_path = {
            "get/hsa00010": (
                "ENTRY       hsa00010\n"
                "MODULE      hsa_M00001 core glycolysis\n"
                "            M00002 branch\n"
            ),
            "get/M00001": (
                "ENTRY       M00001\n"
                "REACTION    R00001 R00002\n"
                "            R00002\n"
            ),
            "get/M00002": ("ENTRY       M00002\n" "REACTION    R00003\n"),
            "get/M99999": "ENTRY       M99999\n",
            "get/rn:R00001": "ENTRY       R00001\nEQUATION    C00001 + C00002 => C00003\n",
            "get/rn:R00002": "ENTRY       R00002\nEQUATION    C00003 <=> C00004\n",
            "get/rn:R00003": "ENTRY       R00003\nEQUATION    C00005 => C00006\n",
            "get/rn:R99999": "ENTRY       R99999\n",
            "get/cpd:C00001": "ENTRY       C00001\nNAME        Water; H2O\n",
            "get/cpd:C00002": "ENTRY       C00002\nNAME        Ethanol\n",
            "get/cpd:C00003": "ENTRY       C00003\nNAME        Acetaldehyde\n",
            "get/cpd:C00004": "ENTRY       C00004\nNAME        Unresolved\n",
            "get/cpd:C00005": "ENTRY       C00005\n",
            "get/cpd:C00006": "ENTRY       C00006\nNAME        Product Six\n",
        }
        optional_text_by_path = {
            "get/cpd:C00001/mol": self.water_molblock,
            "get/cpd:C00002/mol": self.ethanol_molblock,
            "get/cpd:C00003/mol": self.acetaldehyde_molblock,
            "get/cpd:C00004/mol": None,
            "get/cpd:C00005/mol": None,
            "get/cpd:C00006/mol": None,
        }
        client = FakeKEGGClient(text_by_path, optional_text_by_path)
        return KEGGExtractor(client=client, mapper_cls=mapper_cls)

    def test_post_init_creates_default_client(self) -> None:
        extractor = KEGGExtractor(mapper_cls=None)
        self.assertIsNotNone(extractor.client)

    def test_get_modules_from_pathway(self) -> None:
        extractor = self.make_extractor()
        modules = extractor.get_modules_from_pathway("hsa00010")
        self.assertEqual(modules, ["M00001", "M00002"])

    def test_get_reaction_ids_from_module(self) -> None:
        extractor = self.make_extractor()
        reaction_ids = extractor.get_reaction_ids_from_module("M00001")
        self.assertEqual(reaction_ids, ["R00001", "R00002"])

    def test_get_equation_for_reaction_present_and_absent(self) -> None:
        extractor = self.make_extractor()
        self.assertEqual(
            extractor.get_equation_for_reaction("R00001"),
            "C00001 + C00002 => C00003",
        )
        self.assertIsNone(extractor.get_equation_for_reaction("R99999"))

    def test_get_module_and_pathway_equations(self) -> None:
        extractor = self.make_extractor()
        module_eqs = extractor.get_module_equations("M00001")
        self.assertEqual(sorted(module_eqs), ["R00001", "R00002"])
        pathway_eqs = extractor.get_pathway_equations("hsa00010")
        self.assertEqual(sorted(pathway_eqs), ["M00001", "M00002"])
        self.assertEqual(pathway_eqs["M00002"]["R00003"], "C00005 => C00006")

    def test_get_compound_name_and_molblock(self) -> None:
        extractor = self.make_extractor()
        self.assertEqual(extractor.get_compound_name("C00001"), "Water")
        self.assertIsNone(extractor.get_compound_name("C00005"))
        self.assertEqual(extractor.get_compound_molblock("C00001"), self.water_molblock)
        self.assertIsNone(extractor.get_compound_molblock("C00004"))

    def test_build_compound_table(self) -> None:
        extractor = self.make_extractor()
        table = extractor.build_compound_table(["C00001", "C00004", "C00005"])
        self.assertEqual(table["C00001"]["name"], "Water")
        self.assertEqual(table["C00001"]["smiles"], "O")
        self.assertIsNone(table["C00004"]["smiles"])
        self.assertIsNone(table["C00005"]["name"])

    def test_build_reaction_smiles_dict(self) -> None:
        extractor = self.make_extractor()
        equations_by_rid = extractor.get_module_equations("M00001")
        compound_ids = ["C00001", "C00002", "C00003", "C00004"]
        compounds = extractor.build_compound_table(compound_ids)
        _, parsed_by_rid = get_compound_ids_from_equations(equations_by_rid)

        rsmi_by_rid, missing_by_rid = extractor.build_reaction_smiles_dict(
            parsed_by_rid, compounds
        )

        self.assertEqual(rsmi_by_rid["R00001"], "O.CCO>>CC=O")
        self.assertEqual(missing_by_rid["R00001"], {"reactants": [], "products": []})
        self.assertEqual(
            missing_by_rid["R00002"], {"reactants": [], "products": ["C00004"]}
        )

    def test_atom_map_reactions_handles_success_invalid_and_missing_dependency(
        self,
    ) -> None:
        extractor = self.make_extractor(mapper_cls=SuccessfulMapper)
        mapped = extractor.atom_map_reactions(
            {
                "R1": "CCO>>CC=O",
                "R2": ">>CC=O",
                "R3": "CCO>>",
                "R4": "not_a_reaction",
            }
        )
        self.assertEqual(mapped["R1"], "mapped::CCO>>CC=O")
        self.assertIsNone(mapped["R2"])
        self.assertIsNone(mapped["R3"])
        self.assertIsNone(mapped["R4"])

        extractor_no_mapper = self.make_extractor(mapper_cls=None)
        self.assertEqual(
            extractor_no_mapper.atom_map_reactions({"R5": "CCO>>CC=O"}),
            {"R5": None},
        )

    def test_atom_map_reactions_handles_mapper_exception(self) -> None:
        extractor = self.make_extractor(mapper_cls=FailingMapper)
        mapped = extractor.atom_map_reactions({"R1": "CCO>>CC=O"})
        self.assertEqual(mapped, {"R1": None})

    def test_build_missing_compound_report(self) -> None:
        extractor = self.make_extractor()
        equations_by_rid = {
            "R00001": "C00001 + C00002 => C00003",
            "R00002": "C00003 <=> C00004",
            "R00003": None,
        }
        compounds_by_cid = {
            "C00001": {"name": "Water", "smiles": "O"},
            "C00004": {"name": "Unresolved", "smiles": None},
            "C00005": {"name": "Unused Missing", "smiles": None},
        }

        report = extractor.build_missing_compound_report(
            equations_by_rid, compounds_by_cid
        )

        self.assertEqual(report["missing_compound_ids"], ["C00004", "C00005"])
        self.assertEqual(report["reactions_involving_missing"], ["R00002"])
        self.assertEqual(report["missing_compounds"][0]["id"], "C00004")
        self.assertEqual(report["missing_compounds"][1]["reactions"], [])

    def test_build_kegg_json(self) -> None:
        extractor = self.make_extractor()
        data = extractor.build_kegg_json(
            {
                "R00002": "C00003 <=> C00004",
                "R00001": "C00001 + C00002 => C00003",
                "R00003": None,
            },
            smiles_by_rid={"R00001": "O.CCO>>CC=O"},
            rules_by_rid={"R00001": "mapped_rule"},
            molecules_by_cid={"C00001": {"name": "Water", "smiles": "O"}},
        )

        self.assertEqual([r["id"] for r in data["reactions"]], ["R00001", "R00002"])
        self.assertEqual(data["reactions"][0]["rule"], "mapped_rule")
        molecules = {m["id"]: m for m in data["molecules"]}
        self.assertEqual(molecules["C00001"]["name"], "Water")
        self.assertIsNone(molecules["C00002"]["name"])

    def test_build_module_json_with_compounds_and_save(self) -> None:
        extractor = self.make_extractor(mapper_cls=SuccessfulMapper)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "module.json"
            data = extractor.build_module_json(
                "M00001",
                with_compounds=True,
                with_atom_maps=True,
                save_as=str(out),
            )

            self.assertEqual(data["module_id"], "M00001")
            self.assertIn("missing", data)
            self.assertEqual(data["missing"]["missing_compound_ids"], ["C00004"])
            self.assertEqual(data["reactions"][0]["rule"], "mapped::O.CCO>>CC=O")
            self.assertTrue(out.exists())
            written = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(written["module_id"], "M00001")

    def test_build_module_json_without_compounds(self) -> None:
        extractor = self.make_extractor()
        data = extractor.build_module_json(
            "M00001",
            with_compounds=False,
            with_atom_maps=True,
        )
        self.assertEqual(
            data["missing"],
            {
                "missing_compounds": [],
                "missing_compound_ids": [],
                "reactions_involving_missing": [],
            },
        )
        self.assertEqual(data["reactions"][0]["smiles"], None)
        self.assertEqual(data["reactions"][0]["rule"], None)

    def test_build_pathway_json_with_compounds_and_save(self) -> None:
        extractor = self.make_extractor(mapper_cls=SuccessfulMapper)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "pathway.json"
            data = extractor.build_pathway_json(
                "hsa00010",
                with_compounds=True,
                with_atom_maps=False,
                save_as=str(out),
            )

            self.assertEqual(data["modules"], ["M00001", "M00002"])
            self.assertEqual(
                data["missing"]["missing_compound_ids"], ["C00004", "C00005", "C00006"]
            )
            self.assertEqual(
                data["missing"]["reactions_involving_missing"], ["R00002", "R00003"]
            )
            self.assertIn("M00001", data["by_module"])
            self.assertTrue(out.exists())
            written = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(written["pathway_id"], "hsa00010")

    def test_build_pathway_json_without_compounds(self) -> None:
        extractor = self.make_extractor()
        data = extractor.build_pathway_json(
            "hsa00010",
            with_compounds=False,
            with_atom_maps=True,
        )
        self.assertNotIn("missing", data)
        self.assertEqual(sorted(data["by_module"]), ["M00001", "M00002"])


if __name__ == "__main__":
    unittest.main()
