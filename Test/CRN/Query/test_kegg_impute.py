from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from synkit.CRN.Query.kegg_extract import KEGGExtractor
from synkit.CRN.Query.kegg_impute import KEGGImputer


class TrackingExtractor(KEGGExtractor):
    def __init__(self) -> None:
        self.atom_map_calls = []
        self.saved_payloads = []

    def atom_map_reactions(self, reaction_smiles_by_id):
        self.atom_map_calls.append(dict(reaction_smiles_by_id))
        return {rid: f"rule::{rsmi}" for rid, rsmi in reaction_smiles_by_id.items()}

    def save_json(self, data, save_as):
        self.saved_payloads.append((data, save_as))
        super().save_json(data, save_as)


class TestKEGGImputer(unittest.TestCase):
    def make_module_data(self):
        return {
            "module_id": "M00001",
            "molecules": [
                {"id": "C00001", "name": "Water", "smiles": "O"},
                {"id": "C00002", "name": "Missing A", "smiles": None},
                {"id": "C00003", "name": "Ethanol", "smiles": "CCO"},
                {"name": "No ID should be ignored"},
            ],
            "reactions": [
                {
                    "id": "R00001",
                    "reaction": "C00001 + C00002 => C00003",
                    "smiles": ">>CCO",
                    "rule": None,
                },
                {
                    "id": "R00002",
                    "reaction": "C99999 => C00001",
                    "smiles": None,
                    "rule": None,
                },
                {"id": "R00003", "smiles": None, "rule": None},
                {"reaction": "C00002 => C00003", "smiles": None, "rule": None},
            ],
            "missing": {
                "missing_compounds": [
                    {"id": "C00002", "name": "Missing A", "reactions": ["R00001"]}
                ],
                "missing_compound_ids": ["C00002"],
                "reactions_involving_missing": ["R00001"],
            },
        }

    def test_post_init_builds_default_extractor(self):
        imputer = KEGGImputer()
        self.assertIsNotNone(imputer.extractor)
        self.assertIsInstance(imputer.extractor, KEGGExtractor)

    def test_restore_molecule_list_preserves_order_and_appends_new_ids(self):
        restored = KEGGImputer._restore_molecule_list(
            [{"id": "C00002"}, {"id": "C00001"}, {"x": 1}],
            {
                "C00001": {"id": "C00001", "name": "one"},
                "C00002": {"id": "C00002", "name": "two"},
                "C00003": {"id": "C00003", "name": "three"},
            },
        )
        self.assertEqual([m["id"] for m in restored], ["C00002", "C00001", "C00003"])

    def test_infer_impacted_reaction_ids_prefers_missing_block(self):
        reactions = [{"id": "R1", "reaction": "C00002 => C00003"}]
        missing_block = {
            "missing_compounds": [
                {"id": "C00002", "reactions": ["R_missing"]},
                {"id": "C99999", "reactions": ["R_other"]},
            ]
        }
        impacted = KEGGImputer._infer_impacted_reaction_ids(
            reactions,
            missing_block,
            {"C00002"},
            reaction_id_key="id",
            equation_key="reaction",
        )
        self.assertEqual(impacted, {"R_missing"})

    def test_infer_impacted_reaction_ids_falls_back_to_equation_scan(self):
        reactions = [
            {"id": "R1", "reaction": "C00002 => C00003"},
            {"id": "R2", "reaction": "C00005 => C00006"},
            {"id": None, "reaction": "C00002 => C00007"},
        ]
        impacted = KEGGImputer._infer_impacted_reaction_ids(
            reactions,
            {},
            {"C00002"},
            reaction_id_key="id",
            equation_key="reaction",
        )
        self.assertEqual(impacted, {"R1"})

    def test_rebuild_reaction_fields_updates_only_impacted_reactions(self):
        extractor = TrackingExtractor()
        imputer = KEGGImputer(extractor=extractor)
        reactions = [
            {
                "id": "R00001",
                "reaction": "C00001 + C00002 => C00003",
                "smiles": None,
                "rule": None,
            },
            {
                "id": "R00002",
                "reaction": "C00003 => C00001",
                "smiles": "old",
                "rule": "oldrule",
            },
            {"id": "R00003", "smiles": None, "rule": None},
        ]
        molecules = [
            {"id": "C00001", "smiles": "O"},
            {"id": "C00002", "smiles": "C=C"},
            {"id": "C00003", "smiles": "CCO"},
        ]

        imputer._rebuild_reaction_fields(
            reactions, molecules, {"R00001", "R_missing", "R00003"}
        )

        self.assertEqual(reactions[0]["smiles"], "O.C=C>>CCO")
        self.assertEqual(reactions[0]["rule"], "rule::O.C=C>>CCO")
        self.assertEqual(reactions[1]["smiles"], "old")
        self.assertEqual(reactions[1]["rule"], "oldrule")
        self.assertEqual(reactions[2]["smiles"], None)
        self.assertEqual(extractor.atom_map_calls, [{"R00001": "O.C=C>>CCO"}])

    def test_rebuild_reaction_fields_returns_early_when_no_impacted_ids(self):
        extractor = TrackingExtractor()
        imputer = KEGGImputer(extractor=extractor)
        reactions = [
            {"id": "R1", "reaction": "C00001 => C00002", "smiles": None, "rule": None}
        ]
        molecules = [{"id": "C00001", "smiles": "O"}, {"id": "C00002", "smiles": "N"}]
        imputer._rebuild_reaction_fields(reactions, molecules, set())
        self.assertEqual(extractor.atom_map_calls, [])
        self.assertIsNone(reactions[0]["smiles"])

    def test_impute_module_updates_existing_and_new_molecules_and_saves_json(self):
        extractor = TrackingExtractor()
        imputer = KEGGImputer(extractor=extractor)
        module_data = self.make_module_data()

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "module.json"
            updated = imputer.impute_module(
                module_data,
                [
                    {"id": "C00002", "name": "Ethylen", "smiles": "C=C"},
                    {"id": "C00004", "name": "Methane", "smiles": "C"},
                    {"name": "ignored"},
                ],
                save_as=str(out),
            )

            self.assertEqual(updated["molecules"][1]["name"], "Ethylen")
            self.assertEqual(updated["molecules"][1]["smiles"], "C=C")
            self.assertEqual(
                updated["molecules"][-1],
                {"id": "C00004", "name": "Methane", "smiles": "C"},
            )
            self.assertEqual(updated["reactions"][0]["smiles"], "O.C=C>>CCO")
            self.assertEqual(updated["reactions"][0]["rule"], "rule::O.C=C>>CCO")
            self.assertEqual(updated["missing"]["missing_compound_ids"], [])
            self.assertTrue(out.exists())
            saved = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(saved["missing"]["missing_compound_ids"], [])
            self.assertEqual(len(extractor.saved_payloads), 1)
            self.assertEqual(extractor.saved_payloads[0][1], str(out))

        self.assertIsNone(module_data["molecules"][1]["smiles"])
        self.assertEqual(module_data["reactions"][0]["smiles"], ">>CCO")

    def test_impute_module_without_name_keeps_existing_name_and_reports_remaining_missing(
        self,
    ):
        extractor = TrackingExtractor()
        imputer = KEGGImputer(extractor=extractor)
        module_data = self.make_module_data()
        module_data["molecules"].append(
            {"id": "C99999", "name": "Unknown", "smiles": None}
        )

        updated = imputer.impute_module(
            module_data,
            [{"id": "C00002", "smiles": "C=C"}],
            save_as=None,
        )

        self.assertEqual(updated["molecules"][1]["name"], "Missing A")
        self.assertEqual(updated["missing"]["missing_compound_ids"], ["C99999"])
        self.assertEqual(updated["missing"]["reactions_involving_missing"], ["R00002"])
        self.assertEqual(extractor.saved_payloads[0][1], None)

    def test_impute_pathway_updates_each_module_and_aggregates_missing(self):
        extractor = TrackingExtractor()
        imputer = KEGGImputer(extractor=extractor)
        pathway = {
            "pathway_id": "hsa00010",
            "by_module": {
                "M1": self.make_module_data(),
                "M2": {
                    "molecules": [
                        {"id": "C00005", "name": "A", "smiles": None},
                        {"id": "C00006", "name": "B", "smiles": "O"},
                    ],
                    "reactions": [
                        {
                            "id": "R10000",
                            "reaction": "C00005 => C00006",
                            "smiles": None,
                            "rule": None,
                        }
                    ],
                    "missing": {
                        "missing_compounds": [
                            {"id": "C00005", "name": "A", "reactions": ["R10000"]}
                        ],
                        "missing_compound_ids": ["C00005"],
                        "reactions_involving_missing": ["R10000"],
                    },
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "pathway.json"
            updated = imputer.impute_pathway(
                pathway,
                [{"id": "C00002", "smiles": "N"}],
                save_as=str(out),
            )
            self.assertTrue(out.exists())
            saved = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(saved["missing"]["missing_compound_ids"], ["C00005"])

        self.assertEqual(
            updated["by_module"]["M1"]["missing"]["missing_compound_ids"], []
        )
        self.assertEqual(
            updated["by_module"]["M2"]["missing"]["missing_compound_ids"], ["C00005"]
        )
        self.assertEqual(updated["missing"]["missing_compound_ids"], ["C00005"])
        self.assertEqual(updated["missing"]["reactions_involving_missing"], ["R10000"])
        self.assertEqual(len(extractor.saved_payloads), 3)

    def test_impute_module_reaction_fix_rebuilds_smiles_and_rule(self):
        extractor = TrackingExtractor()
        imputer = KEGGImputer(extractor=extractor)
        module_data = {
            "module_id": "M1",
            "molecules": [
                {"id": "C00404", "smiles": "OP"},
                {"id": "C00267", "smiles": "N"},
                {"id": "C00668", "smiles": "O"},
            ],
            "reactions": [
                {
                    "id": "R02189",
                    "reaction": "C00404 + C00267 <=> C00404 + C00668",
                    "smiles": "old",
                    "rule": "old",
                }
            ],
            "missing": {
                "missing_compounds": [],
                "missing_compound_ids": [],
                "reactions_involving_missing": [],
            },
        }

        updated = imputer.impute_module(
            module_data,
            fixes=[
                {"id": "R02189", "reaction": "C00404 + C00267 <=> C99999 + C00668"},
                {"id": "C99999", "smiles": "P(=O)(O)O"},
            ],
        )

        rxn = updated["reactions"][0]
        self.assertEqual(rxn["smiles"], "OP.N>>P(=O)(O)O.O")
        self.assertEqual(rxn["rule"], "rule::OP.N>>P(=O)(O)O.O")
        self.assertEqual(updated["missing"]["missing_compound_ids"], [])


if __name__ == "__main__":
    unittest.main()
