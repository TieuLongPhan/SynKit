from __future__ import annotations

import unittest

from rdkit import Chem

from synkit.CRN.Query.kegg_parse import (
    KEGGEquation,
    get_compound_ids_from_equations,
    expand_stoichiometry,
    get_compound_ids_from_text,
    molblock_to_smiles,
    normalize_module_id,
    parse_side,
    parse_equation,
    parse_kegg_field_blocks,
    reaction_smiles_from_equation,
)


class TestKEGGParsing(unittest.TestCase):
    def test_parse_kegg_field_blocks_with_continuations_and_multiple_occurrences(
        self,
    ) -> None:
        text = (
            "ENTRY       map00010\n"
            "MODULE      M00001 Glycolysis core\n"
            "            additional description\n"
            "NAME        Example pathway\n"
            "MODULE      M00002 Pentose phosphate\n"
            "\tcontinued module text\n"
        )

        payloads = parse_kegg_field_blocks(text, "MODULE")

        self.assertEqual(
            payloads,
            [
                "M00001 Glycolysis core additional description",
                "M00002 Pentose phosphate continued module text",
            ],
        )

    def test_parse_kegg_field_blocks_returns_empty_when_field_missing(self) -> None:
        text = "ENTRY       map00010\nNAME        Example\n"
        self.assertEqual(parse_kegg_field_blocks(text, "REACTION"), [])

    def test_normalize_module_id(self) -> None:
        self.assertEqual(normalize_module_id("hsa_M00001"), "M00001")
        self.assertEqual(normalize_module_id("M12345"), "M12345")
        self.assertIsNone(normalize_module_id("not-a-module"))

    def test_parse_side_parses_coefficients(self) -> None:
        parsed = parse_side("2 C00139 + C00001")
        self.assertEqual(parsed, [("C00139", 2), ("C00001", 1)])

    def test_parse_side_handles_empty_side(self) -> None:
        self.assertEqual(parse_side("   "), [])

    def test_parse_side_raises_on_invalid_term(self) -> None:
        with self.assertRaises(ValueError):
            parse_side("ATP + C00001")

    def test_parse_equation_reversible(self) -> None:
        parsed = parse_equation("C00001 + C00002 <=> C00003")
        self.assertIsInstance(parsed, KEGGEquation)
        self.assertEqual(parsed.reactants, [("C00001", 1), ("C00002", 1)])
        self.assertEqual(parsed.products, [("C00003", 1)])
        self.assertTrue(parsed.reversible)

    def test_parse_equation_forward_irreversible(self) -> None:
        parsed = parse_equation("C00001 => 2 C00002")
        self.assertEqual(parsed.reactants, [("C00001", 1)])
        self.assertEqual(parsed.products, [("C00002", 2)])
        self.assertFalse(parsed.reversible)

    def test_parse_equation_reverse_irreversible(self) -> None:
        parsed = parse_equation("C00003 <= C00001 + C00002")
        self.assertEqual(parsed.reactants, [("C00001", 1), ("C00002", 1)])
        self.assertEqual(parsed.products, [("C00003", 1)])
        self.assertFalse(parsed.reversible)

    def test_parse_equation_raises_on_unknown_arrow(self) -> None:
        with self.assertRaises(ValueError):
            parse_equation("C00001 -> C00002")

    def test_get_compound_ids_from_equations(self) -> None:
        equations = {
            "R00001": "C00001 + 2 C00002 => C00003",
            "R00002": None,
            "R00003": "C00003 <=> C00004 + C00001",
            "R00004": "",
        }

        compound_ids, parsed_by_rid = get_compound_ids_from_equations(equations)

        self.assertEqual(compound_ids, ["C00001", "C00002", "C00003", "C00004"])
        self.assertEqual(sorted(parsed_by_rid), ["R00001", "R00003"])
        self.assertEqual(parsed_by_rid["R00001"].products, [("C00003", 1)])

    def test_get_compound_ids_from_text(self) -> None:
        text = "C00001 appears twice: C00001, plus C00002 and noise."
        self.assertEqual(get_compound_ids_from_text(text), ["C00001", "C00002"])
        self.assertEqual(get_compound_ids_from_text(""), [])

    def test_molblock_to_smiles_success(self) -> None:
        molecule = Chem.MolFromSmiles("CCO")
        assert molecule is not None
        molblock = Chem.MolToMolBlock(molecule)

        smiles = molblock_to_smiles(molblock)

        self.assertEqual(smiles, Chem.MolToSmiles(molecule))

    def test_molblock_to_smiles_none_and_invalid(self) -> None:
        self.assertIsNone(molblock_to_smiles(None))
        self.assertIsNone(molblock_to_smiles("this is not a mol block"))

    def test_expand_stoichiometry(self) -> None:
        expanded = expand_stoichiometry([("C00001", 2), ("C00002", 1), ("C00003", 0)])
        self.assertEqual(expanded, ["C00001", "C00001", "C00002"])

    def test_reaction_smiles_from_equation_complete(self) -> None:
        parsed = KEGGEquation(
            reactants=[("C00001", 2), ("C00002", 1)],
            products=[("C00003", 1)],
            reversible=False,
        )
        compounds_by_cid = {
            "C00001": {"smiles": "O"},
            "C00002": {"smiles": "CCO"},
            "C00003": {"smiles": "CC=O"},
        }

        reaction_smiles, missing = reaction_smiles_from_equation(
            parsed, compounds_by_cid
        )

        self.assertEqual(reaction_smiles, "O.O.CCO>>CC=O")
        self.assertEqual(missing, {"reactants": [], "products": []})

    def test_reaction_smiles_from_equation_with_missing_compounds(self) -> None:
        parsed = KEGGEquation(
            reactants=[("C00001", 1), ("C00002", 2)],
            products=[("C00003", 1), ("C00004", 1)],
            reversible=False,
        )
        compounds_by_cid = {
            "C00001": {"smiles": "O"},
            "C00003": {"smiles": "CC=O"},
            "C00004": {},
        }

        reaction_smiles, missing = reaction_smiles_from_equation(
            parsed, compounds_by_cid
        )

        self.assertEqual(reaction_smiles, "O>>CC=O")
        self.assertEqual(missing["reactants"], ["C00002", "C00002"])
        self.assertEqual(missing["products"], ["C00004"])


if __name__ == "__main__":
    unittest.main()
