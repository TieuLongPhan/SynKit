from __future__ import annotations

import unittest

from rdkit import Chem

from synkit.CRN.Query.kegg_parse import (
    KEGGEquation,
    equation_to_text,
    expand_stoichiometry,
    get_compound_ids_from_equations,
    get_compound_ids_from_text,
    molblock_to_smiles,
    normalize_module_id,
    orient_equation_to_module,
    parse_equation,
    parse_kegg_field_blocks,
    parse_module_reaction_directions,
    parse_side,
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

    def test_parse_kegg_field_blocks_stops_at_next_non_continuation_field(self) -> None:
        text = (
            "MODULE      M00001 First module\n"
            "            continued text\n"
            "NAME        Example name\n"
            "            name continuation\n"
        )

        payloads = parse_kegg_field_blocks(text, "MODULE")

        self.assertEqual(payloads, ["M00001 First module continued text"])

    def test_normalize_module_id_finds_embedded_identifier(self) -> None:
        self.assertEqual(
            normalize_module_id("prefix text M00042 suffix text"),
            "M00042",
        )

    def test_parse_side_handles_extra_spacing(self) -> None:
        parsed = parse_side("  3 C00001   +   C00002  +  2 C00003 ")
        self.assertEqual(
            parsed,
            [("C00001", 3), ("C00002", 1), ("C00003", 2)],
        )

    def test_parse_side_accepts_zero_coefficient(self) -> None:
        parsed = parse_side("0 C00001 + C00002")
        self.assertEqual(parsed, [("C00001", 0), ("C00002", 1)])

    def test_parse_equation_reversible_alternate_arrow(self) -> None:
        parsed = parse_equation("C00001 <-> C00002 + 2 C00003")
        self.assertEqual(parsed.reactants, [("C00001", 1)])
        self.assertEqual(parsed.products, [("C00002", 1), ("C00003", 2)])
        self.assertTrue(parsed.reversible)

    def test_parse_equation_forward_short_arrow(self) -> None:
        parsed = parse_equation("C00001 -> C00002")
        self.assertEqual(parsed.reactants, [("C00001", 1)])
        self.assertEqual(parsed.products, [("C00002", 1)])
        self.assertFalse(parsed.reversible)

    def test_parse_equation_reverse_short_arrow(self) -> None:
        parsed = parse_equation("C00002 <- C00001")
        self.assertEqual(parsed.reactants, [("C00001", 1)])
        self.assertEqual(parsed.products, [("C00002", 1)])
        self.assertFalse(parsed.reversible)

    def test_parse_equation_raises_on_unknown_arrow(self) -> None:
        with self.assertRaises(ValueError):
            parse_equation("C00001 = C00002")

    def test_parse_equation_handles_empty_left_side(self) -> None:
        parsed = parse_equation("=> C00001")
        self.assertEqual(parsed.reactants, [])
        self.assertEqual(parsed.products, [("C00001", 1)])
        self.assertFalse(parsed.reversible)

    def test_get_compound_ids_from_equations_propagates_parse_error(self) -> None:
        equations = {
            "R00001": "C00001 => C00002",
            "R00002": "C00003 = C00004",
        }

        with self.assertRaises(ValueError):
            get_compound_ids_from_equations(equations)

    def test_molblock_to_smiles_empty_string(self) -> None:
        self.assertIsNone(molblock_to_smiles(""))

    def test_expand_stoichiometry_empty_input(self) -> None:
        self.assertEqual(expand_stoichiometry([]), [])

    def test_reaction_smiles_from_equation_empty_both_sides(self) -> None:
        parsed = KEGGEquation(reactants=[], products=[], reversible=False)
        reaction_smiles, missing = reaction_smiles_from_equation(parsed, {})

        self.assertEqual(reaction_smiles, ">>")
        self.assertEqual(missing, {"reactants": [], "products": []})

    def test_reaction_smiles_from_equation_missing_duplicates_are_preserved(
        self,
    ) -> None:
        parsed = KEGGEquation(
            reactants=[("C00001", 3)],
            products=[("C00002", 2)],
            reversible=False,
        )

        reaction_smiles, missing = reaction_smiles_from_equation(parsed, {})

        self.assertEqual(reaction_smiles, ">>")
        self.assertEqual(missing["reactants"], ["C00001", "C00001", "C00001"])
        self.assertEqual(missing["products"], ["C00002", "C00002"])

    def test_parse_module_reaction_directions_single_and_multiple_rids(self) -> None:
        text = (
            "ENTRY       M00001\n"
            "REACTION    R00001 C00001 + C00002 -> C00003\n"
            "            R00002,R00003 C00003 + C00004 <=> C00005 + C00006\n"
            "NAME        Example module\n"
        )

        directions = parse_module_reaction_directions(text)

        self.assertEqual(
            directions["R00001"],
            (["C00001", "C00002"], ["C00003"], "->"),
        )
        self.assertEqual(
            directions["R00002"],
            (["C00003", "C00004"], ["C00005", "C00006"], "<=>"),
        )
        self.assertEqual(
            directions["R00003"],
            (["C00003", "C00004"], ["C00005", "C00006"], "<=>"),
        )

    def test_parse_module_reaction_directions_stops_after_reaction_block(self) -> None:
        text = (
            "ENTRY       M00001\n"
            "REACTION    R00001 C00001 -> C00002\n"
            "NAME        Example module\n"
            "            R99999 C00003 -> C00004\n"
        )

        directions = parse_module_reaction_directions(text)

        self.assertEqual(list(directions), ["R00001"])
        self.assertNotIn("R99999", directions)

    def test_parse_module_reaction_directions_ignores_malformed_lines(self) -> None:
        text = (
            "ENTRY       M00001\n"
            "REACTION    malformed text without proper reaction id\n"
            "            R00001 C00001 + C00002 => C00003\n"
        )

        directions = parse_module_reaction_directions(text)

        self.assertEqual(
            directions,
            {"R00001": (["C00001", "C00002"], ["C00003"], "=>")},
        )

    def test_orient_equation_to_module_keeps_orientation_when_already_matching(
        self,
    ) -> None:
        parsed = KEGGEquation(
            reactants=[("C00001", 1), ("C00002", 1)],
            products=[("C00003", 1)],
            reversible=True,
        )

        oriented = orient_equation_to_module(
            parsed,
            left_ids=["C00001", "C00002"],
            right_ids=["C00003"],
        )

        self.assertEqual(oriented.reactants, [("C00001", 1), ("C00002", 1)])
        self.assertEqual(oriented.products, [("C00003", 1)])
        self.assertTrue(oriented.reversible)

    def test_orient_equation_to_module_flips_orientation_when_module_matches_reverse(
        self,
    ) -> None:
        parsed = KEGGEquation(
            reactants=[("C00001", 1), ("C00002", 1)],
            products=[("C00003", 1)],
            reversible=True,
        )

        oriented = orient_equation_to_module(
            parsed,
            left_ids=["C00003"],
            right_ids=["C00001", "C00002"],
        )

        self.assertEqual(oriented.reactants, [("C00003", 1)])
        self.assertEqual(oriented.products, [("C00001", 1), ("C00002", 1)])
        self.assertTrue(oriented.reversible)

    def test_orient_equation_to_module_tie_keeps_original_orientation(self) -> None:
        parsed = KEGGEquation(
            reactants=[("C00001", 1)],
            products=[("C00002", 1)],
            reversible=False,
        )

        oriented = orient_equation_to_module(
            parsed,
            left_ids=["C00001", "C00002"],
            right_ids=[],
        )

        self.assertEqual(oriented.reactants, [("C00001", 1)])
        self.assertEqual(oriented.products, [("C00002", 1)])

    def test_equation_to_text_default_irreversible_arrow(self) -> None:
        parsed = KEGGEquation(
            reactants=[("C00001", 2), ("C00002", 1)],
            products=[("C00003", 1)],
            reversible=False,
        )

        self.assertEqual(
            equation_to_text(parsed),
            "2 C00001 + C00002 => C00003",
        )

    def test_equation_to_text_default_reversible_arrow(self) -> None:
        parsed = KEGGEquation(
            reactants=[("C00001", 1)],
            products=[("C00002", 1)],
            reversible=True,
        )

        self.assertEqual(equation_to_text(parsed), "C00001 <=> C00002")

    def test_equation_to_text_normalizes_module_forward_arrow(self) -> None:
        parsed = KEGGEquation(
            reactants=[("C00001", 1)],
            products=[("C00002", 1)],
            reversible=False,
        )

        self.assertEqual(equation_to_text(parsed, arrow="->"), "C00001 => C00002")

    def test_equation_to_text_normalizes_module_reverse_arrow(self) -> None:
        parsed = KEGGEquation(
            reactants=[("C00001", 1)],
            products=[("C00002", 1)],
            reversible=False,
        )

        self.assertEqual(equation_to_text(parsed, arrow="<-"), "C00001 <= C00002")

    def test_equation_to_text_forces_explicit_arrow_override(self) -> None:
        parsed = KEGGEquation(
            reactants=[("C00001", 1)],
            products=[("C00002", 2)],
            reversible=True,
        )

        self.assertEqual(equation_to_text(parsed, arrow="=>"), "C00001 => 2 C00002")


if __name__ == "__main__":
    unittest.main()
