import unittest
from synkit.Chem.Reaction.neutralize import Neutralize


class TestNeutralize(unittest.TestCase):

    def test_calculate_charge(self):
        smiles = "[NH4+]"
        charge = Neutralize.calculate_charge(smiles)
        self.assertEqual(charge, 1)

        smiles = "[Cl-]"
        charge = Neutralize.calculate_charge(smiles)
        self.assertEqual(charge, -1)

        smiles = "[OH-].[Na+]"
        charge = Neutralize.calculate_charge(smiles)
        self.assertEqual(charge, 0)

        smiles = "[NH4]"  # can not parse
        charge = Neutralize.calculate_charge(smiles)
        self.assertEqual(charge, 0)

    def test_parse_reaction_success(self):
        """
        Test the parse_reaction method with a valid input.
        """
        reaction_smiles = "C(C)C>>CCO"
        expected_output = ("C(C)C", "CCO")
        result = Neutralize.parse_reaction(reaction_smiles)
        self.assertEqual(
            result,
            expected_output,
            "The parse_reaction method should correctly parse reactants and products.",
        )

    def test_parse_reaction_failure(self):
        """
        Test the parse_reaction method with an invalid input.
        """
        reaction_smiles = "C(C)C"
        expected_output = (None, None)
        result = Neutralize.parse_reaction(reaction_smiles)
        self.assertEqual(
            result,
            expected_output,
            (
                "The parse_reaction method should return None"
                + "for both reactants and products on failure."
            ),
        )

    def test_calculate_charge_dict(self):
        reaction = {"R-id": "R1", "reactions": "CCO>>CCO[O-]"}
        processed_reaction = Neutralize.calculate_charge_dict(reaction, "reactions")
        self.assertEqual(processed_reaction["total_charge_in_products"], -1)

        reaction = {"R-id": "R1", "reactions": "CCO>>CH4CH"}  # can not parse
        processed_reaction = Neutralize.calculate_charge_dict(reaction, "reactions")
        self.assertEqual(processed_reaction["total_charge_in_products"], 0)

    def test_fix_negative_charge(self):
        reaction_dict = {
            "reactants": "H2O",
            "products": "OH-",
            "total_charge_in_products": -1,
            "R-id": "R1",
            "label": "test_label",
        }
        expected = {
            "R-id": "R1",
            "reactions": "H2O.[Na+]>>OH-.[Na+]",
            "reactants": "H2O.[Na+]",
            "products": "OH-.[Na+]",
            "total_charge_in_products": 0,
        }
        results = Neutralize.fix_negative_charge(reaction_dict)
        self.assertEqual(results, expected)

    def test_fix_positive_charge(self):
        reaction_dict = {
            "reactants": "HCl",
            "products": "H+",
            "total_charge_in_products": 1,
            "R-id": "R2",
            "label": "test_label",
        }
        expected = {
            "R-id": "R2",
            "reactions": "HCl.[Cl-]>>H+.[Cl-]",
            "reactants": "HCl.[Cl-]",
            "products": "H+.[Cl-]",
            "total_charge_in_products": 0,
        }
        self.assertEqual(Neutralize.fix_positive_charge(reaction_dict), expected)

    def test_fix_unbalanced_charged_no_change(self):
        reaction_dict = {
            "reactions": "[Na]Cl>>[Na+].[Cl-]",
            "reactants": "NaCl",
            "products": "Na+ + Cl-",
            "total_charge_in_products": 0,
            "R-id": "R3",
            "label": "test_label",
        }

        self.assertEqual(
            Neutralize.fix_unbalanced_charged(reaction_dict, "reactions"), reaction_dict
        )

    def test_fix_unbalanced_charged_negative_change(self):
        reaction_dict = {
            "reactions": "CC(=O)[O-].[H+].[Cl-]>>CC(=O)O.[Cl-]",
            "R-id": "R3",
        }

        self.assertEqual(
            Neutralize.fix_unbalanced_charged(reaction_dict, "reactions")["reactions"],
            "CC(=O)[O-].[H+].[Cl-].[Na+]>>CC(=O)O.[Cl-].[Na+]",
        )

    def test_fix_unbalanced_charged_positive_change(self):
        reaction_dict = {
            "reactions": "CC(=O)[O-].[H+].[Na+]>>CC(=O)O.[Na+]",
            "R-id": "R3",
        }

        self.assertEqual(
            Neutralize.fix_unbalanced_charged(reaction_dict, "reactions")["reactions"],
            "CC(=O)[O-].[H+].[Na+].[Cl-]>>CC(=O)O.[Na+].[Cl-]",
        )

    def test_parallel_fix_unbalanced_charge(self):
        reaction_dicts = [
            {
                "reactions": "CC(=O)[O-].[H+].[Cl-]>>CC(=O)O.[Cl-]",
                "R-id": "R1",
            },
            {
                "reactions": "CC(=O)[O-].[H+].[Na+]>>CC(=O)O.[Na+]",
                "R-id": "R2",
            },
        ]
        fixed_reactions = Neutralize.parallel_fix_unbalanced_charge(
            reaction_dicts, "reactions"
        )
        print(fixed_reactions)
        self.assertEqual(len(fixed_reactions), len(reaction_dicts))
        self.assertEqual(
            fixed_reactions[0]["reactions"],
            "CC(=O)[O-].[H+].[Cl-].[Na+]>>CC(=O)O.[Cl-].[Na+]",
        )
        self.assertEqual(
            fixed_reactions[1]["reactions"],
            "CC(=O)[O-].[H+].[Na+].[Cl-]>>CC(=O)O.[Na+].[Cl-]",
        )


if __name__ == "__main__":
    unittest.main()
