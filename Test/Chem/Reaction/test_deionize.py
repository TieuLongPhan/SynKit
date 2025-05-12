import unittest
from synkit.Chem.Reaction.deionize import (
    Deionize,
)


class TestDeionize(unittest.TestCase):

    def test_uncharge_anion(self):
        smiles = "[OH-]"
        uncharged_smiles = Deionize.uncharge_anion(smiles)
        self.assertEqual(uncharged_smiles, "O")

    def test_uncharge_cation(self):
        smiles = "[NH4+]"
        uncharged_smiles = Deionize.uncharge_cation(smiles)
        self.assertEqual(
            uncharged_smiles, "[NH4]"
        )  # can uncharge to [NH4] but will not pass through

    def test_ammonia_hydroxide_standardize(self):
        reaction_smiles = "[NH4+].[OH-]>>N.O"
        standardized_smiles = Deionize.ammonia_hydroxide_standardize(reaction_smiles)
        self.assertTrue("N.O" in standardized_smiles or "O.N" in standardized_smiles)

    def test_not_uncharge_smiles(self):
        charge_smiles = "[NH4+].[OH-]"
        uncharged_smiles = Deionize.uncharge_smiles(charge_smiles)
        self.assertTrue("N" in uncharged_smiles and "O" in uncharged_smiles)
        self.assertTrue(
            uncharged_smiles in ["[NH4+].[OH-]", "[OH-].[NH4+]"]
        )  # can not uncharge amonium

    def test_uncharge_smiles(self):
        charge_smiles = "[Na+].[OH-]"
        uncharged_smiles = Deionize.uncharge_smiles(charge_smiles)
        self.assertEqual(uncharged_smiles, "[OH][Na]")

    def test_apply_uncharge_smiles_to_reactions(self):
        reactions = [{"reactants": "[NH4+].[OH-]", "products": "N.O"}]
        uncharged_reactions = Deionize.apply_uncharge_smiles_to_reactions(
            reactions, Deionize.uncharge_smiles
        )
        for reaction in uncharged_reactions:
            self.assertTrue(
                "N" in reaction["new_reactants"] and "O" in reaction["new_reactants"]
            )
            self.assertTrue(reaction["success"])


if __name__ == "__main__":
    unittest.main()
