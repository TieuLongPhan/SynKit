import unittest
from rxnmapper import RXNMapper
from synkit.Chem.Reaction.reagent import Reagent


class TestReagent(unittest.TestCase):

    def setUp(self) -> None:
        """
        Set up the necessary instances for testing.
        """
        self.reaction_smiles = "C=O.N#[C-]~[K+].O=C(Cl)c1ccccc1>>N#CC(O)C(=O)c1ccccc1"
        self.mapped_reaction = "[CH2:3]=[O:4].[N:1]#[C-:2]~[K+].[O:6]=[C:5](Cl)[c:7]1[cH:8][cH:9][cH:10][cH:11][cH:12]1>>[N:1]#[C:2][CH:3]([OH:4])[C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][cH:11][cH:12]1"
        self.rsmi = "C1CCCCC1.CCO.CS(=O)(=O)N1CCN(Cc2ccccc2)CC1.[OH-].[OH-].[Pd+2]>>CS(=O)(=O)N1CCNCC1"
        self.rxn_mapper = RXNMapper()
        self.reagent_class = Reagent()

    def test_map_with_rxn_mapper_success(self):
        """
        Test that the map_with_rxn_mapper method successfully maps a reaction.
        """

        result = self.reagent_class.map_with_rxn_mapper(
            self.reaction_smiles, self.rxn_mapper
        )
        self.assertEqual(result, self.mapped_reaction)

    def test_map_with_rxn_mapper_fail(self):
        """
        Test that the map_with_rxn_mapper method handles invalid SMILES input.
        """
        fail_reaction = "fail>>C"
        result = self.reagent_class.map_with_rxn_mapper(fail_reaction)
        self.assertEqual(result, fail_reaction)

    def test_clean_reagents_success_with_clean_reaction(self):
        """
        Test that clean_reagents successfully identifies reagents and returns the cleaned reaction.
        """

        clean_reaction, reagents = self.reagent_class.clean_reagents(
            self.rsmi, return_clean_reaction=True
        )

        self.assertIsNotNone(clean_reaction, "Clean reaction should not be None.")
        self.assertIsNotNone(reagents, "Reagents should not be None.")
        print(clean_reaction)
        self.assertEqual(
            clean_reaction,
            "CS(=O)(=O)N1CCN(Cc2ccccc2)CC1>>CS(=O)(=O)N1CCNCC1",
            "The cleaned reaction should only contain the main product.",
        )
        self.assertEqual(
            reagents,
            "C1CCCCC1.CCO.[OH-].[OH-].[Pd+2]",
            "Reagents should contain the unmapped reactants.",
        )

    def test_clean_reagents_success_only_reagents(self):
        """
        Test that clean_reagents successfully identifies reagents when only reagents are requested.
        """

        clean_reaction, reagents = self.reagent_class.clean_reagents(
            self.rsmi, return_clean_reaction=False
        )

        self.assertIsNone(
            clean_reaction,
            "Clean reaction should be None when only reagents are requested.",
        )
        self.assertEqual(
            reagents,
            "C1CCCCC1.CCO.[OH-].[OH-].[Pd+2]",
            "Reagents should contain the unmapped reactants.",
        )

    def test_clean_reagents_fail_invalid_reaction(self):
        """
        Test that clean_reagents returns None for both clean reaction and reagents when an invalid reaction is passed.
        """
        invalid_reaction_smiles = "INVALID>>CCOC(C)=O"
        clean_reaction, reagents = self.reagent_class.clean_reagents(
            invalid_reaction_smiles, return_clean_reaction=True
        )

        self.assertIsNone(
            clean_reaction, "Clean reaction should be None for invalid input."
        )
        self.assertEqual(
            reagents,
            ".".join(invalid_reaction_smiles.split(">>")),
            "Reagents should be equal to original invalid input.",
        )


if __name__ == "__main__":
    unittest.main()
