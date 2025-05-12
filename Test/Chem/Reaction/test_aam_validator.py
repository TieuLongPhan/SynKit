import unittest
from synkit.Chem.Reaction.aam_validator import AAMValidator


class TestAMMValidator(unittest.TestCase):

    def setUp(self):
        self.true_pair = (
            (
                "[CH:8]=1[S:9][CH:10]=[C:6]([C:5]#[C:4][CH2:3][N:2]([C:11]2=[CH:12]"
                + "[CH:13]=[CH:14][CH:15]=[CH:16]2)[CH3:1])[CH:7]=1.[OH2:17]>>[C:5]([N:2]"
                + "([CH3:1])[C:11]1=[CH:12][CH:13]=[CH:14][CH:15]=[CH:16]1)([C:6]2="
                + "[CH:10][S:9][CH:8]=[CH:7]2)=[CH:4][CH:3]=[O:17]"
            ),
            (
                "[OH2:17].[cH:12]1[cH:13][cH:14][cH:15][cH:16][c:11]1[N:2]([CH3:1])"
                + "[CH2:3][C:4]#[C:5][c:6]1[cH:10][s:9][cH:8][cH:7]1>>[cH:12]1[cH:13]"
                + "[cH:14][cH:15][cH:16][c:11]1[N:2]([CH3:1])[C:5](=[CH:4][CH:3]=[O:17])"
                + "[c:6]1[cH:10][s:9][cH:8][cH:7]1"
            ),
        )
        self.false_pair = (
            (
                "[CH:8]=1[S:9][CH:10]=[C:6]([C:5]#[C:4][CH2:3][N:2]([C:11]2=[CH:12]"
                + "[CH:13]=[CH:14][CH:15]=[CH:16]2)[CH3:1])[CH:7]=1.[OH2:17]>>[C:5]"
                + "([N:2]([CH3:1])[C:11]1=[CH:12][CH:13]=[CH:14][CH:15]=[CH:16]1)"
                + "([C:6]2=[CH:10][S:9][CH:8]=[CH:7]2)=[CH:4][CH:3]=[O:17]"
            ),
            (
                "[CH3:1][N:2]([CH2:3][C:4]#[C:5][c:7]1[cH:8][cH:9][s:10][cH:11]1)"
                + "[c:12]1[cH:13][cH:14][cH:15][cH:16][cH:17]1.[OH2:6]>>[CH3:1][N:2]"
                + "([C:3](=[CH:4][CH:5]=[O:6])[c:7]1[cH:8][cH:9][s:10][cH:11]1)"
                + "[c:12]1[cH:13][cH:14][cH:15][cH:16][cH:17]1"
            ),
        )
        self.tautomer = (
            "[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][CH2:6][OH:7]>>[CH3:1][C:2](=[O:3])"
            + "[O:7][CH2:6][CH3:5].[OH2:4]",
            "[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][CH2:6][OH:7]>>"
            + "[CH3:1][C:2](=[O:4])[O:7][CH2:6][CH3:5].[OH2:3]",
        )

        self.data_dict_1 = {"ref": self.true_pair[0], "map": self.true_pair[1]}
        self.data_dict_2 = {"ref": self.false_pair[0], "map": self.false_pair[1]}
        self.data_dict_3 = {"ref": self.tautomer[0], "map": self.tautomer[1]}
        self.data = [self.data_dict_1, self.data_dict_2, self.data_dict_3]

    def test_smiles_check(self):

        self.assertTrue(
            AAMValidator.smiles_check(
                *self.true_pair, check_method="RC", ignore_aromaticity=False
            )
        )
        self.assertFalse(
            AAMValidator.smiles_check(
                *self.false_pair, check_method="RC", ignore_aromaticity=False
            )
        )

    def test_smiles_check_tautomer(self):
        self.assertFalse(
            AAMValidator.smiles_check(
                self.tautomer[0],
                self.tautomer[1],
                check_method="RC",
                ignore_aromaticity=False,
            )
        )

        self.assertTrue(
            AAMValidator.smiles_check_tautomer(
                self.tautomer[0],
                self.tautomer[1],
                check_method="RC",
                ignore_aromaticity=True,
            )
        )

    def test_validate_smiles_dataframe(self):

        results = AAMValidator.validate_smiles(
            data=self.data,
            ground_truth_col="ref",
            mapped_cols=["map"],
            check_method="RC",
            ignore_aromaticity=False,
            n_jobs=2,
            verbose=0,
            ignore_tautomers=False,
        )
        self.assertEqual(results[0]["accuracy"], 66.67)
        self.assertEqual(results[0]["success_rate"], 100)


if __name__ == "__main__":
    unittest.main()
