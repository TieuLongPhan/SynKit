import unittest
from synkit.Chem.Fingerprint.fp_calculator import FPCalculator


class TestFPCalculator(unittest.TestCase):
    def setUp(self):
        # Sample data setup
        self.data = [
            {
                "smiles": [
                    (
                        "C1CCCCC1.CCO.CS(=O)(=O)N1CCN(Cc2ccccc2)CC1.[OH-].[OH-].[Pd+2]"
                        + ">>CS(=O)(=O)N1CCNCC1"
                    ),
                    (
                        "CCOC(C)=O.Cc1cc([N+](=O)[O-])ccc1NC(=O)c1ccccc1.Cl[Sn]Cl.O.O.O=C([O-])O.[Na+]"
                        + ">>Cc1cc(N)ccc1NC(=O)c1ccccc1"
                    ),
                    (
                        "COc1ccc(-c2coc3ccc(-c4nnc(S)o4)cc23)cc1.COc1ccc(CCl)cc1F"
                        + ">>COc1ccc(-c2coc3ccc(-c4nnc(SCc5ccc(OC)c(F)c5)o4)cc23)cc1"
                    ),
                ],
                "ID": [1, 2, 3],
            }
        ]
        self.smiles_key = "smiles"
        self.fp_type = "drfp"
        self.n_jobs = 2
        self.verbose = 0

        # Instantiate the FPCalculator
        self.fp_calculator = FPCalculator(
            smiles_key=self.smiles_key,
            fp_type=self.fp_type,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

    def test_init_invalid_fp_type(self):
        with self.assertRaises(ValueError):
            FPCalculator(smiles_key=self.smiles_key, fp_type="invalid_type")

    def test_fit_missing_column(self):
        with self.assertRaises(ValueError):
            fp_calculator = FPCalculator(
                smiles_key=self.smiles_key,
                fp_type=self.fp_type,
            )
            fp_calculator.dict_process({"not_smiles": ["C"]}, "smiles")

    def test_constructor_and_attribute_assignment(self):
        self.assertEqual(self.fp_calculator.smiles_key, "smiles")
        self.assertEqual(self.fp_calculator.fp_type, "drfp")
        self.assertEqual(self.fp_calculator.n_jobs, 2)


if __name__ == "__main__":
    unittest.main()
