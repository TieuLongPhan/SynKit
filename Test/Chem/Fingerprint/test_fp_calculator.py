import io
import unittest
from contextlib import redirect_stdout

from synkit.Chem.Fingerprint.fp_calculator import FPCalculator


class TestFPCalculator(unittest.TestCase):
    def setUp(self):
        # Sample single reaction dict
        self.single = {"rsmi": "CCO>>CC=O"}
        # List of dicts for parallel
        self.batch = [
            {"rsmi": "CCO>>CC=O"},
            {"rsmi": "CC(Cl)C>>CCCl"},
        ]
        self.rsmi_key = "rsmi"
        self.fp_type = "ecfp4"
        self.calc = FPCalculator(n_jobs=2, verbose=0)

    def test_constructor_assigns_attributes(self):
        self.assertEqual(self.calc.n_jobs, 2)
        self.assertEqual(self.calc.verbose, 0)

    def test_validate_fp_type_accepts_supported(self):
        # Should not raise
        for ft in FPCalculator.VALID_FP_TYPES:
            self.calc._validate_fp_type(ft)

    def test_validate_fp_type_rejects_unsupported(self):
        with self.assertRaises(ValueError):
            self.calc._validate_fp_type("invalid_fp")

    def test_dict_process_missing_key_raises(self):
        with self.assertRaises(ValueError):
            FPCalculator.dict_process({}, self.rsmi_key, fp_type=self.fp_type)

    def test_dict_process_adds_fingerprint(self):
        data = {"rsmi": "CCO>>CC=O"}
        out = FPCalculator.dict_process(data, "rsmi", fp_type="ecfp4")
        self.assertIn("ecfp4", out)
        # Check it's a list/vector (not None)
        self.assertIsNotNone(out["ecfp4"])

    def test_parallel_process_returns_list_of_dicts(self):
        results = self.calc.parallel_process(self.batch, "rsmi", fp_type="ecfp4")
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        for d in results:
            self.assertIn("ecfp4", d)

    def test_str_and_help_output(self):
        s = str(self.calc)
        self.assertIn("FPCalculator", s)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.calc.help()
        help_out = buf.getvalue()

        # The help text starts with this exact line
        self.assertIn(
            "FPCalculator supports the following fingerprint types:", help_out
        )
        # And lists our parallel jobs config
        self.assertIn(f"Configured for {self.calc.n_jobs} parallel jobs", help_out)


if __name__ == "__main__":
    unittest.main()
