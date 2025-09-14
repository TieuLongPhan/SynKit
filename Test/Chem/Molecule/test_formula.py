import unittest
from typing import Dict
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors

from synkit.Chem.Molecule.formula import Formula


def _parse_formula_string(formula: str) -> Dict[str, int]:
    """
    Simple local mirror of Formula._parse_formula_string used to produce
    expected element counts from RDKit formula strings.
    """
    import re

    pattern = re.compile(r"([A-Z][a-z]*)(\d*)")
    parts = pattern.findall(formula)
    return {elem: (int(n) if n else 1) for elem, n in parts}


def _expected_decompose_from_rdkit(smiles: str) -> Dict[str, int]:
    """
    Compute the expected element counts using RDKit's CalcMolFormula.
    Handles multi-fragment SMILES ('.') by summing fragments.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return {}
    parts = []
    for frag in smiles.split("."):
        mol = Chem.MolFromSmiles(frag)
        if mol is None:
            continue
        f = rdMolDescriptors.CalcMolFormula(mol)
        parts.append(_parse_formula_string(f))
    # sum
    out = {}
    for d in parts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def _expected_molwt_from_rdkit(smiles: str) -> float:
    """Sum RDKit MolWt for fragments; returns None when no valid fragment."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    total = 0.0
    valid = False
    for frag in smiles.split("."):
        mol = Chem.MolFromSmiles(frag)
        if mol is None:
            continue
        total += Descriptors.MolWt(mol)
        valid = True
    return total if valid else None


def _expected_hill_from_counts(counts: Dict[str, int]) -> str:
    """Build expected Hill-order string from counts dict following the same rules."""
    if not counts:
        return ""

    def fmt(elem: str, n: int) -> str:
        return f"{elem}{n if n != 1 else ''}"

    if "C" in counts:
        parts = []
        parts.append(fmt("C", counts.get("C", 0)))
        if counts.get("H", 0) > 0:
            parts.append(fmt("H", counts["H"]))
        others = sorted(k for k in counts.keys() if k not in ("C", "H"))
        parts.extend(fmt(k, counts[k]) for k in others if counts[k] > 0)
        parts = [p for p in parts if not p.endswith("0")]
        return "".join(parts)
    else:
        parts = [fmt(k, counts[k]) for k in sorted(counts.keys()) if counts[k] > 0]
        return "".join(parts)


class TestFormula(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use single job for deterministic behaviour in tests
        cls.f = Formula(n_jobs=1, verbose=0)
        cls.species = ["C#N", "HN=CC#N", "NC(C#N)C#N", "N", "NC=NH"]
        # include an explicit invalid example for negative testing
        cls.invalid = "NotASmiles"

    def test_decompose_against_rdkit(self):
        """decompose() should match RDKit's CalcMolFormula parsing"""
        for s in self.species:
            with self.subTest(smiles=s):
                expected = _expected_decompose_from_rdkit(s)
                got = self.f.decompose(s)
                self.assertEqual(
                    got,
                    expected,
                    msg=f"Decompose mismatch for {s}: expected {expected}, got {got}",
                )

    def test_hill_formula_format(self):
        """hill_formula() should follow Hill rules and omit '1' counts"""
        for s in self.species:
            with self.subTest(smiles=s):
                counts = _expected_decompose_from_rdkit(s)
                expected = _expected_hill_from_counts(counts)
                got = self.f.hill_formula(s)
                self.assertEqual(
                    got,
                    expected,
                    msg=f"Hill format mismatch for {s}: expected '{expected}', got '{got}'",
                )

    def test_mol_weight_against_rdkit(self):
        """mol_weight() should be numerically close to RDKit MolWt sum"""
        for s in self.species:
            with self.subTest(smiles=s):
                expected = _expected_molwt_from_rdkit(s)
                got = self.f.mol_weight(s)
                # both should be None if invalid, otherwise compare floats
                if expected is None:
                    self.assertIsNone(
                        got, msg=f"Expected None molwt for {s}, got {got}"
                    )
                else:
                    # use 3 decimal places tolerance (fits typical MolWt rounding)
                    self.assertIsNotNone(
                        got, msg=f"mol_weight returned None for valid SMILES {s}"
                    )
                    self.assertAlmostEqual(
                        got,
                        expected,
                        places=3,
                        msg=f"MolWt mismatch for {s}: expected {expected}, got {got}",
                    )

    def test_process_list_decompose_and_hill(self):
        """process_list() should map across inputs correctly for both 'decompose' and 'hill'"""
        decomposed = self.f.process_list(self.species, what="decompose")
        expected_decomposed = [_expected_decompose_from_rdkit(s) for s in self.species]
        self.assertEqual(decomposed, expected_decomposed)

        hills = self.f.process_list(self.species, what="hill")
        expected_hills = [
            _expected_hill_from_counts(_expected_decompose_from_rdkit(s))
            for s in self.species
        ]
        self.assertEqual(hills, expected_hills)

    def test_process_list_dict_with_list_and_dataframe(self):
        """process_list_dict should accept list-of-dicts and DataFrame and attach output properly"""
        records = [{"smiles": s, "idx": i} for i, s in enumerate(self.species)]
        out = self.f.process_list_dict(
            records, smiles_key="smiles", out_key="hill", what="hill", copy=True
        )
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), len(self.species))
        for i, rec in enumerate(out):
            expected = _expected_hill_from_counts(
                _expected_decompose_from_rdkit(self.species[i])
            )
            self.assertIn("hill", rec)
            self.assertEqual(rec["hill"], expected)
            # ensure original keys still present
            self.assertEqual(rec["idx"], i)

        # Test DataFrame input as well (converted internally to list-of-dicts)
        df = pd.DataFrame(records)
        out2 = self.f.process_list_dict(
            df, smiles_key="smiles", out_key="hill", what="hill", copy=False
        )
        self.assertIsInstance(out2, list)
        self.assertEqual(len(out2), len(self.species))
        for i, rec in enumerate(out2):
            expected = _expected_hill_from_counts(
                _expected_decompose_from_rdkit(self.species[i])
            )
            self.assertEqual(rec["hill"], expected)

    def test_invalid_smiles_behavior(self):
        """Invalid SMILES should return empty dict '', or None as appropriate."""
        # decompose -> {}
        self.assertEqual(self.f.decompose(self.invalid), {})
        # hill -> ''
        self.assertEqual(self.f.hill_formula(self.invalid), "")
        # mol_weight -> None
        self.assertIsNone(self.f.mol_weight(self.invalid))
        # process_list should contain these outputs in sequence
        seq = [self.species[0], self.invalid, self.species[1]]
        res = self.f.process_list(seq, what="decompose")
        self.assertEqual(res[1], {})  # middle invalid -> {}
        res_hill = self.f.process_list(seq, what="hill")
        self.assertEqual(res_hill[1], "")  # middle invalid -> ''


if __name__ == "__main__":
    unittest.main()
