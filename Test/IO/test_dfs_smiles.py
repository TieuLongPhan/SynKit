import unittest
from synkit.IO.chem_converter import (
    dfs_to_smiles,
    smiles_to_dfs,
    normalize_dfs_for_compare,
)

r1f_str = (
    "[H]1[C]2([*]10)([*]11)[C]3([*]12){=}[O]4.[H+]5"
    + ">>"
    + "[H+]1.[*]10[C]2([*]11){=}[C]3([*]12)[O]4[H]5"
)

r2f_str = (
    "[H]1[O]2[C]3{=}[C]4([*]10)[*]11.[*]12[C]5({=}[O]6)[*]13.[H+]7"
    + ">>"
    + "[H+]1.[O]2{=}[C]3[C]4([*]10)([*]11)[C]5([*]12)([*]13)[O]6[H]7"
)

r3f_str = (
    "[*]1[C]2({=}[O]3)[C]4([H]5)([*]6)[C]7([*]8)([*]9)[O]10[H]11"
    + ">>"
    + "[H]11[O]10[H]5.[*]1[C]2({=}[O]3)[C]4([*]6){=}[C]7([*]8)[*]9"
)

r4f_str = (
    "[*]1[C]2([*]3)=[C]4([*]5)[*]6.[H]7[O]8[H]9"
    + ">>"
    + "[*]1[C]2([*]3)([O]8[H]7)[C]4([H]9)([*]5)[*]6"
)

r_n1 = "[N]1#[C]2[H]3.[N]4#[C]5[H]6" + ">>" + "[N]1#[C]2[C]5([H]6)=[N]4[H]3"

r_n2 = (
    "[N]1#[C]2[C]5([H]6)=[N]4[H]3.[N]7#[C]8[H]9"
    + ">>"
    + "[N]1#[C]2[C]5([H]6)([C]8#[N]7)[N]4([H]9)[H]3"
)

r_n3 = "[N]1#[C]2[*]3.[H]5[N]4([H]6)[H]7" + ">>" + "[H]5[N]1=[C]2([*]3)[N]4([H]6)[H]7"

r_n4 = (
    "[N]1#[C]2[C]3([N]7([H]10)[H]11)[C]4(=[N]8[H]12)[N]5([H]9)[H]6.[H]14[N]13([H]15)[H]16"
    + ">>"
    + "[H]14[N]1=[C]2([N]13([H]15)[H]16)[C]3([N]7([H]10)[H]11)[C]4(=[N]8[H]12)[N]5([H]6)[H]9"
)

r_n7 = (
    "[H]9[N]8=[C]6[N]5=[C]2([*]1)[C]3([H]12)([*]4)[N]7([H]10)[H]11"
    + ">>"
    + "[H]12[N]5[C]2([*]1)=[C]3([*]4)[N]8=[C]6-5.[H]9[N]7([H]10)[H]11"
)

r_n8 = (
    "[C]2[N]3([H]8)[H]9.[H]10[N]4=[C]5([*]6)[*]7"
    + ">>"
    + "[C]2[N]3=[C]5([*]6)[*]7.[H]8[N]4([H]9)[H]10"
)

r_n9 = (
    "[H]1[N]2=[C]3[N]4=[C]5([*]6)[C]7=[C]8([*]9)[N]10([H]11)[H]12"
    + ">>"
    + "[N]2=[C]3[N]4=[C]5([*]6)[C]7=[C]8([*]9)2.[H]11[N]10([H]12)[H]1"
)

INPUT_RULES = [
    r1f_str,
    r2f_str,
    r3f_str,
    r4f_str,
    r_n1,
    r_n2,
    r_n3,
    r_n4,
    r_n7,
    r_n8,
    r_n9,
]


class TestDFSSmiles(unittest.TestCase):
    def test_roundtrip_rules(self):
        """Round-trip conversion (DFS -> SMILES -> DFS) should match original after normalization."""
        for dfs in INPUT_RULES:
            with self.subTest(dfs=dfs[:80]):
                smiles = dfs_to_smiles(dfs, keep_map=True)
                back = smiles_to_dfs(smiles)
                self.assertEqual(
                    normalize_dfs_for_compare(dfs),
                    normalize_dfs_for_compare(back),
                    msg=f"Round-trip failed.\norig: {dfs}\nsmiles: {smiles}\nback: {back}",
                )

    def test_keep_map_false(self):
        """When keep_map=False, DFS digits should be removed and no ':n' remains."""
        dfs = "[H]1[N]2([H]4)[]3.[H]5"
        s = dfs_to_smiles(dfs, keep_map=False)
        self.assertNotIn(":", s)
        # make sure wildcard present but unmapped
        self.assertIn("[*]", s)

    def test_multidigit_maps(self):
        """Multi-digit indices should be preserved in mapping and round-trip correctly."""
        dfs = "[H]12[N]234([]56)"
        s = dfs_to_smiles(dfs, keep_map=True)
        self.assertIn("[H:12]", s)
        self.assertIn("[N:234]", s)
        self.assertIn("[*:56]", s)
        back = smiles_to_dfs(s)
        self.assertEqual(
            normalize_dfs_for_compare(dfs), normalize_dfs_for_compare(back)
        )

    def test_wildcard_conversion(self):
        """Test that [] -> [*] on DFS->SMILES and reversed on SMILES->DFS."""
        dfs = "[]3.C[]4>>C[]4.[]3"
        s = dfs_to_smiles(dfs, keep_map=True)
        self.assertIn("[*:3]", s)
        back = smiles_to_dfs(s)
        self.assertIn("[]3", back)

    def test_smiles_to_dfs_specific(self):
        """Specific pattern: wildcard mapping inside SMILES yields []n in DFS."""
        s = "[H:1][*:3].C[O:2]>>C[O:2].[H:1][*:3]"
        back = smiles_to_dfs(s)
        self.assertIn("[H]1", back)
        self.assertIn("[]3", back)
        self.assertIn("[O]2", back)


if __name__ == "__main__":
    unittest.main()
