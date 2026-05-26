import unittest

from synkit.Chem.Reaction.standardize import Standardize
from synkit.IO.chem_converter import rsmi_to_its
from synkit.Synthesis.Reactor.syn_reactor import SynReactor


class TestSynReactorBugCases(unittest.TestCase):
    def test_tuple_backward_cross_coupling_keeps_aromatic_role_context(self):
        smart = (
            "[CH3:10][CH2:11][O:12][C:13](=[O:14])[c:15]1[cH:16][cH:18][cH:19]"
            "[c:20]([B:21]([OH:22])[OH:23])[cH:17]1."
            "[CH3:1][c:2]1[cH:3][cH:5][cH:6][c:7]([Br:8])[c:4]1[I:9]"
            ">>"
            "[CH3:1][c:2]1[cH:3][cH:5][cH:6][c:7]([Br:8])[c:4]1-"
            "[c:20]1[cH:17][c:15]([C:13]([O:12][CH2:11][CH3:10])=[O:14])"
            "[cH:16][cH:18][cH:19]1.[I:9][B:21]([OH:22])[OH:23]"
        )
        expected = Standardize().fit(smart)
        reactants, products = expected.split(">>")
        rc = rsmi_to_its(smart, core=True, format="tuple")

        forward = SynReactor(
            substrate=reactants,
            template=rc,
            implicit_temp=False,
            explicit_h=False,
        )
        backward = SynReactor(
            substrate=products,
            template=rc,
            implicit_temp=False,
            explicit_h=False,
            invert=True,
        )

        forward_smis = [
            Standardize().fit(candidate, remove_aam=True)
            for candidate in forward.smarts
        ]
        backward_smis = [
            Standardize().fit(candidate, remove_aam=True)
            for candidate in backward.smarts
        ]

        self.assertIn(expected, forward_smis)
        self.assertIn(expected, backward_smis)


if __name__ == "__main__":
    unittest.main()
