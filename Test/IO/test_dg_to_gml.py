import unittest
import importlib

from synkit.IO.chem_converter import smart_to_gml
from synkit.Chem.Reaction.standardize import Standardize
from synkit.IO.dg_to_gml import DGToGML

# from synkit.Synthesis.Reactor.reactor_engine import ReactorEngine
from synkit.Synthesis.Reactor.mod_reactor import MODReactor


MOD_AVAILABLE = importlib.util.find_spec("mod") is not None


@unittest.skipUnless(MOD_AVAILABLE, "requires `mod` package for rule backend")
class TestGMLToNX(unittest.TestCase):

    def setUp(self) -> None:
        self.smart = (
            "[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])"
            + "[N:8]1[CH2:9][CH2:11][N:12]([H:13])[CH2:14][CH2:10]1."
            + "[O:15]=[S:16](=[O:17])([c:18]1[cH:20][cH:22][cH:23]"
            + "[cH:24][cH:21]1)[N:19]1[CH2:25][CH2:27][O:28][c:29]2[c:26]1"
            + "[cH:35][c:33]([Cl:34])[cH:32][c:30]2[Br:31]>>[CH3:1][C:2]"
            + "([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[N:8]1[CH2:9][CH2:11]"
            + "[N:12]([c:30]2[c:29]3[c:26]([cH:35][c:33]([Cl:34])[cH:32]2)"
            + "[N:19]([S:16](=[O:15])(=[O:17])[c:18]2[cH:20][cH:22][cH:23]"
            + "[cH:24][cH:21]2)[CH2:25][CH2:27][O:28]3)[CH2:14][CH2:10]1"
            + ".[H:13][Br:31]"
        )
        self.standardizer = Standardize()
        self.standardized_rsmi = self.standardizer.fit(self.smart)
        self.gml = smart_to_gml(self.smart)
        self.dg_to_gml = DGToGML()

    def test_getReactionSmiles(self):
        reactor = MODReactor(
            substrate=self.standardized_rsmi.split(">>")[0].split("."),
            rule_file=self.gml,
        ).run()
        dg = reactor.get_dg()
        rsmis = self.dg_to_gml.getReactionSmiles(dg)
        self.assertIsInstance(rsmis, dict)
        self.assertGreater(len(rsmis), 0)

    def test_fit(self):
        reactor = MODReactor(
            substrate=self.standardized_rsmi.split(">>")[0].split("."),
            rule_file=self.gml,
        ).run()
        dg = reactor.get_dg()
        rsmis = self.dg_to_gml.fit(dg=dg, origSmiles=self.standardized_rsmi)

        self.assertGreater(len(rsmis), 0)


if __name__ == "__main__":
    unittest.main()
