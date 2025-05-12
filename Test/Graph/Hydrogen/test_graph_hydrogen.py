import unittest
from synkit.Chem.Reaction.aam_validator import AAMValidator
from synkit.Graph.Matcher.graph_morphism import graph_isomorphism
from synkit.IO.chem_converter import rsmi_to_graph, graph_to_rsmi
from synkit.Graph.Hyrogen._misc import implicit_hydrogen, h_to_explicit


class TestGraphH(unittest.TestCase):

    def setUp(self) -> None:

        self.rsmi = (
            "[C:1]([C:2]([C:3]([H:43])([H:44])[H:45])([C:4]([H:46])"
            + "([H:47])[H:48])[O:5][C:6](=[O:7])[N:8]([C:9]([C:10](=[O:11])"
            + "[N:12]([c:13]1[c:14]([H:52])[c:15]2[c:16]3[c:17]([c:18]"
            + "(-[c:19]4[c:20]([H:53])[c:21]([H:54])[c:22]([H:55])[c:23]"
            + "([H:56])[c:24]4[H:57])[n:25]([H:39])[c:26]3[c:27]1[H:58])"
            + "[C:28]([H:59])=[N:29][N:30]([H:60])[C:31]2=[O:32])[H:51])"
            + "([C:33]1([H:61])[C:34]([H:62])([H:63])[C:35]([H:64])([H:65])"
            + "[C:36]([H:66])([H:67])[C:37]([H:68])([H:69])[C:38]1([H:70])"
            + "[H:71])[H:50])[H:49])([H:40])([H:41])[H:42].[O:72]([H:73])[H:74]"
            + ">>[C:1]([C:2]([C:3]([H:43])([H:44])[H:45])([C:4]([H:46])([H:47])"
            + "[H:48])[O:5][C:6](=[O:7])[O:72][H:74])([H:40])([H:41])[H:42]."
            + "[N:8]([C:9]([C:10](=[O:11])[N:12]([c:13]1[c:14]([H:52])[c:15]2"
            + "[c:16]3[c:17]([c:18](-[c:19]4[c:20]([H:53])[c:21]([H:54])[c:22]"
            + "([H:55])[c:23]([H:56])[c:24]4[H:57])[n:25]([H:39])[c:26]3[c:27]1"
            + "[H:58])[C:28]([H:59])=[N:29][N:30]([H:60])[C:31]2=[O:32])[H:51])"
            + "([C:33]1([H:61])[C:34]([H:62])([H:63])[C:35]([H:64])([H:65])[C:36]"
            + "([H:66])([H:67])[C:37]([H:68])([H:69])[C:38]1([H:70])[H:71])"
            + "[H:50])([H:49])[H:73]"
        )

        self.implicit_rsmi = (
            "[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[NH:8]"
            + "[CH:9]([C:10](=[O:11])[NH:12][c:13]1[cH:14][c:15]2"
            + "[c:16]3[c:17]([c:18](-[c:19]4[cH:20][cH:21][cH:22][cH:23]"
            + "[cH:24]4)[nH:25][c:26]3[cH:27]1)[CH:28]=[N:29][NH:30]"
            + "[C:31]2=[O:32])[CH:33]1[CH2:34][CH2:35][CH2:36][CH2:37]"
            + "[CH2:38]1.[OH:72][H:73]>>[CH3:1][C:2]([CH3:3])([CH3:4])"
            + "[O:5][C:6](=[O:7])[OH:72].[NH:8]([CH:9]([C:10](=[O:11])"
            + "[NH:12][c:13]1[cH:14][c:15]2[c:16]3[c:17]([c:18](-[c:19]4"
            + "[cH:20][cH:21][cH:22][cH:23][cH:24]4)[nH:25][c:26]3"
            + "[cH:27]1)[CH:28]=[N:29][NH:30][C:31]2=[O:32])[CH:33]1"
            + "[CH2:34][CH2:35][CH2:36][CH2:37][CH2:38]1)[H:73]"
        )

    def test_implicit_hydrogen(self):
        # Keep hydrogen atom map 73
        r, p = rsmi_to_graph(self.rsmi)
        r, p = implicit_hydrogen(r, [73]), implicit_hydrogen(p, [73])
        new_rsmi = graph_to_rsmi(r, p, None, sanitize=True, explicit_hydrogen=False)
        self.assertTrue(AAMValidator().smiles_check(new_rsmi, self.implicit_rsmi))

    def test_explicit_hydrogen(self):
        r_imp, p_imp = rsmi_to_graph(self.implicit_rsmi)
        r_ex, p_ex = h_to_explicit(r_imp), h_to_explicit(p_imp)
        r, p = rsmi_to_graph(self.rsmi)
        self.assertTrue(graph_isomorphism(r, r_ex))
        self.assertTrue(graph_isomorphism(p, p_ex))
