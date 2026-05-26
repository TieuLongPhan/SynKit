import unittest

from synkit.Graph.ITS.its_reverter import ITSReverter
from synkit.Chem.Reaction.aam_validator import AAMValidator
from synkit.Chem.Reaction.standardize import Standardize
from synkit.IO.chem_converter import rsmi_to_its
from synkit.Synthesis.Reactor.syn_reactor import SynReactor


class TestSynReactorElectronCases(unittest.TestCase):
    @staticmethod
    def _has_equivalent_candidate(smart: str, candidates: list[str]) -> bool:
        return any(
            AAMValidator().smiles_check(smart, candidate) for candidate in candidates
        )

    @staticmethod
    def _has_standardized_candidate(smart: str, candidates: list[str]) -> bool:
        standardizer = Standardize()
        expected = standardizer.fit(smart)
        return expected in [
            standardizer.fit(candidate, remove_aam=True) for candidate in candidates
        ]

    @staticmethod
    def _tuple_reactors(
        smart: str,
        *,
        core: bool,
        implicit_temp: bool,
        explicit_h: bool,
    ) -> tuple[SynReactor, SynReactor]:
        substrate, product = Standardize().fit(smart, remove_aam=True).split(">>")
        rc = rsmi_to_its(smart, core=core, format="tuple")
        return (
            SynReactor(
                substrate=substrate,
                template=rc,
                electron_diagnostics=True,
                implicit_temp=implicit_temp,
                explicit_h=explicit_h,
            ),
            SynReactor(
                substrate=product,
                template=rc,
                electron_diagnostics=True,
                implicit_temp=implicit_temp,
                explicit_h=explicit_h,
                invert=True,
            ),
        )

    def _assert_bidirectional_equivalent(
        self,
        smart: str,
        *,
        core: bool,
        implicit_temp: bool,
        explicit_h: bool,
    ) -> tuple[SynReactor, SynReactor]:
        forward, backward = self._tuple_reactors(
            smart,
            core=core,
            implicit_temp=implicit_temp,
            explicit_h=explicit_h,
        )
        self.assertTrue(forward.smarts)
        self.assertTrue(backward.smarts)
        self.assertTrue(self._has_equivalent_candidate(smart, forward.smarts))
        self.assertTrue(self._has_equivalent_candidate(smart, backward.smarts))
        return forward, backward

    def _assert_bidirectional_standardized(
        self,
        smart: str,
        *,
        core: bool,
        implicit_temp: bool,
        explicit_h: bool,
    ) -> tuple[SynReactor, SynReactor]:
        forward, backward = self._tuple_reactors(
            smart,
            core=core,
            implicit_temp=implicit_temp,
            explicit_h=explicit_h,
        )
        self.assertTrue(forward.smarts)
        self.assertTrue(backward.smarts)
        self.assertTrue(self._has_standardized_candidate(smart, forward.smarts))
        self.assertTrue(self._has_standardized_candidate(smart, backward.smarts))
        return forward, backward

    def test_lone_pair_donation_recomputes_product_charge(self):
        smart = "[NH3:1].[CH3:2][Cl:3]>>[NH3+:1][CH3:2].[Cl-:3]"
        rc = rsmi_to_its(smart, core=True, format="tuple")

        # The rewrite should not need the product charge labels from the RC.
        # Keep the electron-state changes, but erase direct product charge.
        rc.nodes[1]["charge"] = (0, 0)
        rc.nodes[3]["charge"] = (0, 0)
        n_types = list(rc.nodes[1]["typesGH"])
        cl_types = list(rc.nodes[3]["typesGH"])
        rc.nodes[1]["typesGH"] = (
            n_types[0],
            n_types[1][:3] + (0,) + n_types[1][4:],
        )
        rc.nodes[3]["typesGH"] = (
            cl_types[0],
            cl_types[1][:3] + (0,) + cl_types[1][4:],
        )

        reactor = SynReactor(
            "[NH3:1].[CH3:2][Cl:3]",
            rc,
            implicit_temp=False,
            explicit_h=False,
        )
        product = ITSReverter(reactor.its_list[0]).to_product_graph()

        self.assertEqual(product.nodes[1]["lone_pairs"], 0)
        self.assertEqual(product.nodes[3]["lone_pairs"], 4)
        self.assertEqual(product.nodes[1]["charge"], 1.0)
        self.assertEqual(product.nodes[3]["charge"], -1.0)
        self.assertEqual(
            reactor.smarts,
            ["[CH3:2][Cl:3].[NH3:1]>>[Cl-:3].[NH3+:1][CH3:2]"],
        )

        reverse_rc = rsmi_to_its(smart, core=True, format="tuple")
        backward = SynReactor(
            "[NH3+:1][CH3:2].[Cl-:3]",
            reverse_rc,
            implicit_temp=False,
            explicit_h=False,
            invert=True,
        )
        self.assertTrue(self._has_equivalent_candidate(smart, backward.smarts))

    def test_tuple_reactor_assigns_fresh_atom_maps_for_unmapped_substrate(self):
        smart = "[NH3:1].[CH3:2][Cl:3]>>[NH3+:1][CH3:2].[Cl-:3]"
        forward, backward = self._assert_bidirectional_equivalent(
            smart,
            core=False,
            implicit_temp=True,
            explicit_h=False,
        )

        for reactor in (forward, backward):
            self.assertEqual(len(reactor.smarts), 1)
            self.assertTrue(
                all(
                    pair[0] > 0 and pair[1] > 0
                    for _, data in reactor.its_list[0].nodes(data=True)
                    for pair in [data["atom_map"]]
                )
            )

    def test_tuple_reactor_assigns_fresh_atom_maps_to_expanded_hydrogens(self):
        smart = "[CH3:1][Cl:2].[O:3]([H:4])[H:5]" ">>[CH3:1][O:3][H:4].[Cl:2][H:5]"
        forward, backward = self._assert_bidirectional_standardized(
            smart,
            core=False,
            implicit_temp=False,
            explicit_h=True,
        )

        for reactor in (forward, backward):
            self.assertEqual(len(reactor.smarts), 1)
            self.assertTrue(all("[H]" not in smarts for smarts in reactor.smarts))
            hydrogen_maps = [
                data["atom_map"]
                for _, data in reactor.its_list[0].nodes(data=True)
                if data["element"] == ("H", "H")
            ]
            self.assertTrue(hydrogen_maps)
            self.assertTrue(all(pair[0] > 0 and pair[1] > 0 for pair in hydrogen_maps))

    def test_tuple_explicit_h_only_reconstructs_template_explicit_hydrogens(self):
        smart = "[H:4][NH2:1].[CH3:2][Cl:3]>>[NH2:1][CH3:2].[Cl:3][H:4]"
        forward, backward = self._assert_bidirectional_equivalent(
            smart,
            core=True,
            implicit_temp=False,
            explicit_h=True,
        )

        for reactor in (forward, backward):
            self.assertEqual(len(reactor.smarts), 1)
            self.assertIn("[H:4]", reactor.smarts[0])

    def test_tuple_implicit_output_omits_mapped_explicit_hydrogens(self):
        smart = "[H:4][NH2:1].[CH3:2][Cl:3]>>[NH2:1][CH3:2].[Cl:3][H:4]"
        forward, backward = self._tuple_reactors(
            smart,
            core=True,
            implicit_temp=False,
            explicit_h=False,
        )

        for reactor in (forward, backward):
            self.assertEqual(len(reactor.smarts), 1)
            self.assertNotIn("[H:", reactor.smarts[0])

    def test_tuple_reactor_keeps_removable_hydrogens_implicit_when_requested(self):
        smart = "[CH3:1][Cl:2].[O:3]([H:4])[H:5]" ">>[CH3:1][O:3][H:4].[Cl:2][H:5]"
        forward, backward = self._tuple_reactors(
            smart,
            core=False,
            implicit_temp=False,
            explicit_h=False,
        )

        self.assertEqual(
            forward.smarts, ["[CH3:1][Cl:2].[OH2:3]>>[CH3:1][OH:3].[ClH:2]"]
        )
        self.assertTrue(self._has_standardized_candidate(smart, backward.smarts))
        self.assertTrue(all("[H:" not in candidate for candidate in backward.smarts))

    def test_tuple_explicit_h_renders_remaining_water_hydrogens(self):
        smart = (
            "[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[C:7]([H:23])=[O:8]."
            "[cH:9]1[cH:10][cH:11][cH:12][cH:13][c:14]1[C:15]([H:19])=[O:16]."
            "[C-:17]#[N:18].[O:20]([H:21])[H:22]"
            ">>"
            "[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[C:7]([H:23])([O:8][H:21])"
            "[C:15](=[O:16])[c:14]1[cH:13][cH:12][cH:11][cH:10][cH:9]1."
            "[C-:17]#[N:18].[O:20]([H:19])[H:22]"
        )
        forward, backward = self._tuple_reactors(
            smart,
            core=False,
            implicit_temp=False,
            explicit_h=True,
        )

        self.assertEqual(len(forward.smarts), 1)
        self.assertEqual(len(backward.smarts), 1)
        for reactor in (forward, backward):
            self.assertTrue(all("[OH:1]" in smarts for smarts in reactor.smarts))
            self.assertTrue(all("[cH:" in smarts for smarts in reactor.smarts))
            atom_maps = [
                pair
                for _, data in reactor.its_list[0].nodes(data=True)
                if data["element"] == ("H", "H")
                for pair in [data["atom_map"]]
            ]
            self.assertEqual(len(atom_maps), len(set(atom_maps)))

    def test_tuple_explicit_h_has_equivalent_candidate_for_aromatic_case(self):
        smart = (
            "[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[CH:7]=[O:8]."
            "[cH:9]1[cH:10][cH:11][cH:12][cH:13][c:14]1[C:15]([H:19])=[O:16]."
            "[C-:17]#[N:18].[OH:20]([H:21])>>"
            "[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[CH:7]([O:8][H:21])"
            "[C:15](=[O:16])[c:14]1[cH:13][cH:12][cH:11][cH:10][cH:9]1."
            "[C-:17]#[N:18].[OH:20]([H:19])"
        )
        forward, backward = self._assert_bidirectional_equivalent(
            smart,
            core=False,
            implicit_temp=False,
            explicit_h=True,
        )
        self.assertTrue(forward.smarts)
        self.assertTrue(backward.smarts)

    def test_tuple_hh_reaction_keeps_molecular_hydrogen_explicit(self):
        smart = (
            "[C:1](#[C:2][CH3:6])[CH3:5].[H:3][H:4]"
            ">>[C:1](=[C:2]([H:4])[CH3:6])([H:3])[CH3:5]"
        )
        for explicit_h in (False, True):
            with self.subTest(explicit_h=explicit_h):
                forward, backward = self._assert_bidirectional_equivalent(
                    smart,
                    core=True,
                    implicit_temp=False,
                    explicit_h=explicit_h,
                )

                for reactor in (forward, backward):
                    self.assertTrue(
                        all("[H:" in candidate for candidate in reactor.smarts)
                    )

    def test_tuple_implicit_hh_reaction_consumes_hydrogen_into_hcount(self):
        smart = (
            "[C:1](#[C:2][CH3:6])[CH3:5].[H:3][H:4]" ">>[CH:1](=[CH:2][CH3:6])[CH3:5]"
        )
        forward, backward = self._assert_bidirectional_equivalent(
            smart,
            core=True,
            implicit_temp=True,
            explicit_h=False,
        )

        self.assertTrue(forward.smarts)
        self.assertTrue(backward.smarts)

    def test_tuple_implicit_hydrogen_permutations_are_pruned_before_rewrite(self):
        smart = (
            "[CH3:1][C:2]#[C:3][CH3:4].[H:5][H:6].[H:7][H:8]"
            ">>[CH3:1][CH2:2][CH2:3][CH3:4]"
        )
        forward, backward = self._tuple_reactors(
            smart,
            core=True,
            implicit_temp=True,
            explicit_h=False,
        )

        # Swapping equivalent H atoms within one H2 or swapping two equivalent
        # H2 reagent components is provenance only and is pruned.
        self.assertEqual(len(forward.mappings), 2)
        self.assertEqual(len(forward.its_list), 1)
        self.assertTrue(self._has_standardized_candidate(smart, forward.smarts))
        self.assertTrue(self._has_standardized_candidate(smart, backward.smarts))

    def test_tuple_real_backward_explicit_h_case_is_reproducible(self):
        smart = (
            "[CH3:1][CH2:2][C:3]([CH2:4][CH3:7])([c:5]1[cH:8][cH:10][c:11]"
            "([O:12][CH2:14][CH:15]([O:16][Si:18]([CH3:19])([CH3:20])[C:21]"
            "([CH3:22])([CH3:23])[CH3:24])[C:17]([CH3:25])([CH3:26])[CH3:27])"
            "[c:13]([CH3:28])[cH:9]1)[c:6]1[cH:29][c:31]([CH3:32])[c:33]"
            "([CH:34]=[O:35])[s:30]1.[CH3:36][O:37][C:38](=[O:39])[CH2:40]"
            "[NH:41][H:42].[H:43][H:44]>>[CH3:1][CH2:2][C:3]([CH2:4][CH3:7])"
            "([c:5]1[cH:8][cH:10][c:11]([O:12][CH2:14][CH:15]([O:16][Si:18]"
            "([CH3:19])([CH3:20])[C:21]([CH3:22])([CH3:23])[CH3:24])[C:17]"
            "([CH3:25])([CH3:26])[CH3:27])[c:13]([CH3:28])[cH:9]1)[c:6]1[cH:29]"
            "[c:31]([CH3:32])[c:33]([CH:34]([NH:41][CH2:40][C:38]([O:37][CH3:36])"
            "=[O:39])[H:43])[s:30]1.[O:35]([H:42])[H:44]"
        )
        standardizer = Standardize()
        expected = standardizer.fit(smart)
        substrate, product = standardizer.fit(smart, remove_aam=True).split(">>")
        rc = rsmi_to_its(smart, core=True, format="tuple")

        forward = SynReactor(
            substrate=substrate,
            template=rc,
            electron_diagnostics=True,
            implicit_temp=False,
            explicit_h=False,
        )
        backward = SynReactor(
            substrate=product,
            template=rc,
            electron_diagnostics=True,
            implicit_temp=False,
            explicit_h=False,
            invert=True,
        )

        self.assertIn(
            expected,
            [
                standardizer.fit(candidate, remove_aam=True)
                for candidate in forward.smarts
            ],
        )
        self.assertIn(
            expected,
            [
                standardizer.fit(candidate, remove_aam=True)
                for candidate in backward.smarts
            ],
        )
        self.assertTrue(backward.its_list)
        product_graph = ITSReverter(backward.its_list[0]).to_product_graph()
        self.assertTrue(
            any(
                attrs.get("element") == "O" and attrs.get("hcount") == -1
                for _, attrs in product_graph.nodes(data=True)
            )
        )

    def test_radical_homolytic_cc_cleavage(self):
        self._assert_radicals(
            "[CH3:1][CH3:2]>>[CH3:1].[CH3:2]",
            expected_product_radicals={1: 1, 2: 1},
        )

    def test_radical_cc_recombination(self):
        self._assert_radicals(
            "[CH3:1].[CH3:2]>>[CH3:1][CH3:2]",
            expected_product_radicals={1: 0, 2: 0},
        )

    def test_radical_homolytic_cbr_cleavage(self):
        self._assert_radicals(
            "[CH3:1][Br:2]>>[CH3:1].[Br:2]",
            expected_product_radicals={1: 1, 2: 1},
        )

    def test_radical_cbr_recombination(self):
        self._assert_radicals(
            "[CH3:1].[Br:2]>>[CH3:1][Br:2]",
            expected_product_radicals={1: 0, 2: 0},
        )

    def _assert_radicals(
        self,
        smart: str,
        *,
        expected_product_radicals: dict[int, int],
    ) -> None:
        reactants, _ = smart.split(">>")
        rc = rsmi_to_its(smart, core=True, format="tuple")
        reactor = SynReactor(
            reactants,
            rc,
            implicit_temp=False,
            explicit_h=False,
            electron_diagnostics=True,
        )
        product = ITSReverter(reactor.its_list[0]).to_product_graph()

        for node, radical in expected_product_radicals.items():
            self.assertEqual(product.nodes[node]["radical"], radical)
            self.assertEqual(product.nodes[node]["charge"], 0.0)

        _, products = smart.split(">>")
        backward = SynReactor(
            products,
            rc,
            implicit_temp=False,
            explicit_h=False,
            electron_diagnostics=True,
            invert=True,
        )
        self.assertTrue(self._has_equivalent_candidate(smart, reactor.smarts))
        self.assertTrue(self._has_equivalent_candidate(smart, backward.smarts))


if __name__ == "__main__":
    unittest.main()
