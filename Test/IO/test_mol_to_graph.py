import unittest

import networkx as nx
from rdkit import Chem

from synkit.IO.mol_to_graph import MolToGraph


class TestMolToGraph(unittest.TestCase):
    def setUp(self):
        self.converter = MolToGraph()
        self.mol = Chem.MolFromSmiles("CCO")  # ethanol

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    def test_add_partial_charges(self):
        MolToGraph.add_partial_charges(self.mol)
        for atom in self.mol.GetAtoms():
            self.assertTrue(atom.HasProp("_GasteigerCharge"))

    def test_get_stereochemistry_chiral(self):
        mol = Chem.MolFromSmiles("CC[C@@H](C)O")
        atom = mol.GetAtomWithIdx(2)
        stereo = MolToGraph.get_stereochemistry(atom)
        self.assertIn(stereo, ["R", "S"])

    def test_get_stereochemistry_non_chiral(self):
        atom = self.mol.GetAtomWithIdx(0)
        self.assertEqual(MolToGraph.get_stereochemistry(atom), "N")

    def test_get_bond_stereochemistry_double(self):
        mol = Chem.MolFromSmiles("C/C=C/C")
        bond = mol.GetBondWithIdx(1)
        stereo = MolToGraph.get_bond_stereochemistry(bond)
        self.assertIn(stereo, ["E", "Z", "N"])

    def test_get_bond_stereochemistry_single_returns_n(self):
        bond = self.mol.GetBondWithIdx(0)
        self.assertEqual(MolToGraph.get_bond_stereochemistry(bond), "N")

    def test_has_atom_mapping_true(self):
        mol = Chem.MolFromSmiles("[CH3:1][OH:2]")
        self.assertTrue(MolToGraph.has_atom_mapping(mol))

    def test_has_atom_mapping_false(self):
        self.assertFalse(MolToGraph.has_atom_mapping(self.mol))

    def test_random_atom_mapping(self):
        mol = Chem.MolFromSmiles("CCO")
        MolToGraph.random_atom_mapping(mol)
        maps = [a.GetAtomMapNum() for a in mol.GetAtoms()]
        self.assertEqual(sorted(maps), list(range(1, mol.GetNumAtoms() + 1)))

    # ------------------------------------------------------------------
    # Constructor validation
    # ------------------------------------------------------------------

    def test_invalid_profile_raises(self):
        with self.assertRaises(ValueError):
            MolToGraph(attr_profile="invalid")

    def test_valid_profiles(self):
        for profile in ("minimal", "full"):
            c = MolToGraph(attr_profile=profile)
            self.assertEqual(c.attr_profile, profile)

    def test_repr_contains_profile(self):
        r = repr(MolToGraph(attr_profile="minimal"))
        self.assertIn("minimal", r)

    # ------------------------------------------------------------------
    # transform / transform_store / graph property
    # ------------------------------------------------------------------

    def test_transform_returns_graph(self):
        g = self.converter.transform(self.mol)
        self.assertIsInstance(g, nx.Graph)
        self.assertEqual(g.number_of_nodes(), self.mol.GetNumAtoms())
        self.assertEqual(g.number_of_edges(), self.mol.GetNumBonds())

    def test_transform_node_keys_minimal(self):
        g = self.converter.transform(self.mol)
        node_data = dict(list(g.nodes(data=True))[0][1])
        for key in (
            "element",
            "charge",
            "hcount",
            "aromatic",
            "radical",
            "lone_pairs",
            "valence_electrons",
            "available_lp",
            "oxidation_state",
        ):
            self.assertIn(key, node_data, f"missing key: {key}")
        # minimal profile must NOT include these verbose intermediate fields
        for key in (
            "bond_order_sum",
            "lp_bond_order_sum",
            "estimated_lone_pairs",
            "available_lone_pairs",
            "implicit_hcount",
        ):
            self.assertNotIn(key, node_data, f"minimal profile should not have: {key}")

    def test_transform_edge_keys(self):
        g = self.converter.transform(self.mol)
        edge_data = dict(list(g.edges(data=True))[0][2])
        for key in (
            "order",
            "bond_type",
            "aromatic",
            "kekule_order",
            "sigma_order",
            "pi_order",
            "kekule_bond_type",
            "ez_isomer",
            "conjugated",
            "in_ring",
        ):
            self.assertIn(key, edge_data, f"missing edge key: {key}")

    def test_transform_store_and_graph_property(self):
        c = MolToGraph()
        result = c.transform_store(self.mol)
        self.assertIs(result, c)
        self.assertIsInstance(c.graph, nx.Graph)

    def test_graph_property_raises_before_store(self):
        c = MolToGraph()
        with self.assertRaises(RuntimeError):
            _ = c.graph

    def test_transform_full_profile(self):
        g = MolToGraph(attr_profile="full").transform(self.mol)
        self.assertIsInstance(g, nx.Graph)
        node_data = dict(list(g.nodes(data=True))[0][1])
        for key in (
            "bond_order_sum",
            "lp_bond_order_sum",
            "estimated_lone_pairs",
            "available_lone_pairs",
            "valence_electrons",
        ):
            self.assertIn(key, node_data, f"full profile missing: {key}")

    def test_transform_with_topology(self):
        g = MolToGraph(with_topology=True).transform(self.mol)
        self.assertIsInstance(g, nx.Graph)

    def test_transform_node_whitelist(self):
        g = MolToGraph(node_attrs=["element", "charge"]).transform(self.mol)
        for _, data in g.nodes(data=True):
            self.assertEqual(set(data.keys()), {"element", "charge"})

    def test_transform_edge_whitelist(self):
        g = MolToGraph(edge_attrs=["order"]).transform(self.mol)
        for _, _, data in g.edges(data=True):
            self.assertEqual(set(data.keys()), {"order"})

    def test_sigma_pi_order_split_matches_kekule_order(self):
        mol = Chem.MolFromSmiles("C#CC=C")
        g = MolToGraph().transform(mol)
        for _, _, data in g.edges(data=True):
            self.assertEqual(
                data["kekule_order"],
                data["sigma_order"] + data["pi_order"],
            )

    def test_aromatic_bonds_preserve_matching_and_rewrite_views(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        g = MolToGraph().transform(mol)

        self.assertEqual({data["order"] for _, _, data in g.edges(data=True)}, {1.5})
        self.assertEqual(
            {
                (data["kekule_order"], data["sigma_order"], data["pi_order"])
                for _, _, data in g.edges(data=True)
            },
            {(1.0, 1.0, 0.0), (2.0, 1.0, 1.0)},
        )

    def test_drop_non_aam_requires_use_index(self):
        with self.assertRaises(ValueError):
            self.converter.transform(
                self.mol, drop_non_aam=True, use_index_as_atom_map=False
            )

    def test_drop_non_aam_filters_unmapped(self):
        mol = Chem.MolFromSmiles("[CH3:1][CH2:2]Br")  # Br unmapped
        g = MolToGraph().transform(mol, drop_non_aam=True, use_index_as_atom_map=True)
        # Only the two mapped atoms should remain
        self.assertEqual(g.number_of_nodes(), 2)

    def test_use_index_as_atom_map_node_ids(self):
        mol = Chem.MolFromSmiles("[CH3:1][OH:2]")
        g = MolToGraph().transform(mol, use_index_as_atom_map=True)
        self.assertIn(1, g.nodes)
        self.assertIn(2, g.nodes)

    # ------------------------------------------------------------------
    # mol_to_graph class-method (backward-compatible API)
    # ------------------------------------------------------------------

    def test_mol_to_graph_basic(self):
        g = MolToGraph.mol_to_graph(self.mol)
        self.assertIsInstance(g, nx.Graph)
        self.assertEqual(len(g.nodes), self.mol.GetNumAtoms())
        self.assertEqual(len(g.edges), self.mol.GetNumBonds())
        node_data = list(g.nodes(data=True))[0][1]
        self.assertIn("element", node_data)
        self.assertIn("order", list(g.edges(data=True))[0][2])

    def test_mol_to_graph_light_weight(self):
        g = MolToGraph.mol_to_graph(self.mol, light_weight=True)
        self.assertIsInstance(g, nx.Graph)
        self.assertEqual(g.number_of_nodes(), self.mol.GetNumAtoms())
        node_data = list(g.nodes(data=True))[0][1]
        for key in (
            "element",
            "charge",
            "radical",
            "hcount",
            "lone_pairs",
            "available_lp",
            "oxidation_state",
        ):
            self.assertIn(key, node_data, f"light-weight missing: {key}")
        for key in (
            "bond_order_sum",
            "lp_bond_order_sum",
            "estimated_lone_pairs",
            "available_lone_pairs",
            "implicit_hcount",
        ):
            self.assertNotIn(key, node_data, f"light-weight should not have: {key}")

    def test_mol_to_graph_drop_non_aam_requires_use_index(self):
        with self.assertRaises(ValueError):
            MolToGraph.mol_to_graph(
                self.mol, drop_non_aam=True, use_index_as_atom_map=False
            )

    # ------------------------------------------------------------------
    # Radical attribute
    # ------------------------------------------------------------------

    def test_radical_zero_for_closed_shell(self):
        g = self.converter.transform(self.mol)
        for _, data in g.nodes(data=True):
            self.assertEqual(data["radical"], 0)

    def test_radical_nonzero_for_radical_species(self):
        mol = Chem.MolFromSmiles("[CH3]")  # methyl radical
        g = MolToGraph().transform(mol)
        # Carbon in methyl radical has 1 radical electron
        c_radical = g.nodes[1]["radical"]
        self.assertGreaterEqual(c_radical, 1)

    def test_radical_in_light_weight_graph(self):
        g = MolToGraph.mol_to_graph(self.mol, light_weight=True)
        for _, data in g.nodes(data=True):
            self.assertIn("radical", data)

    # ------------------------------------------------------------------
    # Lone-pair chemistry audit
    # ------------------------------------------------------------------

    def test_lone_pair_audit_matrix(self):
        cases = {
            "O": [("O", 2)],
            "[OH-]": [("O", 3)],
            "N": [("N", 1)],
            "[NH4+]": [("N", 0)],
            "[Cl-]": [("Cl", 4)],
            "n1ccccc1": [("N", 1)],
            "[nH]1cccc1": [("N", 1)],
            "C=O": [("O", 2)],
            "[CH3]": [("C", 0)],
            "S": [("S", 2)],
            "[SH-]": [("S", 3)],
            "[SH3+]": [("S", 1)],
            "P": [("P", 1)],
            "[PH4+]": [("P", 0)],
            "P(=O)(O)(O)O": [("P", 0)],
            "S(=O)(=O)(O)O": [("S", 0)],
        }

        for smiles, expected_atoms in cases.items():
            with self.subTest(smiles=smiles):
                mol = Chem.MolFromSmiles(smiles)
                observed = [
                    (atom.GetSymbol(), MolToGraph.estimate_lone_pairs(atom))
                    for atom in mol.GetAtoms()
                    if atom.GetSymbol() in {symbol for symbol, _ in expected_atoms}
                ]
                self.assertEqual(observed[: len(expected_atoms)], expected_atoms)

    def test_lone_pairs_match_for_explicit_and_implicit_hydrogen_forms(self):
        equivalent_pairs = [
            ("O", "[OH2]"),
            ("N", "[NH3]"),
            ("[nH]1cccc1", "[n]1([H])cccc1"),
            ("Oc1ccccc1", "[OH]c1ccccc1"),
            ("Nc1ccccc1", "[NH2]c1ccccc1"),
        ]

        for implicit_smiles, explicit_smiles in equivalent_pairs:
            with self.subTest(
                implicit_smiles=implicit_smiles,
                explicit_smiles=explicit_smiles,
            ):
                implicit_mol = Chem.MolFromSmiles(implicit_smiles)
                explicit_mol = Chem.MolFromSmiles(explicit_smiles)
                implicit_values = [
                    MolToGraph.estimate_lone_pairs(atom)
                    for atom in implicit_mol.GetAtoms()
                ]
                explicit_values = [
                    MolToGraph.estimate_lone_pairs(atom)
                    for atom in explicit_mol.GetAtoms()
                ]
                self.assertEqual(implicit_values, explicit_values)

    # ------------------------------------------------------------------
    # Lone-pair estimation
    # ------------------------------------------------------------------

    def test_estimate_lone_pairs_water_oxygen(self):
        mol = Chem.MolFromSmiles("O")
        o_atom = mol.GetAtomWithIdx(0)
        lp = MolToGraph.estimate_lone_pairs(o_atom)
        self.assertEqual(lp, 2)

    def test_estimate_lone_pairs_pyrrolic_n(self):
        mol = Chem.MolFromSmiles("c1cc[nH]c1")
        n_atom = next(a for a in mol.GetAtoms() if a.GetSymbol() == "N")
        lp = MolToGraph.estimate_lone_pairs(n_atom)
        self.assertGreater(lp, 0)

    def test_estimate_lone_pairs_fused_aromatic_bridgehead_n(self):
        mol = Chem.MolFromSmiles("c1nc2cnccn2c1")
        n_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "N"]
        self.assertEqual(
            [MolToGraph.estimate_lone_pairs(atom) for atom in n_atoms],
            [1, 1, 1],
        )

    def test_estimate_available_lone_pairs_pyrrolic_n_zero(self):
        # [nH] lone pair is conjugated into the ring — not available for donation
        mol = Chem.MolFromSmiles("c1cc[nH]c1")
        n_atom = next(a for a in mol.GetAtoms() if a.GetSymbol() == "N")
        alp = MolToGraph.estimate_available_lone_pairs(n_atom)
        self.assertEqual(alp, 0)

    def test_available_lp_flag(self):
        mol = Chem.MolFromSmiles("O")
        g = MolToGraph().transform(mol)
        o_data = g.nodes[1]
        self.assertTrue(o_data["available_lp"])
        # full profile exposes the numeric count
        o_data_full = MolToGraph(attr_profile="full").transform(mol).nodes[1]
        self.assertGreater(o_data_full["available_lone_pairs"], 0)

    # ------------------------------------------------------------------
    # Oxidation-state estimation
    # ------------------------------------------------------------------

    def test_estimate_oxidation_states_returns_dict(self):
        ox = MolToGraph.estimate_oxidation_states(self.mol)
        self.assertIsInstance(ox, dict)
        self.assertEqual(len(ox), self.mol.GetNumAtoms())

    def test_oxidation_states_by_atom_map(self):
        mol = Chem.MolFromSmiles("[CH3:1][OH:2]")
        result = MolToGraph.oxidation_states_by_atom_map(mol)
        self.assertIn(1, result)
        self.assertIn(2, result)
        self.assertIn("oxidation_state", result[1])

    def test_reaction_oxidation_state_delta_raises_without_arrow(self):
        with self.assertRaises(ValueError):
            MolToGraph.reaction_oxidation_state_delta_from_rsmi("CCO")

    def test_reaction_oxidation_state_delta_methanol_formaldehyde(self):
        rsmi = "[CH3:1][OH:2]>>[CH2:1]=[O:2]"
        changes = MolToGraph.reaction_oxidation_state_delta_from_rsmi(rsmi)
        # Carbon is oxidised in this reaction
        self.assertIn(1, changes)
        self.assertEqual(changes[1]["classification"], "oxidized")


if __name__ == "__main__":
    unittest.main()
