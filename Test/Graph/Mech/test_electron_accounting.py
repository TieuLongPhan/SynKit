import unittest

import networkx as nx
from rdkit import Chem

from synkit.Graph.Mech.electron_accounting import (
    bond_order_sum,
    graph_to_sanitized_kekule_mol,
    recompute_charge,
    refresh_changed_atom_charge,
    refresh_electron_fields,
)


class TestElectronAccounting(unittest.TestCase):
    @staticmethod
    def _graph_from_kekule_smiles(smiles):
        mol = Chem.MolFromSmiles(smiles)
        kekule = Chem.Mol(mol)
        Chem.Kekulize(kekule, clearAromaticFlags=True)

        valence_electrons = {
            "C": 4,
            "N": 5,
            "O": 6,
        }
        graph = nx.Graph()
        for atom in kekule.GetAtoms():
            graph.add_node(
                atom.GetIdx(),
                element=atom.GetSymbol(),
                charge=atom.GetFormalCharge(),
                hcount=atom.GetTotalNumHs(),
                lone_pairs=0,
                radical=atom.GetNumRadicalElectrons(),
                valence_electrons=valence_electrons[atom.GetSymbol()],
            )
        for bond in kekule.GetBonds():
            order = bond.GetBondTypeAsDouble()
            graph.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                sigma_order=1.0,
                pi_order=order - 1.0,
            )
        return graph

    def test_refresh_recomputes_kekule_order_and_charge(self):
        graph = nx.Graph()
        graph.add_node(
            1,
            element="O",
            charge=0,
            hcount=0,
            lone_pairs=2,
            radical=0,
            valence_electrons=6,
        )
        graph.add_node(
            2,
            element="C",
            charge=0,
            hcount=2,
            lone_pairs=0,
            radical=0,
            valence_electrons=4,
        )
        graph.add_edge(1, 2, sigma_order=1.0, pi_order=1.0)

        refreshed = refresh_electron_fields(graph)

        self.assertEqual(refreshed.edges[1, 2]["kekule_order"], 2.0)
        self.assertEqual(bond_order_sum(refreshed, 1), 2.0)
        self.assertEqual(recompute_charge(refreshed, 1), 0.0)
        self.assertFalse(refreshed.nodes[1]["charge_mismatch"])

    def test_refresh_changed_atom_charge_only_updates_selected_maps(self):
        graph = nx.Graph()
        graph.add_node(
            "o",
            atom_map=1,
            element="O",
            charge=99,
            hcount=1,
            lone_pairs=3,
            radical=0,
            valence_electrons=6,
        )
        graph.add_node(
            "c",
            atom_map=2,
            element="C",
            charge=99,
            hcount=3,
            lone_pairs=0,
            radical=0,
            valence_electrons=4,
        )

        reports = refresh_changed_atom_charge(graph, [1])

        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].atom_map, 1)
        self.assertEqual(graph.nodes["o"]["charge"], -1)
        self.assertEqual(graph.nodes["c"]["charge"], 99)

    def test_refresh_detects_charge_mismatch(self):
        graph = nx.Graph()
        graph.add_node(
            1,
            element="O",
            charge=0,
            hcount=1,
            lone_pairs=3,
            radical=0,
            valence_electrons=6,
        )
        graph.add_node(
            2,
            element="H",
            charge=0,
            hcount=0,
            lone_pairs=0,
            radical=0,
            valence_electrons=1,
        )
        graph.add_edge(1, 2, sigma_order=1.0, pi_order=0.0)

        refreshed = refresh_electron_fields(graph)

        self.assertEqual(refreshed.nodes[1]["recomputed_charge"], -2.0)
        self.assertTrue(refreshed.nodes[1]["charge_mismatch"])

    def test_radical_count_prevents_false_charge_on_hydroxyl_radical(self):
        graph = nx.Graph()
        graph.add_node(
            1,
            element="O",
            charge=0,
            hcount=1,
            lone_pairs=2,
            radical=1,
            valence_electrons=6,
        )

        refreshed = refresh_electron_fields(graph)

        self.assertEqual(refreshed.nodes[1]["recomputed_charge"], 0)
        self.assertFalse(refreshed.nodes[1]["charge_mismatch"])

    def test_radical_count_prevents_false_charge_on_methyl_radical(self):
        graph = nx.Graph()
        graph.add_node(
            1,
            element="C",
            charge=0,
            hcount=3,
            lone_pairs=0,
            radical=1,
            valence_electrons=4,
        )

        refreshed = refresh_electron_fields(graph)

        self.assertEqual(refreshed.nodes[1]["recomputed_charge"], 0)
        self.assertFalse(refreshed.nodes[1]["charge_mismatch"])

    def test_kekule_reconstruction_reperceives_aromatic_examples(self):
        cases = {
            "c1ccccc1": "c1ccccc1",
            "n1ccccc1": "c1ccncc1",
            "[nH]1cccc1": "c1cc[nH]c1",
            "o1cccc1": "c1ccoc1",
            "[nH+]1ccccc1": "c1cc[nH+]cc1",
        }

        for input_smiles, expected_smiles in cases.items():
            with self.subTest(smiles=input_smiles):
                graph = self._graph_from_kekule_smiles(input_smiles)
                mol = graph_to_sanitized_kekule_mol(graph)

                self.assertEqual(Chem.MolToSmiles(mol), expected_smiles)
                self.assertTrue(all(bond.GetIsAromatic() for bond in mol.GetBonds()))

    def test_kekule_reconstruction_does_not_invent_aromaticity(self):
        cases = {
            "C=CC=C": "C=CC=C",
            "C1=CC=CCC1": "C1=CCCC=C1",
        }

        for input_smiles, expected_smiles in cases.items():
            with self.subTest(smiles=input_smiles):
                graph = self._graph_from_kekule_smiles(input_smiles)
                mol = graph_to_sanitized_kekule_mol(graph)

                self.assertEqual(Chem.MolToSmiles(mol), expected_smiles)
                self.assertFalse(any(bond.GetIsAromatic() for bond in mol.GetBonds()))


if __name__ == "__main__":
    unittest.main()
