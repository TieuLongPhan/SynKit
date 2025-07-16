import unittest
from rdkit import Chem
from synkit.Graph.ITS.its_relabel import ITSRelabel
from synkit.Graph.syn_graph import SynGraph
from synkit.IO.chem_converter import smiles_to_graph


class TestITSRelabel(unittest.TestCase):
    def setUp(self):
        self.its = ITSRelabel()

    def test_get_nodes_with_atom_map(self):
        # Build a graph from SMILES with explicit atom mapping
        raw = smiles_to_graph(
            "[CH3:1][CH2:2][OH:3]", use_index_as_atom_map=True, drop_non_aam=False
        )
        sg = SynGraph(raw)
        nodes = ITSRelabel._get_nodes_with_atom_map(sg)
        # Every atom should have a non-zero atom_map
        self.assertCountEqual(nodes, list(sg.raw.nodes()))

    def test_remove_internal_edges(self):
        # Linear 3‑carbon chain; after removing internal edges none remain
        raw = smiles_to_graph("CCC", use_index_as_atom_map=True, drop_non_aam=False)
        sg = SynGraph(raw)
        all_nodes = list(sg.raw.nodes())
        pruned = ITSRelabel._remove_internal_edges(sg, all_nodes)
        self.assertEqual(pruned.raw.number_of_edges(), 0)

    def test_dict_to_tuple_list_sorting(self):
        mapping = {3: 1, 2: 2, 1: 3}
        # Sort by key
        by_key = ITSRelabel._dict_to_tuple_list(mapping, sort_by_key=True)
        self.assertEqual(by_key, [(1, 3), (2, 2), (3, 1)])
        # Sort by value
        by_val = ITSRelabel._dict_to_tuple_list(mapping, sort_by_value=True)
        self.assertEqual(by_val, [(3, 1), (2, 2), (1, 3)])
        # No sorting
        no_sort = ITSRelabel._dict_to_tuple_list(mapping)
        self.assertCountEqual(no_sort, [(1, 3), (2, 2), (3, 1)])

    def test_fit_simple_reaction(self):
        # CCO to CC=O, mapping preserved
        input_rsmi = "CC[CH2:3][Cl:1].[N:2]>>CC[CH2:3][N:2].[Cl:1]"
        out = self.its.fit(input_rsmi)
        react, prod = out.split(">>")
        self.assertIsNotNone(Chem.MolFromSmiles(react))
        self.assertIsNotNone(Chem.MolFromSmiles(prod))

    def test_fit_invalid_format_raises(self):
        with self.assertRaises(ValueError):
            self.its.fit("invalid_format")

    def test_fit_non_isomorphic_raises(self):
        # Wrong reaction format
        with self.assertRaises(ValueError):
            self.its.fit("C:1>CC:1")


if __name__ == "__main__":
    unittest.main()
