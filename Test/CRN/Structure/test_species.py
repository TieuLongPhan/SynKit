import unittest

from synkit.CRN.Structure.species import Species


class TestSpecies(unittest.TestCase):
    def test_init_minimal(self):
        species = Species(
            id="s_1",
            source_node_id="node_1",
            label="H2O",
        )
        self.assertEqual(species.id, "s_1")
        self.assertEqual(species.source_node_id, "node_1")
        self.assertEqual(species.label, "H2O")
        self.assertIsNone(species.smiles)
        self.assertEqual(species.source_attrs, {})
        self.assertEqual(species.metadata, {})

    def test_init_full(self):
        species = Species(
            id="s_2",
            source_node_id=42,
            label="ethanol",
            smiles="CCO",
            source_attrs={"kind": "species", "charge": 0},
            metadata={"mass": 46.07, "origin": "input_graph"},
        )
        self.assertEqual(species.id, "s_2")
        self.assertEqual(species.source_node_id, 42)
        self.assertEqual(species.label, "ethanol")
        self.assertEqual(species.smiles, "CCO")
        self.assertEqual(species.source_attrs, {"kind": "species", "charge": 0})
        self.assertEqual(species.metadata, {"mass": 46.07, "origin": "input_graph"})

    def test_to_dict_minimal(self):
        species = Species(
            id="s_1",
            source_node_id="node_1",
            label="A",
        )
        self.assertEqual(
            species.to_dict(),
            {
                "id": "s_1",
                "source_node_id": "node_1",
                "label": "A",
                "smiles": None,
                "source_attrs": {},
                "metadata": {},
            },
        )

    def test_to_dict_full(self):
        species = Species(
            id="s_3",
            source_node_id="orig_3",
            label="acetone",
            smiles="CC(=O)C",
            source_attrs={"kind": "species", "step": 2},
            metadata={"inchi_key": "CSCPPACGZOOCGX-UHFFFAOYSA-N"},
        )
        self.assertEqual(
            species.to_dict(),
            {
                "id": "s_3",
                "source_node_id": "orig_3",
                "label": "acetone",
                "smiles": "CC(=O)C",
                "source_attrs": {"kind": "species", "step": 2},
                "metadata": {"inchi_key": "CSCPPACGZOOCGX-UHFFFAOYSA-N"},
            },
        )

    def test_to_dict_returns_copied_dicts(self):
        species = Species(
            id="s_4",
            source_node_id="node_4",
            label="B",
            source_attrs={"foo": 1},
            metadata={"bar": 2},
        )
        d = species.to_dict()

        self.assertEqual(d["source_attrs"], {"foo": 1})
        self.assertEqual(d["metadata"], {"bar": 2})
        self.assertIsNot(d["source_attrs"], species.source_attrs)
        self.assertIsNot(d["metadata"], species.metadata)

    def test_mutable_default_dicts_are_not_shared(self):
        s1 = Species(id="s_1", source_node_id="n1", label="A")
        s2 = Species(id="s_2", source_node_id="n2", label="B")

        s1.source_attrs["x"] = 1
        s1.metadata["y"] = 2

        self.assertEqual(s1.source_attrs, {"x": 1})
        self.assertEqual(s1.metadata, {"y": 2})
        self.assertEqual(s2.source_attrs, {})
        self.assertEqual(s2.metadata, {})

    def test_source_node_id_can_be_any_hashable(self):
        species_str = Species(id="s_1", source_node_id="node_A", label="A")
        species_int = Species(id="s_2", source_node_id=123, label="B")
        species_tuple = Species(id="s_3", source_node_id=("node", 3), label="C")

        self.assertEqual(species_str.source_node_id, "node_A")
        self.assertEqual(species_int.source_node_id, 123)
        self.assertEqual(species_tuple.source_node_id, ("node", 3))


if __name__ == "__main__":
    unittest.main()
