import tempfile
import unittest
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

from synkit.CRN.IO.sbml import (
    SBML_NS,
    _as_sid,
    _format_stoich,
    _is_true,
    _parse_stoich,
    crn_from_sbml,
    crn_to_sbml,
    read_sbml,
    write_sbml,
)
from synkit.CRN.Structure.syncrn import SynCRN

SBML_L2 = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
  <model id="m1" name="Hexokinase">
    <listOfCompartments>
      <compartment id="c"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="X" name="Glucose" compartment="c"/>
      <species id="Y" name="G6P" compartment="c"/>
      <species id="Z" name="ATP" compartment="c" boundaryCondition="true"/>
    </listOfSpecies>
    <listOfReactions>
      <reaction id="hk" reversible="true">
        <listOfReactants>
          <speciesReference species="X" stoichiometry="1"/>
          <speciesReference species="Z"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="Y" stoichiometry="2"/>
        </listOfProducts>
        <kineticLaw/>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


class TestSbmlHelpers(unittest.TestCase):
    def test_as_sid_passes_valid_ids(self):
        self.assertEqual(_as_sid("s_1", fallback="x"), "s_1")

    def test_as_sid_sanitizes_smiles(self):
        self.assertEqual(_as_sid("CC(=O)O", fallback="x"), "CC__O_O")

    def test_as_sid_prefixes_leading_digit(self):
        self.assertEqual(_as_sid("12", fallback="x"), "_12")

    def test_as_sid_falls_back_when_unsalvageable(self):
        self.assertEqual(_as_sid("", fallback="x"), "x")

    def test_format_stoich_drops_trailing_zero(self):
        self.assertEqual(_format_stoich(2), "2")
        self.assertEqual(_format_stoich(2.0), "2")
        self.assertEqual(_format_stoich(0.5), "0.5")

    def test_parse_stoich_defaults_to_one(self):
        self.assertEqual(_parse_stoich(None), 1.0)
        self.assertEqual(_parse_stoich(""), 1.0)
        self.assertEqual(_parse_stoich("junk"), 1.0)
        self.assertEqual(_parse_stoich("3"), 3.0)

    def test_is_true(self):
        self.assertTrue(_is_true("true"))
        self.assertTrue(_is_true("1"))
        self.assertFalse(_is_true("false"))
        self.assertFalse(_is_true(None))


class TestSbmlExport(unittest.TestCase):
    def setUp(self):
        self.crn = SynCRN.from_reaction_strings(["2A>>B+3C", "B+C>>D"])
        self.xml = crn_to_sbml(self.crn)

    def test_is_well_formed_level_3(self):
        root = ET.fromstring(self.xml)
        self.assertEqual(root.tag, f"{{{SBML_NS}}}sbml")
        self.assertEqual(root.get("level"), "3")
        self.assertEqual(root.get("version"), "2")

    def test_declares_every_species_and_reaction(self):
        root = ET.fromstring(self.xml)
        species = [e for e in root.iter() if e.tag.endswith("}species")]
        reactions = [e for e in root.iter() if e.tag.endswith("}reaction")]
        self.assertEqual(len(species), 4)
        self.assertEqual(len(reactions), 2)

    def test_carries_labels_as_names(self):
        root = ET.fromstring(self.xml)
        names = {
            e.get("name") for e in root.iter() if e.tag.endswith("}species")
        }
        self.assertEqual(names, {"A", "B", "C", "D"})

    def test_writes_stoichiometry(self):
        root = ET.fromstring(self.xml)
        coefficients = {
            e.get("stoichiometry")
            for e in root.iter()
            if e.tag.endswith("}speciesReference")
        }
        self.assertIn("2", coefficients)
        self.assertIn("3", coefficients)

    def test_rejects_non_syncrn(self):
        with self.assertRaises(TypeError):
            crn_to_sbml({"not": "a crn"})  # type: ignore[arg-type]

    def test_empty_network_is_still_valid(self):
        xml = crn_to_sbml(SynCRN.from_reaction_strings([]))
        root = ET.fromstring(xml)
        self.assertEqual(root.tag, f"{{{SBML_NS}}}sbml")

    def test_model_id_and_name(self):
        xml = crn_to_sbml(self.crn, model_id="my_model", model_name="My Model")
        root = ET.fromstring(xml)
        model = next(e for e in root.iter() if e.tag.endswith("}model"))
        self.assertEqual(model.get("id"), "my_model")
        self.assertEqual(model.get("name"), "My Model")


class TestSbmlRoundTrip(unittest.TestCase):
    RXNS = ["2A>>B+3C", "B+C>>D", "D>>B+C"]

    def test_equations_survive(self):
        crn = SynCRN.from_reaction_strings(self.RXNS)
        back = crn_from_sbml(crn_to_sbml(crn))
        self.assertEqual(
            back.to_equations(species="label", include_id=False),
            crn.to_equations(species="label", include_id=False),
        )

    def test_ids_survive(self):
        crn = SynCRN.from_reaction_strings(self.RXNS)
        back = crn_from_sbml(crn_to_sbml(crn))
        self.assertEqual(back.species_ids, crn.species_ids)
        self.assertEqual(back.reaction_ids, crn.reaction_ids)

    def test_annotation_does_not_create_phantom_reactions(self):
        # The SynKit annotation elements must not be mistaken for SBML content.
        crn = SynCRN.from_reaction_strings(self.RXNS)
        back = crn_from_sbml(crn_to_sbml(crn))
        self.assertEqual(back.n_reactions, 3)
        self.assertEqual(back.n_species, 4)

    def test_smiles_survive(self):
        crn = SynCRN.from_reaction_strings(["CCO>>CC=O"])
        crn.species["s_1"].smiles = "CCO"
        back = crn_from_sbml(crn_to_sbml(crn))
        self.assertEqual(back.species["s_1"].smiles, "CCO")

    def test_deficiency_is_preserved(self):
        from synkit.CRN.Props import deficiency

        crn = SynCRN.from_reaction_strings(
            ["A>>2A", "A+B>>C", "C>>A+B", "C>>B"]
        )
        self.assertEqual(deficiency(crn_from_sbml(crn_to_sbml(crn))), deficiency(crn))

    def test_file_roundtrip(self):
        crn = SynCRN.from_reaction_strings(self.RXNS)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_sbml(crn, Path(tmp) / "net.xml")
            self.assertTrue(path.exists())
            back = read_sbml(path)
        self.assertEqual(back.n_reactions, crn.n_reactions)


class TestSbmlImport(unittest.TestCase):
    def test_reads_level_2(self):
        crn = crn_from_sbml(SBML_L2)
        self.assertEqual(crn.n_species, 3)

    def test_reversible_reaction_is_split(self):
        crn = crn_from_sbml(SBML_L2)
        self.assertEqual(crn.n_reactions, 2)
        self.assertEqual(
            crn.to_equations(species="label", include_id=False),
            ["Glucose + ATP >> 2G6P", "2G6P >> Glucose + ATP"],
        )

    def test_reversible_expansion_can_be_disabled(self):
        crn = crn_from_sbml(SBML_L2, expand_reversible=False)
        self.assertEqual(crn.n_reactions, 1)

    def test_boundary_species_can_be_dropped(self):
        crn = crn_from_sbml(SBML_L2, drop_boundary_species=True)
        labels = {sp.label for sp in crn.species.values()}
        self.assertNotIn("ATP", labels)
        self.assertEqual(crn.n_species, 2)

    def test_missing_stoichiometry_defaults_to_one(self):
        crn = crn_from_sbml(SBML_L2, expand_reversible=False)
        rxn = next(iter(crn.reactions.values()))
        self.assertEqual(sorted(rxn.lhs.to_dict().values()), [1, 1])

    def test_kinetic_law_is_ignored(self):
        self.assertEqual(crn_from_sbml(SBML_L2).n_species, 3)

    def test_species_names_become_labels(self):
        labels = {sp.label for sp in crn_from_sbml(SBML_L2).species.values()}
        self.assertEqual(labels, {"Glucose", "G6P", "ATP"})

    def test_accepts_parsed_element(self):
        crn = crn_from_sbml(ET.fromstring(SBML_L2))
        self.assertEqual(crn.n_species, 3)

    def test_accepts_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "l2.xml"
            path.write_text(SBML_L2, encoding="utf-8")
            self.assertEqual(crn_from_sbml(path).n_species, 3)
            self.assertEqual(crn_from_sbml(str(path)).n_species, 3)

    def test_document_without_model_raises(self):
        with self.assertRaises(ValueError):
            crn_from_sbml('<sbml xmlns="http://x" level="3" version="2"/>')

    def test_duplicate_species_references_are_summed(self):
        xml = """<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core">
          <model id="m">
            <listOfSpecies>
              <species id="A"/><species id="B"/>
            </listOfSpecies>
            <listOfReactions>
              <reaction id="r1" reversible="false">
                <listOfReactants>
                  <speciesReference species="A" stoichiometry="1"/>
                  <speciesReference species="A" stoichiometry="1"/>
                </listOfReactants>
                <listOfProducts>
                  <speciesReference species="B" stoichiometry="1"/>
                </listOfProducts>
              </reaction>
            </listOfReactions>
          </model>
        </sbml>"""
        crn = crn_from_sbml(xml)
        self.assertEqual(
            crn.to_equations(species="label", include_id=False), ["2A >> B"]
        )

    def test_reaction_free_model_warns(self):
        # An SBML-qual logical model has transitions, not reactions. Returning
        # an empty network silently looks like a parse failure to the caller.
        xml = """<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core">
          <model id="qual_model">
            <qual:listOfQualitativeSpecies xmlns:qual="http://q"/>
            <qual:listOfTransitions xmlns:qual="http://q"/>
          </model>
        </sbml>"""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            crn = crn_from_sbml(xml)
        self.assertEqual(crn.n_reactions, 0)
        self.assertTrue(caught)
        self.assertIn("qual", str(caught[0].message))

    def test_model_with_species_but_no_reactions_warns(self):
        xml = """<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core">
          <model id="m"><listOfSpecies><species id="A"/></listOfSpecies></model>
        </sbml>"""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            crn = crn_from_sbml(xml)
        self.assertEqual(crn.n_reactions, 0)
        self.assertTrue(caught)
        self.assertIn("no reactions", str(caught[0].message))

    def test_normal_model_does_not_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            crn_from_sbml(SBML_L2)
        self.assertEqual([w for w in caught if "no reactions" in str(w.message)], [])

    def test_reference_to_unknown_species_is_skipped(self):
        xml = """<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core">
          <model id="m">
            <listOfSpecies><species id="A"/></listOfSpecies>
            <listOfReactions>
              <reaction id="r1" reversible="false">
                <listOfReactants>
                  <speciesReference species="A"/>
                </listOfReactants>
                <listOfProducts>
                  <speciesReference species="GHOST"/>
                </listOfProducts>
              </reaction>
            </listOfReactions>
          </model>
        </sbml>"""
        crn = crn_from_sbml(xml)
        self.assertEqual(crn.n_species, 1)
        self.assertEqual(crn.n_reactions, 1)


if __name__ == "__main__":
    unittest.main()
