"""Independent XML quantities and controlled SBML participant variants."""

import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from Experiment.CRN.studies.sbml_reference import (
    crnt_from_sbml,
    reaction_side,
    read_structure,
    stoichiometric_columns,
)
from Experiment.CRN.studies import participant_conventions as study
from Experiment.CRN.studies import sbml_reference
from Experiment.CRN.tools.common import DATA_DIR
from synkit.CRN import native_backend


def sbml(reactions, *, level=3):
    return ET.fromstring(f"""<sbml level="{level}" version="1"
      xmlns="http://www.sbml.org/sbml/level{level}/version1/core">
      <model id="audit"><listOfCompartments><compartment id="c"/></listOfCompartments>
      <listOfSpecies><species id="A" compartment="c"/>
      <species id="B" compartment="c"/><species id="C" compartment="c"/>
      </listOfSpecies><listOfReactions>{reactions}</listOfReactions></model></sbml>""")


def sides(left, right):
    def listing(name, entries):
        return (
            f"<{name}>"
            + "".join(f'<speciesReference species="{key}" stoichiometry="{value}"/>' for key, value in entries)
            + f"</{name}>"
        )

    return listing("listOfReactants", left) + listing("listOfProducts", right)


def test_independent_level2_default_expands_both_directions():
    root = sbml('<reaction id="r">' + sides([("A", 1)], [("B", 1)]) + "</reaction>", level=2)
    result = crnt_from_sbml(root)
    assert result["n_reactions"] == 2
    assert result["n_omitted_reversible"] == 1
    assert result["weakly_reversible"] is True
    assert result["rank"] == 1
    assert result["deficiency"] == 0
    assert result["n_conservation_laws"] == 2  # includes the isolated C


def test_independent_level3_requires_explicit_reversible():
    root = sbml('<reaction id="r">' + sides([("A", 1)], [("B", 1)]) + "</reaction>")
    with pytest.raises(ValueError, match="explicit reversible"):
        crnt_from_sbml(root)


def test_rational_reference_runs_without_site_packages(tmp_path):
    root = sbml('<reaction id="r" reversible="false">' + sides([("A", "0.5")], [("B", 1)]) + "</reaction>")
    path = tmp_path / "rational.xml"
    ET.ElementTree(root).write(path)
    script = Path(sbml_reference.__file__)
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(script), str(path)], capture_output=True, text=True, check=True
    )
    report = json.loads(result.stdout)
    assert report["rank"] == 1
    assert report["deficiency"] == 0


def test_partial_common_stoichiometry_is_cancelled_exactly():
    root = sbml(
        '<reaction id="r" reversible="true">'
        + sides([("A", 2), ("C", 1)], [("A", 1), ("B", 1), ("C", 1)])
        + "</reaction>"
    )
    variant, changed = study.participant_variant(root, "cancelled")
    assert changed == 2
    assert stoichiometric_columns(root) == stoichiometric_columns(variant)
    reaction = next(item for item in variant.iter() if item.tag.endswith("}reaction"))
    assert reaction.get("reversible") == "true"
    assert reaction_side(reaction, "listOfReactants") == {"A": 1}
    assert reaction_side(reaction, "listOfProducts") == {"B": 1}
    assert len(read_structure(variant)["reactions"]) == 2


def test_modifier_promotion_adds_to_existing_participant():
    root = sbml(
        '<reaction id="r" reversible="false">'
        + sides([("A", 1), ("C", 1)], [("B", 1), ("C", 1)])
        + '<listOfModifiers><modifierSpeciesReference species="C"/>'
        "</listOfModifiers></reaction>"
    )
    variant, changed = study.participant_variant(root, "modifiers")
    reaction = next(item for item in variant.iter() if item.tag.endswith("}reaction"))
    assert changed == 1
    assert reaction_side(reaction, "listOfReactants")["C"] == 2
    assert reaction_side(reaction, "listOfProducts")["C"] == 2
    assert stoichiometric_columns(root) == stoichiometric_columns(variant)


def test_native_and_independent_audits_detect_deficiency_change(tmp_path, exact_library):
    root = sbml(
        '<reaction id="r1" reversible="false">'
        + sides([("A", 1), ("C", 1)], [("B", 1), ("C", 1)])
        + '</reaction><reaction id="r2" reversible="false">'
        + sides([("B", 1)], [("A", 1)])
        + '<listOfModifiers><modifierSpeciesReference species="C"/>'
        "</listOfModifiers></reaction>"
    )
    path = tmp_path / "toy.xml"
    ET.ElementTree(root).write(path)
    with native_backend(exact_library):
        row = study.analyse_participant_conventions(path)
    assert row["agrees"] and row["net_stoichiometry_unchanged"]
    assert row["deficiency_kept"] == 1
    assert row["deficiency_cancelled"] == 0
    assert row["deficiency_modifiers"] == 0


def test_manifest_checksum_failure_is_a_report_failure(monkeypatch, tmp_path):
    path = tmp_path / "BIOMD0000000001.xml"
    path.write_text("<invalid/>")
    monkeypatch.setattr(study, "fetch", lambda *args, **kwargs: path)
    report = study.participant_conventions_report(["BIOMD0000000001"], offline=True)
    assert report["status"] == "FAIL"
    assert "snapshot" in report["models"][0]["error"]
    assert report["checks"]["all_inputs_match_manifest"] is False


@pytest.mark.parametrize("model_id", study.DEFAULT_MODELS)
def test_cached_biomodels_match_independent_xml(model_id, exact_library):
    path = DATA_DIR / "biomodels" / f"{model_id}.xml"
    if not path.exists():
        pytest.skip("Optional BioModels XML cache is not distributed")
    manifest = json.loads((DATA_DIR / "biomodels_manifest.json").read_text())
    import hashlib

    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["models"][model_id]["sha256"]
    with native_backend(exact_library):
        row = study.analyse_participant_conventions(path)
    assert row["agrees"], row
    assert row["net_stoichiometry_unchanged"], row
    if model_id == "BIOMD0000000255":
        assert row["n_reactions"] == 1654
    if model_id == "BIOMD0000000404":
        assert row["deficiency_kept"] == 13
        assert row["deficiency_cancelled"] == 11
    if model_id == "BIOMD0000000637":
        reference = row["variants"]["kept"]["independent_reference"]
        assert reference["rank"] == 9 and reference["deficiency"] == 9
