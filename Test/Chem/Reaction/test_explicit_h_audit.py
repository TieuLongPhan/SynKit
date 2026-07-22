import pytest

from synkit.Chem.Reaction import (
    MappedBondChange,
    audit_explicit_h_reaction,
    reaction_smiles_from_annotated_text,
)

HYDROGENATION = (
    "[CH3:1]/[CH:2]=[CH:3]/[CH3:4].[H:5][H:6]>>"
    "[CH3:1][CH:2]([H:5])[CH:3]([H:6])[CH3:4]"
)


def test_accepts_balanced_reaction_with_mapped_hydrogen_bond_changes():
    report = audit_explicit_h_reaction(HYDROGENATION)

    assert report.accepted
    assert report.errors == ()
    assert report.reactant_formula == report.product_formula == "C4H10"
    assert report.reactant_charge == report.product_charge == 0
    assert report.atom_maps == (1, 2, 3, 4, 5, 6)
    assert report.explicit_hydrogen_maps == (5, 6)
    assert report.changed_hydrogen_maps == (5, 6)
    assert set(report.changed_bonds) == {
        MappedBondChange((2, 3), 2.0, 1.0),
        MappedBondChange((2, 5), 0.0, 1.0),
        MappedBondChange((3, 6), 0.0, 1.0),
        MappedBondChange((5, 6), 1.0, 0.0),
    }


def test_extracts_reaction_from_legacy_electron_flow_annotation():
    annotated = f"{HYDROGENATION} 5,6-2,5;2,3=3,6"

    assert reaction_smiles_from_annotated_text(annotated) == HYDROGENATION
    assert audit_explicit_h_reaction(annotated).accepted


@pytest.mark.parametrize(
    ("reaction", "error"),
    [
        (
            "[H:3][O:1][CH3:2]>>[H:3][O:1].[CH3:2]",
            "NO_CHANGED_EXPLICIT_HYDROGEN",
        ),
        ("[H][CH3:1]>>[CH4:1]", "UNMAPPED_REACTANT_ATOM"),
        ("[CH3:1][H:2]>>[CH3:1]", "MAP_INVENTORY_MISMATCH"),
        ("[CH2:1][H:2]>>[CH3:1][H:2]", "FORMULA_IMBALANCE"),
        ("[N:1][H:2]>>[N+:1][H:2]", "CHARGE_IMBALANCE"),
        ("[CH3:1][Cl:2]>>[CH3:1].[Cl:2]", "NO_EXPLICIT_MAPPED_HYDROGEN"),
    ],
)
def test_rejects_records_that_only_look_suitable(reaction, error):
    report = audit_explicit_h_reaction(reaction)

    assert not report.accepted
    assert error in report.errors


def test_charge_change_does_not_masquerade_as_formula_imbalance():
    report = audit_explicit_h_reaction("[N:1][H:2]>>[N+:1][H:2]")

    assert "CHARGE_IMBALANCE" in report.errors
    assert "FORMULA_IMBALANCE" not in report.errors


def test_rejects_duplicate_maps_and_atom_identity_changes():
    duplicate = audit_explicit_h_reaction("[H:2][CH3:1].[Cl:1]>>[H:2][CH3:1].[Cl:1]")
    identity_change = audit_explicit_h_reaction("[H:2][CH3:1]>>[H:2][NH2:1]")

    assert "DUPLICATE_REACTANT_MAP" in duplicate.errors
    assert "DUPLICATE_PRODUCT_MAP" in duplicate.errors
    assert "MAPPED_ATOM_IDENTITY_MISMATCH" in identity_change.errors


def test_reports_invalid_shape_and_parse_failure_without_raising():
    assert audit_explicit_h_reaction("CCO").errors == ("INVALID_REACTION_SEPARATOR",)
    assert audit_explicit_h_reaction("invalid>>[H:1][H:2]").errors == (
        "REACTANT_PARSE_FAILED",
    )
