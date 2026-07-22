from synkit.Chem import utils
from synkit.Chem.Reaction import (
    audit_explicit_h_reaction,
    remove_explicit_H_from_rsmi,
)


def test_reaction_reexports_remove_explicit_h_from_rsmi():
    assert remove_explicit_H_from_rsmi is utils.remove_explicit_H_from_rsmi


def test_reaction_reexports_explicit_h_audit():
    report = audit_explicit_h_reaction("[H:2][CH3:1]>>[H:2][CH3:1]")

    assert "NO_CHANGED_EXPLICIT_HYDROGEN" in report.errors
