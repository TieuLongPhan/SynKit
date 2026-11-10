"""Distinguish aggregate balance, initial enabling and complete route firing."""

import pytest

from Experiment.CRN.studies.representation import inventory_controls
from synkit.CRN import native_backend
from synkit.CRN.backend import python_backend


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_glycolysis_requires_priming_atp_and_net_cofactor_supply(backend, exact_library):
    context = python_backend() if backend == "python" else native_backend(exact_library)
    with context:
        controls = inventory_controls()
    cases = {row["name"]: row for row in controls["cases"]}
    assert controls["adenylate_siphon"]["valid_on_prescribed_transitions"]
    assert cases["glucose_only"]["outcome"]["explored_states"] == 1
    assert cases["two_atp_with_redox_and_phosphate"]["net_inventory_deficits"] == {"ADP": 2}
    blocked = cases["one_atp_with_adp_redox_and_phosphate"]
    assert blocked["net_inventory_feasible"]
    assert blocked["outcome"]["status"] == "unrealizable"
    assert blocked["outcome"]["exhaustive"]
    supplied = cases["two_atp_with_adp_redox_and_phosphate"]["outcome"]
    assert supplied["status"] == "realizable"
    assert supplied["independent_replay_ok"]
    assert len(supplied["certificate"]) == 15
    assert supplied["final_marking"]["Pyruvate"] == 2
    assert supplied["final_marking"]["ATP"] == 4
    assert supplied["final_marking"]["NADH"] == 2
