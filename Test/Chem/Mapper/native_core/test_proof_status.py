"""A completed minimum proof is distinct from enumeration and deadline expiry."""

import pytest

from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph
from synkit.Chem.Mapper.solvers.native.core import enumerate_core_mappings
from synkit.Chem.Mapper.solvers.native.runtime import CancellationToken


def endpoints():
    return (
        LabeledGraph({0: {1: 0.5}, 1: {0: 0.5}}, [6, 8]),
        LabeledGraph({0: {1: 1.0}, 1: {0: 1.0}}, [6, 8]),
    )


@pytest.mark.parametrize("backend", ["pabs", "astar", "cpabs"])
def test_proved_minimum_is_not_an_enumeration_timeout(mapper_core_library, backend):
    proof = enumerate_core_mappings(
        endpoints(), library_path=mapper_core_library, backend=backend,
        proof_only=True, time_limit_seconds=2,
    )
    assert proof.status == "minimum_proven"
    assert proof.minimum_cost == 0.5
    assert proof.backend_statistics["minimum_proven"]
    assert not proof.backend_statistics["enumeration_requested"]
    assert not proof.complete and not proof.mappings
    assert proof.truncation_reason == "proof_only"
    # A real enumeration of the same problem still returns its complete map.
    complete = enumerate_core_mappings(
        endpoints(), library_path=mapper_core_library, backend=backend,
        time_limit_seconds=2,
    )
    assert complete.status == "complete" and complete.complete
    assert complete.mappings == [[0, 1]] and complete.minimum_cost == 0.5


def test_zero_deadline_does_not_reuse_a_proven_status(mapper_core_library):
    enumerate_core_mappings(endpoints(), library_path=mapper_core_library, proof_only=True)
    timed = enumerate_core_mappings(
        endpoints(), library_path=mapper_core_library, proof_only=True,
        time_limit_seconds=0,
    )
    assert timed.status == "timeout" and timed.truncation_reason == "time_limit"
    assert timed.minimum_cost is None and not timed.backend_statistics["minimum_proven"]


def test_cancelled_proof_is_not_reported_as_proved(mapper_core_library):
    token = CancellationToken(mapper_core_library)
    token.cancel()
    result = enumerate_core_mappings(
        endpoints(), library_path=mapper_core_library, proof_only=True,
        cancel_token=token,
    )
    assert result.status != "minimum_proven"
    assert result.truncation_reason == "cancelled"
    assert result.minimum_cost is None and not result.complete
