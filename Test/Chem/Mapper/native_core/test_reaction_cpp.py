"""Optional standalone C++ reaction pipeline: chemistry and class-ID parity."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from synkit.Chem.Mapper import Mapper
from synkit.Chem.Mapper._native.core import map_reaction_core
from synkit.Chem.Mapper._native.reaction import map_reaction_native
from synkit.Chem.Mapper._native.runtime import CancellationToken


@pytest.fixture(scope="module")
def bundle():
    path = os.environ.get("SYNKIT_REACTION_LIBRARY")
    if not path:
        pytest.skip("Explicit optional RDKit C++ build not supplied")
    library = Path(path)
    core = library.with_name("libsynkit_mapper.so")
    assert core.is_file()
    return library, core


@pytest.mark.parametrize(
    "reaction,options",
    [
        ("O=CC=O>>OC#CO", {}),
        ("[NH4+].[OH-]>>N.O", {}),
        ("[CH3]>>[CH3]", {}),
        ("[13CH3]O>>[13CH3]O", {}),
        ("F/C=C/F>>F/C=C\\F", {}),
        ("F[C@H](Cl)Br>>F[C@@H](Cl)Br", {}),
        ("c1cc[nH]c1>>c1cc[nH]c1", {}),
        ("c1ccsc1>>c1ccsc1", {}),
        ("C>>C.O", {"balance": "dummy"}),
        ("O>>[OH-]", {"balance": "dummy"}),
        ("[2H]O>>O[2H]", {}),
        (
            "CC>>CC",
            {"CD": 4, "objective": "bond", "hydrogen": "heavy", "CD_mode": "at_most"},
        ),
        ("O=CC=O>>OC#CO", {"CD": 8}),
    ],
)
def test_chemistry_and_legacy_class_ids_match_python_frontend(
    bundle, reaction, options
):
    library, core = bundle
    actual = map_reaction_native(
        reaction,
        library_path=library,
        max_mappings=None,
        time_limit_seconds=10,
        **options,
    )
    reference = map_reaction_core(
        reaction,
        library_path=core,
        native_classification=False,
        max_mappings=None,
        time_limit_seconds=10,
        **options,
    )
    assert actual["complete"] and reference.complete
    assert actual["minimum_cost"] == reference.minimum_cost
    assert {(r["class_id"], r["distance"]) for r in actual["classes"]} == {
        (r.class_id, r.distance) for r in reference.reactions
    }
    assert actual["pipeline"] == "rdkit_cpp"


@pytest.mark.parametrize(
    "reaction,options,expected_fallback",
    [
        ("O=CC=O>>OC#CO", {}, False),
        ("CC>>CC", {"CD": 4}, False),
        ("O>>[OH-]", {"balance": "dummy"}, False),
        ("[2H]O>>O[2H]", {}, True),
        ("F[C@H](Cl)Br>>F[C@H](Cl)Br", {}, True),
    ],
)
def test_native_hydrogen_quotient_or_explicit_fallback(
    bundle, reaction, options, expected_fallback
):
    library, _ = bundle
    reduced = map_reaction_native(
        reaction,
        library_path=library,
        hydrogen_quotient=True,
        max_mappings=None,
        **options,
    )
    explicit = map_reaction_native(
        reaction, library_path=library, max_mappings=None, **options
    )
    assert reduced["complete"] and explicit["complete"]
    assert reduced["minimum_cost"] == explicit["minimum_cost"]
    assert {(r["class_id"], r["distance"]) for r in reduced["classes"]} == {
        (r["class_id"], r["distance"]) for r in explicit["classes"]
    }
    assert reduced["hydrogen_quotient_fallback"] == expected_fallback


def test_thin_adapter_calls_no_python_chemistry_and_executable_needs_no_python(
    bundle, monkeypatch
):
    from rdkit import Chem

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Python chemical preparation/rendering was called")

    for name in ("MolFromSmiles", "MolToSmiles", "AddHs", "RemoveHs"):
        monkeypatch.setattr(Chem, name, forbidden)
    library, _ = bundle
    result = map_reaction_native("[NH4+].[OH-]>>N.O", library_path=library)
    assert result["complete"] and result["minimum_cost"] == 4
    executable = library.with_name("mapper_reaction")
    output = subprocess.check_output([str(executable), "O=CC=O>>OC#CO"], text=True)
    assert json.loads(output)["minimum_cost"] == 8


def test_empty_fractional_shell_and_interruption_are_honest(bundle):
    library, _ = bundle
    empty = map_reaction_native("O>>O", library_path=library, CD=0.25)
    assert (
        empty["complete"] and empty["classes"] == [] and empty["minimum_cost"] is None
    )
    stopped = map_reaction_native(
        "O=CC=O>>OC#CO", library_path=library, time_limit_seconds=0
    )
    assert not stopped["complete"] and stopped["statistics"]["reason"] == 1
    assert stopped.keys() == empty.keys()
    assert stopped["statistics"].keys() == empty["statistics"].keys()
    assert stopped["statistics"]["backend"] == 1
    assert stopped["statistics"]["elapsed_seconds"] >= 0
    proof = map_reaction_native("O=CC=O>>OC#CO", library_path=library, proof_only=True)
    assert proof["minimum_proven"] and not proof["complete"] and proof["classes"] == []
    limited = map_reaction_native("O=CC=O>>OC#CO", library_path=library, max_classes=1)
    assert not limited["complete"] and limited["minimum_proven"]


def test_cancellation_and_agents_preserve_result_schema(bundle):
    library, core = bundle
    completed = map_reaction_native("O>CC>O", library_path=library)
    assert completed["complete"] and completed["agents"] == "CC"
    with CancellationToken(core) as token:
        token.cancel()
        stopped = map_reaction_native(
            "O>CC>O", library_path=library, cancel_token=token
        )
    assert stopped.keys() == completed.keys()
    assert stopped["statistics"]["reason"] == 10 and not stopped["complete"]


@pytest.mark.parametrize("backend", ["kbb", "kmilp"])
def test_native_kernel_pipeline_exposes_restricted_scope(bundle, backend):
    library, _ = bundle
    actual = map_reaction_native("CCO>>CCO", library_path=library, backend=backend)
    if backend == "kmilp" and actual["statistics"]["status"] == 2:
        pytest.skip("SCIP optional engine not linked")
    assert actual["complete"] and actual["minimum_cost"] == 0
    assert (
        actual["search_scope"] == "restricted_kernel"
        and actual["kernel_seed_candidates"] > 0
    )


def test_public_hydrogen_completion_tracks_transport_interruption(bundle):
    _, core = bundle
    result = Mapper("native", library_path=core, hydrogen_quotient=True).map(
        "CC>>CC",
        CD=8,
        CD_mode="at_most",
        max_hydrogen_plans=1,
        max_mappings=None,
    )
    assert not result.complete
    assert not result.reaction_result.hydrogen_complete
