import json
import platform
from collections import Counter
from pathlib import Path

import pytest
from rdkit import Chem, rdBase
from rdkit.Chem import rdCIPLabeler

import synkit
from synkit.Chem.Reaction import audit_explicit_h_reaction
from synkit.Graph.Stereo import PlanarBondStereo, StereoOutcome
from synkit.IO.chem_converter import rsmi_to_its
from synkit.Rule import SynRule
from synkit.Synthesis.Reactor.syn_reactor import SynReactor

ROOT = Path(__file__).parents[3]
DATA_PATH = ROOT / "Data/MechanismBench/stereo.json"
PAYLOAD = json.loads(DATA_PATH.read_text(encoding="utf-8"))
CASES = PAYLOAD["cases"]
TRANSFORMATIONS = [case for case in CASES if case["case_kind"] == "transformation"]
NEGATIVE_ASSERTIONS = [
    case for case in CASES if case["case_kind"] == "negative_assertion"
]
STEPS = [(case, step) for case in TRANSFORMATIONS for step in case["steps"]]
RUNNABLE_STEPS = [
    (case, step)
    for case, step in STEPS
    if case["status"] in {"executable", "graph_only"}
]
HYDROGEN_ABSENCE_ERRORS = {
    "NO_EXPLICIT_MAPPED_HYDROGEN",
    "NO_CHANGED_EXPLICIT_HYDROGEN",
}
SMILES_PARAMS = Chem.SmilesParserParams()
SMILES_PARAMS.removeHs = False


def _id(value):
    return value["case_id"]


def _rule(step):
    serialized = step.get("rule_metadata", {}).get("stereo_outcomes", {})
    if not serialized:
        return step["mapped_reaction"]
    outcomes = {
        key: StereoOutcome(value["kind"], tuple(value["weights"]))
        for key, value in serialized.items()
    }
    return SynRule.from_smart(
        step["mapped_reaction"],
        format="tuple",
        stereo_outcomes=outcomes,
    )


def _reactor(step, substrate=None):
    options = {
        "explicit_h": False,
        "radical_policy": "strict",
        "stereo_mode": "strict",
    }
    options.update(step["application"].get("reactor_options", {}))
    return SynReactor(
        substrate or step["application"]["substrate"],
        _rule(step),
        template_format="tuple",
        **options,
    )


def _map_neutral_molecule(smiles):
    molecule = Chem.MolFromSmiles(smiles, SMILES_PARAMS)
    assert molecule is not None
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
        if atom.HasProp("_CIPCode"):
            atom.ClearProp("_CIPCode")
    for bond in molecule.GetBonds():
        if bond.HasProp("_CIPCode"):
            bond.ClearProp("_CIPCode")
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    rdCIPLabeler.AssignCIPLabels(molecule)
    return molecule


def _canonical_stereoisomer(smiles):
    return Chem.MolToSmiles(
        _map_neutral_molecule(smiles),
        canonical=True,
        isomericSmiles=True,
    )


def _cip_signature(smiles):
    molecule = _map_neutral_molecule(smiles)
    return tuple(
        sorted(
            atom.GetProp("_CIPCode")
            for atom in molecule.GetAtoms()
            if atom.HasProp("_CIPCode")
        )
    )


def _double_bond_cip(smiles):
    molecule = _map_neutral_molecule(smiles)
    return {
        bond.GetProp("_CIPCode")
        for bond in molecule.GetBonds()
        if bond.HasProp("_CIPCode") and bond.GetProp("_CIPCode") in {"E", "Z"}
    }


def _selected_steps(case):
    if case["step_relationship"] == "sequential":
        return case["steps"][-1:]
    return case["steps"]


def _generated_products(case):
    products = []
    for step in _selected_steps(case):
        products.extend(
            candidate.split(">>", 1)[1] for candidate in _reactor(step).smarts
        )
    return products


def test_dataset_matches_the_local_validation_environment():
    assert PAYLOAD["schema"] == "SynKit-stereo-conformance-v0.3"
    assert PAYLOAD["toolkit"] == {
        "python": platform.python_version(),
        "rdkit": rdBase.rdkitVersion,
        "synkit": synkit.__version__,
        "platform": f"{platform.system()} {platform.machine()}",
        "cip_labeller": "rdCIPLabeler.AssignCIPLabels",
    }
    assert len(CASES) == 48
    assert len({case["case_id"] for case in CASES}) == len(CASES)
    assert Counter(case["status"] for case in CASES) == {
        "executable": 33,
        "graph_only": 4,
        "deferred_isotope_support": 3,
        "specification_only": 8,
    }


@pytest.mark.parametrize(
    ("case", "step"),
    STEPS,
    ids=[f"{case['case_id']}-{step['step_id']}" for case, step in STEPS],
)
def test_transformation_step_parses_balances_and_stores_declared_stereo(case, step):
    left, product = step["mapped_reaction"].split(">>", 1)
    assert Chem.MolFromSmiles(left, SMILES_PARAMS) is not None
    assert Chem.MolFromSmiles(product, SMILES_PARAMS) is not None

    report = audit_explicit_h_reaction(step["mapped_reaction"])
    assert not set(report.errors) - HYDROGEN_ABSENCE_ERRORS

    its = rsmi_to_its(
        step["mapped_reaction"],
        format="tuple",
        drop_non_aam=False,
        use_index_as_atom_map=True,
    )
    assert {
        key: change.change
        for key, change in its.graph.get("stereo_changes", {}).items()
    } == step["expected_stereo_changes"]


@pytest.mark.parametrize(
    ("case", "step"),
    RUNNABLE_STEPS,
    ids=[f"{case['case_id']}-{step['step_id']}" for case, step in RUNNABLE_STEPS],
)
def test_runnable_step_applies_and_rejects_wrong_stereoisomers(case, step):
    reactor = _reactor(step)
    unique_products = {
        _canonical_stereoisomer(candidate.split(">>", 1)[1])
        for candidate in reactor.smarts
    }

    assert reactor.mapping_count > 0
    assert (
        len(unique_products)
        == step["application"]["expected_unique_stereoisomer_count"]
    )
    for rejected_substrate in step["application"]["rejected_substrates"]:
        rejected = _reactor(step, rejected_substrate)
        assert rejected.mapping_count == 0
        assert rejected.smarts == []


@pytest.mark.parametrize(
    "case",
    [case for case in TRANSFORMATIONS if case["status"] == "executable"],
    ids=_id,
)
def test_executable_case_emits_declared_product_stereo(case):
    products = _generated_products(case)
    expected_cip = {
        tuple(sorted(values.values())) for values in case["oracle"]["product_cip_sets"]
    }

    assert {_cip_signature(product) for product in products} == expected_cip

    expected_bonds = set(case["oracle"].get("product_descriptor_by_locus", {}).values())
    if expected_bonds:
        assert set().union(*(_double_bond_cip(product) for product in products)) == (
            expected_bonds
        )


@pytest.mark.parametrize(
    "case",
    [case for case in TRANSFORMATIONS if case["status"] == "graph_only"],
    ids=_id,
)
def test_graph_only_case_preserves_planar_descriptor_in_product_graph(case):
    for step in _selected_steps(case):
        reactor = _reactor(step)
        assert reactor.its_list
        for its in reactor.its_list:
            product_registry = its.graph["stereo_descriptors"]["product"]
            assert any(
                isinstance(descriptor, PlanarBondStereo)
                for descriptor in product_registry.values()
            )
    assert case["known_blocker"]["code"].endswith("NOT_SERIALIZED")


@pytest.mark.parametrize("case", NEGATIVE_ASSERTIONS, ids=_id)
def test_negative_assertion_is_well_formed_specification(case):
    assertion = case["assertion"]
    target = assertion["target"]

    assert case["status"] == "specification_only"
    assert assertion["must_not_assign"]
    assert assertion["required_annotation"]
    if assertion["target_kind"] == "reaction":
        left, product = target.split(">>", 1)
        assert Chem.MolFromSmiles(left, SMILES_PARAMS) is not None
        assert Chem.MolFromSmiles(product, SMILES_PARAMS) is not None
    else:
        assert assertion["target_kind"] in {"molecule", "annotation_pair"}
        assert Chem.MolFromSmiles(target, SMILES_PARAMS) is not None


def test_deferred_cases_declare_the_isotope_blocker():
    deferred = [
        case for case in TRANSFORMATIONS if case["status"] == "deferred_isotope_support"
    ]

    assert {case["case_id"] for case in deferred} == {"ST-04", "ST-06", "ST-19"}
    assert all(
        case["known_blocker"]["code"] == "ISOTOPE_LABEL_NOT_PRESERVED"
        for case in deferred
    )
