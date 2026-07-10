import pytest
from rdkit import Chem

from synkit.Graph.FG import FunctionalGroupDetector
from synkit.IO.mol_to_graph import MolToGraph


def _detect(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    graph = MolToGraph().transform(mol)
    return FunctionalGroupDetector().detect(graph)


@pytest.mark.parametrize(
    "smiles, expected",
    [
        ("C=O", [("aldehyde", (1, 2))]),
        ("C(=O)N", [("amide", (1, 2, 3))]),
        ("CC(=O)O", [("carboxylic_acid", (2, 3, 4))]),
        ("COC(C)=O", [("ester", (2, 3, 5))]),
        ("NCC(=O)O", [("amine", (1,)), ("carboxylic_acid", (3, 4, 5))]),
        ("CCSCC", [("thioether", (3,))]),
        ("CSC(=O)c1ccccc1", [("thioester", (2, 3, 4))]),
        (
            "O=C(C)Oc1ccccc1C(=O)O",
            [("ester", (1, 2, 4)), ("carboxylic_acid", (11, 12, 13))],
        ),
        ("CC(C)(C)OO", [("peroxide", (5, 6))]),
        ("CC(=O)OO", [("peroxy_acid", (2, 3, 4, 5))]),
        ("CCO", [("primary_alcohol", (2, 3))]),
        ("CC(C)O", [("secondary_alcohol", (2, 4))]),
        ("CC(C)(C)O", [("tertiary_alcohol", (2, 5))]),
        ("C=CO", [("enol", (1, 2, 3))]),
        ("C1CO1", [("epoxide", (1, 2, 3))]),
        ("c1ccccc1O", [("phenol", (7,))]),
        ("c1ccccc1N", [("aniline", (7,))]),
        ("CC(=O)Cl", [("acyl_chloride", (2, 3, 4))]),
        ("CC#N", [("nitrile", (2, 3))]),
        ("CN=O", [("nitroso", (2, 3))]),
        ("C[N+](=O)[O-]", [("nitro", (2, 3, 4))]),
        ("COC(=O)N", [("carbamate", (2, 3, 4, 5))]),
        ("CC(=O)OC(C)=O", [("anhydride", (2, 3, 4, 5, 7))]),
        ("COC(O)(C)C", [("hemiketal", (2, 3, 4))]),
        ("CO[CH](O)C", [("hemiacetal", (2, 3, 4))]),
        ("COC(OC)(C)C", [("ketal", (2, 3, 4))]),
        ("CO[CH](OC)C", [("acetal", (2, 3, 4))]),
        (
            "n1ccccc1",
            [
                ("heteroaromatic_ring", (1, 2, 3, 4, 5, 6)),
                ("pyridine", (1, 2, 3, 4, 5, 6)),
            ],
        ),
        (
            "[nH]1cccc1",
            [
                ("heteroaromatic_ring", (1, 2, 3, 4, 5)),
                ("pyrrole", (1, 2, 3, 4, 5)),
            ],
        ),
        (
            "Cn1cccc1",
            [
                ("heteroaromatic_ring", (2, 3, 4, 5, 6)),
                ("pyrrole", (2, 3, 4, 5, 6)),
            ],
        ),
        (
            "o1cccc1",
            [
                ("furan", (1, 2, 3, 4, 5)),
                ("heteroaromatic_ring", (1, 2, 3, 4, 5)),
            ],
        ),
        (
            "s1cccc1",
            [
                ("heteroaromatic_ring", (1, 2, 3, 4, 5)),
                ("thiophene", (1, 2, 3, 4, 5)),
            ],
        ),
        (
            "n1ccncc1",
            [
                ("diazine", (1, 2, 3, 4, 5, 6)),
                ("heteroaromatic_ring", (1, 2, 3, 4, 5, 6)),
            ],
        ),
        (
            "c1ncc[nH]1",
            [
                ("heteroaromatic_ring", (1, 2, 3, 4, 5)),
                ("imidazole", (1, 2, 3, 4, 5)),
            ],
        ),
        (
            "c1cn[nH]c1",
            [
                ("heteroaromatic_ring", (1, 2, 3, 4, 5)),
                ("pyrazole", (1, 2, 3, 4, 5)),
            ],
        ),
        (
            "c1ccc2[nH]cnc2c1",
            [
                ("benzimidazole", (1, 2, 3, 4, 5, 6, 7, 8, 9)),
                ("heteroaromatic_ring", (1, 2, 3, 4, 5, 6, 7, 8, 9)),
                ("imidazole", (4, 5, 6, 7, 8)),
            ],
        ),
        (
            "c1ccc2[nH]ccc2c1",
            [
                ("heteroaromatic_ring", (1, 2, 3, 4, 5, 6, 7, 8, 9)),
                ("indole", (1, 2, 3, 4, 5, 6, 7, 8, 9)),
                ("pyrrole", (4, 5, 6, 7, 8)),
            ],
        ),
        (
            "c1ccc2[nH]ncc2c1",
            [
                ("heteroaromatic_ring", (1, 2, 3, 4, 5, 6, 7, 8, 9)),
                ("indazole", (1, 2, 3, 4, 5, 6, 7, 8, 9)),
                ("pyrazole", (4, 5, 6, 7, 8)),
            ],
        ),
        (
            "c1ccc2ocnc2c1",
            [
                ("benzoxazole", (1, 2, 3, 4, 5, 6, 7, 8, 9)),
                ("heteroaromatic_ring", (1, 2, 3, 4, 5, 6, 7, 8, 9)),
                ("oxazole", (4, 5, 6, 7, 8)),
            ],
        ),
        (
            "c1ccc2scnc2c1",
            [
                ("benzothiazole", (1, 2, 3, 4, 5, 6, 7, 8, 9)),
                ("heteroaromatic_ring", (1, 2, 3, 4, 5, 6, 7, 8, 9)),
                ("thiazole", (4, 5, 6, 7, 8)),
            ],
        ),
        (
            "c1nccs1",
            [
                ("heteroaromatic_ring", (1, 2, 3, 4, 5)),
                ("thiazole", (1, 2, 3, 4, 5)),
            ],
        ),
        (
            "c1ccns1",
            [
                ("heteroaromatic_ring", (1, 2, 3, 4, 5)),
                ("isothiazole", (1, 2, 3, 4, 5)),
            ],
        ),
        (
            "c1ncco1",
            [
                ("heteroaromatic_ring", (1, 2, 3, 4, 5)),
                ("oxazole", (1, 2, 3, 4, 5)),
            ],
        ),
        (
            "c1ccon1",
            [
                ("heteroaromatic_ring", (1, 2, 3, 4, 5)),
                ("isoxazole", (1, 2, 3, 4, 5)),
            ],
        ),
        (
            "c1n[nH]cn1",
            [
                ("heteroaromatic_ring", (1, 2, 3, 4, 5)),
                ("triazole", (1, 2, 3, 4, 5)),
            ],
        ),
        (
            "c1nocn1",
            [
                ("heteroaromatic_ring", (1, 2, 3, 4, 5)),
                ("oxadiazole", (1, 2, 3, 4, 5)),
            ],
        ),
        (
            "c1nn[nH]n1",
            [
                ("heteroaromatic_ring", (1, 2, 3, 4, 5)),
                ("tetrazole", (1, 2, 3, 4, 5)),
            ],
        ),
        (
            "c1nscn1",
            [
                ("heteroaromatic_ring", (1, 2, 3, 4, 5)),
                ("thiadiazole", (1, 2, 3, 4, 5)),
            ],
        ),
        (
            "c1ncnnc1",
            [
                ("heteroaromatic_ring", (1, 2, 3, 4, 5, 6)),
                ("triazine", (1, 2, 3, 4, 5, 6)),
            ],
        ),
        ("CCCl", [("organohalide", (2, 3))]),
        ("c1ccccc1Cl", [("aryl_halide", (6, 7))]),
        ("CS(=O)C", [("sulfoxide", (2, 3))]),
        ("CS(=O)(=O)C", [("sulfone", (2, 3, 4))]),
        ("CS(=O)(=O)N", [("sulfonamide", (2, 3, 4, 5))]),
        ("OB(O)c1ccccc1", [("boronic_acid", (1, 2, 3))]),
        ("B(OC)(OC)c1ccccc1", [("boronate_ester", (1, 2, 4))]),
        ("CO[Si](C)(C)C", [("silyl_ether", (2, 3))]),
        ("COP(=O)(OC)OC", [("phosphate", (2, 3, 4, 5, 7))]),
        ("CP(=O)(OC)OC", [("phosphonate", (1, 2, 3, 4, 6))]),
        ("CP(=O)(C)C", [("phosphine_oxide", (1, 2, 3, 4, 5))]),
        ("COP(OC)OC", [("phosphite", (2, 3, 4, 6))]),
        ("O=C=Nc1ccccc1", [("isocyanate", (1, 2, 3))]),
        ("ON=Cc1ccccc1", [("oxime", (1, 2, 3))]),
        ("CNN=C(C)C", [("hydrazone", (2, 3, 4))]),
        ("CC=NC", [("imine", (2, 3))]),
        ("N=C(N)c1ccccc1", [("amidine", (1, 2, 3))]),
        ("NC(=NO)c1ccccc1", [("amidoxime", (1, 2, 3, 4))]),
        ("CN=[N+]=[N-]", [("azide", (2, 3, 4))]),
        ("c1ccccc1N=Nc1ccccc1", [("azo", (7, 8))]),
        ("S=C=Nc1ccccc1", [("isothiocyanate", (1, 2, 3))]),
        ("NC(=S)N", [("thiourea", (1, 2, 3, 4))]),
        ("CC(=S)N", [("thioamide", (2, 3, 4))]),
    ],
)
def test_detects_compatibility_groups(smiles, expected):
    assert _detect(smiles) == expected


def test_raw_matches_keep_parent_before_resolution():
    mol = Chem.MolFromSmiles("CC(=O)O")
    graph = MolToGraph().transform(mol)
    detector = FunctionalGroupDetector()

    raw_names = {match.name for match in detector.raw_matches(graph)}
    assert {"carbonyl", "carboxylic_acid"}.issubset(raw_names)
    assert detector.detect(graph) == [("carboxylic_acid", (2, 3, 4))]


def test_internal_prerequisite_patterns_do_not_leak_into_public_results():
    mol = Chem.MolFromSmiles("COC")
    graph = MolToGraph().transform(mol)
    detector = FunctionalGroupDetector()

    assert {match.name for match in detector.raw_matches(graph)} == {"ether"}
    assert {
        match.name for match in detector.raw_matches(graph, include_internal=True)
    } == {"ether", "oxygen_link"}


def test_water_is_not_alcohol():
    assert _detect("O") == []


def test_heteroaromatic_ring_suppresses_generic_amine():
    assert _detect("n1ccccc1") == [
        ("heteroaromatic_ring", (1, 2, 3, 4, 5, 6)),
        ("pyridine", (1, 2, 3, 4, 5, 6)),
    ]


def test_diazine_keeps_generic_heteroaromatic_coverage():
    assert _detect("n1ccncc1") == [
        ("diazine", (1, 2, 3, 4, 5, 6)),
        ("heteroaromatic_ring", (1, 2, 3, 4, 5, 6)),
    ]


@pytest.mark.parametrize(
    "implicit_smiles, explicit_smiles, expected",
    [
        ("CCO", "[CH3][CH2][OH]", [("primary_alcohol", (2, 3))]),
        ("C=O", "[CH2]=O", [("aldehyde", (1, 2))]),
        ("CC(=O)O", "[CH3][C](=O)[OH]", [("carboxylic_acid", (2, 3, 4))]),
    ],
)
def test_implicit_and_explicit_hydrogen_inputs_agree(
    implicit_smiles,
    explicit_smiles,
    expected,
):
    assert _detect(implicit_smiles) == expected
    assert _detect(explicit_smiles) == expected
