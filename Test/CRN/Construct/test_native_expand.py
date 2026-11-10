"""Native rule expansion must reproduce the RDKit Python worker exactly."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from synkit.CRN.Construct import CRNExpand, build_native_expander, native_expander

FORMOSE = [
    "[C:1]([C:2]=[O:3])[H:4]>>[C:1]=[C:2][O:3][H:4]",
    "[C:1]=[C:2][O:3][H:4].[O:5]=[C:6]>>[C:1]([C:2]=[O:3])[C:6][O:5][H:4]",
    "[C:1]=[C:2][O:3][H:4]>>[C:1]([C:2]=[O:3])[H:4]",
    "[C:1]([C:2]=[O:3])[C:6][O:5][H:4]>>[C:1]=[C:2][O:3][H:4].[O:5]=[C:6]",
]
CATALYSIS = [
    "[C:1]=[C:2].[Pd:3]>>[C:1]([Pd:3])[C:2]",
    "[C:1]([Pd:3])[C:2]>>[C:1]=[C:2].[Pd:3]",
    "[C:1]=[O:2].[N:3]>>[C:1]=[N:3].[O:2]",
]


def _sdk():
    configured = os.environ.get("SYNKIT_RDKIT_CPP_DIR")
    candidates = [Path(configured)] if configured else []
    candidates += sorted((Path.home() / ".cache/synkit").glob("rdkit-cpp-*"), reverse=True)
    for candidate in candidates:
        if (candidate / "include/rdkit/GraphMol/ChemReactions/Reaction.h").is_file():
            return candidate
    return None


@pytest.fixture(scope="session")
def expander(tmp_path_factory):
    sdk = _sdk()
    if sdk is None:
        pytest.skip("Set SYNKIT_RDKIT_CPP_DIR to an RDKit C++ SDK for native expansion")
    return build_native_expander(tmp_path_factory.mktemp("crn-expand"), rdkit_dir=sdk)


def _signature(graph):
    from rdkit import Chem

    species = {
        node: Chem.MolToSmiles(Chem.MolFromSmiles(data["smiles"]))
        for node, data in graph.nodes(data=True)
        if data["kind"] == "species"
    }
    reactions = []
    for node, data in graph.nodes(data=True):
        if data["kind"] != "rule":
            continue
        left = sorted((species[u], e["stoich"]) for u, _, e in graph.in_edges(node, data=True))
        right = sorted((species[v], e["stoich"]) for _, v, e in graph.out_edges(node, data=True))
        reactions.append((node, data["label"], data["step"], left, right))
    return [(n, species[n]) for n in graph.nodes if n in species], reactions


@pytest.mark.parametrize(
    "rules, seeds, repeats, options",
    [
        (FORMOSE, ["C=O", "OCC=O"], 4, {}),
        (FORMOSE, ["C=O", "OCC=O"], 5, {}),
        (FORMOSE, ["C=O", "OCC=O"], 4, {"allow_self_mixtures": True}),
        (FORMOSE, ["C=O", "OCC=O"], 3, {"use_frontier": False}),
        (FORMOSE, ["C=O", "OCC=O"], 4, {"strip_spectators": True, "dedup_across_rules": True}),
        (CATALYSIS, ["C=CC=O", "[Pd]", "N"], 3, {"allow_self_mixtures": True}),
    ],
)
def test_native_matches_rdkit_worker(expander, rules, seeds, repeats, options):
    common = dict(rules=rules, repeats=repeats, keep_aam=False, **options)
    reference = CRNExpand(rule_backend="rdkit", **common).build(seeds)
    builder = CRNExpand(rule_backend="native", native_expander=str(expander), **common)
    native = builder.build(seeds)
    assert _signature(native) == _signature(reference)
    assert builder.native_report["stop_reason"] in {"repeats", "fixed_point", "no_tasks"}


def test_formose_size_and_pruning(expander):
    builder = CRNExpand(rules=FORMOSE, repeats=4, keep_aam=False, rule_backend="native")
    with native_expander(expander):
        graph = builder.build(["C=O", "OCC=O"])
    kinds = [data["kind"] for _, data in graph.nodes(data=True)]
    assert kinds.count("species") == 37 and kinds.count("rule") == 44
    steps = builder.native_report["steps"]
    # Only mixtures that bind every reactant template reach the reaction engine.
    assert all(step["tasks"] == step["candidate_mixtures"] for step in steps)


def test_threads_are_deterministic(expander):
    common = dict(rules=FORMOSE, repeats=5, keep_aam=False, rule_backend="native", native_expander=str(expander))
    one = CRNExpand(threads=1, **common).build(["C=O", "OCC=O"])
    four = CRNExpand(threads=4, **common).build(["C=O", "OCC=O"])
    assert _signature(one) == _signature(four)


def test_native_requires_expander_and_smarts():
    with pytest.raises(ValueError):
        CRNExpand(rules=FORMOSE, rule_backend="native", keep_aam=False).build(["C=O"])
    with pytest.raises(ValueError):
        CRNExpand(rules=FORMOSE, rule_backend="native", keep_aam=True)


def _balanced(graph):
    """Return True when every reaction conserves atoms (with H) and charge."""
    from collections import defaultdict

    from rdkit import Chem

    def inventory(smiles):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        counts = defaultdict(int)
        for atom in mol.GetAtoms():
            counts[atom.GetSymbol()] += 1
        counts["charge"] = Chem.GetFormalCharge(mol)
        return counts

    smiles = {n: d["smiles"] for n, d in graph.nodes(data=True) if d["kind"] == "species"}
    for node, data in graph.nodes(data=True):
        if data["kind"] != "rule":
            continue
        net = defaultdict(int)
        for u, _, e in graph.in_edges(node, data=True):
            for key, value in inventory(smiles[u]).items():
                net[key] -= value * e["stoich"]
        for _, v, e in graph.out_edges(node, data=True):
            for key, value in inventory(smiles[v]).items():
                net[key] += value * e["stoich"]
        if any(net.values()):
            return False
    return True


def _sets(graph):
    species, reactions = _signature(graph)
    return sorted(s for _, s in species), sorted((r[3], r[4]) for r in reactions)


@pytest.mark.parametrize("repeats", [4, 5])
def test_lewis_hydrogens_reproduce_explicit_formose(expander, repeats):
    common = dict(rules=FORMOSE, repeats=repeats, keep_aam=False, rule_backend="native", native_expander=str(expander))
    explicit = CRNExpand(hydrogen_model="explicit", **common).build(["C=O", "OCC=O"])
    builder = CRNExpand(hydrogen_model="lewis", **common)
    lewis = builder.build(["C=O", "OCC=O"])
    assert _sets(lewis) == _sets(explicit)
    assert [r["hydrogen_model"] for r in builder.native_report["rules"]] == ["lewis"] * 4
    assert builder.native_report["rules"][0]["hydrogen_deltas"] == {"1": -1, "3": 1}


@pytest.mark.parametrize(
    "rules, seeds, expected",
    [
        # Hydrogen-atom transfer: the Lewis model yields ethyl radical + water.
        (["[C:1][H:2].[O:3]>>[C:1].[O:3][H:2]"], ["CC", "[OH]"], {"O", "[CH2]C"}),
        # Proton transfer between bracket atoms with fixed hydrogen counts.
        (["[N+:1][H:2].[O-:3]>>[N:1].[O:3][H:2]"], ["[NH4+]", "CC(=O)[O-]"], {"N", "CC(=O)O"}),
    ],
)
def test_lewis_hydrogens_conserve_atoms_and_charge(expander, rules, seeds, expected):
    builder = CRNExpand(
        rules=rules,
        repeats=1,
        keep_aam=False,
        rule_backend="native",
        native_expander=str(expander),
        hydrogen_model="lewis",
    )
    graph = builder.build(seeds)
    species = {d["smiles"] for _, d in graph.nodes(data=True) if d["kind"] == "species"}
    assert expected <= species
    assert _balanced(graph)


def test_lewis_never_fabricates_hydrogen_for_incomplete_rules(expander):
    common = dict(rules=CATALYSIS[:2], repeats=3, keep_aam=False, rule_backend="native", native_expander=str(expander))
    assert not _balanced(CRNExpand(hydrogen_model="explicit", **common).build(["C=CC=O", "[Pd]"]))
    assert _balanced(CRNExpand(hydrogen_model="lewis", **common).build(["C=CC=O", "[Pd]"]))


def test_lewis_falls_back_for_uncompilable_rules(expander):
    rules = ["[C:1]=[C:2].[H:3][H:4]>>[C:1]([H:3])[C:2][H:4]"]
    builder = CRNExpand(
        rules=rules,
        repeats=1,
        keep_aam=False,
        rule_backend="native",
        native_expander=str(expander),
        hydrogen_model="lewis",
    )
    builder.build(["C=C", "[H][H]"])
    (rule,) = builder.native_report["rules"]
    assert rule["hydrogen_model"] == "explicit" and rule["note"]


def test_lewis_requires_native_backend():
    with pytest.raises(ValueError):
        CRNExpand(rules=FORMOSE, keep_aam=False, rule_backend="rdkit", hydrogen_model="lewis")
