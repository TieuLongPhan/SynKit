"""The implicit-H proof must equal full Lewis optimization without H expansion."""

import pytest
from rdkit import Chem

from synkit.Chem.Mapper import map_reaction
from synkit.Chem.Mapper.exact.native.pmcd import (
    _implicit_quotient_graphs,
    prove_reaction_pmcd,
)
from synkit.Chem.Mapper.reaction_search import _graphs, _parse


@pytest.mark.parametrize(
    "reaction,balance",
    [
        ("N.Cl>>[NH4+].[Cl-]", "strict"),
        ("O=CC=O>>OC#CO", "strict"),
        ("CO>>[CH2-][OH2+]", "strict"),
        ("CO>>C", "dummy"), ("C>>CO", "dummy"),
        ("O>>[OH-]", "dummy"), ("[OH-]>>O", "dummy"),
        ("C[C@H](O)F>>C[C@@H](O)F", "strict"),
        ("F/C=C/F>>F/C=C\\F", "strict"),
        ("c1cc[nH]c1>>c1cc[nH]c1", "strict"),
    ],
)
def test_implicit_proof_matches_full_minimum_without_h_or_class_expansion(
    native_library, monkeypatch, reaction, balance
):
    explicit = map_reaction(
        reaction, backend="cpp", library_path=native_library, balance=balance,
        max_mappings=None,
    )
    assert explicit.complete

    def forbidden_expansion(*args, **kwargs):
        raise AssertionError("The proof-only path must not add H atoms")

    monkeypatch.setattr(Chem, "AddHs", forbidden_expansion)
    result = prove_reaction_pmcd(
        reaction, library_path=native_library, balance=balance
    )
    assert result.optimality_proven and result.status == "proven"
    assert result.minimum_cost == explicit.minimum_cost
    assert result.hydrogen_nodes_added == result.hydrogen_vertices_searched == 0
    assert not result.enumeration_complete
    assert len(result.heavy_mapping) + 2 == result.search_atoms


@pytest.mark.parametrize("smiles", ["CO", "c1cc[nH]c1", "c1ncccc1", "[NH4+]", "OP(=O)(O)O"])
def test_implicit_electron_and_hydrogen_counts_equal_explicit_endpoint(smiles):
    reduced, counts, totals = _implicit_quotient_graphs(smiles + ">>" + smiles, "strict")
    molecules, _ = _parse(smiles + ">>" + smiles, "explicit")
    explicit = _graphs(molecules, "strict", "lewis")
    heavy = [atom for atom in molecules[0].GetAtoms() if atom.GetAtomicNum() != 1]
    electron_anchor = len(explicit[0].labels) - 1
    assert counts[0] == len(heavy)
    assert totals[0] == sum(atom.GetAtomicNum() == 1 for atom in molecules[0].GetAtoms())
    for i, atom in enumerate(heavy):
        full_e = 2 * explicit[0].graph[atom.GetIdx()].get(electron_anchor, 0)
        assert reduced[0].props["electrons"][i] == full_e
        assert reduced[0].props["hcounts"][i] == sum(
            neighbor.GetAtomicNum() == 1 for neighbor in atom.GetNeighbors()
        )


@pytest.mark.parametrize("reaction", ["C=C.[H][H]>>CC", "[OH].[H]>>O", "[2H]O>>[2H]O"])
def test_unsupported_hydrogen_never_silently_becomes_heavy_only_proof(
    native_library, monkeypatch, reaction
):
    monkeypatch.setattr(Chem, "AddHs", lambda *a, **kw: pytest.fail("Unexpected expansion"))
    result = prove_reaction_pmcd(reaction, library_path=native_library, balance="dummy")
    assert result.status == "unsupported" and result.reason
    assert result.minimum_cost is None and not result.optimality_proven


def test_zero_deadline_and_balance_contract(native_library):
    timed = prove_reaction_pmcd("CO>>CO", library_path=native_library, time_limit_seconds=0)
    assert timed.status == "time_limit" and timed.minimum_cost is None
    assert not timed.optimality_proven
    with pytest.raises(ValueError, match="Unbalanced"):
        prove_reaction_pmcd("CO>>C", library_path=native_library)
