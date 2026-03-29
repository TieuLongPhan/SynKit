from __future__ import annotations

from typing import Dict, List, Tuple, Any
from unittest.mock import patch

SEEDS = ["C=O", "OCC=O"]
RULES = [
    "[C:1]([C:2]=[O:3])[H:4]>>[C:1]=[C:2][O:3][H:4]",
    "[C:1]=[C:2][O:3][H:4].[O:5]=[C:6]>>[C:1]([C:2]=[O:3])[C:6][O:5][H:4]",
    "[C:1]=[C:2][O:3][H:4]>>[C:1]([C:2]=[O:3])[H:4]",
    "[C:1]([C:2]=[O:3])[C:6][O:5][H:4]>>[C:1]=[C:2][O:3][H:4].[O:5]=[C:6]",
]

A = SEEDS[0]
B = SEEDS[1]
X = "s_enol_1"
Y = "s_adduct_1"
Z = "s_adduct_2"
P = "s_p"
Q = "s_q"
R = "s_r"
S = "s_s"

# Explicit task-level chemistry stub used by builder / worker facing tests.
# Keys are (rule_index, sorted_reactant_tuple).
FAKE_PRODUCTS: Dict[Tuple[int, Tuple[str, ...]], List[str]] = {
    # Step 1
    (0, (A,)): [X],
    (2, (B,)): [A],
    # Step 2
    (1, tuple(sorted((A, X)))): [Y],
    (1, tuple(sorted((B, X)))): [Z],
    # Step 3
    (0, (Y,)): [P],
    (0, (Z,)): [Q],
    (1, tuple(sorted((A, Y)))): [R],
    (1, tuple(sorted((B, Y)))): [S],
    (1, tuple(sorted((A, Z)))): [X],
    (1, tuple(sorted((B, Z)))): [A],
}

# Step 4: 12 unary events total, 8 new species + 4 existing species.
_step4_unary = {
    (0, (P,)): ["u01"],
    (2, (P,)): ["u02"],
    (3, (P,)): [A],
    (0, (Q,)): ["u03"],
    (2, (Q,)): ["u04"],
    (3, (Q,)): [B],
    (0, (R,)): ["u05"],
    (2, (R,)): ["u06"],
    (3, (R,)): [X],
    (0, (S,)): ["u07"],
    (2, (S,)): ["u08"],
    (3, (S,)): [Y],
}
FAKE_PRODUCTS.update(_step4_unary)

# Step 4: choose 24 of 26 frontier-binary mixtures.
_step4_binary_mixtures = [
    tuple(sorted((A, P))),
    tuple(sorted((B, P))),
    tuple(sorted((X, P))),
    tuple(sorted((Y, P))),
    tuple(sorted((Z, P))),
    tuple(sorted((P, Q))),
    tuple(sorted((P, R))),
    tuple(sorted((P, S))),
    tuple(sorted((A, Q))),
    tuple(sorted((B, Q))),
    tuple(sorted((X, Q))),
    tuple(sorted((Y, Q))),
    tuple(sorted((Z, Q))),
    tuple(sorted((Q, R))),
    tuple(sorted((Q, S))),
    tuple(sorted((A, R))),
    tuple(sorted((B, R))),
    tuple(sorted((X, R))),
    tuple(sorted((Y, R))),
    tuple(sorted((Z, R))),
    tuple(sorted((R, S))),
    tuple(sorted((A, S))),
    tuple(sorted((B, S))),
    tuple(sorted((X, S))),
]

for idx, mix in enumerate(_step4_binary_mixtures, start=9):
    # First 18 binary events create new species u09..u26.
    if idx <= 26:
        FAKE_PRODUCTS[(1, mix)] = [f"u{idx:02d}"]
    # Remaining six binary events produce existing species while still
    # creating distinct deltas because the reactant multisets differ.

_existing_binary = {
    tuple(sorted((Y, R))): [B],
    tuple(sorted((Z, R))): [A],
    tuple(sorted((R, S))): [X],
    tuple(sorted((A, S))): [Y],
    tuple(sorted((B, S))): [Z],
    tuple(sorted((X, S))): [P],
}
for mix, prods in _existing_binary.items():
    FAKE_PRODUCTS[(1, mix)] = prods

EXPECTED_COUNTS = {
    2: {"nodes": 9, "edges": 10, "species": 5, "rules": 4},
    3: {"nodes": 19, "edges": 26, "species": 9, "rules": 10},
    4: {"nodes": 81, "edges": 122, "species": 35, "rules": 46},
}


def fake_apply_rule_worker(args):
    idx, rule, substrate, explicit_h, implicit_temp, strategy, reactant_keys = args
    del rule, explicit_h, implicit_temp, strategy, substrate
    return (
        idx,
        reactant_keys,
        list(FAKE_PRODUCTS.get((int(idx), tuple(reactant_keys)), [])),
    )


class FakeReactor:
    last_kwargs: Dict[str, Any] | None = None

    def __init__(self, smiles_list: List[str]):
        self.smiles_list = smiles_list

    @classmethod
    def from_smiles(cls, **kwargs):
        cls.last_kwargs = dict(kwargs)
        substrate = kwargs["smiles"]
        reactant_keys = tuple(sorted(x for x in substrate.split(".") if x))
        rule = kwargs["template"]
        idx = RULES.index(rule)
        products = list(FAKE_PRODUCTS.get((idx, reactant_keys), []))
        # include duplicates/blank to test worker-level cleanup
        smiles_list = []
        for p in products:
            smiles_list.extend([p, p])
        smiles_list.append("")
        return cls(smiles_list)


def identity_standardizer(smiles: str, *, keep_aam: bool):
    del keep_aam
    return smiles or None


def patch_builder_chemistry(builder_mod):
    return patch.multiple(
        builder_mod,
        apply_rule_worker=fake_apply_rule_worker,
        standardize_smiles_rdkit=identity_standardizer,
        Chem=None,
    )
