"""Frozen public A/B/C case definitions, independent of execution policy."""

from dataclasses import dataclass

_BUDGETS = ("A", "B", "C")


@dataclass(frozen=True)
class CaseSpec:
    budget: str
    source: str
    record_id: str
    composition: tuple[tuple[str, int], ...]
    raw_product: int
    formal_assignments: int

    def __post_init__(self) -> None:
        if self.budget not in _BUDGETS:
            raise ValueError(f"Unknown global benchmark budget: {self.budget}.")
        if self.source not in {"acs", "cip", "rota"}:
            raise ValueError(f"Unknown global benchmark source: {self.source}.")
        if len(self.composition) < 2:
            raise ValueError("ABC global cases must mix at least two families.")


def _composition(**families: int) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(families.items()))


_CASES = (
    # A: every mixed-family pairing available in the molecular datasets.
    CaseSpec("A", "cip", "VS141", _composition(cumulene_axis=1, planar_bond=1), 64, 4),
    CaseSpec(
        "A",
        "cip",
        "VS154",
        _composition(extended_cis_trans=1, planar_bond=1),
        64,
        4,
    ),
    CaseSpec(
        "A", "rota", "RotA-0071", _composition(atrop_bond=1, planar_bond=1), 64, 4
    ),
    CaseSpec(
        "A", "rota", "RotA-0073", _composition(atrop_bond=1, planar_bond=1), 64, 4
    ),
    CaseSpec("A", "acs", "VS032", _composition(planar_bond=1, tetrahedral=1), 192, 4),
    CaseSpec("A", "cip", "VS072", _composition(atrop_bond=1, tetrahedral=1), 192, 4),
    CaseSpec("A", "cip", "VS078", _composition(cumulene_axis=1, tetrahedral=1), 192, 4),
    CaseSpec(
        "A",
        "cip",
        "VS164",
        _composition(extended_cis_trans=1, tetrahedral=1),
        192,
        4,
    ),
    CaseSpec(
        "A",
        "rota",
        "RotA-0072",
        _composition(atrop_bond=1, planar_bond=2),
        512,
        8,
    ),
    CaseSpec(
        "A",
        "rota",
        "RotA-0468",
        _composition(atrop_bond=2, planar_bond=1),
        512,
        8,
    ),
    # B: increasing multiplicity and matched symmetry/non-symmetry cases.
    CaseSpec("B", "acs", "VS192", _composition(planar_bond=2, tetrahedral=1), 1_536, 8),
    CaseSpec(
        "B",
        "rota",
        "RotA-0551",
        _composition(atrop_bond=2, tetrahedral=1),
        1_536,
        8,
    ),
    CaseSpec("B", "acs", "VS018", _composition(planar_bond=1, tetrahedral=2), 4_608, 8),
    CaseSpec(
        "B",
        "cip",
        "VS063",
        _composition(extended_cis_trans=1, tetrahedral=2),
        4_608,
        8,
    ),
    CaseSpec(
        "B",
        "cip",
        "VS120",
        _composition(cumulene_axis=1, tetrahedral=2),
        4_608,
        8,
    ),
    CaseSpec(
        "B",
        "cip",
        "VS243",
        _composition(cumulene_axis=1, tetrahedral=2),
        4_608,
        8,
    ),
    CaseSpec(
        "B",
        "rota",
        "RotA-0262",
        _composition(atrop_bond=3, tetrahedral=1),
        12_288,
        16,
    ),
    CaseSpec(
        "B",
        "rota",
        "RotA-0195",
        _composition(atrop_bond=2, tetrahedral=2),
        36_864,
        16,
    ),
    CaseSpec(
        "B",
        "rota",
        "RotA-0497",
        _composition(atrop_bond=2, tetrahedral=2),
        36_864,
        16,
    ),
    CaseSpec(
        "B",
        "acs",
        "VS053",
        _composition(planar_bond=4, tetrahedral=1),
        98_304,
        32,
    ),
    CaseSpec(
        "B",
        "rota",
        "RotA-0273",
        _composition(atrop_bond=5, tetrahedral=1),
        786_432,
        64,
    ),
    # C: mixed-family stress cases, capped at 1,024 formal assignments.
    CaseSpec(
        "C",
        "acs",
        "VS146",
        _composition(planar_bond=4, tetrahedral=2),
        2_359_296,
        64,
    ),
    CaseSpec(
        "C",
        "acs",
        "VS245",
        _composition(planar_bond=1, tetrahedral=4),
        2_654_208,
        32,
    ),
    CaseSpec(
        "C",
        "rota",
        "RotA-0256",
        _composition(atrop_bond=1, tetrahedral=4),
        2_654_208,
        32,
    ),
    CaseSpec(
        "C",
        "acs",
        "VS191",
        _composition(planar_bond=6, tetrahedral=1),
        6_291_456,
        128,
    ),
    CaseSpec(
        "C",
        "rota",
        "RotA-0272",
        _composition(atrop_bond=7, tetrahedral=1),
        50_331_648,
        256,
    ),
    CaseSpec(
        "C",
        "acs",
        "VS259",
        _composition(planar_bond=1, tetrahedral=5),
        63_700_992,
        64,
    ),
    CaseSpec(
        "C",
        "rota",
        "RotA-0378",
        _composition(planar_bond=1, tetrahedral=5),
        63_700_992,
        64,
    ),
    CaseSpec(
        "C",
        "rota",
        "RotA-0290",
        _composition(atrop_bond=2, tetrahedral=6),
        12_230_590_464,
        256,
    ),
    CaseSpec(
        "C",
        "acs",
        "VS126",
        _composition(planar_bond=4, tetrahedral=6),
        782_757_789_696,
        1_024,
    ),
)

CASES_BY_BUDGET = {
    budget: tuple(case for case in _CASES if case.budget == budget)
    for budget in _BUDGETS
}
