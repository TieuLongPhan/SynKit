"""Portable 2D fixtures adapted from StereoMolGraph's public tests.

Source: maxim-papusha/StereoMolGraph commit
2189f610f23eaaf992e2e01a12ea4d0532496601 (MIT).
Copyright (c) 2025 Maxim Papusha.

Only connectivity and stereochemistry encoded in InChI or SMILES are kept.
XYZ geometries, coordinate inference, and transition-state trajectories are
outside this fixture module's scope.
"""

from __future__ import annotations

from dataclasses import dataclass


PINNED_COMMIT = "2189f610f23eaaf992e2e01a12ea4d0532496601"

MAPPED_CONNECTIVITY_SMILES = (
    "[C:1]([O:2][C:33]([C:4]([O:5][H:13])([H:111])[H:12])"
    "([H:9])[H:10])([H:6])([H:77])[H:8]"
)
MAPPED_CONNECTIVITY_ATOMS = frozenset(
    {1, 2, 33, 4, 5, 6, 77, 8, 9, 10, 111, 12, 13}
)


@dataclass(frozen=True)
class InchiCase:
    """A portable InChI round-trip fixture."""

    name: str
    inchi: str
    expected_stereo_descriptors: int


@dataclass(frozen=True)
class NonTetrahedralCase:
    """A non-tetrahedral RDKit fixture with a recorded SynKit boundary."""

    name: str
    stereo_class: str
    smiles: str


INCHI_ROUND_TRIP_CASES = (
    InchiCase(
        "isopropanol",
        "InChI=1S/C3H8O/c1-3(2)4/h3-4H,1-2H3",
        0,
    ),
    InchiCase(
        "caffeine",
        "InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/"
        "h4H,1-3H3",
        0,
    ),
    InchiCase(
        "alpha-D-gulopyranose",
        "InChI=1S/C6H12O6/c7-1-2-3(8)4(9)5(10)6(11)12-2/"
        "h2-11H,1H2/t2-,3-,4+,5-,6+/m1/s1",
        5,
    ),
    InchiCase(
        "(R)-bromochlorofluoromethane",
        "InChI=1S/CHBrClF/c2-1(3)4/h1H/t1-/m0/s1",
        1,
    ),
    InchiCase(
        "trans-1,2-dichloroethylene",
        "InChI=1S/C2H2Cl2/c3-1-2-4/h1-2H/b2-1+",
        1,
    ),
    InchiCase(
        "cis-1,2-dichloroethylene",
        "InChI=1S/C2H2Cl2/c3-1-2-4/h1-2H/b2-1-",
        1,
    ),
    InchiCase(
        "butadiene",
        "InChI=1S/C4H6/c1-3-4-2/h3-4H,1-2H2",
        0,
    ),
)


SQUARE_PLANAR_CASES = (
    NonTetrahedralCase(
        "hydridofluorochlorobromoplatinum(II)",
        "square_planar",
        "[H][Pt@SP1](F)(Cl)Br",
    ),
    NonTetrahedralCase(
        "(SP-4-2)-diamminedichloroplatinum",
        "square_planar",
        "Cl[Pt@SP1](Cl)([NH3])[NH3]",
    ),
    NonTetrahedralCase(
        "(SP-4-1)-diamminedichloroplatinum",
        "square_planar",
        "Cl[Pt@SP2](Cl)([NH3])[NH3]",
    ),
    NonTetrahedralCase(
        "square-planar-equivalent-SP1",
        "square_planar",
        "C[Pt@SP1](F)(Cl)[H]",
    ),
    NonTetrahedralCase(
        "square-planar-equivalent-SP2",
        "square_planar",
        "C[Pt@SP2](Cl)(F)[H]",
    ),
    NonTetrahedralCase(
        "square-planar-equivalent-SP3",
        "square_planar",
        "C[Pt@SP3](F)([H])Cl",
    ),
)


TRIGONAL_BIPYRAMIDAL_SMILES = (
    "S[As@TB1](F)(Cl)(Br)N",
    "S[As@TB2](F)(Br)(Cl)N",
    "S[As@TB3](F)(Cl)(N)Br",
    "S[As@TB4](F)(Br)(N)Cl",
    "S[As@TB5](F)(N)(Cl)Br",
    "S[As@TB6](F)(N)(Br)Cl",
    "S[As@TB7](N)(F)(Cl)Br",
    "S[As@TB8](N)(F)(Br)Cl",
    "F[As@TB9](S)(Cl)(Br)N",
    "F[As@TB11](S)(Br)(Cl)N",
    "F[As@TB10](S)(Cl)(N)Br",
    "F[As@TB12](S)(Br)(N)Cl",
    "F[As@TB13](S)(N)(Cl)Br",
    "F[As@TB14](S)(N)(Br)Cl",
    "F[As@TB15](Cl)(S)(Br)N",
    "F[As@TB20](Br)(S)(Cl)N",
    "F[As@TB16](Cl)(S)(N)Br",
    "F[As@TB19](Br)(S)(N)Cl",
    "F[As@TB17](Cl)(Br)(S)N",
    "F[As@TB18](Br)(Cl)(S)N",
)

TRIGONAL_BIPYRAMIDAL_CASES = tuple(
    NonTetrahedralCase(
        f"trigonal-bipyramidal-input-{index:02d}",
        "trigonal_bipyramidal",
        smiles,
    )
    for index, smiles in enumerate(TRIGONAL_BIPYRAMIDAL_SMILES, start=1)
)


OCTAHEDRAL_DISTINCT_LIGAND_SMILES = (
    "O[Co@OH1](Cl)(C)(N)(F)P",
    "O[Co@OH2](Cl)(F)(N)(C)P",
    "O[Co@OH3](Cl)(C)(N)(P)F",
    "O[Co@OH16](Cl)(F)(N)(P)C",
    "O[Co@OH6](Cl)(C)(P)(N)F",
    "O[Co@OH18](Cl)(F)(P)(N)C",
    "O[Co@OH19](Cl)(P)(C)(N)F",
    "O[Co@OH24](Cl)(P)(F)(N)C",
    "O[Co@OH25](P)(Cl)(C)(N)F",
    "O[Co@OH30](P)(Cl)(F)(N)C",
    "O[Co@OH4](Cl)(C)(F)(N)P",
    "O[Co@OH14](Cl)(F)(C)(N)P",
    "O[Co@OH5](Cl)(C)(F)(P)N",
    "O[Co@OH15](Cl)(F)(C)(P)N",
    "O[Co@OH7](Cl)(C)(P)(F)N",
    "O[Co@OH17](Cl)(F)(P)(C)N",
    "O[Co@OH20](Cl)(P)(C)(F)N",
    "O[Co@OH23](Cl)(P)(F)(C)N",
    "O[Co@OH26](P)(Cl)(C)(F)N",
    "O[Co@OH29](P)(Cl)(F)(C)N",
    "O[Co@OH10](Cl)(N)(F)(C)P",
    "O[Co@OH8](Cl)(N)(C)(F)P",
    "O[Co@OH11](Cl)(N)(F)(P)C",
    "O[Co@OH9](Cl)(N)(C)(P)F",
    "O[Co@OH13](Cl)(N)(P)(F)C",
    "O[Co@OH12](Cl)(N)(P)(C)F",
    "O[Co@OH22](Cl)(P)(N)(F)C",
    "O[Co@OH21](Cl)(P)(N)(C)F",
    "O[Co@OH28](P)(Cl)(N)(F)C",
    "O[Co@OH27](P)(Cl)(N)(C)F",
)


OCTAHEDRAL_REPEATED_LIGAND_SMILES = (
    "Cl[Co@OH1](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH2](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH3](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH4](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH5](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH14](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH15](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH16](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH21](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH22](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH27](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH28](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH6](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH7](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH17](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH18](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH19](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH20](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH23](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH24](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH25](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH26](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH29](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH30](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH8](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH9](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH10](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH11](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH12](N)(N)(O)(Cl)Cl",
    "Cl[Co@OH13](N)(N)(O)(Cl)Cl",
)

OCTAHEDRAL_CASES = tuple(
    NonTetrahedralCase(
        f"octahedral-distinct-input-{index:02d}",
        "octahedral",
        smiles,
    )
    for index, smiles in enumerate(OCTAHEDRAL_DISTINCT_LIGAND_SMILES, start=1)
) + tuple(
    NonTetrahedralCase(
        f"octahedral-repeated-input-{index:02d}",
        "octahedral",
        smiles,
    )
    for index, smiles in enumerate(OCTAHEDRAL_REPEATED_LIGAND_SMILES, start=1)
)


NON_TETRAHEDRAL_RDKIT_CASES = (
    *SQUARE_PLANAR_CASES,
    *TRIGONAL_BIPYRAMIDAL_CASES,
    *OCTAHEDRAL_CASES,
)

# These counts are part of the fixture contract. A source update must be
# reviewed instead of silently changing the conformance population.
assert len(INCHI_ROUND_TRIP_CASES) == 7
assert len(SQUARE_PLANAR_CASES) == 6
assert len(TRIGONAL_BIPYRAMIDAL_CASES) == 20
assert len(OCTAHEDRAL_CASES) == 60
assert len(NON_TETRAHEDRAL_RDKIT_CASES) == 86
