"""Reference stereo actions built from geometry and raw value fields.

The reference functions do not call production canonical forms, orbit tables,
stereo equality, graph encoders, mirror helpers, or descriptor relabel methods.
The two adapters share only the declared input wire fields and constructors.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import permutations, product

from Experiment.Stereo.Canonicalization.theory_audit import (
    octahedral_rotations, sign, tbp_rotations,
)

ATOM_FAMILIES = frozenset({"tetrahedral", "square_planar", "trigonal_bipyramidal", "octahedral"})
AXIS_FAMILIES = frozenset({"cumulene_axis", "extended_cis_trans"})
BOND_FAMILIES = frozenset({"planar_bond", "atrop_bond", *AXIS_FAMILIES})
MIRROR_SENSITIVE = frozenset({
    "tetrahedral", "trigonal_bipyramidal", "octahedral", "atrop_bond",
    "cumulene_axis", "helical", "planar_chirality", "framework",
})


@dataclass(frozen=True)
class State:
    """Raw, independently transported configuration, without cached identity."""

    family: str
    atoms: tuple = ()
    parity: int | None = None
    path: tuple = ()
    cyclic: bool = False
    pilot: int | None = None
    coupling_id: str | None = None
    support: tuple = ()
    frames: tuple = ()


def from_descriptor(value):
    """Read wire fields only; no production semantic methods are invoked."""
    family = value.descriptor_class
    if family in ATOM_FAMILIES or family in {"planar_bond", "atrop_bond"}:
        return State(family, tuple(value.atoms), value.parity)
    if family in AXIS_FAMILIES:
        left, right = value.terminal_frames
        path = value.axis_path if family == "cumulene_axis" else value.path
        atoms = (*left, path[0], path[-1], *right)
        return State(family, atoms, value.parity, tuple(path))
    if family == "helical":
        return State(family, parity=value.parity, path=tuple(value.path),
                     cyclic=value.cyclic, coupling_id=value.coupling_id)
    if family == "planar_chirality":
        return State(family, parity=value.parity, path=tuple(value.plane_atoms), pilot=value.pilot)
    if family == "framework":
        frames = tuple((frame.center, tuple(frame.references), frame.relation) for frame in value.frames)
        return State(family, parity=value.orientation, support=tuple(sorted(value.support_atoms)), frames=frames)
    raise ValueError(f"Reference model does not support {family!r}.")


def to_descriptor(state):
    """Construct production inputs; this adapter is outside reference truth."""
    from synkit.Graph.Stereo import (
        AtropBondStereo, CumuleneAxisStereo, ExtendedCisTransStereo,
        FrameworkFrame, FrameworkStereo, HelicalStereo, OctahedralStereo,
        PlanarBondStereo, PlanarChiralityStereo, SquarePlanarStereo,
        TetrahedralStereo, TrigonalBipyramidalStereo,
    )

    constructors = {
        "tetrahedral": TetrahedralStereo, "square_planar": SquarePlanarStereo,
        "trigonal_bipyramidal": TrigonalBipyramidalStereo, "octahedral": OctahedralStereo,
        "planar_bond": PlanarBondStereo, "atrop_bond": AtropBondStereo,
    }
    if state.family in constructors:
        return constructors[state.family](state.atoms, state.parity)
    if state.family in AXIS_FAMILIES:
        constructor = CumuleneAxisStereo if state.family == "cumulene_axis" else ExtendedCisTransStereo
        return constructor(state.path, (state.atoms[:2], state.atoms[4:]), state.parity)
    if state.family == "helical":
        return HelicalStereo(state.path, state.parity, cyclic=state.cyclic, coupling_id=state.coupling_id)
    if state.family == "planar_chirality":
        return PlanarChiralityStereo(state.path, state.pilot, state.parity)
    if state.family == "framework":
        frames = tuple(FrameworkFrame(center, references, relation) for center, references, relation in state.frames)
        return FrameworkStereo(frozenset(state.support), frames, state.parity)
    raise ValueError(f"Unsupported reference family: {state.family!r}.")


def _with_center(images):
    return tuple((0, *(position + 1 for position in image)) for image in images)


def _terminal_actions(axial):
    """Terminal sign changes preserve E/Z or the oriented axis triple product.

    Reversing an axial viewing direction changes the triple-product sign;
    a planar dot-product relation has no such viewing-direction factor.
    """
    result = []
    for exchange, left_flip, right_flip in product(range(2), repeat=3):
        if (left_flip ^ right_flip) != (exchange if axial else 0):
            continue
        if exchange:
            result.append((4 + left_flip, 5 - left_flip, 3, 2, right_flip, 1 - right_flip))
        else:
            result.append((left_flip, 1 - left_flip, 2, 3, 4 + right_flip, 5 - right_flip))
    return tuple(result)


@lru_cache(maxsize=None)
def proper_group(family):
    """Geometry-derived slot maps, never copied from production shape tables."""
    if family == "tetrahedral":
        return _with_center(tuple(image for image in permutations(range(4)) if sign(image) == 1))
    if family == "square_planar":
        return _with_center(tuple(tuple((offset + sense * i) % 4 for i in range(4))
                                  for sense in (-1, 1) for offset in range(4)))
    if family == "trigonal_bipyramidal":
        return _with_center(tbp_rotations())
    if family == "octahedral":
        return _with_center(octahedral_rotations())
    if family in BOND_FAMILIES:
        return _terminal_actions(family in {"atrop_bond", "cumulene_axis"})
    raise ValueError(f"No finite geometry for {family!r}.")


def reflect_frame(family, atoms):
    """One explicit improper operation in the declared geometry coordinates."""
    positions = {
        "tetrahedral": (1, 2), "trigonal_bipyramidal": (4, 5), "octahedral": (1, 2),
        "atrop_bond": (0, 1), "cumulene_axis": (0, 1),
        "planar_bond": (0, 1), "extended_cis_trans": (0, 1),
    }
    first, second = positions[family]
    result = list(atoms)
    result[first], result[second] = result[second], result[first]
    return tuple(result)


def _rotations(path):
    return tuple(path[offset:] + path[:offset] for offset in range(len(path)))


def _effective_atoms(state):
    if state.parity is None:
        raise ValueError("Independent fixed-configuration oracle refuses unspecified states.")
    return reflect_frame(state.family, state.atoms) if state.parity == -1 else state.atoms


def state_key(state):
    """Exact local equivalence by reference rotations/support transformations."""
    if state.parity is None:
        raise ValueError("Unspecified configuration is outside this oracle's domain.")
    family = state.family
    if family in ATOM_FAMILIES or family in BOND_FAMILIES:
        atoms = _effective_atoms(state)
        orbit = [tuple(atoms[index] for index in image) for image in proper_group(family)]
        path = min((state.path, state.path[::-1]), key=repr) if state.path else ()
        return family, path, min(orbit, key=repr)
    if family == "helical":
        variants = (state.path, state.path[::-1])
        if state.cyclic:
            variants = (*_rotations(state.path), *_rotations(state.path[::-1]))
        return family, min(variants, key=repr), state.cyclic, state.parity, state.coupling_id
    if family == "planar_chirality":
        variants = [(path, state.parity) for path in _rotations(state.path)]
        variants += [(path, -state.parity) for path in _rotations(state.path[::-1])]
        return family, state.pilot, min(variants, key=repr)
    if family == "framework":
        frames = []
        for center, references, relation in state.frames:
            local = State("tetrahedral", (center, *references), relation * state.parity)
            frames.append(state_key(local))
        return family, tuple(sorted(state.support)), tuple(sorted(frames, key=repr))
    raise ValueError(f"Unsupported reference family: {family!r}.")


def _reframe(state, atoms):
    path = state.path
    if state.family in AXIS_FAMILIES and atoms[2] != path[0]:
        path = path[::-1]
    return replace(state, atoms=atoms, path=path)


def raw_states(seed):
    """Independently enumerate the full admissible local representation set."""
    family = seed.family
    if family in ATOM_FAMILIES:
        parity = 0 if family == "square_planar" else 1
        return tuple(replace(seed, atoms=(seed.atoms[0], *references), parity=parity)
                     for references in permutations(seed.atoms[1:]))
    if family in BOND_FAMILIES:
        parity = 0 if family in {"planar_bond", "extended_cis_trans"} else 1
        result = []
        for exchange, left_flip, right_flip in product(range(2), repeat=3):
            if exchange:
                image = (4 + left_flip, 5 - left_flip, 3, 2, right_flip, 1 - right_flip)
            else:
                image = (left_flip, 1 - left_flip, 2, 3, 4 + right_flip, 5 - right_flip)
            result.append(_reframe(replace(seed, parity=parity), tuple(seed.atoms[i] for i in image)))
        return tuple(result)
    if family in {"helical", "planar_chirality"}:
        variants = (seed.path, seed.path[::-1])
        if seed.cyclic or family == "planar_chirality":
            variants = (*_rotations(seed.path), *_rotations(seed.path[::-1]))
        return tuple(replace(seed, path=path, parity=parity) for path in variants for parity in (-1, 1))
    if family == "framework":
        return tuple(replace(seed, parity=parity) for parity in (-1, 1))
    raise ValueError(f"Cannot enumerate {family!r}.")


def formal_states(seed):
    grouped = {}
    for state in raw_states(seed):
        grouped.setdefault(repr(state_key(state)), state)
    return tuple(grouped[key] for key in sorted(grouped))


def inventory_key(state):
    """Erase this locus's orientation while retaining its declared coupling."""
    family = state.family
    if family in ATOM_FAMILIES:
        return family, state.atoms[0], tuple(sorted(state.atoms[1:], key=repr))
    if family in BOND_FAMILIES:
        frames = tuple(value.atoms for value in raw_states(state))
        path = min((state.path, state.path[::-1]), key=repr) if state.path else ()
        return family, path, min(frames, key=repr)
    if family == "helical":
        key = state_key(replace(state, parity=1))
        return key[:3], state.coupling_id
    if family == "planar_chirality":
        return family, state.pilot, min((*_rotations(state.path), *_rotations(state.path[::-1])), key=repr)
    if family == "framework":
        return tuple(sorted((state_key(replace(state, parity=1)), state_key(replace(state, parity=-1))), key=repr))
    raise ValueError(f"Unsupported reference inventory family: {family!r}.")


def mirror(state):
    return replace(state, parity=-state.parity) if state.family in MIRROR_SENSITIVE else state


def map_reference(reference, mapping):
    if type(reference) is int:
        return mapping[reference]
    if isinstance(reference, str) and reference.startswith(("@H:", "@LP:")):
        kind, owner = reference.split(":", 1)
        return f"{kind}:{mapping[int(owner)]}"
    raise ValueError(f"Undeclared reference type: {reference!r}.")


def transport(state, mapping):
    """Entrywise graph-map transport, including owner-scoped virtual ports."""
    def ref(value):
        return map_reference(value, mapping)

    return replace(
        state, atoms=tuple(ref(value) for value in state.atoms),
        path=tuple(mapping[value] for value in state.path),
        pilot=None if state.pilot is None else mapping[state.pilot],
        support=tuple(sorted(mapping[value] for value in state.support)),
        frames=tuple((mapping[center], tuple(ref(value) for value in references), relation)
                     for center, references, relation in state.frames),
    )


def assignment_key(states):
    return tuple(sorted((state_key(state) for state in states), key=repr))


def raw_representation_key(state):
    """A raw encoding key for checking exhaustive input enumeration itself."""
    if state.family in ATOM_FAMILIES or state.family in BOND_FAMILIES:
        return repr((state.family, _effective_atoms(state), state.path))
    return repr((state.family, state.path, state.parity, state.cyclic, state.pilot, state.coupling_id))
