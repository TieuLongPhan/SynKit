"""Executable public reaction-stereo workflows for the RX14 documentation."""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from synkit.Graph.Stereo import (  # noqa: E402
    PlanarBondStereo,
    StereoChange,
    StereoCoupling,
    StereoOutcome,
    StereoReactionComposition,
    StereoReactionValue,
    TetrahedralStereo,
    compose_reaction_stereo,
    project_reaction_stereo,
    reaction_stereo_from_graph,
)
from synkit.Mechanism import (  # noqa: E402
    ElectronLocus,
    ElectronMove,
    ElectronMoveGroup,
    ElectrocyclicStereoMotion,
    MechanismRecord,
    MechanisticStep,
)
from synkit.Rule import (  # noqa: E402
    GenericStereoRuleExtractor,
    GenericStereoRulePolicy,
)

SN2 = "[CH3:1][C@H:2]([F:3])[Cl:4].[OH-:5]>>" "[CH3:1][C@@H:2]([F:3])[OH:5].[Cl-:4]"


def exact_and_generic_rules():
    exact = GenericStereoRuleExtractor(
        GenericStereoRulePolicy(domain_source="exact")
    ).extract(SN2)
    generic = GenericStereoRuleExtractor(
        GenericStereoRulePolicy(
            domain_source="class",
            explicit_domains={
                1: {"elements": {"C"}},
                3: {"elements": {"F", "Br"}},
            },
        )
    ).extract(SN2)
    assert exact.certificate.source_replay_exact
    assert generic.certificate.source_replay_exact
    assert generic.certificate.ports
    return exact, generic


def branching_and_coupling():
    formed = TetrahedralStereo((2, 1, 3, 4, "@H:2"), 1)
    enantiomers = StereoOutcome.racemic().alternatives(formed)
    planar = PlanarBondStereo((1, 7, 2, 3, 8, 4), 0)
    coupling = StereoCoupling.vicinal_addition(
        "ANTI",
        centers=(2, 3),
        ligands=(5, 6),
    )
    paired_faces = coupling.tetrahedral_product_pairs(planar)
    assert len(enantiomers) == 2
    assert len(paired_faces) == 2
    assert all(len(pair) == 2 for pair in paired_faces)
    return enantiomers, paired_faces


def composition_and_interchange():
    state = TetrahedralStereo((1, 2, 3, 4, 5), 1)
    inverse = state.invert()
    first = StereoReactionValue(
        guards={"atom:1": state},
        effects={"atom:1": StereoChange.from_endpoints(state, inverse)},
    )
    second = StereoReactionValue(
        guards={"atom:1": inverse},
        effects={"atom:1": StereoChange.from_endpoints(inverse, state)},
    )
    proof = compose_reaction_stereo(first, second)
    restored_proof = StereoReactionComposition.from_dict(proof.to_dict())
    graph, report = project_reaction_stereo(
        proof.result,
        "internal_graph",
    )
    assert restored_proof == proof
    assert proof.result.effects["atom:1"].change == "RETAINED"
    assert report.lossless
    assert reaction_stereo_from_graph(graph) == proof.result
    return proof


def mechanism_replay():
    move = ElectronMove(
        ElectronLocus.atom("lp", atom_map=1),
        ElectronLocus.bond("sigma", atom_maps=(1, 2)),
        2,
        "curved",
        "g1",
    )
    record = MechanismRecord(
        "[OH-:1].[CH3+:2]>>[CH3:2][OH:1]",
        (
            MechanisticStep(
                "association",
                (ElectronMoveGroup("g1", (move,)),),
            ),
        ),
    )
    certificate = record.verify(
        electron="strict",
        stereo="stepwise",
    )
    assert certificate.status == "VALID"
    assert certificate.final_match["stereo_verification_performed"]
    return record, certificate


def electrocyclic_figure_11():
    """Verify the substituted thermal 4π Figure 11 path and its reversal."""
    moves = ElectronMoveGroup(
        "g1",
        (
            ElectronMove(
                ElectronLocus("sigma", (1, 4)),
                ElectronLocus("pi", (3, 4)),
                2,
                "curved",
                "g1",
            ),
            ElectronMove(
                ElectronLocus("pi", (2, 3)),
                ElectronLocus("pi", (1, 2)),
                2,
                "curved",
                "g1",
            ),
        ),
    )
    motion = ElectrocyclicStereoMotion(
        "CONROTATORY",
        "RING_OPENING",
        termini=(1, 4),
        substituents=(5, 6),
        terminal_motion=(1, 1),
        pi_electrons=4,
        activation="THERMAL",
        provenance="figure-11",
    )
    record = MechanismRecord(
        "[CH:1]1([Cl:5])[CH:2]=[CH:3][CH:4]1[Cl:6]>>"
        "[CH:1]([Cl:5])=[CH:2][CH:3]=[CH:4][Cl:6]",
        (
            MechanisticStep(
                "electrocyclic",
                (moves,),
                stereo_motions=(motion,),
            ),
        ),
        provenance={"figure": "Figure 11"},
    )
    certificate = record.verify(stereo="stepwise")
    reverse = record.reversed()
    assert certificate.status == "VALID"
    assert len(certificate.final_match["canonical_neighbor_changes"]) == 2
    assert reverse.verify(stereo="stepwise").status == "VALID"
    assert reverse.steps[0].stereo_motions[0].direction == "RING_CLOSURE"
    return record, certificate, reverse


if __name__ == "__main__":
    exact_and_generic_rules()
    branching_and_coupling()
    composition_and_interchange()
    mechanism_replay()
    electrocyclic_figure_11()
