import pytest
from copy import deepcopy
from hashlib import sha256

from synkit.Chem.Mapper import (
    GlobalShellConfig,
    enumerate_mapped_reaction_its_alternatives,
)
from synkit.Chem.Mapper.template_adapter import (
    correspondence_from_export,
    executable_rule_from_correspondence,
    mapped_reaction_from_correspondence,
    prospective_products,
    replay_executable_rule_on_source,
    score_product_recovery,
    class_correspondence_from_export,
)

REACTION = "[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:3].[Br-:2]"
IDENTITY = ((1, 1), (2, 2), (3, 3))
FLOWER_FULL_ATOM_CONTROL = (
    "[Cl:1][CH2:2][CH2:3][CH2:4][N:5]1[CH2:6][CH2:7][CH2:8][CH2:9]1."
    "[OH2:10]>>[CH2:2]1[CH2:3][CH2:4][N+:5]12[CH2:6][CH2:7][CH2:8][CH2:9]2."
    "[Cl-:1].[OH2:10]"
)
FLOWER_EXPLICIT_HYDROGEN_CONTROL = (
    "[C:1]1(=[O:2])[O:3][C:4](=[O:5])[CH:6]=[CH:7]1."
    "[CH2:8]([C:9]([CH3:10])=[CH2:11])[H:14]>>"
    "[C:1]1(=[O:2])[O:3][C:4](=[O:5])[CH:6]([CH2:11][C:9](=[CH2:8])[CH3:10])"
    "[CH:7]1[H:14]"
)


def test_complete_correspondence_builds_executable_rule_and_source_replays():
    mapped = mapped_reaction_from_correspondence(REACTION, IDENTITY)
    rule = executable_rule_from_correspondence(REACTION, IDENTITY, heavy_only=False)
    replay = replay_executable_rule_on_source(REACTION, IDENTITY, heavy_only=False)

    assert mapped.split(">>", 1)[0] == REACTION.split(">>", 1)[0]
    assert set(mapped.split(">>", 1)[1].split(".")) == {
        "[CH3:1][OH:3]",
        "[Br-:2]",
    }
    assert rule.rc.raw.number_of_nodes() == 3
    assert replay.recovered is True
    assert replay.expected_product == "CO.[Br-]"
    assert replay.generated_products == ("CO.[Br-]",)


def test_adapter_refuses_incomplete_or_heavy_only_correspondences():
    with pytest.raises(ValueError, match="cover each reactant"):
        mapped_reaction_from_correspondence(REACTION, ((1, 1), (2, 2)))
    with pytest.raises(ValueError, match="heavy-only"):
        executable_rule_from_correspondence(REACTION, IDENTITY, heavy_only=True)


def test_prospective_application_receives_only_reactants_and_rules():
    rule = executable_rule_from_correspondence(REACTION, IDENTITY, heavy_only=False)

    products = prospective_products("CBr.[OH-]", [rule])

    assert products == ("CO.[Br-]",)


def test_target_scoring_is_separate_from_prospective_application():
    score = score_product_recovery(["[Br-].CO", "CO.[Br-]"], "CO.[Br-]")

    assert score.candidate_count == 1
    assert score.recovered is True


def test_export_reader_requires_explicit_class_correspondence():
    assert correspondence_from_export(
        {"atom_map_correspondence": [[1, 3], [2, 2], [3, 1]]}
    ) == ((1, 3), (2, 2), (3, 1))
    with pytest.raises(ValueError, match="lacks"):
        correspondence_from_export({})


def test_full_atom_synister_export_feeds_executable_source_replay():
    result = enumerate_mapped_reaction_its_alternatives(
        REACTION,
        CD="minimal",
        seed_mode="none",
        heavy_only=False,
        config=GlobalShellConfig(
            binary=True,
            time_limit_seconds=3,
            max_bijections=None,
            max_mappings=None,
            symmetry_pruning=False,
        ),
    )
    payload = result.as_dict()
    correspondence = correspondence_from_export(payload["shell"]["classes"][0])

    replay = replay_executable_rule_on_source(
        REACTION,
        correspondence,
        heavy_only=payload["heavy_only"],
    )

    assert result.shell.complete is True
    assert replay.recovered is True


def test_full_atom_flower_control_export_feeds_executable_source_replay():
    result = enumerate_mapped_reaction_its_alternatives(
        FLOWER_FULL_ATOM_CONTROL,
        CD="minimal",
        seed_mode="none",
        heavy_only=False,
        config=GlobalShellConfig(
            binary=False,
            time_limit_seconds=3,
            max_bijections=None,
            max_mappings=None,
            symmetry_pruning=False,
        ),
    )
    payload = result.as_dict()
    correspondence = correspondence_from_export(payload["shell"]["classes"][0])

    replay = replay_executable_rule_on_source(
        FLOWER_FULL_ATOM_CONTROL,
        correspondence,
        heavy_only=payload["heavy_only"],
    )

    assert result.shell.complete is True
    assert result.shell.elapsed_seconds < 3
    assert replay.recovered is True


def test_adapter_preserves_mapped_explicit_hydrogens_in_flower_control():
    result = enumerate_mapped_reaction_its_alternatives(
        FLOWER_EXPLICIT_HYDROGEN_CONTROL,
        CD="minimal",
        seed_mode="none",
        heavy_only=False,
        config=GlobalShellConfig(
            binary=False,
            time_limit_seconds=3,
            max_bijections=None,
            max_mappings=None,
            symmetry_pruning=False,
        ),
    )
    payload = result.as_dict()
    replays = [
        replay_executable_rule_on_source(
            FLOWER_EXPLICIT_HYDROGEN_CONTROL,
            correspondence_from_export(class_record),
            heavy_only=payload["heavy_only"],
        )
        for class_record in payload["shell"]["classes"]
    ]

    assert result.shell.complete is True
    assert len(replays) == 2
    assert all(replay.recovered for replay in replays)


@pytest.mark.parametrize("value", [1.9, 1.0, True, "1", None, 0, -1])
def test_correspondence_rejects_noninteger_or_nonpositive_ids(value):
    with pytest.raises(ValueError, match="positive integers"):
        mapped_reaction_from_correspondence(REACTION, ((value, 1), (2, 2), (3, 3)))
    with pytest.raises(ValueError, match="positive integers"):
        correspondence_from_export(
            {"atom_map_correspondence": [[value, 1], [2, 2], [3, 3]]}
        )


@pytest.mark.parametrize("pairs", [((1, 1), (2, 2), (3, 2)), ((1, 1), (1, 2), (3, 3))])
def test_correspondence_rejects_duplicate_images_and_domains(pairs):
    with pytest.raises(ValueError, match="more than once"):
        mapped_reaction_from_correspondence(REACTION, pairs)


def test_correspondence_rejects_element_incompatible_maps():
    with pytest.raises(ValueError, match="preserve atom elements"):
        mapped_reaction_from_correspondence(REACTION, ((1, 3), (2, 2), (3, 1)))


def test_source_replay_accepts_a_single_pass_correspondence_generator():
    replay = replay_executable_rule_on_source(
        REACTION, iter(IDENTITY), heavy_only=False
    )
    assert replay.recovered


def _declared_export():
    return {
        "schema_version": 1,
        "kind": "synister_mapped_reaction_its_alternatives",
        "reaction_sha256": sha256(REACTION.encode()).hexdigest(),
        "heavy_only": False,
        "reactant_atom_maps": [1, 2, 3],
        "product_atom_maps": [1, 2, 3],
        "shell": {
            "complete": True,
            "shell_complete": True,
            "classification_complete": True,
            "classes": [
                {
                    "its_class_id": "test-class",
                    "representative_mapping": [0, 1, 2],
                    "atom_map_correspondence": [[1, 1], [2, 2], [3, 3]],
                }
            ],
        },
    }


def test_typed_export_binds_reaction_and_explicit_legacy_compatibility():
    payload = _declared_export()
    result = class_correspondence_from_export(payload, REACTION, "test-class")
    assert result.correspondence == IDENTITY and result.heavy_only is False
    legacy = deepcopy(payload)
    legacy.pop("schema_version")
    legacy.pop("kind")
    with pytest.raises(ValueError, match="schema"):
        class_correspondence_from_export(legacy, REACTION, "test-class")
    assert (
        class_correspondence_from_export(
            legacy, REACTION, "test-class", allow_legacy=True
        )
        == result
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "schema",
        "hash",
        "scope",
        "inventory",
        "coordinates",
        "element",
        "enumeration",
        "classification",
        "duplicate_class",
    ],
)
def test_typed_export_rejects_wrong_identity_coordinates_or_incomplete_status(
    corruption,
):
    payload = _declared_export()
    record = payload["shell"]["classes"][0]
    if corruption == "schema":
        payload["schema_version"] = True
    elif corruption == "hash":
        payload["reaction_sha256"] = "0" * 64
    elif corruption == "scope":
        payload["heavy_only"] = "false"
    elif corruption == "inventory":
        payload["reactant_atom_maps"] = [11, 2, 3]
        record["atom_map_correspondence"][0][0] = 11
    elif corruption == "coordinates":
        record["representative_mapping"] = [2, 1, 0]
    elif corruption == "element":
        record["representative_mapping"] = [2, 1, 0]
        record["atom_map_correspondence"] = [[1, 3], [2, 2], [3, 1]]
    elif corruption in ("enumeration", "classification"):
        payload["shell"][
            (
                "shell_complete"
                if corruption == "enumeration"
                else "classification_complete"
            )
        ] = False
    else:
        payload["shell"]["classes"].append(deepcopy(record))
    with pytest.raises(ValueError):
        class_correspondence_from_export(payload, REACTION, "test-class")
