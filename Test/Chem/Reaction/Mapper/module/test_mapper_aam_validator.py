from synkit.Chem.Reaction.Mapper import AAMValidator as MapperAAMValidator
from synkit.Chem.Reaction.aam_validator import AAMValidator as LegacyAAMValidator


def test_mapper_aam_validator_is_legacy_compatible():
    assert MapperAAMValidator is LegacyAAMValidator
    assert MapperAAMValidator().strip_unbalanced_maps is True
