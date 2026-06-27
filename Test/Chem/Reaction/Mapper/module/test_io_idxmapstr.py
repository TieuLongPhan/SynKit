from synkit.Chem.Reaction.Mapper.io.idxmapstr import (
    fmt_idxs,
    parse_index_mapping_string,
    parse_index_string,
)


def test_idxmapstr_parses_and_formats_index_runs():
    assert parse_index_string("1,3-5", base=1) == [0, 2, 3, 4]
    assert parse_index_mapping_string("1-2>>3-4", base=1) == [([0, 1], [2, 3])]

    assert fmt_idxs([0, 1, 3], base=1) == "1-2,4"
