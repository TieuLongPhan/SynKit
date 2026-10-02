from Experiment.Synister.select_confirmation import choose


def row(group, identifier, endpoint):
    return dict(source_sequence=group, r_id=identifier, endpoint_sha256=endpoint)


def test_group_priority_is_independent_of_number_of_rows():
    rows = [row("a", "1", "a"), row("b", "2", "b")]
    selected = choose(rows, 2)[0]
    expanded = rows + [row("a", str(i), "a") for i in range(3, 103)]
    new = choose(expanded, 2)[0]
    assert [x["source_sequence"] for x in selected] == [x["source_sequence"] for x in new]


def test_endpoint_dedup_precedes_rank_and_is_input_order_invariant():
    rows = [row("b", "2", "same"), row("a", "1", "same"), row("c", "3", "different")]
    selected, frame, _ = choose(rows, 2)
    assert {x["source_sequence"] for x in frame} == {"a", "c"}
    assert choose(list(reversed(rows)), 2)[0] == selected
