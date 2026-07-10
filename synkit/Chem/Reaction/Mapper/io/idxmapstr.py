def parse_index_string(s, base=0):
    """Parse a string like '1,3-5' into a sorted list of unique indices."""
    offset = 1 if base == 1 else 0
    indices = {
        idx - offset for token in s.split(",") for idx in _expand_index_token(token)
    }
    return sorted(indices)


def _expand_index_token(token):
    token = token.strip()
    if "-" not in token:
        return [int(token)]

    start, end = map(int, token.split("-"))
    return range(start, end + 1)


def parse_index_mapping_string(pair_str, base=0):
    """Parse a string of the form '1-3,5>>4-6;7>>8'."""
    result = []

    for pair in pair_str.split(";"):
        if ">>" not in pair:
            raise ValueError(f"Missing '>>' in pair: {pair}")

        left_str, right_str = pair.split(">>")
        left_indices = parse_index_string(left_str, base=base)
        right_indices = parse_index_string(right_str, base=base)

        if len(left_indices) != len(right_indices):
            raise ValueError(
                f"Mismatch in number of unique elements: {left_indices} vs {right_indices}"
            )

        result.append((left_indices, right_indices))

    return result


def fmt_idxs(idxs, base=0):
    """
    Convert a sorted list of indices into a compact string such as '1-3,5,7'.

    Parameters
    ----------
    idxs : list[int]
        Sorted list of indices.
    base : int, optional (default=0)
        If 1, convert output to 1-based indices.

    Returns
    -------
    str
        Compact string representation.
    """
    shifted = [idx + base for idx in idxs]
    return ",".join(_format_index_run(start, end) for start, end in _runs(shifted))


def _runs(values):
    if not values:
        return

    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        yield start, prev
        start = prev = value

    yield start, prev


def _format_index_run(start, end):
    return str(start) if start == end else f"{start}-{end}"


def lgp2idx_map_str(lgp, base=0):
    left_labels = lgp[0].label2idxs
    right_labels = lgp[1].label2idxs

    return ";".join(
        f"{fmt_idxs(left_labels[label], base)}>>{fmt_idxs(right_labels[label], base)}"
        for label in left_labels
    )
