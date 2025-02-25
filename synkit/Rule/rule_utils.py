import re

# Regex patterns for nodes and edges
NODE_REGEX = re.compile(r'node \[ id (\d+) label "(\w+)" \]')
EDGE_REGEX = re.compile(r'edge \[ source (\d+) target (\d+) label "(.+?)" \]')


def find_block(lines, keyword):
    """
    Finds the start and end indices of a block (e.g., "left [", "context [", etc.)
    in the given lines of GML. Returns (start_idx, end_idx) or (None, None) if not found.
    """
    start_idx = None
    depth = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if start_idx is None and stripped.startswith(keyword):
            start_idx = i
            depth = 1
        elif start_idx is not None:
            # Check brackets to maintain correct depth
            if stripped.endswith("["):
                depth += 1
            elif stripped == "]":
                depth -= 1
                if depth == 0:
                    return start_idx, i
    return None, None


def get_nodes_from_edges(block_lines):
    """
    Extract node IDs from edges in the given block lines.
    Returns a set of node IDs found in the edges.
    """
    node_set = set()
    for line in block_lines:
        m = EDGE_REGEX.search(line.strip())
        if m:
            source, target, _ = m.groups()
            node_set.update([source, target])
    return node_set


def parse_context(context_lines, node_regex=None, edge_regex=None):
    """
    Parse the context lines to identify nodes and edges.
    Returns two structures:
    - context_nodes: {node_id: label}
    - context_edges: list of (source, target, label)
    """

    context_nodes = {}
    context_edges = []
    for line in context_lines:
        stripped = line.strip()
        nm = NODE_REGEX.search(stripped)
        if nm:
            nid, lbl = nm.groups()
            context_nodes[nid] = lbl
        else:
            em = EDGE_REGEX.search(stripped)
            if em:
                source, target, label = em.groups()
                context_edges.append((source, target, label))
    return context_nodes, context_edges


def filter_context(context_lines, relevant_nodes):
    """
    Given the context lines and a set of relevant nodes, remove hydrogen nodes
    not in relevant_nodes and all edges connected to them. Returns filtered lines.
    """
    context_nodes, context_edges = parse_context(context_lines)

    # Identify hydrogen nodes to remove
    hydrogen_nodes_to_remove = {
        nid
        for nid, lbl in context_nodes.items()
        if lbl == "H" and nid not in relevant_nodes
    }

    filtered_context = []
    for line in context_lines:
        stripped = line.strip()
        nm = NODE_REGEX.search(stripped)
        em = EDGE_REGEX.search(stripped)

        if nm:
            nid, lbl = nm.groups()
            if nid not in hydrogen_nodes_to_remove:
                filtered_context.append(line)
        elif em:
            source, target, label = em.groups()
            if (
                source not in hydrogen_nodes_to_remove
                and target not in hydrogen_nodes_to_remove
            ):
                filtered_context.append(line)
        else:
            # Keep section lines like "context [" or "]"
            filtered_context.append(line)

    return filtered_context


def strip_context(gml_text: str, remove_all: bool = True) -> str:
    """
    Filters or clears the 'context' section of GML-like content based on the remove_all flag.
    If remove_all is True, all edges in the 'context' section are removed.
    If False, it removes hydrogen nodes that do not appear in both 'left' and 'right' sections,
    along with their edges, while preserving the original structure and formatting of the GML.

    Parameters:
    - gml_text (str): GML-like content describing a chemical reaction rule.
    - remove_all (bool): Flag to determine if all edges should be removed from the 'context'.

    Returns:
    - str: The modified GML content with the filtered 'context' section.
    """
    lines = gml_text.split("\n")

    # Locate main sections: rule, left, context, right
    rule_start, rule_end = find_block(lines, "rule [")
    left_start, left_end = find_block(lines, "left [")
    context_start, context_end = find_block(lines, "context [")
    right_start, right_end = find_block(lines, "right [")

    # If we cannot find proper structure, return original text
    if any(
        x is None
        for x in [
            rule_start,
            rule_end,
            left_start,
            left_end,
            context_start,
            context_end,
            right_start,
            right_end,
        ]
    ):
        return gml_text

    # fmt: off
    context_lines = lines[context_start: context_end + 1]

    # Determine relevant nodes by intersection of nodes in left and right edges
    left_nodes = get_nodes_from_edges(lines[left_start: left_end + 1])
    right_nodes = get_nodes_from_edges(lines[right_start: right_end + 1])
    # fmt: on
    relevant_nodes = left_nodes.intersection(right_nodes)

    # Filter the context section based on relevant nodes
    filtered_context = filter_context(context_lines, relevant_nodes)

    if remove_all:
        # Remove all edges from the context
        # Retain only node lines and other structural lines
        final_context = []
        for line in filtered_context:
            if not EDGE_REGEX.search(line.strip()):
                final_context.append(line)
        filtered_context = final_context

    # Rebuild the full GML text
    # Replace the original context lines with the filtered or cleared context lines
    # fmt: off
    new_lines = lines[:context_start] + filtered_context + lines[context_end + 1:]
    # fmt: on

    return "\n".join(new_lines)


def _increment_gml_ids(gml_content: str) -> str:
    """
    Increment the numerical IDs within a GML content string if node id 0 exists.

    Parameters:
    - gml_content (str): The GML content as a string.

    Returns:
    - str: The modified GML content with incremented IDs.
    """
    if "node [ id 0 " not in gml_content:
        return gml_content

    def increment_id(match):
        return f"{match.group(1)} {int(match.group(2)) + 1}"

    return re.sub(r"(id|source|target) (\d+)", increment_id, gml_content)
