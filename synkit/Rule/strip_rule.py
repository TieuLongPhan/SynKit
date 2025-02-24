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


def filter_context(context_lines, left_edges):
    """
    Given the context lines and a set of edges from the left graph, remove edges
    from the context that are also present in the left graph (ignoring labels).
    Returns filtered lines.
    """
    # Create a set of edges from the left graph (ignoring labels)
    left_edge_set = set((source, target) for source, target, _ in left_edges)

    filtered_context = []
    for line in context_lines:
        stripped = line.strip()
        nm = NODE_REGEX.search(stripped)
        em = EDGE_REGEX.search(stripped)

        if nm:
            # Keep node lines as they are
            filtered_context.append(line)
        elif em:
            source, target, label = em.groups()
            # Only keep the edge if it is not present in the left edges (ignoring label)
            if (source, target) not in left_edge_set:
                filtered_context.append(line)
        else:
            # Keep section lines like "context [" or "]"
            filtered_context.append(line)

    return filtered_context


def strip_context(gml_text: str, remove_all: bool = True) -> str:
    """
    Filters or clears the 'context' section of GML-like content based on the remove_all flag.
    If remove_all is True, all edges in the 'context' section are removed.
    If False, it removes edges in the 'context' that are also present in the 'left' section.
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

    # Determine edges in the left graph
    left_edges = []
    for line in lines[left_start:left_end + 1]:
        em = EDGE_REGEX.search(line.strip())
        if em:
            source, target, label = em.groups()
            left_edges.append((source, target, label))
    # fmt: on

    # Filter the context section based on edges in the left graph
    filtered_context = filter_context(context_lines, left_edges)

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
