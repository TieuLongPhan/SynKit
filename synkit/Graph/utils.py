import networkx as nx
from pprint import pformat
from typing import Any, Dict, List, Mapping, Tuple


def _stereo_value(value: Any, key: str, default: Any = None) -> Any:
    """Read one field from a descriptor/change object or serialized mapping."""
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _is_stereo_descriptor(value: Any) -> bool:
    return bool(
        _stereo_value(value, "descriptor_class")
        and _stereo_value(value, "atoms") is not None
    )


def _format_stereo_descriptor(value: Any) -> str:
    """Format one relative descriptor with its ordered local reference frame."""
    if value is None:
        return "none"
    descriptor_class = _stereo_value(value, "descriptor_class", type(value).__name__)
    atoms = tuple(_stereo_value(value, "atoms", ()))
    parity = _stereo_value(value, "parity")
    provenance = _stereo_value(value, "provenance")

    if descriptor_class in {"planar_bond", "atrop_bond"} and len(atoms) >= 6:
        locus = (
            f"bond={atoms[2]}-{atoms[3]} "
            f"left_ordered_refs={atoms[:2]!r} "
            f"right_ordered_refs={atoms[4:]!r}"
        )
    elif atoms:
        locus = f"center={atoms[0]} ordered_refs={atoms[1:]!r}"
    else:
        locus = "atoms=()"

    suffix = f" provenance={provenance}" if provenance is not None else ""
    return f"{descriptor_class} {locus} parity={parity!r}{suffix}"


def _descriptor_atoms(value: Any) -> tuple[Any, ...]:
    return tuple(_stereo_value(value, "atoms", ())) if value is not None else ()


def _reference_delta(before: Any, after: Any) -> dict[str, list[Any]]:
    remaining_after = list(_descriptor_atoms(after))
    removed = []
    for reference in _descriptor_atoms(before):
        if reference in remaining_after:
            remaining_after.remove(reference)
        else:
            removed.append(reference)
    return {"removed": removed, "added": remaining_after}


def _print_stereo_registry(value: Any, *, indent: str = "    ") -> None:
    if not isinstance(value, Mapping) or not value:
        print(f"{indent}(none)")
        return

    nested = all(
        isinstance(registry, Mapping) and not _is_stereo_descriptor(registry)
        for registry in value.values()
    )
    if nested:
        for side, registry in value.items():
            print(f"{indent}{side}:")
            _print_stereo_registry(registry, indent=indent + "  ")
        return

    for target, descriptor in value.items():
        print(f"{indent}{target}: {_format_stereo_descriptor(descriptor)}")


def _print_stereo_changes(value: Any, *, indent: str = "    ") -> None:
    if not isinstance(value, Mapping) or not value:
        print(f"{indent}(none)")
        return
    for target, change in value.items():
        label = _stereo_value(change, "change", "UNSPECIFIED")
        before = _stereo_value(change, "before")
        after = _stereo_value(change, "after")
        transition = _stereo_value(change, "transition")
        print(f"{indent}{target}: {label}")
        print(f"{indent}  before: {_format_stereo_descriptor(before)}")
        print(f"{indent}  after: {_format_stereo_descriptor(after)}")
        if before is not None and after is not None:
            delta = _reference_delta(before, after)
            if delta["removed"] or delta["added"]:
                print(
                    f"{indent}  reference_delta: "
                    f"removed={delta['removed']!r} added={delta['added']!r}"
                )
        if transition is not None:
            print(f"{indent}  transition: {_format_stereo_descriptor(transition)}")


def _print_graph_mapping(value: Any, *, indent: str = "    ") -> None:
    if not isinstance(value, Mapping) or not value:
        print(f"{indent}(none)")
        return
    for key, item in value.items():
        if hasattr(item, "to_dict"):
            item = item.to_dict()
        formatted = pformat(item, width=100, sort_dicts=False)
        continuation = "\n" + indent + "  "
        print(f"{indent}{key}: {formatted.replace(chr(10), continuation)}")


def print_graph_attributes(G: nx.Graph) -> None:
    """Print node, edge, and graph-level attributes from a NetworkX graph.

    Relative stereo registries and ITS stereo changes receive a structured
    rendering that exposes ordered local references, parity, provenance,
    endpoint states, and reference replacements.

    Parameters:
        G (nx.Graph): A NetworkX graph (Graph, DiGraph, MultiGraph, etc.).
    """
    print("🔹 Nodes and their attributes:")
    for node, attr in G.nodes(data=True):
        print(f"  Node {node}: {attr}")

    print("\n🔸 Edges and their attributes:")
    if G.is_multigraph():
        for u, v, key, attr in G.edges(data=True, keys=True):
            print(f"  Edge {u}-{v} (key={key}): {attr}")
    else:
        for u, v, attr in G.edges(data=True):
            print(f"  Edge {u}-{v}: {attr}")

    print("\n🔷 Graph-level attributes:")
    if not G.graph:
        print("  (none)")
        return
    for key, value in G.graph.items():
        print(f"  {key}:")
        if key == "stereo_descriptors":
            _print_stereo_registry(value)
        elif key == "stereo_changes":
            _print_stereo_changes(value)
        elif key.startswith("stereo_") and isinstance(value, Mapping):
            _print_graph_mapping(value)
        else:
            formatted = pformat(value, width=100, sort_dicts=False)
            continuation = "\n    "
            print(f"    {formatted.replace(chr(10), continuation)}")


def remove_wildcard_nodes(G: nx.Graph, inplace: bool = True) -> nx.Graph:
    """Remove all wildcard nodes from the graph.

    A wildcard node is identified by having its 'element' attribute equal to '*'.

    Parameters
    ----------
    G : nx.Graph
        The input graph from which wildcard nodes will be removed.
    inplace : bool, optional
        If True, modify the input graph in place and return it.
        If False (default), a copy of the graph is created and the removal is applied to the copy.

    Returns
    -------
    nx.Graph
        The graph after removing all wildcard nodes.
    """
    if not inplace:
        G = G.copy()

    # Identify and remove wildcard nodes
    wildcard_nodes = [
        node for node, data in G.nodes(data=True) if data.get("element") == "*"
    ]
    G.remove_nodes_from(wildcard_nodes)
    return G


def has_wildcard_node(
    G: nx.Graph,
    element_attr: str = "element",
    wildcard: Any = "*",
) -> bool:
    """
    Fast check: return True if any node has its `element_attr` equal to the wildcard,
    using the public API with minimal overhead.

    :param G: Graph to inspect.
    :type G: nx.Graph
    :param element_attr: Node attribute key to check.
    :type element_attr: str
    :param wildcard: Value considered wildcard (e.g., "*").
    :type wildcard: Any
    :returns: True if at least one node's element_attr == wildcard.
    :rtype: bool
    """
    # iterate over just the attribute value, not the full dict
    for _, elem in G.nodes(data=element_attr):
        if elem == wildcard:
            return True
    return False


def add_wildcard_subgraph_for_unmapped(
    G: nx.Graph,
    L: nx.Graph,
    mapping: Dict[Any, Any],
    edge_keys: List[str] = ["order"],
    inplace: bool = False,
    tuple_mode: bool = False,
) -> Tuple[nx.Graph, Dict[Any, Any]]:
    """Extend G with wildcard nodes/edges for every L-node not already mapped,
    preserving original L->G mapping and returning the full mapping.

    Parameters
    ----------
    G : nx.Graph
        Target graph. If inplace=False (default), operates on a shallow copy.
    L : nx.Graph
        Pattern/reference graph containing full nodes and edges.
    mapping : Dict[L_node, G_node]
        Partial mapping from pattern L nodes to graph G nodes.
    edge_keys : List[str], optional
        Edge attributes to copy (first element if list/tuple). Default ['order'].
    inplace : bool, optional
        If True, modify G in place; otherwise modify a copy.
    tuple_mode : bool, optional
        If True, scalarize tuple ITS node attrs onto the left side before
        adding wildcard placeholders to the host graph.

    Returns
    -------
    G_ext : nx.Graph
        Extended graph with added wildcard nodes and edges.
    full_map : Dict[L_node, G_node]
        Combined L->G mapping, original plus newly added wildcard nodes.
    """
    # Use a copy if not in-place
    G_ext = G if inplace else G.copy()

    # Start from L->G mapping
    L_to_G: Dict[Any, Any] = mapping.copy()

    # Identify unmapped L nodes
    unmapped = set(L.nodes()) - set(L_to_G.keys())

    # Prepare new node IDs
    next_id = max(G_ext.nodes, default=-1) + 1
    used_atom_maps = {
        data.get("atom_map")
        for _, data in G_ext.nodes(data=True)
        if data.get("atom_map") not in (None, 0)
    }

    def _next_unused_atom_map(start: int) -> int:
        candidate = start
        while candidate in used_atom_maps:
            candidate += 1
        return candidate

    # Add wildcard nodes for each unmapped L node
    for l_node in unmapped:
        attrs = L.nodes[l_node].copy()
        if tuple_mode:
            attrs = {
                key: (
                    value[0] if isinstance(value, tuple) and len(value) == 2 else value
                )
                for key, value in attrs.items()
                if key != "typesGH"
            }
            left_types = L.nodes[l_node].get("typesGH", (None, None))[0]
            if left_types is not None:
                attrs["typesGH"] = (left_types, left_types)
        attrs["element"] = "*"
        atom_map = attrs.get("atom_map")
        if atom_map in (None, 0) or atom_map in used_atom_maps:
            atom_map = _next_unused_atom_map(next_id)
        attrs["atom_map"] = atom_map
        used_atom_maps.add(atom_map)
        G_ext.add_node(next_id, **attrs)
        L_to_G[l_node] = next_id
        next_id += 1

    # Add edges matching pattern L, mapping endpoints via L_to_G
    for u_l, v_l, data in L.edges(data=True):
        g_u = L_to_G.get(u_l)
        g_v = L_to_G.get(v_l)
        if g_u is None or g_v is None:
            continue
        edge_data: Dict[Any, Any] = {}
        for key in edge_keys:
            if key in data:
                val = data[key]
                edge_data[key] = val[0] if isinstance(val, (list, tuple)) else val
        G_ext.add_edge(g_u, g_v, **edge_data)

    # full mapping now includes original and new nodes
    full_map = L_to_G
    return G_ext, full_map


def clean_graph_keep_largest_component(graph: nx.Graph) -> nx.Graph:
    """Return a shallow copy of the input graph with all edges removed where
    the 'standard_order' attribute is exactly 0, then retain only the largest
    connected component.

    Parameters
    ----------
    graph : nx.Graph
        The input molecular graph.

    Returns
    -------
    nx.Graph
        A modified copy of the original graph with specified edges removed
        and only the largest connected component preserved.
    """
    # Work on a copy to avoid side effects
    G = graph.copy()

    # Remove edges with 'standard_order' == 0
    to_remove = [
        (u, v) for u, v, data in G.edges(data=True) if data.get("standard_order") == 0
    ]
    G.remove_edges_from(to_remove)

    # If no nodes remain, return the empty graph
    if G.number_of_nodes() == 0:
        return G

    # Identify the largest connected component
    largest_cc = max(nx.connected_components(G), key=len)

    # Return the subgraph induced by the largest component
    return G.subgraph(largest_cc).copy()
