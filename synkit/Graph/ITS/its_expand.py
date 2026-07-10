from __future__ import annotations

from synkit.IO.chem_converter import rsmi_to_graph, graph_to_rsmi, smiles_to_graph
from synkit.Graph.ITS.its_decompose import its_decompose
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Graph.ITS.its_builder import ITSBuilder
from synkit.Chem.Reaction.standardize import Standardize
from synkit.Graph.ITS.its_relabel import ITSRelabel

std = Standardize()


class ITSExpand:
    """Partially expand a reaction SMILES (RSMI) by reconstructing intermediate
    transition states (ITS) and applying transformation rules based on the
    reaction center graph.

    This class identifies the reaction center from an RSMI, builds and
    reconstructs the ITS graph, decomposes it back into reactants and products,
    and standardizes atom mappings to produce a fully mapped AAM RSMI.

    The optional ``preserve_older_map`` mode keeps existing atom-map numbers
    from the input RSMI by reindexing the side graph before ITS reconstruction.

    Notes
    -----
    ``preserve_older_map=True`` is intended for the ITS expansion path only.
    It should not be combined with ``relabel=True``, because ``ITSRelabel``
    globally renumbers atom maps.

    :cvar std: Standardize instance for reaction SMILES standardization.
    :type std: Standardize
    """

    def __init__(self) -> None:
        """Initialize ITSExpand.

        No instance-specific attributes are required.
        """
        pass

    @staticmethod
    def _split_rsmi(rsmi: str) -> tuple[str, str]:
        """Split a reaction SMILES into reactant and product sides.

        :param rsmi: Reaction SMILES string in ``reactant>>product`` format.
        :type rsmi: str
        :returns: Reactant-side SMILES and product-side SMILES.
        :rtype: tuple[str, str]
        :raises ValueError: If the input is not a valid two-sided RSMI.
        """
        try:
            react_smi, prod_smi = rsmi.split(">>")
        except ValueError as e:
            raise ValueError("Input RSMI must be 'reactant>>product'") from e

        if not react_smi or not prod_smi:
            raise ValueError("Input RSMI must contain both reactant and product sides")

        return react_smi, prod_smi

    @staticmethod
    def _atom_map(data: dict) -> int:
        """Safely extract an atom-map number from node attributes.

        :param data: Node attribute dictionary.
        :type data: dict
        :returns: Atom-map number. Returns ``0`` if absent or falsy.
        :rtype: int
        """
        return int(data.get("atom_map", 0) or 0)

    @staticmethod
    def _nonzero_atom_maps(graph) -> list[int]:
        """Collect all nonzero atom-map numbers from a graph.

        :param graph: Molecular graph.
        :type graph: networkx.Graph
        :returns: List of nonzero atom-map numbers.
        :rtype: list[int]
        """
        return [
            ITSExpand._atom_map(data)
            for _, data in graph.nodes(data=True)
            if ITSExpand._atom_map(data) != 0
        ]

    @staticmethod
    def _validate_unique_atom_maps(atom_maps: list[int]) -> None:
        """Validate that all nonzero atom-map numbers are unique.

        :param atom_maps: Nonzero atom-map numbers.
        :type atom_maps: list[int]
        :raises ValueError: If duplicate nonzero atom-map numbers are found.
        """
        if len(atom_maps) != len(set(atom_maps)):
            raise ValueError(
                "Duplicate nonzero atom_map values found in side graph. "
                "Cannot safely reindex graph by atom_map."
            )

    @staticmethod
    def _validate_atom_maps_within_range(
        atom_maps: list[int],
        n_nodes: int,
    ) -> None:
        """Validate that atom-map numbers can be used as contiguous node IDs.

        In the side graph, we want final node IDs to remain exactly ``1..N``.
        Therefore, a mapped atom can only be moved to its atom-map number if
        that number is within ``1..N``.

        :param atom_maps: Nonzero atom-map numbers.
        :type atom_maps: list[int]
        :param n_nodes: Number of nodes in the side graph.
        :type n_nodes: int
        :raises ValueError: If any atom-map number is outside ``1..N``.
        """
        bad_targets = [
            atom_map for atom_map in atom_maps if atom_map < 1 or atom_map > n_nodes
        ]

        if bad_targets:
            raise ValueError(
                "Cannot keep side graph node ids contiguous from 1..N while "
                f"also using atom_map as node id. The following atom maps are "
                f"outside 1..{n_nodes}: {bad_targets}"
            )

    @staticmethod
    def _validate_positive_atom_maps(atom_maps: list[int]) -> None:
        """Validate that atom-map numbers are valid positive node IDs."""
        bad_targets = [atom_map for atom_map in atom_maps if atom_map < 1]

        if bad_targets:
            raise ValueError(
                "Cannot use non-positive atom_map values as node ids. "
                f"Invalid atom maps: {bad_targets}"
            )

    @staticmethod
    def _assign_mapped_nodes(graph) -> tuple[dict, set[int]]:
        """Assign mapped atoms to node IDs equal to their atom-map numbers.

        For example, if a node has ``atom_map=20``, the returned mapping assigns
        that old node to new node ID ``20``.

        :param graph: Molecular side graph.
        :type graph: networkx.Graph
        :returns: A partial old-node to new-node mapping and the used node IDs.
        :rtype: tuple[dict, set[int]]
        """
        mapping = {}
        used_ids = set()

        for node, data in graph.nodes(data=True):
            atom_map = ITSExpand._atom_map(data)

            if atom_map == 0:
                continue

            mapping[node] = atom_map
            used_ids.add(atom_map)

        return mapping, used_ids

    @staticmethod
    def _assign_unmapped_nodes(
        graph,
        mapping: dict,
        used_ids: set[int],
    ) -> dict:
        """Assign unmapped atoms while preserving contiguous node IDs.

        Unmapped atoms keep their original node ID when possible. If an
        unmapped atom's node ID conflicts with a mapped atom's target ID, it is
        moved into one of the remaining free IDs inside ``1..N``.

        :param graph: Molecular side graph.
        :type graph: networkx.Graph
        :param mapping: Existing mapping from mapped atoms.
        :type mapping: dict
        :param used_ids: Node IDs already occupied by mapped atoms.
        :type used_ids: set[int]
        :returns: Complete old-node to new-node mapping.
        :rtype: dict
        """
        n_nodes = graph.number_of_nodes()
        free_ids = set(range(1, n_nodes + 1)) - used_ids
        pending_unmapped = []

        for node, data in graph.nodes(data=True):
            atom_map = ITSExpand._atom_map(data)

            if atom_map != 0:
                continue

            if isinstance(node, int) and node in free_ids:
                mapping[node] = node
                free_ids.remove(node)
            else:
                pending_unmapped.append(node)

        for old_node, new_node in zip(pending_unmapped, sorted(free_ids)):
            mapping[old_node] = new_node

        return mapping

    @staticmethod
    def _assign_unmapped_nodes_sparse(
        graph,
        mapping: dict,
        used_ids: set[int],
    ) -> dict:
        """Assign unmapped atoms without changing preserved sparse map IDs."""
        next_candidate = 1

        for node, data in graph.nodes(data=True):
            if ITSExpand._atom_map(data) != 0:
                continue

            if isinstance(node, int) and node > 0 and node not in used_ids:
                mapping[node] = node
                used_ids.add(node)
                next_candidate = max(next_candidate, node + 1)
                continue

            while next_candidate in used_ids:
                next_candidate += 1
            mapping[node] = next_candidate
            used_ids.add(next_candidate)
            next_candidate += 1

        return mapping

    @staticmethod
    def _validate_contiguous_mapping(mapping: dict, n_nodes: int) -> None:
        """Validate that a mapping produces exactly node IDs ``1..N``.

        :param mapping: Old-node to new-node mapping.
        :type mapping: dict
        :param n_nodes: Number of nodes in the graph.
        :type n_nodes: int
        :raises RuntimeError: If the mapped node IDs are not exactly ``1..N``.
        """
        expected_ids = set(range(1, n_nodes + 1))
        actual_ids = set(mapping.values())

        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)
            extra = sorted(actual_ids - expected_ids)
            raise RuntimeError(
                f"Reindexing failed. Missing node ids: {missing}; "
                f"extra node ids: {extra}"
            )

    @staticmethod
    def _validate_complete_unique_mapping(mapping: dict, n_nodes: int) -> None:
        """Validate that every source node maps to one unique target node."""
        if len(mapping) != n_nodes:
            raise RuntimeError(
                f"Reindexing failed. Expected {n_nodes} mapped nodes; "
                f"got {len(mapping)}."
            )

        if len(set(mapping.values())) != n_nodes:
            raise RuntimeError(
                "Reindexing failed. Multiple source nodes map to the same "
                "target node id."
            )

    @staticmethod
    def _build_side_graph_reindex_mapping(
        graph,
        contiguous: bool = True,
    ) -> dict:
        """Build an old-node to new-node mapping for a side graph.

        The mapping satisfies two conditions:

        1. Every atom with ``atom_map != 0`` is assigned to node ID
           ``atom_map``.
        2. The final node IDs are exactly contiguous from ``1..N`` when
           ``contiguous=True``. With ``contiguous=False``, existing positive,
           sparse atom-map IDs are retained and unmapped atoms receive unused
           positive IDs.

        :param graph: Molecular side graph.
        :type graph: networkx.Graph
        :returns: Old-node to new-node mapping.
        :rtype: dict
        :raises ValueError: If atom-map values are duplicated or incompatible
            with contiguous node IDs.
        """
        n_nodes = graph.number_of_nodes()
        atom_maps = ITSExpand._nonzero_atom_maps(graph)

        ITSExpand._validate_unique_atom_maps(atom_maps)
        if contiguous:
            ITSExpand._validate_atom_maps_within_range(atom_maps, n_nodes)
        else:
            ITSExpand._validate_positive_atom_maps(atom_maps)

        mapping, used_ids = ITSExpand._assign_mapped_nodes(graph)
        if contiguous:
            mapping = ITSExpand._assign_unmapped_nodes(graph, mapping, used_ids)
            ITSExpand._validate_contiguous_mapping(mapping, n_nodes)
        else:
            mapping = ITSExpand._assign_unmapped_nodes_sparse(
                graph,
                mapping,
                used_ids,
            )
            ITSExpand._validate_complete_unique_mapping(mapping, n_nodes)

        return mapping

    @staticmethod
    def _copy_nodes_with_mapping(graph, new_graph, mapping: dict) -> None:
        """Copy graph nodes into a new graph using a node mapping.

        :param graph: Source graph.
        :type graph: networkx.Graph
        :param new_graph: Destination graph.
        :type new_graph: networkx.Graph
        :param mapping: Old-node to new-node mapping.
        :type mapping: dict
        """
        for old_node, new_node in mapping.items():
            attrs = dict(graph.nodes[old_node])
            new_graph.add_node(new_node, **attrs)

    @staticmethod
    def _copy_edges_with_mapping(graph, new_graph, mapping: dict) -> None:
        """Copy graph edges into a new graph using a node mapping.

        Supports both simple graphs and multigraphs.

        :param graph: Source graph.
        :type graph: networkx.Graph
        :param new_graph: Destination graph.
        :type new_graph: networkx.Graph
        :param mapping: Old-node to new-node mapping.
        :type mapping: dict
        """
        if graph.is_multigraph():
            for u, v, key, attrs in graph.edges(keys=True, data=True):
                new_graph.add_edge(
                    mapping[u],
                    mapping[v],
                    key=key,
                    **dict(attrs),
                )
            return

        for u, v, attrs in graph.edges(data=True):
            new_graph.add_edge(
                mapping[u],
                mapping[v],
                **dict(attrs),
            )

    @staticmethod
    def _rebuild_graph_with_mapping(graph, mapping: dict):
        """Rebuild a graph with remapped node IDs.

        This avoids in-place relabeling collisions, for example when node ``27``
        must become node ``20`` while old node ``20`` must move elsewhere.

        :param graph: Source graph.
        :type graph: networkx.Graph
        :param mapping: Old-node to new-node mapping.
        :type mapping: dict
        :returns: Rebuilt graph with remapped node IDs.
        :rtype: networkx.Graph
        """
        new_graph = graph.__class__()
        new_graph.graph.update(graph.graph)

        ITSExpand._copy_nodes_with_mapping(graph, new_graph, mapping)
        ITSExpand._copy_edges_with_mapping(graph, new_graph, mapping)

        return new_graph

    @staticmethod
    def reindex_side_graph_by_atom_map(graph, contiguous: bool = True):
        """Reindex a side graph so mapped atoms use ``atom_map`` as node ID.

        By default, the returned graph keeps node IDs contiguous from ``1..N``.
        Set ``contiguous=False`` to preserve valid sparse map IDs such as
        ``:10`` and ``:21``; this is needed by the EF-SMIRKS expansion path.

        This is useful because the reaction-center graph produced by
        ``ITSConstruction().ITSGraph(...)`` uses atom-map numbers as node IDs,
        whereas the side graph produced by ``smiles_to_graph(...)`` may use
        RDKit-style atom indices as node IDs.

        Example
        -------
        Before reindexing:

        .. code-block:: text

            Node 20: atom_map = 0
            Node 27: atom_map = 20

        After reindexing:

        .. code-block:: text

            Node 20: atom_map = 20
            Node 27: atom_map = 0

        or another unmapped atom may be moved into the freed node position.

        :param graph: Molecular side graph.
        :type graph: networkx.Graph
        :param contiguous: Whether the returned node IDs must be ``1..N``.
        :type contiguous: bool
        :returns: Reindexed side graph.
        :rtype: networkx.Graph
        :raises ValueError: If atom-map numbers cannot be safely used as node
            IDs while preserving ``1..N`` indexing.
        """
        mapping = ITSExpand._build_side_graph_reindex_mapping(graph, contiguous)
        return ITSExpand._rebuild_graph_with_mapping(graph, mapping)

    @staticmethod
    def expand_aam_with_its(
        rsmi: str,
        relabel: bool = False,
        use_G: bool = True,
        preserve_older_map: bool = False,
    ) -> str:
        """Expand a partial reaction SMILES to a full AAM RSMI using ITS
        reconstruction.

        :param rsmi: Reaction SMILES string in the format
            ``reactant>>product``.
        :type rsmi: str
        :param relabel: If True, directly apply ``ITSRelabel().fit(rsmi)``.
            This globally renumbers atom maps.
        :type relabel: bool
        :param use_G: If True, expand using the reactant side. If False,
            expand using the product side.
        :type use_G: bool
        :param preserve_older_map: If True, preserve existing nonzero atom-map
            numbers by reindexing the side graph before ITS reconstruction.
            This keeps old maps such as ``:20`` attached to the same atom.
            This option is incompatible with ``relabel=True``.
        :type preserve_older_map: bool
        :returns: Fully atom-mapped reaction SMILES after ITS expansion and
            standardization.
        :rtype: str
        :raises ValueError: If input RSMI format is invalid, if incompatible
            options are used, or if side-graph reindexing is unsafe.

        :example:
        >>> expander = ITSExpand()
        >>> expander.expand_aam_with_its(
        ...     "CC[CH2:3][Cl:1].[N:2]>>CC[CH2:3][N:2].[Cl:1]",
        ...     preserve_older_map=True,
        ... )
        '[CH3:1][CH2:2][CH2:3][Cl:4].[N:5]>>[CH3:1][CH2:2][CH2:3][N:5].[Cl:4]'
        """
        if relabel and preserve_older_map:
            raise ValueError(
                "preserve_older_map=True cannot be combined with relabel=True. "
                "ITSRelabel globally renumbers atom maps. Use relabel=False "
                "with preserve_older_map=True."
            )

        if relabel:
            return ITSRelabel().fit(rsmi)

        react_smi, prod_smi = ITSExpand._split_rsmi(rsmi)

        # Build graphs for reactants and products.
        react_graph, prod_graph = rsmi_to_graph(rsmi)

        # Construct the ITS reaction-center graph.
        #
        # Do NOT reindex rc_graph here.
        # The reaction-center graph already uses atom-map numbers as node IDs,
        # for example nodes 10, 11, 12, and 20.
        rc_graph = ITSConstruction().ITSGraph(react_graph, prod_graph)

        # Choose which side to expand.
        smi_side = react_smi if use_G else prod_smi

        side_graph = smiles_to_graph(
            smi_side,
            sanitize=True,
            drop_non_aam=False,
            use_index_as_atom_map=False,
        )

        # Node IDs remain contiguous from 1..N.
        if preserve_older_map:
            side_graph = ITSExpand.reindex_side_graph_by_atom_map(
                side_graph,
                contiguous=False,
            )

        # Reconstruct the full ITS graph.
        its_graph = ITSBuilder().ITSGraph(side_graph, rc_graph)

        # Decompose ITS back into reactant and product graphs.
        new_react, new_prod = its_decompose(its_graph)

        # Convert graphs back to RSMI and standardize atom mappings.
        expanded_rsmi = graph_to_rsmi(
            new_react,
            new_prod,
            its_graph,
            True,
            False,
        )

        return std.fit(expanded_rsmi, remove_aam=False)
