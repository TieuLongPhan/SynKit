from __future__ import annotations

from dataclasses import asdict, dataclass

import networkx as nx
from rdkit import Chem

from synkit.IO.chem_converter import rsmi_to_graph, graph_to_rsmi, smiles_to_graph
from synkit.Graph.ITS.its_decompose import get_rc, its_decompose
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Graph.ITS.its_builder import ITSBuilder
from synkit.Chem.Reaction.standardize import Standardize
from synkit.Graph.ITS.its_relabel import ITSRelabel

std = Standardize()

CONSTITUTIONAL_RC_NODE_ATTRS = (
    "element",
    "aromatic",
    "hcount",
    "charge",
    "neighbors",
    "atom_map",
)
CONSTITUTIONAL_SIDE_NODE_ATTRS = (
    *CONSTITUTIONAL_RC_NODE_ATTRS,
    "radical",
)
CONSTITUTIONAL_RC_EDGE_ATTRS = ("order",)
CONSTITUTIONAL_SIDE_EDGE_ATTRS = ("order",)


class ITSExpansionError(ValueError):
    """Raised when partial-AAM ITS reconstruction cannot produce a safe RSMI."""


@dataclass(frozen=True)
class ITSExpansionResult:
    """Structured evidence for one successful partial-AAM expansion."""

    rsmi: str
    preferred_side: str
    selected_side: str
    fallback_used: bool
    constitution_checked: bool
    constitution_guard_passed: bool | None
    unmapped_explicit_hydrogens_folded: bool = False
    folded_unmapped_explicit_hydrogen_count: int = 0
    stereochemistry_ignored_for_expansion: bool = False
    explicit_hydrogen_serialization: bool = False
    radical_state_preserved: bool = False
    fallback_reason: str | None = None

    def to_dict(self) -> dict:
        """Return JSON-serializable expansion evidence."""
        return asdict(self)


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
    def _canonical_unmapped_constitution(smiles: str) -> str:
        """Return an atom-map- and stereo-independent constitutional key.

        Isotopes, formal charges, explicit hydrogens, bond orders, and molecular
        fragments remain part of the key. Only atom-map labels and
        stereochemical annotations are removed.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ITSExpansionError(
                f"Cannot parse endpoint while checking constitution: {smiles!r}"
            )

        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    @classmethod
    def endpoint_constitutions_match(
        cls, source_rsmi: str, candidate_rsmi: str
    ) -> bool:
        """Check that reactant and product constitutions are independently equal.

        This is a fail-closed guard for partial-AAM completion, not an atom-map
        accuracy metric. It prevents a fallback reconstruction from silently
        deleting fragments or changing either reaction endpoint.
        """
        source_react, source_prod = cls._split_rsmi(source_rsmi)
        candidate_react, candidate_prod = cls._split_rsmi(candidate_rsmi)
        return cls._canonical_unmapped_constitution(
            source_react
        ) == cls._canonical_unmapped_constitution(
            candidate_react
        ) and cls._canonical_unmapped_constitution(
            source_prod
        ) == cls._canonical_unmapped_constitution(
            candidate_prod
        )

    @staticmethod
    def _fold_unmapped_explicit_hydrogens_in_side_with_count(
        smiles: str,
    ) -> tuple[str, int]:
        """Fold removable unmapped H atoms into their heavy-atom H count.

        Mapped hydrogen atoms, isotopic hydrogens, and non-removable standalone
        hydrogen fragments remain explicit. This prevents the partial-AAM
        reaction-center path from deleting an unmapped explicit H without
        transferring its hydrogen count to the attached atom.
        """
        parser = Chem.SmilesParserParams()
        parser.removeHs = False
        mol = Chem.MolFromSmiles(smiles, parser)
        if mol is None:
            raise ITSExpansionError(
                f"Cannot parse endpoint while folding unmapped H: {smiles!r}"
            )

        explicit_unmapped_before = sum(
            atom.GetAtomicNum() == 1 and atom.GetAtomMapNum() == 0
            for atom in mol.GetAtoms()
        )

        parameters = Chem.RemoveHsParameters()
        parameters.removeMapped = False
        parameters.updateExplicitCount = True
        try:
            folded = Chem.RemoveHs(mol, parameters, sanitize=True)
        except Exception as exc:
            raise ITSExpansionError(
                f"Cannot safely fold unmapped explicit H: {smiles!r}; {exc}"
            ) from exc
        explicit_unmapped_after = sum(
            atom.GetAtomicNum() == 1 and atom.GetAtomMapNum() == 0
            for atom in folded.GetAtoms()
        )
        return Chem.MolToSmiles(
            folded,
            canonical=False,
            isomericSmiles=True,
        ), int(explicit_unmapped_before - explicit_unmapped_after)

    @classmethod
    def _fold_unmapped_explicit_hydrogens_in_side(cls, smiles: str) -> str:
        """Fold removable unmapped H atoms and return the normalized side."""
        folded, _count = cls._fold_unmapped_explicit_hydrogens_in_side_with_count(
            smiles
        )
        return folded

    @classmethod
    def fold_unmapped_explicit_hydrogens(cls, rsmi: str) -> str:
        """Normalize removable unmapped H on both reaction endpoints."""
        reactant, product = cls._split_rsmi(rsmi)
        return (
            f"{cls._fold_unmapped_explicit_hydrogens_in_side(reactant)}>>"
            f"{cls._fold_unmapped_explicit_hydrogens_in_side(product)}"
        )

    @classmethod
    def _fold_unmapped_explicit_hydrogens_with_count(cls, rsmi: str) -> tuple[str, int]:
        """Normalize both endpoints and count the explicit H atoms folded."""
        reactant, product = cls._split_rsmi(rsmi)
        folded_reactant, reactant_count = (
            cls._fold_unmapped_explicit_hydrogens_in_side_with_count(reactant)
        )
        folded_product, product_count = (
            cls._fold_unmapped_explicit_hydrogens_in_side_with_count(product)
        )
        return (
            f"{folded_reactant}>>{folded_product}",
            reactant_count + product_count,
        )

    @staticmethod
    def _side_has_stereochemistry(smiles: str) -> bool:
        """Return whether an endpoint contains atom or bond stereochemistry."""
        parser = Chem.SmilesParserParams()
        parser.removeHs = False
        mol = Chem.MolFromSmiles(smiles, parser)
        if mol is None:
            raise ITSExpansionError(
                f"Cannot parse endpoint while inspecting stereochemistry: {smiles!r}"
            )
        return any(
            atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
            for atom in mol.GetAtoms()
        ) or any(
            bond.GetStereo() != Chem.BondStereo.STEREONONE
            or bond.GetBondDir() != Chem.BondDir.NONE
            for bond in mol.GetBonds()
        )

    @staticmethod
    def _remove_stereochemistry_in_side(smiles: str) -> str:
        """Remove atom/bond stereo while retaining explicit mapped H atoms."""
        parser = Chem.SmilesParserParams()
        parser.removeHs = False
        mol = Chem.MolFromSmiles(smiles, parser)
        if mol is None:
            raise ITSExpansionError(
                f"Cannot parse endpoint while removing stereochemistry: {smiles!r}"
            )
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol, canonical=False, isomericSmiles=False)

    @classmethod
    def remove_stereochemistry(cls, rsmi: str) -> str:
        """Remove endpoint stereo without folding explicit mapped H atoms."""
        reactant, product = cls._split_rsmi(rsmi)
        return (
            f"{cls._remove_stereochemistry_in_side(reactant)}>>"
            f"{cls._remove_stereochemistry_in_side(product)}"
        )

    @classmethod
    def _rsmi_has_stereochemistry(cls, rsmi: str) -> bool:
        """Return whether either reaction endpoint carries stereo metadata."""
        if not any(marker in rsmi for marker in ("@", "/", "\\")):
            return False
        return any(
            cls._side_has_stereochemistry(side) for side in cls._split_rsmi(rsmi)
        )

    @staticmethod
    def _build_expansion_reaction_center(
        rsmi: str,
        *,
        constitutional_only: bool,
    ) -> nx.Graph:
        """Convert partial endpoints and construct their ITS reaction centre."""
        react_graph, prod_graph = rsmi_to_graph(
            rsmi,
            node_attrs=(CONSTITUTIONAL_RC_NODE_ATTRS if constitutional_only else None),
            edge_attrs=(CONSTITUTIONAL_RC_EDGE_ATTRS if constitutional_only else None),
            include_stereo_descriptors=not constitutional_only,
        )
        if react_graph is None or prod_graph is None:
            raise ITSExpansionError("reaction graph conversion returned None")
        rc_graph = ITSConstruction().ITSGraph(react_graph, prod_graph)
        if rc_graph is None:
            raise ITSExpansionError("ITS reaction-center construction returned None")
        return rc_graph

    @staticmethod
    def _convert_expansion_side(
        smiles: str,
        *,
        constitutional_only: bool,
        preserve_radical_state: bool,
    ) -> nx.Graph:
        """Convert one complete endpoint with only expansion-required fields."""
        graph = smiles_to_graph(
            smiles,
            sanitize=True,
            drop_non_aam=False,
            use_index_as_atom_map=False,
            node_attrs=(
                (
                    CONSTITUTIONAL_SIDE_NODE_ATTRS
                    if preserve_radical_state
                    else CONSTITUTIONAL_RC_NODE_ATTRS
                )
                if constitutional_only
                else None
            ),
            edge_attrs=(
                CONSTITUTIONAL_SIDE_EDGE_ATTRS if constitutional_only else None
            ),
            include_stereo_descriptors=not constitutional_only,
        )
        if graph is None:
            raise ITSExpansionError("full endpoint conversion returned None")
        return graph

    @classmethod
    def _expansion_side_graphs(
        cls,
        react_smi: str,
        prod_smi: str,
        *,
        use_G: bool,
        preserve_radical_state: bool,
        constitutional_only: bool,
    ) -> tuple[nx.Graph, nx.Graph | None, nx.Graph | None]:
        """Return selected side and optional radical-reference endpoints."""
        selected_smiles = react_smi if use_G else prod_smi
        side_graph = cls._convert_expansion_side(
            selected_smiles,
            constitutional_only=constitutional_only,
            preserve_radical_state=preserve_radical_state,
        )
        full_react_graph = side_graph if use_G else None
        full_prod_graph = side_graph if not use_G else None
        if preserve_radical_state:
            other_graph = cls._convert_expansion_side(
                prod_smi if use_G else react_smi,
                constitutional_only=constitutional_only,
                preserve_radical_state=True,
            )
            if use_G:
                full_prod_graph = other_graph
            else:
                full_react_graph = other_graph
        return side_graph, full_react_graph, full_prod_graph

    @staticmethod
    def _reacting_hydrogen_output_maps(
        rc_graph: nx.Graph,
        side_graph: nx.Graph,
        *,
        explicit_hydrogen: bool,
    ) -> list[int]:
        """Translate reacting H labels to post-builder atom-map identifiers."""
        if explicit_hydrogen:
            return []
        reacting_hydrogen_maps = {
            data["atom_map"]
            for _, data in get_rc(rc_graph).nodes(data=True)
            if data.get("element") == "H"
        }
        return [
            node
            for node, data in side_graph.nodes(data=True)
            if data.get("atom_map") in reacting_hydrogen_maps
        ]

    @staticmethod
    def _serialize_expansion(
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        its_graph: nx.Graph,
        *,
        explicit_hydrogen: bool,
        preserve_hydrogen_maps: list[int],
        standardize_output: bool,
    ) -> str:
        """Serialize a reconstructed ITS and optionally standardize its RSMI."""
        expanded_rsmi = graph_to_rsmi(
            react_graph,
            prod_graph,
            its_graph,
            sanitize=True,
            explicit_hydrogen=explicit_hydrogen,
            preserve_hydrogen_maps=preserve_hydrogen_maps,
        )
        if not isinstance(expanded_rsmi, str) or not expanded_rsmi:
            raise ITSExpansionError("graph-to-RSMI serialization returned no result")
        if not standardize_output:
            return expanded_rsmi
        standardized = std.fit(
            expanded_rsmi,
            remove_aam=False,
            remove_invalid=False,
        )
        if not standardized:
            raise ITSExpansionError(
                "expanded RSMI contains an invalid fragment; no fragments were dropped"
            )
        return standardized

    @classmethod
    def _expand_aam_once(
        cls,
        rsmi: str,
        *,
        use_G: bool,
        preserve_older_map: bool,
        explicit_hydrogen: bool,
        preserve_radical_state: bool,
        constitutional_only: bool,
        standardize_output: bool,
    ) -> str:
        """Run one side-specific ITS expansion and fail on invalid output."""
        side_name = "reactant" if use_G else "product"
        try:
            react_smi, prod_smi = cls._split_rsmi(rsmi)
            rc_graph = cls._build_expansion_reaction_center(
                rsmi,
                constitutional_only=constitutional_only,
            )
            side_graph, full_react_graph, full_prod_graph = (
                cls._expansion_side_graphs(
                    react_smi,
                    prod_smi,
                    use_G=use_G,
                    preserve_radical_state=preserve_radical_state,
                    constitutional_only=constitutional_only,
                )
            )

            if preserve_older_map:
                side_graph = cls.reindex_side_graph_by_atom_map(
                    side_graph,
                    contiguous=False,
                )

            preserve_hydrogen_maps = cls._reacting_hydrogen_output_maps(
                rc_graph,
                side_graph,
                explicit_hydrogen=explicit_hydrogen,
            )

            its_graph = ITSBuilder().ITSGraph(side_graph, rc_graph)
            if its_graph is None:
                raise ITSExpansionError("ITS reconstruction returned None")

            new_react, new_prod = its_decompose(its_graph)
            if preserve_radical_state:
                assert full_react_graph is not None
                assert full_prod_graph is not None
                cls._transfer_endpoint_radicals(
                    full_react_graph,
                    new_react,
                    preserve_atom_maps=preserve_older_map,
                )
                cls._transfer_endpoint_radicals(
                    full_prod_graph,
                    new_prod,
                    preserve_atom_maps=preserve_older_map,
                )
            return cls._serialize_expansion(
                new_react,
                new_prod,
                its_graph,
                explicit_hydrogen=explicit_hydrogen,
                preserve_hydrogen_maps=preserve_hydrogen_maps,
                standardize_output=standardize_output,
            )
        except ITSExpansionError as exc:
            raise ITSExpansionError(
                f"{side_name}-side ITS expansion failed: {exc}"
            ) from exc
        except Exception as exc:
            raise ITSExpansionError(
                f"{side_name}-side ITS expansion failed: {exc}"
            ) from exc

    @staticmethod
    def _mapped_radical_assignments(source_graph, target_graph):
        """Return direct map-anchored radical assignments when compatible."""
        radical_sources = [
            data
            for _, data in source_graph.nodes(data=True)
            if int(data.get("radical", 0) or 0) > 0
        ]
        target_by_map = {
            int(data.get("atom_map", 0) or 0): node
            for node, data in target_graph.nodes(data=True)
            if int(data.get("atom_map", 0) or 0) > 0
        }
        assignments = []
        for source_data in radical_sources:
            target_node = target_by_map.get(int(source_data.get("atom_map", 0) or 0))
            if target_node is None:
                return None
            target_data = target_graph.nodes[target_node]
            if any(
                source_data.get(name) != target_data.get(name)
                for name in ("element", "aromatic", "hcount", "charge")
            ):
                return None
            assignments.append((target_node, int(source_data.get("radical", 0) or 0)))
        return assignments

    @staticmethod
    def _transfer_endpoint_radicals(
        source_graph,
        target_graph,
        *,
        preserve_atom_maps: bool = True,
    ) -> None:
        """Transfer radical counts through exact endpoint identity.

        Existing atom maps constrain the correspondence only when the caller
        requested anchor preservation.  In normal, unanchored expansion the
        reconstructed endpoint has deliberately received fresh map numbers,
        so requiring equality to the source labels would make a valid graph
        isomorphism impossible.
        """

        radical_sources = [
            (node, data)
            for node, data in source_graph.nodes(data=True)
            if int(data.get("radical", 0) or 0) > 0
        ]
        if not radical_sources:
            nx.set_node_attributes(target_graph, 0, "radical")
            return

        if preserve_atom_maps:
            direct_assignments = ITSExpand._mapped_radical_assignments(
                source_graph, target_graph
            )
            if direct_assignments is not None:
                nx.set_node_attributes(target_graph, 0, "radical")
                for target_node, radical_count in direct_assignments:
                    target_graph.nodes[target_node]["radical"] = radical_count
                return

        def node_match(source: dict, target: dict) -> bool:
            if any(
                source.get(name) != target.get(name)
                for name in ("element", "aromatic", "hcount", "charge")
            ):
                return False
            if not preserve_atom_maps:
                return True
            required_map = int(source.get("atom_map", 0) or 0)
            target_map = int(target.get("atom_map", 0) or 0)
            return required_map == 0 or required_map == target_map

        def edge_match(source: dict, target: dict) -> bool:
            return source.get("order", 1.0) == target.get("order", 1.0)

        matcher = nx.algorithms.isomorphism.GraphMatcher(
            source_graph,
            target_graph,
            node_match=node_match,
            edge_match=edge_match,
        )
        try:
            mapping = next(matcher.isomorphisms_iter())
        except StopIteration as exc:
            raise ITSExpansionError(
                "Cannot align a reconstructed endpoint with its supplied radical state"
            ) from exc

        for source_node, target_node in mapping.items():
            target_graph.nodes[target_node]["radical"] = int(
                source_graph.nodes[source_node].get("radical", 0) or 0
            )

    @classmethod
    def expand_aam_with_its_report(
        cls,
        rsmi: str,
        relabel: bool = False,
        use_G: bool = True,
        preserve_older_map: bool = False,
        fallback_to_other_side: bool = False,
        require_constitution_preservation: bool = False,
        fold_unmapped_explicit_hydrogens: bool = False,
        ignore_stereochemistry: bool = False,
        explicit_hydrogen: bool = False,
        preserve_radical_state: bool = False,
        constitutional_only: bool = False,
        standardize_output: bool = True,
    ) -> ITSExpansionResult:
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
        :param fallback_to_other_side: Retry from the opposite reaction side if
            the preferred side cannot be expanded. Defaults to False.
        :type fallback_to_other_side: bool
        :param require_constitution_preservation: Require the expanded RSMI to
            preserve the reactant and product constitutions independently.
            This guard ignores atom maps and stereochemistry but retains
            isotopes, charges, hydrogens, fragments, and bond orders.
        :type require_constitution_preservation: bool
        :param fold_unmapped_explicit_hydrogens: Fold removable unmapped
            explicit H atoms into heavy-atom hydrogen counts before partial
            reaction-center construction, while retaining mapped H atoms.
        :type fold_unmapped_explicit_hydrogens: bool
        :param ignore_stereochemistry: Remove atom and bond stereo from the
            working RSMI before ITS construction. This is useful when stereo is
            explicitly outside the expansion task and malformed directional
            annotations prevent graph conversion. The constitution guard still
            compares both original endpoints independently.
        :type ignore_stereochemistry: bool
        :param explicit_hydrogen: Preserve explicit hydrogen graph nodes during
            graph-to-RSMI serialization. This does not materialize hydrogen
            atoms already represented only by ``hcount``.
        :type explicit_hydrogen: bool
        :param preserve_radical_state: Transfer supplied endpoint radical
            counts through exact endpoint identity instead of asking RDKit to
            infer them solely from valence during serialization. Existing maps
            constrain identity only when ``preserve_older_map=True``.
        :type preserve_radical_state: bool
        :param constitutional_only: Build only attributes required for
            constitutional partial-AAM completion and omit graph-level stereo
            descriptors. Defaults to False.
        :type constitutional_only: bool
        :param standardize_output: Reparse and standardize the already
            sanitized graph serialization. Disable only for the minimal RSMI
            adapter, whose validation is performed separately.
        :type standardize_output: bool
        :returns: Fully mapped RSMI and structured expansion evidence.
        :rtype: ITSExpansionResult
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
            expanded = ITSRelabel().fit(rsmi)
            if (
                require_constitution_preservation
                and not cls.endpoint_constitutions_match(rsmi, expanded)
            ):
                raise ITSExpansionError(
                    "Relabelled RSMI does not preserve both endpoint constitutions."
                )
            return ITSExpansionResult(
                expanded,
                preferred_side="relabel",
                selected_side="relabel",
                fallback_used=False,
                constitution_checked=require_constitution_preservation,
                constitution_guard_passed=(
                    True if require_constitution_preservation else None
                ),
                explicit_hydrogen_serialization=explicit_hydrogen,
                radical_state_preserved=preserve_radical_state,
            )

        cls._split_rsmi(rsmi)
        source_rsmi = rsmi
        if fold_unmapped_explicit_hydrogens:
            working_rsmi, folded_hydrogen_count = (
                cls._fold_unmapped_explicit_hydrogens_with_count(rsmi)
            )
        else:
            working_rsmi = rsmi
            folded_hydrogen_count = 0
        stereochemistry_ignored = bool(
            ignore_stereochemistry and cls._rsmi_has_stereochemistry(working_rsmi)
        )
        if stereochemistry_ignored:
            working_rsmi = cls.remove_stereochemistry(working_rsmi)
        preferred_name = "reactant" if use_G else "product"
        fallback_name = "product" if use_G else "reactant"

        def expand_and_guard(side: bool) -> str:
            expanded = cls._expand_aam_once(
                working_rsmi,
                use_G=side,
                preserve_older_map=preserve_older_map,
                explicit_hydrogen=explicit_hydrogen,
                preserve_radical_state=preserve_radical_state,
                constitutional_only=constitutional_only,
                standardize_output=standardize_output,
            )
            if (
                require_constitution_preservation
                and not cls.endpoint_constitutions_match(source_rsmi, expanded)
            ):
                side_name = "reactant" if side else "product"
                raise ITSExpansionError(
                    f"{side_name}-side expansion changed an endpoint constitution"
                )
            return expanded

        try:
            expanded = expand_and_guard(use_G)
            return ITSExpansionResult(
                expanded,
                preferred_side=preferred_name,
                selected_side=preferred_name,
                fallback_used=False,
                constitution_checked=require_constitution_preservation,
                constitution_guard_passed=(
                    True if require_constitution_preservation else None
                ),
                unmapped_explicit_hydrogens_folded=(folded_hydrogen_count > 0),
                folded_unmapped_explicit_hydrogen_count=folded_hydrogen_count,
                stereochemistry_ignored_for_expansion=stereochemistry_ignored,
                explicit_hydrogen_serialization=explicit_hydrogen,
                radical_state_preserved=preserve_radical_state,
            )
        except ITSExpansionError as preferred_error:
            if not fallback_to_other_side:
                raise

            try:
                expanded = expand_and_guard(not use_G)
                return ITSExpansionResult(
                    expanded,
                    preferred_side=preferred_name,
                    selected_side=fallback_name,
                    fallback_used=True,
                    constitution_checked=require_constitution_preservation,
                    constitution_guard_passed=(
                        True if require_constitution_preservation else None
                    ),
                    unmapped_explicit_hydrogens_folded=(folded_hydrogen_count > 0),
                    folded_unmapped_explicit_hydrogen_count=folded_hydrogen_count,
                    stereochemistry_ignored_for_expansion=stereochemistry_ignored,
                    explicit_hydrogen_serialization=explicit_hydrogen,
                    radical_state_preserved=preserve_radical_state,
                    fallback_reason=str(preferred_error),
                )
            except ITSExpansionError as fallback_error:
                raise ITSExpansionError(
                    f"Both ITS expansion directions failed. {preferred_name}: "
                    f"{preferred_error}; {fallback_name}: {fallback_error}"
                ) from fallback_error

    @classmethod
    def expand_rsmi(cls, rsmi: str, *, use_G: bool = True) -> str:
        """Complete partial AAM through the minimal RSMI reconstruction path.

        This adapter parses the mapped reaction centre and one complete
        endpoint, performs ITS glue/decomposition, and serializes the result.
        It deliberately assigns fresh maps and excludes optional hydrogen,
        stereo, radical, fallback, and post-expansion constitution guards.
        Use :meth:`expand_aam_with_its_report` when those policies or their
        structured evidence are required.

        :param rsmi: Partially mapped reaction SMILES.
        :type rsmi: str
        :param use_G: Reconstruct from reactants when True, products otherwise.
        :type use_G: bool
        :returns: Completely mapped reaction SMILES.
        :rtype: str
        """
        return cls.expand_aam_with_its(
            rsmi,
            use_G=use_G,
            preserve_older_map=False,
            constitutional_only=True,
            standardize_output=False,
        )

    @classmethod
    def expand_aam_with_its(
        cls,
        rsmi: str,
        relabel: bool = False,
        use_G: bool = True,
        preserve_older_map: bool = False,
        fallback_to_other_side: bool = False,
        require_constitution_preservation: bool = False,
        fold_unmapped_explicit_hydrogens: bool = False,
        ignore_stereochemistry: bool = False,
        explicit_hydrogen: bool = False,
        preserve_radical_state: bool = False,
        constitutional_only: bool = False,
        standardize_output: bool = True,
    ) -> str:
        """Expand partial AAM and return the completed reaction SMILES.

        Use :meth:`expand_aam_with_its_report` when selected-side and guard
        evidence are also required. This method preserves the historical
        string-returning API.
        """
        return cls.expand_aam_with_its_report(
            rsmi,
            relabel=relabel,
            use_G=use_G,
            preserve_older_map=preserve_older_map,
            fallback_to_other_side=fallback_to_other_side,
            require_constitution_preservation=require_constitution_preservation,
            fold_unmapped_explicit_hydrogens=fold_unmapped_explicit_hydrogens,
            ignore_stereochemistry=ignore_stereochemistry,
            explicit_hydrogen=explicit_hydrogen,
            preserve_radical_state=preserve_radical_state,
            constitutional_only=constitutional_only,
            standardize_output=standardize_output,
        ).rsmi
