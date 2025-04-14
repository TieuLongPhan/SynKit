import copy
import networkx as nx
from operator import eq
from typing import Dict, List, Optional, Set, Tuple, Any
from networkx.algorithms.isomorphism import generic_node_match, generic_edge_match
from synkit.IO.chem_converter import gml_to_its


class RuleMapping:
    @staticmethod
    def enumerate_all_unique_mappings(
        child: nx.Graph, parent: nx.Graph
    ) -> List[Dict[Any, Any]]:
        """
        Generate all unique mappings (as dictionaries) from the child graph to the parent graph.
        A mapping is valid if:
        - Every node from the child graph is assigned exactly one parent node.
        - The parent's node has the same 'element' attribute as the child node.
        - No parent's node is repeated in a mapping.

        Parameters:
        - child (nx.Graph): The child graph whose nodes will be mapped.
        - parent (nx.Graph): The parent graph in which to search for matching nodes.

        Returns:
        - List[dict]: A list of mapping dictionaries. Each dictionary maps a child node to a unique
                      parent node with the same 'element'. If no valid mapping exists, returns an empty list.
        """
        # Build candidate sets for each node in child, based on matching 'element' in parent
        candidate_map: Dict[Any, List[Any]] = {}
        for node, attrs in child.nodes(data=True):
            element = attrs.get("element")
            # Gather all parent nodes with the same element
            candidates = [
                pnode
                for pnode, p_attrs in parent.nodes(data=True)
                if p_attrs.get("element") == element
            ]
            candidate_map[node] = candidates

        all_mappings: List[Dict[Any, Any]] = []
        child_nodes = list(child.nodes())

        def backtrack(
            i: int, current_mapping: Dict[Any, Any], used_parents: Set[Any]
        ) -> None:
            # If we've assigned every child node, store a copy of the mapping.
            if i == len(child_nodes):
                all_mappings.append(current_mapping.copy())
                return

            child_node = child_nodes[i]
            for candidate in candidate_map.get(child_node, []):
                if candidate not in used_parents:
                    current_mapping[child_node] = candidate
                    used_parents.add(candidate)
                    backtrack(i + 1, current_mapping, used_parents)
                    used_parents.remove(candidate)
                    del current_mapping[child_node]

        # Backtracking to explore all valid mappings
        backtrack(0, {}, set())
        return all_mappings

    @staticmethod
    def standardize_order(
        order_tuple: Tuple[float, ...],
    ) -> Optional[Tuple[float, ...]]:
        """
        Standardizes an order tuple by adding 1 to every element repeatedly until no element is negative.
        If the resulting tuple becomes all zeros, returns None, which indicates that the edge should be dropped.

        For example:
          (-1.0, 0.0) --> add 1 gives (0.0, 1.0)
          (-2.0, -1.0) --> add 1 yields (-1.0, 0.0) --> add 1 yields (0.0, 1.0)
          (0.0, 0.0) remains (0.0, 0.0) and then returns None.

        Parameters:
        - order_tuple (Tuple[float, ...]): The order attribute (tuple of floats).

        Returns:
        - Optional[Tuple[float, ...]]: The standardized tuple, or None if it becomes all zeros.
        """
        order_list = list(order_tuple)
        while any(x < 0 for x in order_list):
            order_list = [x + 1 for x in order_list]
        if all(x == 0 for x in order_list):
            return None
        return tuple(order_list)

    @staticmethod
    def keep_largest_component(graph: nx.Graph) -> nx.Graph:
        """
        Given an undirected graph, returns the subgraph corresponding to the largest connected component.

        Parameters:
        - graph (nx.Graph): The input graph from which the largest component is extracted.

        Returns:
        - nx.Graph: A subgraph induced by the largest connected component of the input graph.
        """
        if graph.number_of_nodes() == 0:
            return graph
        # Find all connected components
        components = list(nx.connected_components(graph))
        # Identify the largest by number of nodes
        largest = max(components, key=len)
        # Return the induced subgraph (as a new, independent graph)
        return graph.subgraph(largest).copy()

    @staticmethod
    def subtract_parent_from_child(
        child: nx.Graph, parent: nx.Graph, mapping: Dict[Any, Any]
    ) -> nx.Graph:
        """
        Create a new graph by performing a (parent - child) subtraction of edge attributes
        using a given mapping from child nodes to parent nodes. The result is then reduced
        to its largest connected component.

        Steps:
        1. Make a deep copy of the parent graph and remove all its edges.
        2. Build the union of the parent's edges plus the child's edges mapped into the parent's node IDs.
        3. For each edge in the union (using parent node IDs):
            - new_standard_order = parent's standard_order - child's standard_order.
            - If an 'order' tuple exists:
                a. If one side is missing, assume zeros of appropriate length.
                b. Compute (parent_order - child_order) element-wise.
                c. Standardize the resulting tuple via standardize_order().
                d. If None, omit the edge entirely.
        4. Add each valid edge to the new graph.
        5. Keep only the largest connected component.

        Parameters:
        - child (nx.Graph): The child graph (provides edge attributes to subtract).
        - parent (nx.Graph): The parent graph (provides baseline edge/node attributes).
        - mapping (Dict[Any, Any]): A one-to-one mapping from child nodes to parent nodes.

        Returns:
        - nx.Graph: A new graph (deep copy of parent, with edges recomputed),
                    reduced to its largest connected component.
        """
        # 1. Deep copy the parent and remove its edges
        new_graph = copy.deepcopy(parent)
        new_graph.remove_edges_from(list(new_graph.edges()))

        # 2. Build union of edges. We'll store them in a dictionary (u, v) -> {"parent", "child"}
        union_edges: Dict[Tuple[Any, Any], Dict[str, Dict[str, Any]]] = {}

        # Parent edges
        for u, v, pdata in parent.edges(data=True):
            key = tuple(sorted([u, v], key=lambda x: str(x)))
            union_edges.setdefault(key, {})["parent"] = pdata

        # Child edges (mapped)
        for u, v, cdata in child.edges(data=True):
            parent_u = mapping.get(u)
            parent_v = mapping.get(v)
            if parent_u is None or parent_v is None:
                continue
            key = tuple(sorted([parent_u, parent_v], key=lambda x: str(x)))
            union_edges.setdefault(key, {})["child"] = cdata

        # 3. Compute new edge attributes
        for (u, v), entry in union_edges.items():
            parent_data = entry.get("parent", {})
            child_data = entry.get("child", {})

            parent_so = parent_data.get("standard_order", 0)
            child_so = child_data.get("standard_order", 0)
            new_so = parent_so - child_so

            parent_order = parent_data.get("order", None)
            child_order = child_data.get("order", None)

            new_order = None
            if parent_order is not None or child_order is not None:
                # If one side is missing, assume zero tuple
                if parent_order is None and child_order is not None:
                    parent_order = tuple(0 for _ in child_order)
                if child_order is None and parent_order is not None:
                    child_order = tuple(0 for _ in parent_order)

                # Subtract if they match in length
                if (
                    isinstance(parent_order, tuple)
                    and isinstance(child_order, tuple)
                    and len(parent_order) == len(child_order)
                ):
                    computed_order = tuple(
                        p - c for p, c in zip(parent_order, child_order)
                    )
                    new_order = RuleMapping.standardize_order(computed_order)

            new_edge_data = {"standard_order": new_so}
            # Only add the 'order' attribute if new_order is not None
            if new_order is not None:
                new_edge_data["order"] = new_order
                # If new_order is None, we skip adding this edge

            if new_order is not None:
                new_graph.add_edge(u, v, **new_edge_data)

        # 4. Return the largest connected component
        return RuleMapping.keep_largest_component(new_graph)

    @staticmethod
    def graph_alignment(
        child: nx.Graph,
        parent: nx.Graph,
        node_label_names: List[str] = ["element"],
        node_label_default: List[str] = ["*"],
        edge_attribute: str = "standard_order",
    ) -> Tuple[bool, Optional[Dict[Any, Any]]]:
        """
        Check whether the child and parent graphs are isomorphic using specified node and edge match criteria.
        If they are isomorphic, return the mapping from child to parent.

        Parameters:
        - child (nx.Graph): The child graph to align.
        - parent (nx.Graph): The parent graph to align with.
        - node_label_names (List[str]): Node attribute names for matching (default: ["element"]).
        - node_label_default (List[str]): Default values for those attributes if missing (default: ["*"]).
        - edge_attribute (str): The edge attribute to match (default: "standard_order").

        Returns:
        - Tuple[bool, Optional[Dict[Any, Any]]]:
            A tuple (is_iso, mapping):
              - is_iso (bool): True if the graphs are isomorphic; otherwise, False.
              - mapping (dict or None): The child→parent node mapping if isomorphic, else None.
        """
        node_match = generic_node_match(
            node_label_names, node_label_default, [eq] * len(node_label_names)
        )
        edge_match = generic_edge_match(edge_attribute, 1, eq)

        gm = nx.algorithms.isomorphism.GraphMatcher(
            child, parent, node_match=node_match, edge_match=edge_match
        )
        is_iso = gm.is_isomorphic()
        return is_iso, (gm.mapping if is_iso else None)

    @staticmethod
    def get_child1_to_child2_mapping(
        mapping_child1_to_parent: Dict[Any, Any],
        mapping_child2_to_parent: Dict[Any, Any],
    ) -> Dict[Any, Optional[Any]]:
        """
        Build a mapping from Child1 to Child2 using each child's mapping to a common Parent.

        If a Parent node in Child1's mapping is not in Child2's inverted mapping,
        that Child1 node will map to None.

        Parameters:
        - mapping_child1_to_parent (dict): Mapping from Child1 nodes → Parent nodes.
        - mapping_child2_to_parent (dict): Mapping from Child2 nodes → Parent nodes.

        Returns:
        - dict: A dictionary from Child1 node → Child2 node based on the shared Parent node.
        """
        # Invert Child2→Parent to get Parent→Child2
        inverted_child2 = {
            parent_node: child2_node
            for child2_node, parent_node in mapping_child2_to_parent.items()
        }

        # Build Child1→Child2 by looking up each Parent node in the inverted Child2 mapping
        mapping_child1_to_child2: Dict[Any, Optional[Any]] = {}
        for child1_node, parent_node in mapping_child1_to_parent.items():
            mapping_child1_to_child2[child1_node] = inverted_child2.get(
                parent_node, None
            )
        mapping_child1_to_child2 = {
            key: value
            for key, value in mapping_child1_to_child2.items()
            if value is not None
        }
        return mapping_child1_to_child2

    def fit(self, rule_1: str, rule_2: str, comp_rule: str) -> Optional[Dict[Any, Any]]:
        """
        Demonstrate an alignment-based composition workflow using the class methods.

        1. Convert each GML-based rule into an internal graph (via gml_to_its).
        2. Enumerate all unique mappings from rule_2 to comp_rule.
        3. For each mapping, subtract rule_2 from comp_rule using that mapping.
        4. Check if rule_1 is isomorphic to the resulting new graph.
           - If isomorphic, build a child1→child2 mapping and return it.

        Parameters:
        - rule_1 (str): GML representation of the first rule.
        - rule_2 (str): GML representation of the second rule.
        - comp_rule (str): GML representation of a composite rule.

        Returns:
        - Optional[dict]: A dictionary mapping rule_1's nodes to the new_graph's nodes if alignment is found.
                          Returns None otherwise.
        """
        # Convert GML to internal graph structures
        rc_1 = gml_to_its(rule_1)
        rc_2 = gml_to_its(rule_2)
        comp_its = gml_to_its(comp_rule)

        # Enumerate mappings from rule_2 → comp_rule
        maps_2 = self.enumerate_all_unique_mappings(rc_2, comp_its)
        for map_2 in maps_2:
            # Subtract rule_2 from comp_rule with this mapping
            new_graph = self.subtract_parent_from_child(rc_2, comp_its, map_2)
            # Check if rule_1 is isomorphic to new_graph
            is_iso, map_1 = self.graph_alignment(rc_1, new_graph)
            if is_iso and map_1 is not None:
                # If isomorphic, build a final mapping from rule_1 → new_graph
                mappings = self.get_child1_to_child2_mapping(map_1, map_2)
                return mappings
        return None
