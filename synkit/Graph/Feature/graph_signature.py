import hashlib
import networkx as nx


class GraphSignature:
    """
    Provides methods to generate canonical signatures for graph edges (with flexible 'order' and 'state' attributes,
    and node degrees/neighbor information), various spectral invariants, adjacency matrix, and complete graphs.
    Aims for high uniqueness without relying solely on isomorphism checks.
    """

    def __init__(self, graph: nx.Graph):
        """
        Initializes the GraphSignature class with a specified graph.

        Parameters:
        - graph (nx.Graph): A NetworkX graph instance.
        """
        self.graph = graph
        self._validate_graph()

    def _validate_graph(self):
        """
        Validates that all nodes have the required attributes ('element' and 'charge'),
        and all edges have the required 'order' attribute as int, float, or tuple of two floats,
        and optionally the 'state' attribute.

        Raises:
        - ValueError: If any node is missing the 'element' or 'charge' attribute,
                      or if any edge is missing the 'order' attribute or has an invalid type.
        """
        for node, data in self.graph.nodes(data=True):
            if "element" not in data:
                raise ValueError(f"Node {node} is missing the 'element' attribute.")
            if "charge" not in data:
                raise ValueError(f"Node {node} is missing the 'charge' attribute.")

        for u, v, data in self.graph.edges(data=True):
            if "order" not in data:
                raise ValueError(f"Edge ({u}, {v}) is missing the 'order' attribute.")
            order = data["order"]
            if isinstance(order, tuple):
                if len(order) != 2 or not all(
                    isinstance(o, (int, float)) for o in order
                ):
                    raise ValueError(
                        f"Edge ({u}, {v}) has an invalid 'order'. It must be a tuple of two ints/floats."
                    )
            elif not isinstance(order, (int, float)):
                raise ValueError(
                    f"Edge ({u}, {v}) has an invalid 'order'. It must be an int, float, or a tuple of two ints/floats."
                )

            # Optional: Validate 'state' attribute if present
            state = data.get("state", "steady")  # Default to 'steady' if missing
            if state not in {"break", "form", "steady"}:
                raise ValueError(
                    f"Edge ({u}, {v}) has an invalid 'state'. It must be 'break', 'form', or 'steady'."
                )

    def create_edge_signature(
        self, include_neighbors: bool = False, max_hop: int = 2
    ) -> str:
        """
        Generates a canonical edge signature by formatting each edge with sorted node elements (including charge),
        node degrees, bond order, bond state, and optionally including neighbor information and topological context.

        Parameters:
        - include_neighbors (bool): Whether to include neighbors' details in the edge signature.
        - max_hop (int): Maximum number of hops to include for neighbor-level structural information.

        Returns:
        - str: A concatenated and sorted string of edge representations.
        """
        edge_signature_parts = []

        for u, v, data in self.graph.edges(data=True):
            # Retrieve bond order (default to (1.0, 1.0) if missing)
            order = data.get("order", (1.0, 1.0))

            # Format order as a tuple (default or actual value)
            if isinstance(order, tuple):
                order_str = f"{{{order[0]:.1f},{order[1]:.1f}}}"
            else:
                order_str = f"{float(order):.1f}"

            # Get node elements and charges for both nodes
            node1_element = self.graph.nodes[u].get(
                "element", "X"
            )  # Default to 'X' if missing
            node1_charge = self.graph.nodes[u].get(
                "charge", 0
            )  # Default to 0 if missing
            node2_element = self.graph.nodes[v].get("element", "X")
            node2_charge = self.graph.nodes[v].get("charge", 0)

            # Construct node representation with element and charge
            node1 = f"{node1_element}{node1_charge}"
            node2 = f"{node2_element}{node2_charge}"

            # Optionally include neighbors in the signature
            if include_neighbors:
                neighbors_u = sorted(
                    [
                        f"{self.graph.nodes[neighbor].get('element', 'X')}{self.graph.nodes[neighbor].get('charge', 0)}"
                        + f"d{self.graph.degree(neighbor)}"
                        for neighbor in self.graph.neighbors(u)
                    ]
                )
                neighbors_v = sorted(
                    [
                        f"{self.graph.nodes[neighbor].get('element', 'X')}{self.graph.nodes[neighbor].get('charge', 0)}"
                        + f"d{self.graph.degree(neighbor)}"
                        for neighbor in self.graph.neighbors(v)
                    ]
                )

                # Represent neighbors within square brackets
                node1_neighbors = "".join(neighbors_u)
                node2_neighbors = "".join(neighbors_v)
                node1 = f"{node1}[{node1_neighbors}]"
                node2 = f"{node2}[{node2_neighbors}]"

            # Include k-hop neighborhood information
            if max_hop > 1:
                node1_neighbors_khop = self._get_khop_neighbors(u, max_hop)
                node2_neighbors_khop = self._get_khop_neighbors(v, max_hop)
                node1 += f"[{node1_neighbors_khop}]"
                node2 += f"[{node2_neighbors_khop}]"

            # Sort nodes to ensure consistency in edge signature (avoid direction dependency)
            node1, node2 = sorted([node1, node2])

            # Format the edge signature and append it
            edge_part = f"{node1}{order_str}{node2}"
            edge_signature_parts.append(edge_part)

        # Sort all edge signatures to ensure consistency in the final representation
        return "/".join(sorted(edge_signature_parts))

    def _get_khop_neighbors(self, node, max_hop):
        """
        Retrieves the k-hop neighborhood information for a given node.

        Parameters:
        - node (int): The node for which to get neighborhood information.
        - max_hop (int): Maximum number of hops for neighborhood exploration.

        Returns:
        - str: A concatenated string representing the k-hop neighborhood information.
        """
        k_hop_neighbors = []
        current_hop_neighbors = [node]
        for _ in range(max_hop):
            next_hop_neighbors = []
            for n in current_hop_neighbors:
                next_hop_neighbors.extend(list(self.graph.neighbors(n)))
            # Filter out already seen nodes to avoid loops
            next_hop_neighbors = set(next_hop_neighbors) - set(k_hop_neighbors)
            k_hop_neighbors.extend(next_hop_neighbors)
            current_hop_neighbors = next_hop_neighbors

        # Return sorted k-hop neighborhood info
        return "".join(
            sorted(
                [
                    f"{self.graph.nodes[neighbor].get('element', 'X')}{self.graph.nodes[neighbor].get('charge', 0)}"
                    for neighbor in k_hop_neighbors
                ]
            )
        )

    def create_wl_hash(self, iterations: int = 3) -> str:
        """
        Generates a Weisfeiler-Lehman (WL) hash for the graph to capture its structural features.

        Parameters:
        - iterations (int): Number of WL iterations to perform.

        Returns:
        - str: A hexadecimal hash representing the WL feature.
        """
        # Initialize labels with both 'element' and 'charge'
        labels = {
            node: f"{data['element']}{data.get('charge', 0)}"
            for node, data in self.graph.nodes(data=True)
        }
        for _ in range(iterations):
            new_labels = {}
            for node in self.graph.nodes():
                # Gather sorted labels of neighbors
                neighbor_labels = sorted(
                    labels[neighbor] for neighbor in self.graph.neighbors(node)
                )
                # Concatenate current label with neighbor labels
                concatenated = labels[node] + "".join(neighbor_labels)
                # Hash the concatenated string to obtain a new label
                new_label = hashlib.sha256(concatenated.encode()).hexdigest()
                new_labels[node] = new_label
            labels = new_labels
        # Aggregate all node labels into a sorted string and hash it
        sorted_labels = sorted(labels.values())
        aggregated = "".join(sorted_labels)
        graph_hash = hashlib.sha256(aggregated.encode()).hexdigest()
        return graph_hash

    def create_graph_signature(
        self,
        include_wl_hash: bool = True,
        include_neighbors: bool = True,
        max_hop: int = 1,
    ) -> str:
        """
        Combines edge, various spectral invariants, and WL hash into a single comprehensive graph signature.

        Parameters:
        - include_wl_hash (bool): Whether to include the Weisfeiler-Lehman hash.
        - include_spectral (bool): Whether to include spectral invariants.
        - include_combined_hash (bool): Whether to include the combined hash.
        - include_neighbors (bool): Whether to include neighbor information in edge signatures.

        Returns:
        - str: A concatenated string representing the complete graph signature.
        """
        signatures = []

        if include_wl_hash:
            wl_signature = self.create_wl_hash()
            signatures.append(f"{wl_signature}")

        edge_signature = self.create_edge_signature(
            include_neighbors=include_neighbors, max_hop=max_hop
        )
        signatures.append(f"{edge_signature}")

        return "|".join(signatures)
