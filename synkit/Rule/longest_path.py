import networkx as nx
from collections import deque
from typing import Tuple, List, Set


class LongestPath:
    def __init__(self, G: nx.Graph):
        """
        Initializes the LongestPath object with a graph.

        Parameters:
        - G (nx.Graph): The networkx graph.
        """
        self.G = G
        self.vertices = len(G.nodes)

    def BFS(self, u: int) -> Tuple[int, int]:
        """
        Performs a Breadth-First Search (BFS) from a given node `u` to
        find the farthest node and its distance.

        Parameters:
        - u (int): The starting node for the BFS.

        Returns:
        - Tuple[int, int]: The farthest node from `u` and its distance.
        """
        # Initialize visited and distance dictionaries
        visited = {i: False for i in self.G.nodes}
        distance = {i: -1 for i in self.G.nodes}

        # Distance of `u` from itself is 0
        distance[u] = 0
        queue = deque([u])
        visited[u] = True

        while queue:
            front = queue.popleft()

            # Explore neighbors of the current node
            for neighbor in self.G.neighbors(front):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    distance[neighbor] = distance[front] + 1
                    queue.append(neighbor)

        # Find the farthest node and its distance
        maxDis = 0
        nodeIdx = u  # Default to start node if no farther node is found
        for node, dist in distance.items():
            if dist > maxDis:
                maxDis = dist
                nodeIdx = node

        return nodeIdx, maxDis

    def LongestPathInDisconnectedGraph(self) -> int:
        """
        Finds the longest path in a potentially disconnected graph.
        The graph can consist of multiple components.

        This method performs a BFS on every unvisited component to find the
        farthest node and computes the longest path across all components.

        Returns:
            int: The length of the longest path in the graph across all components.
        """
        visited_components: Set[int] = set()
        longest_path: int = 0
        component_paths: List[Tuple[int, int, int]] = []

        # Iterate over all nodes to ensure all components are covered
        for node in self.G.nodes:
            if node not in visited_components:
                # First BFS to find one end point of the longest path in this component
                farthest_node, _ = self.BFS(node)

                # Second BFS from that node to find the actual longest path
                node_2, component_longest_path = self.BFS(farthest_node)

                # Track the longest path found
                longest_path = max(longest_path, component_longest_path)
                component_paths.append((farthest_node, node_2, component_longest_path))

                # Mark all nodes in this component as visited
                queue = deque([farthest_node])
                while queue:
                    current = queue.popleft()
                    if current not in visited_components:
                        visited_components.add(current)
                        for neighbor in self.G.neighbors(current):
                            if neighbor not in visited_components:
                                queue.append(neighbor)

        return longest_path
