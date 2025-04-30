from collections import defaultdict
import heapq
from typing import List, Dict, Any
from synkit.Chem.Reaction.cleanning import Cleanning
from synkit.Chem.utils import (
    count_carbons,
    process_smiles_list,
    get_max_fragment,
)
from synkit.Synthesis.reactor_utils import _remove_reagent
from synkit.Synthesis.core_engine import CoreEngine


class DCRN:
    def __init__(
        self,
        rule_list: List[Dict[str, Any]],
        smiles_list: List[str],
        starting_compound: str,
        target_compound: str,
    ) -> None:
        self.rule_list = rule_list
        self.smiles_list = smiles_list
        self.starting_compound = starting_compound
        self.target_compound = target_compound
        self.count_starting = count_carbons(starting_compound)
        self.count_target = count_carbons(target_compound)
        self.adjacency = defaultdict(list)
        self.visited = set()
        self.expansion_cache = {}  # Cache for expanded nodes

    @staticmethod
    def _get_valid_node(molecules, lower, upper):
        """
        Filters molecules by their carbon count within the given range.
        """
        return [mol for mol in molecules if lower <= count_carbons(mol) <= upper]

    def _expand(self, smiles_list: List[str]) -> List[str]:
        """
        Expands molecules based on transformation rules. Uses caching to avoid redundant computation.
        """
        smiles_tuple = tuple(smiles_list)
        if smiles_tuple in self.expansion_cache:
            return self.expansion_cache[smiles_tuple]

        results = []
        processed_smiles = process_smiles_list(smiles_list)
        for rule_dict in self.rule_list:
            expansions = CoreEngine()._inference(rule_dict["gml"], processed_smiles)
            expansions = Cleanning().clean_smiles(expansions)
            expansions = [_remove_reagent(e) for e in expansions]
            for r in expansions:
                product = r.split(">>")[1]
                product = get_max_fragment(product)
                results.append(product)

        # Filter valid nodes within the carbon count range
        valid_nodes = self._get_valid_node(
            results, self.count_starting, self.count_target
        )
        self.expansion_cache[smiles_tuple] = valid_nodes
        return valid_nodes

    def _heuristic(self, a: str, b: str) -> int:
        """
        Returns the heuristic estimate (absolute difference in carbon count) between two compounds.
        """
        return abs(count_carbons(a) - count_carbons(b))

    def _dynamic_expand_node(self, node: str, smiles_list: list) -> None:
        """
        Dynamically expands the given node to generate new possible compounds.
        """
        if node not in self.visited:
            self.visited.add(node)
            expanded_nodes = self._expand(
                smiles_list + [node]
            )  # Expand only the current node
            self.adjacency[node] = expanded_nodes

    def build_and_search(
        self,
        starting_compound: str,
        target_compound: str,
        max_solutions: int = 5,
        fast_process: bool = True,
    ) -> Dict[str, Any]:
        """
        Builds the search graph and searches for paths from the starting compound to the target compound.
        Ensures depth levels follow a sequential order starting from 0.
        """
        # Initialize the heap with the starting compound at depth 0
        heap = [
            (0, starting_compound, [starting_compound])
        ]  # Start with depth 0 and path
        visited_nodes = set([starting_compound])
        solutions = []

        # Dictionary to track the minimum depth at which a node was visited
        node_depths = {starting_compound: 0}

        while heap and len(solutions) < max_solutions:
            # Pop the node with the smallest depth (simulating a depth-first search)
            depth, node, path = heapq.heappop(heap)
            print(f"{node} (depth {depth})")  # Print node and its depth for clarity

            # Check if we reached the target compound
            if node == target_compound:
                solutions.append(path)
                continue

            # Expand the node if it hasn't been visited at this depth
            if node not in self.adjacency:
                self._dynamic_expand_node(node, self.smiles_list)

            expanded_nodes = self.adjacency.get(node, [])
            for nbr in expanded_nodes:
                if nbr not in visited_nodes:
                    # Mark as visited and add to the heap
                    visited_nodes.add(nbr)
                    new_path = path + [nbr]

                    # Increment depth for the child node
                    new_depth = depth + 1

                    # Heuristic estimate to the target (optional, could be omitted if not needed)
                    h = self._heuristic(nbr, target_compound)

                    # Use depth as the primary factor for heap priority
                    new_cost = (
                        new_depth + h
                    )  # Prioritize by depth first, then heuristic
                    heapq.heappush(heap, (new_cost, nbr, new_path))

                    # Track the node depth for each node
                    node_depths[nbr] = new_depth

        return {"adjacency": self.adjacency, "paths": solutions}
