import heapq
from collections import deque
from typing import List, Dict, Optional
from synkit.Chem.utils import count_carbons


class PathFinder:
    def __init__(
        self,
        reaction_rounds: List[Dict[str, List[str]]],
    ):
        """
        Initialize with a list of dictionaries, each representing a reaction round,
        plus an optional random state for reproducible Monte Carlo search.

        Parameters:
        - reaction_rounds (List[Dict[str, List[str]]]): A list where each dictionary
          contains the reaction SMILES strings for a given round
          (e.g. {"Round 1": [...] }).
        """
        self.reaction_rounds = reaction_rounds

        # Build adjacency structure:
        # self._adjacency[round_idx] => { reactant_smiles: [(reaction_smiles, [products])] }
        self._adjacency = []
        for round_idx, round_dict in enumerate(self.reaction_rounds):
            round_key = f"Round {round_idx + 1}"
            reactions = round_dict.get(round_key, [])
            adjacency_map = {}
            for rxn in reactions:
                reactants, products = rxn.split(">>")
                reactant_list = reactants.split(".")
                product_list = products.split(".")
                for rct in reactant_list:
                    adjacency_map.setdefault(rct, []).append((rxn, product_list))
            self._adjacency.append(adjacency_map)

    def _valid_intermediate(
        self, smiles: str, input_smiles: str, target_smiles: str
    ) -> bool:
        """
        Checks whether the given molecule satisfies the carbon_distance constraint:
            n_C(input_smiles) <= n_C(smiles) <= n_C(target_smiles)
        """
        return (
            count_carbons(input_smiles)
            <= count_carbons(smiles)
            <= count_carbons(target_smiles)
        )

    def search_paths(
        self,
        input_smiles: str,
        target_smiles: str,
        method: str = "bfs",
        max_solutions: Optional[int] = None,
        cheapest: bool = True,
    ) -> List[List[str]]:
        """
        Search for reaction pathways from the input molecule to the target molecule
        using a specified method, optionally limiting the number of solutions.

        Additionally, `cheapest` can be set to True or False:
          - If cheapest=True, BFS uses a visited set and A* prunes costlier routes (typical approach).
          - If cheapest=False, BFS does *not* track visited states (returns more solutions),
            and A* does *not* prune costlier routes (also returns more solutions).
            (May lead to duplicates or many solutions if cycles exist.)

        Parameters:
        - input_smiles (str): SMILES of the starting molecule.
        - target_smiles (str): SMILES of the target molecule.
        - method (str, optional): 'bfs', 'astar', or 'mc'.
        - iterations (int, optional): Number of MC iterations (if method='mc').
        - max_solutions (int, optional): If set, stop after finding this many solutions.
        - cheapest (bool, optional): Controls pruning.
          Default True => standard BFS/A*; False => "unrestricted" BFS/A*.

        Returns:
        - List[List[str]]: Each solution path is a list of reaction SMILES from start to target.
        """
        if method == "bfs":
            return self._bfs(input_smiles, target_smiles, max_solutions, cheapest)
        elif method == "astar":
            return self._astar(input_smiles, target_smiles, max_solutions, cheapest)
        else:
            raise ValueError("Invalid method. Choose 'bfs', 'astar', or 'mc'.")

    def _bfs(
        self,
        input_smiles: str,
        target_smiles: str,
        max_solutions: Optional[int],
        cheapest: bool,
    ) -> List[List[str]]:
        """
        Perform a BFS search. If cheapest=True, use a visited set to avoid re-processing
        the same (molecule, round_index). If cheapest=False, skip that pruning and collect
        *all* possible solutions (potentially large if cycles exist).

        Returns a list of successful reaction pathways, up to max_solutions if specified.
        """

        queue = deque([(input_smiles, [], 0)])
        pathways = []
        visited = set() if cheapest else None  # Only track visited if cheapest=True

        while queue:
            current_smiles, current_path, round_index = queue.popleft()
            if round_index >= len(self._adjacency):
                continue

            # For BFS with cheapest=True, skip repeated states
            if cheapest:
                if (current_smiles, round_index) in visited:
                    continue
                visited.add((current_smiles, round_index))

            adjacency_map = self._adjacency[round_index]
            if current_smiles not in adjacency_map:
                continue

            for rxn_string, product_list in adjacency_map[current_smiles]:
                updated_path = current_path + [rxn_string]
                # If any product is the target, record a solution
                if target_smiles in product_list:
                    pathways.append(updated_path)
                    if max_solutions is not None and len(pathways) >= max_solutions:
                        return pathways
                # Otherwise queue each valid product
                for product in product_list:
                    if self._valid_intermediate(product, input_smiles, target_smiles):
                        queue.append((product, updated_path, round_index + 1))

        return pathways

    def _heuristic(self, smiles: str, target_smiles: str) -> int:
        """
        Heuristic function for A* search.
        Returns difference in SMILES lengths as a stand-in for "distance."
        """
        return abs(len(smiles) - len(target_smiles))

    def _astar(
        self,
        input_smiles: str,
        target_smiles: str,
        max_solutions: Optional[int],
        cheapest: bool,
    ) -> List[List[str]]:
        """
        A* search. If cheapest=True, we track the best cost visited for each state
        and prune costlier paths. If cheapest=False, we do not prune, so we collect
        all solutions (but it may be large).

        Returns a list of successful reaction pathways, up to max_solutions if specified.
        """
        start_cost = self._heuristic(input_smiles, target_smiles)
        # Heap stores (cost, current_smiles, current_path, round_index)
        heap = [(start_cost, input_smiles, [], 0)]
        pathways = []
        visited_cost = {} if cheapest else None

        while heap:
            cost, current_smiles, current_path, round_index = heapq.heappop(heap)

            if cheapest and visited_cost is not None:
                # If we've seen this state with a cheaper or equal cost, skip
                if (current_smiles, round_index) in visited_cost:
                    if visited_cost[(current_smiles, round_index)] <= cost:
                        continue
                visited_cost[(current_smiles, round_index)] = cost

            if round_index >= len(self._adjacency):
                continue

            adjacency_map = self._adjacency[round_index]
            if current_smiles not in adjacency_map:
                continue

            for rxn_string, product_list in adjacency_map[current_smiles]:
                updated_path = current_path + [rxn_string]

                # If any product is the target, record a solution
                if target_smiles in product_list:
                    pathways.append(updated_path)
                    if max_solutions is not None and len(pathways) >= max_solutions:
                        return pathways
                else:
                    # Otherwise expand each product
                    next_round = round_index + 1
                    for product in product_list:
                        if self._valid_intermediate(
                            product, input_smiles, target_smiles
                        ):
                            new_cost = len(updated_path) + self._heuristic(
                                product, target_smiles
                            )
                            if cheapest and visited_cost is not None:
                                old_cost = visited_cost.get(
                                    (product, next_round), float("inf")
                                )
                                # Only push if cheaper route
                                if new_cost < old_cost:
                                    heapq.heappush(
                                        heap,
                                        (new_cost, product, updated_path, next_round),
                                    )
                            else:
                                # cheapest=False => always push
                                heapq.heappush(
                                    heap, (new_cost, product, updated_path, next_round)
                                )

        return pathways
