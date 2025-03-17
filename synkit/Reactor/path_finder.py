import heapq
import random
from collections import deque
from typing import List, Dict


class ReactionPathFinder:
    def __init__(self, reaction_rounds: List[Dict[str, List[str]]]):
        """
        Initialize with a list of dictionaries, each representing a reaction round.

        Parameters:
        - reaction_rounds (List[Dict[str, List[str]]]): A list where each dictionary
        contains the reaction SMILES strings per round, with round numbers as keys.
        """
        self.reaction_rounds = reaction_rounds

    def search_paths(
        self,
        input_smiles: str,
        target_smiles: str,
        method: str = "bfs",
        iterations: int = 1000,
    ) -> List[List[str]]:
        """
        Search for reaction pathways from the input molecule to the target molecule
        using a specified method.

        Parameters:
        - input_smiles (str): SMILES representation of the starting molecule.
        - target_smiles (str): SMILES representation of the target molecule.
        - method (str, optional): The search method to use.
        Can be 'bfs' for Breadth-First Search, 'astar' for A* Search,
        or 'mc' for Monte Carlo search. Default is 'bfs'.
        - iterations (int, optional): The number of iterations to perform if using
        the Monte Carlo search method. Default is 1000.

        Returns:
        - List[List[str]]: A list of paths, where each path is a list of SMILES strings
        representing the reactions leading from the input to the target molecule.

        Raises:
            ValueError: If an invalid method is provided.
        """
        if method == "bfs":
            return self._bfs(input_smiles, target_smiles)
        elif method == "astar":
            return self._astar(input_smiles, target_smiles)
        elif method == "mc":
            return self._monte_carlo_search(input_smiles, target_smiles, iterations)
        else:
            raise ValueError("Invalid method. Choose 'bfs', 'astar', or 'montecarlo'.")

    def _bfs(self, input_smiles: str, target_smiles: str) -> List[List[str]]:
        """
        Private method to perform Breadth-First Search (BFS) for finding
        reaction pathways.

        Parameters:
        - input_smiles (str): SMILES representation of the starting molecule.
        - target_smiles (str): SMILES representation of the target molecule.

        Returns:
            List[List[str]]: A list of successful reaction pathways found.
        """
        queue = deque([(input_smiles, [], 0)])
        pathways = []

        while queue:
            current_smiles, current_path, round_index = queue.popleft()
            if round_index >= len(self.reaction_rounds):
                continue

            reactions = self.reaction_rounds[round_index].get(
                f"Round {round_index + 1}", []
            )

            for reaction_smiles in reactions:
                reactants, products = reaction_smiles.split(">>")
                reactant_smiles = set(reactants.split("."))

                if current_smiles in reactant_smiles:
                    product_smiles = products.split(".")
                    updated_path = current_path + [reaction_smiles]

                    if target_smiles in product_smiles:
                        pathways.append(updated_path)
                    else:
                        for product in product_smiles:
                            queue.append((product, updated_path, round_index + 1))
        return pathways

    def _heuristic(self, smiles: str, target_smiles: str) -> int:
        """
        Heuristic function for A* search, estimating the difference in molecule size
        between current and target.

        Parameters:
        - smiles (str): SMILES of the current molecule.
        - target_smiles (str): SMILES of the target molecule.

        Returns:
            int: Estimated cost (difference in length of SMILES strings).
        """
        return abs(len(smiles) - len(target_smiles))

    def _astar(self, input_smiles: str, target_smiles: str) -> List[List[str]]:
        """
        Private method to perform A* search for finding reaction pathways.

        Parameters:
        - input_smiles (str): SMILES representation of the starting molecule.
        - target_smiles (str): SMILES representation of the target molecule.

        Returns:
            List[List[str]]: A list of successful reaction pathways found.
        """
        heap = [(self._heuristic(input_smiles, target_smiles), input_smiles, [], 0)]
        pathways = []

        while heap:
            _, current_smiles, current_path, round_index = heapq.heappop(heap)
            if round_index >= len(self.reaction_rounds):
                continue

            reactions = self.reaction_rounds[round_index].get(
                f"Round {round_index + 1}", []
            )

            for reaction_smiles in reactions:
                reactants, products = reaction_smiles.split(">>")
                reactant_smiles = set(reactants.split("."))

                if current_smiles in reactant_smiles:
                    product_smiles = products.split(".")
                    updated_path = current_path + [reaction_smiles]

                    if target_smiles in product_smiles:
                        pathways.append(updated_path)
                    else:
                        for product in product_smiles:
                            cost = len(updated_path) + self._heuristic(
                                product, target_smiles
                            )
                            heapq.heappush(
                                heap, (cost, product, updated_path, round_index + 1)
                            )
        return pathways

    def _monte_carlo_search(
        self, input_smiles: str, target_smiles: str, iterations: int
    ) -> List[List[str]]:
        """
        Private method to perform Monte Carlo search for finding reaction pathways.

        Parameters:
        - input_smiles (str): SMILES representation of the starting molecule.
        - target_smiles (str): SMILES representation of the target molecule.
        - iterations (int): Number of iterations to randomly explore potential pathways.

        Returns:
            List[List[str]]: A list of successful reaction pathways found.
        """
        pathways = set()

        for _ in range(iterations):
            current_smiles = input_smiles
            current_path = []

            for round_index in range(len(self.reaction_rounds)):
                reactions = [
                    r
                    for r in self.reaction_rounds[round_index].get(
                        f"Round {round_index + 1}", []
                    )
                    if current_smiles in set(r.split(">>")[0].split("."))
                ]

                if not reactions:
                    break

                reaction_smiles = random.choice(reactions)
                _, products = reaction_smiles.split(">>")
                product_smiles = products.split(".")
                current_path.append(reaction_smiles)

                if target_smiles in product_smiles:
                    pathways.add(tuple(current_path))
                    break
                else:
                    current_smiles = random.choice(product_smiles)

        return [list(pathway) for pathway in pathways]
