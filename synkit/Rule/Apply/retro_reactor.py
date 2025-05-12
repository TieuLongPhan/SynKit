import heapq
import importlib.util
from typing import Dict, List, Tuple


from synkit.IO.chem_converter import gml_to_smart
from synkit.Rule.Modify.molecule_rule import MoleculeRule
from synkit.Chem.utils import (
    get_sanitized_smiles,
    remove_duplicates,
    filter_smiles,
    count_carbons,
)


if importlib.util.find_spec("mod"):
    from mod import ruleGMLString, RCMatch
else:
    ruleGMLString = None
    RCMatch = None
    print("Optional 'mod' package not found")


class RetroReactor:
    def __init__(self) -> None:
        """
        Initialize the RuleFrag class with caches and null initial values.

        Attributes:
        - backward_cache: A dictionary cache (keyed by (smiles, rule)) to avoid redundant computations.
        """
        self.backward_cache: Dict[Tuple[str, str], List[str]] = {}

    def _apply_backward(self, smiles: str, rule: str) -> List[str]:
        """
        Apply a transformation rule in backward mode to a SMILES string, returning possible precursors.
        Uses caching to avoid redundant computations.

        Parameters:
        - smiles (str): SMILES string to transform.
        - rule (str): Transformation rule.

        Returns:
        - List[str]: List of possible precursor SMILES strings.
        """
        cache_key = (smiles, rule)
        if cache_key in self.backward_cache:
            return self.backward_cache[cache_key]

        # Convert rule to GML in backward mode
        rule_str = ruleGMLString(rule, invert=True, add=False)
        mol_rule = MoleculeRule().generate_molecule_rule(smiles)
        mol_rule_str = ruleGMLString(mol_rule, add=False)

        matcher = RCMatch(mol_rule_str, rule_str)
        mod_results = matcher.composeAll()

        results_set = set()
        for match_rule in mod_results:
            # In user-defined backward mode, "reactants"
            # appear in smarts.split(">>")[1].
            smarts = gml_to_smart(match_rule.getGMLString(), sanitize=False)
            reactants = smarts.split(">>")[1].split(".")
            reactants = get_sanitized_smiles(reactants)
            results_set.update(reactants)

        # Filter out SMILES that are invalid relative to the original
        results_list = filter_smiles(results_set, smiles)
        results_list = remove_duplicates(results_list)
        self.backward_cache[cache_key] = list(results_list)
        return self.backward_cache[cache_key]

    def _heuristic(self, current_smiles: str, precursor_smiles: str) -> int:
        """
        Heuristic function for A* search. Here, we define the "distance" as the
        absolute difference in the carbon count between the current SMILES and
        the known precursor SMILES.

        Parameters:
        - current_smiles (str): The SMILES of the node being expanded.
        - precursor_smiles (str): The SMILES of the known precursor (our target).

        Returns:
        - int: Estimated cost (distance) based on difference in carbon count.
        """
        return abs(count_carbons(current_smiles) - count_carbons(precursor_smiles))

    def backward_synthesis_search(
        self,
        product_smiles: str,
        known_precursor_smiles: str,
        rules: List[str],
        max_solutions: int = 1,
        fast_process: bool = True,
    ) -> List[Dict[str, List]]:
        """
        Perform a backward synthesis search from a product to a known precursor using A* search.

        Constrains any intermediate X to satisfy:
            n_C(known_precursor_smiles) <= n_C(X) <= n_C(product_smiles).

        If fast_process=True, we prune expansions by storing the best cost at which
        we have visited each SMILES. If a new path to the same SMILES has a higher cost,
        we do not expand it again.

        If fast_process=False, we disable cost-based pruning, which can yield more solutions
        (possibly duplicates) but also potentially more computational expense.

        Parameters:
        - product_smiles (str): SMILES string of the product molecule.
        - known_precursor_smiles (str): SMILES string of the known precursor molecule.
        - rules (List[str]): List of transformation rules to apply in backward mode.
        - max_solutions (int): Maximum number of solution pathways to return. Defaults to 1.
        - fast_process (bool): If True, enable pruning (classic A*). If False,
          do not prune (which can discover more solutions but is slower).

        Returns:
        - List[Dict[str, List]]: A list of solution pathways, each represented as a dictionary with:
            {{
                'rule_index': List[int],  # The sequence of rule indices used
                'smiles': List[str]       # The sequence of SMILES (excluding the final known precursor)
            }}
        """
        # If the product is already the known precursor
        if product_smiles == known_precursor_smiles:
            return [
                {
                    "rule_index": [],
                    "smiles": [],  # no intermediate SMILES if product == precursor
                }
            ]

        # Carbon constraint check for the *starting* product
        product_C = count_carbons(product_smiles)
        precursor_C = count_carbons(known_precursor_smiles)
        if not (precursor_C <= product_C):
            # If the product has *fewer* carbons than the precursor, no solutions
            return []

        # Priority queue of expansions: each element is (cost, path_so_far, rules_used)
        # where path_so_far is [product_smiles, ..., intermediate].
        # We'll do "backward" expansions until we find known_precursor_smiles.
        heap = []
        start_cost = 0 + self._heuristic(product_smiles, known_precursor_smiles)
        # Initialize with no rules used so far
        heapq.heappush(heap, (start_cost, [product_smiles], []))

        solutions: List[Dict[str, List]] = []
        # For fast_process pruning, store the best cost found for each SMILES
        visited_cost = {} if fast_process else None

        while heap and len(solutions) < max_solutions:
            cost, path, rule_path = heapq.heappop(heap)
            current_smiles = path[-1]

            # If we've arrived at the precursor, we have a solution
            if current_smiles == known_precursor_smiles:
                # Exclude the final known precursor from 'smiles' to be consistent
                solutions.append({"rule_index": rule_path, "smiles": path[:-1]})
                continue  # keep searching in case we want more solutions

            # If cost-based pruning is active, skip expansions if there's a cheaper route
            if fast_process and visited_cost is not None:
                if (current_smiles in visited_cost) and (
                    visited_cost[current_smiles] < cost
                ):
                    continue
                visited_cost[current_smiles] = cost

            # Expand all possible predecessors from the current node
            for i, rule in enumerate(rules):
                precursors = self._apply_backward(current_smiles, rule)
                for prec in precursors:
                    # Filter out invalid arrow-based or extraneous data
                    if "->" in prec:
                        continue

                    # Carbon distance constraint:
                    # n_C(known_precursor_smiles) <= n_C(prec) <= n_C(product_smiles)
                    nc_prec = count_carbons(prec)
                    if not (precursor_C <= nc_prec <= product_C):
                        continue

                    new_path = path + [prec]
                    new_rule_path = rule_path + [i]
                    new_cost = len(new_path) + self._heuristic(
                        prec, known_precursor_smiles
                    )

                    # If pruning is active, skip if we've found a cheaper route before
                    if fast_process and visited_cost is not None:
                        prev_cost = visited_cost.get(prec, float("inf"))
                        if new_cost >= prev_cost:
                            continue

                    # If we've reached the known precursor, record solution
                    if prec == known_precursor_smiles:
                        transformations = [
                            f"{new_path[j+1]}>>{new_path[j]}"
                            for j in range(len(new_path) - 1)
                        ]
                        solutions.append(
                            {"rule_index": new_rule_path, "rsmi": transformations}
                        )
                        if len(solutions) >= max_solutions:
                            break
                    else:
                        # Otherwise push onto the heap
                        heapq.heappush(heap, (new_cost, new_path, new_rule_path))

                if len(solutions) >= max_solutions:
                    break
            # End for each rule

        return solutions
