from rdkit import Chem
from rdkit.Chem import rdFMCS
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional
from synkit.IO.chem_converter import gml_to_smart
from synkit.Rule.molecule_rule import MoleculeRule
from synkit.Chem.utils import get_sanitized_smiles, remove_duplicates, filter_smiles
from mod import ruleGMLString, RCMatch


class RuleFrag:
    def __init__(self) -> None:
        """Initialize the RuleFrag class with caches and null initial values."""
        self.backward_cache: Dict[Tuple[str, str], List[str]] = {}
        self.mcs_mol: Optional[Chem.Mol] = None

    def _compute_mcs(self, smiles_a: str, smiles_b: str) -> None:
        """
        Compute the Maximum Common Substructure (MCS) between two SMILES strings.
        Store the result as an RDKit Mol object in self.mcs_mol.

        Parameters:
        - smiles_a (str): First SMILES string.
        - smiles_b (str): Second SMILES string.
        """
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)
        if mol_a is None or mol_b is None:
            self.mcs_mol = None
            return

        mcs_result = rdFMCS.FindMCS([mol_a, mol_b], completeRingsOnly=False, timeout=10)
        if mcs_result.smartsString:
            self.mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
        else:
            self.mcs_mol = None

    def has_mcs_substructure(self, smiles: str) -> bool:
        """
        Check if the stored MCS is a substructure of the given SMILES string.

        Parameters:
        - smiles (str): SMILES string to check against the MCS.

        Returns:
        - bool: True if MCS is a substructure, False otherwise. If no MCS is stored, returns True.
        """
        if self.mcs_mol is None:
            # No valid MCS => skip the substructure requirement
            return True

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        return mol.HasSubstructMatch(self.mcs_mol)

    def _apply_backward(self, smiles: str, rule: str) -> list:
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
        mol_rule_str = ruleGMLString(mol_rule)

        matcher = RCMatch(mol_rule_str, rule_str)
        mod_results = matcher.composeAll()

        results_set = set()
        for match_rule in mod_results:
            # In this user-defined backward mode, the "reactants"
            # appear in smarts.split(">>")[1].
            smarts = gml_to_smart(match_rule.getGMLString(), sanitize=False)[0]
            reactants = smarts.split(">>")[1].split(".")
            reactants = get_sanitized_smiles(reactants)
            results_set.update(reactants)

        results_list = filter_smiles(results_set, smiles)
        results_list = remove_duplicates(results_list)
        self.backward_cache[cache_key] = list(results_list)
        return self.backward_cache[cache_key]

    def backward_synthesis_search(
        self,
        product_smiles: str,
        known_precursor_smiles: str,
        rules: List[str],
        max_solutions: int = 1,
    ) -> List[List[str]]:
        """
        Perform a backward synthesis search to find pathways from a product to a known precursor using specified rules.

        Parameters:
        - product_smiles (str): SMILES string of the product molecule.
        - known_precursor_smiles (str): SMILES string of the known precursor molecule.
        - rules (List[str]): List of transformation rules to apply.
        - max_solutions (int, optional): Maximum number of solution pathways to return. Defaults to 1.

        Returns:
        - List[List[str]]: List of pathways, each a list of SMILES strings from product to precursor.
        """
        # 1) Compute and store MCS
        self._compute_mcs(product_smiles, known_precursor_smiles)

        # If the product is already the known precursor
        if product_smiles == known_precursor_smiles:
            return [[product_smiles]]

        # BFS queue of paths (each path is a list of SMILES steps)
        queue = deque([[product_smiles]])
        visited = set([product_smiles])

        # For reconstructing multiple minimal solutions:
        #   parents[child] = [possible_parents]
        parents = defaultdict(list)

        solutions = []
        found_depth = None
        depth = 0
        max_depth = None

        while queue and len(solutions) < max_solutions:
            depth += 1
            layer_size = len(queue)

            # If we have a user-defined max_depth, do not expand beyond it
            if max_depth is not None and depth > max_depth:
                break

            for _ in range(layer_size):
                path = queue.popleft()
                current = path[-1]

                # If we've already found solutions at depth `found_depth`,
                # skip expansions deeper than that.
                if found_depth is not None and depth > found_depth:
                    break

                # Apply all backward rules to get precursors
                for rule in rules:
                    precursors = self._apply_backward(current, rule)

                    for prec in precursors:

                        # If the reaction rule output contains something
                        # like '->' or other undesired strings, skip
                        if "->" in prec:
                            continue

                        if not self.has_mcs_substructure(prec):
                            # print(prec)
                            continue
                        # print(prec)
                        if prec not in visited:
                            visited.add(prec)
                            parents[prec].append(current)
                            new_path = path + [prec]
                            queue.append(new_path)

                        else:
                            # If we've seen it, but didn't record the current node
                            # as a parent, add it
                            if current not in parents[prec]:
                                parents[prec].append(current)

                        # Check if we've exactly reached the known precursor
                        if prec == known_precursor_smiles:
                            # Mark the BFS depth at which we found a solution
                            if found_depth is None:
                                found_depth = depth
                            if depth <= found_depth:
                                found_depth = depth

                            # Reconstruct all minimal paths for this newly found solution
                            new_solutions = self._reconstruct_paths(
                                start=product_smiles,
                                end=known_precursor_smiles,
                                parents=parents,
                            )
                            # print(found_depth)
                            # De-duplicate
                            deduped = []
                            seen_set = set()
                            for sol in new_solutions:
                                tup = tuple(sol)
                                if tup not in seen_set:
                                    seen_set.add(tup)
                                    deduped.append(sol)

                            # Add them to overall solutions (found_depthup to max_solutions)
                            needed = max_solutions - len(solutions)
                            solutions.extend(deduped[:needed])
                            if len(solutions) >= max_solutions:
                                break
                    # End loop over precursors
                    if len(solutions) >= max_solutions:
                        break
                # End loop over rules
                if len(solutions) >= max_solutions:
                    break
            # End BFS layer

            # If we found solutions at this depth, do NOT expand deeper
            if found_depth is not None and depth >= found_depth:
                break

        return solutions

    def _reconstruct_paths(self, start: str, end: str, parents: dict) -> list:
        """
        Given a BFS parent dictionary that maps child -> list of parents,
        reconstruct all distinct minimal paths from 'start' to 'end'.
        We do a simple stack-based DFS from 'end' back to 'start',
        then reverse each path to get correct forward order.
        """
        all_paths = []
        stack = [(end, [end])]
        while stack:
            node, path_so_far = stack.pop()
            if node == start:
                all_paths.append(path_so_far[::-1])  # reverse it
            else:
                for p in parents[node]:
                    stack.append((p, path_so_far + [p]))
        return all_paths
