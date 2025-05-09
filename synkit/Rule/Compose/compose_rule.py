import importlib.util
from synkit.IO.debug import setup_logging
from typing import List, Set, Any, Dict, Optional
from synkit.IO.chem_converter import gml_to_smart, smart_to_gml
from synkit.Rule.Modify.rule_utils import _increment_gml_ids
from synkit.Chem.Reaction.standardize import Standardize
from synkit.Chem.Reaction.cleanning import Cleanning
from synkit.Chem.Reaction.rsmi_utils import find_longest_fragment


logger = setup_logging()

if importlib.util.find_spec("mod"):
    from mod import RCMatch, ruleGMLString
    from synkit.Synthesis.Reactor.mod_reactor import MODReactor
else:
    RCMatch = None
    ruleGMLString = None
    logger.warning("Optional 'mod' package not found.")


class ComposeRule:

    @staticmethod
    def filter_smallest_vertex(combo: List[object]) -> List[object]:
        """
        Filters and returns the elements from a list that have the smallest
        number of vertices in their context.

        Parameters:
        - combo (List[object]): A list of objects, each with a 'context'
        attribute that has a 'numVertices' attribute.

        Returns:
        - List[object]: A list of objects from the input list that have
        the minimum number of vertices in their context.
        """
        # Extract the number of vertices from each rule's context and find the minimum
        num_vertices = [rule.context.numVertices for rule in combo]
        min_vertex = min(num_vertices)

        # Collect all rules that have the minimum number of vertices
        new_combo = [
            rule
            for rule, vertices in zip(combo, num_vertices)
            if vertices == min_vertex
        ]

        return new_combo

    @staticmethod
    def rule_cluster(graphs: List[Any]) -> List[Any]:
        """
        Cluster graphs based on their isomorphic relationships and
        return a representative from each cluster.

        Parameters:
        - graphs (List[Any]): A list of graph objects.

        Returns:
        - List[Any]: A list of graphs where each graph is a representative from a different cluster.
        """
        visited: Set[int] = set()
        clusters: List[Set[int]] = []

        for i, graph_i in enumerate(graphs):
            if i in visited:
                continue
            cluster: Set[int] = {i}
            visited.add(i)
            for j, graph_j in enumerate(graphs):
                if j in visited or j <= i:
                    continue
                # Assuming isomorphism() returns 1 for isomorphic graphs.
                if graph_i.isomorphism(graph_j) == 1:
                    cluster.add(j)
                    visited.add(j)
            clusters.append(cluster)

        representative_graphs = [graphs[list(cluster)[0]] for cluster in clusters]
        return representative_graphs

    @staticmethod
    def _compose_mapping(
        rule_1: str, rule_2: str, mapping: Dict[int, int], return_string: bool = True
    ) -> Any:
        """
        Compose two rule graphs from their GML representations using a mapping between external IDs.

        Parameters:
        - rule_1 (str): The GML representation for the first rule.
        - rule_2 (str): The GML representation for the second rule.
        - mapping (Dict[int, int]): A dictionary mapping external IDs in the first rule (child side)
                                    to corresponding external IDs in the second rule (parent side).
        - return_string (bool): If True, returns the composed rule as a GML string.

        Returns:
        - Any: The composed rule object or its GML string if return_string is True.
        """
        # Create rule objects from the GML inputs.
        r1 = ruleGMLString(rule_1)
        r2 = ruleGMLString(rule_2)

        # Create an RCMatch object with r1 and r2.
        m = RCMatch(r1, r2)

        # Push alignments between vertices according to the mapping.
        for child_ext_id, parent_ext_id in mapping.items():
            v1 = r1.getVertexFromExternalId(child_ext_id)
            v2 = r2.getVertexFromExternalId(parent_ext_id)
            m.push(v1.right, v2.left)

        # Compose the mapping.
        composed_rule = m.compose()
        if return_string:
            composed_rule = composed_rule.getGMLString()
        return composed_rule

    @staticmethod
    def _compose(rule_1: str, rule_2: str, return_string: bool = True) -> List[Any]:
        """
        Compose two rules and return a list of modifications that pass chemical valence checks.

        Parameters:
        - rule_1 (str): The first rule (in GML format) to compose.
        - rule_2 (str): The second rule (in GML format) to compose.
        - return_string (bool): If True, returns the composed rules as GML strings.

        Returns:
        - List[Any]: A list of valid composed rules (either as rule objects or as GML strings).
                      Returns an empty list if an error occurs.
        """
        try:
            m = RCMatch(
                ruleGMLString(rule_1, add=False), ruleGMLString(rule_2, add=False)
            )
            modRes = m.composeAll()
            modRes = ComposeRule.rule_cluster(modRes)
            if return_string:
                modRes = [i.getGMLString() for i in modRes]
            return modRes
        except Exception as e:
            print("Error during rule composition:", e)
            return []

    @staticmethod
    def _get_valid_rule(rules: List[str], format: str = "gml") -> List[str]:
        """
        Validate and convert a list of rule GML strings to either SMARTS or GML format.

        Parameters:
        - rules (List[str]): A list of rule GML strings.
        - format (str): The output format. 'smart' returns SMARTS strings; otherwise, returns GML strings.

        Returns:
        - List[str]: A list of valid rules in the desired format.
        """
        new_rules: List[str] = []
        for value in rules:
            new = gml_to_smart(value, sanitize=True, explicit_hydrogen=False)[0]
            if "Error" not in new:
                if format == "smart":
                    new_rules.append(new)
                else:
                    new_rules.append(
                        smart_to_gml(
                            new, sanitize=True, explicit_hydrogen=False, reindex=False
                        )
                    )
        return new_rules

    @staticmethod
    def _get_comp_reaction(smart_1: str, smart_2: str) -> str:
        """
        Compute a representative reaction SMILES for the composed rule from two SMARTS strings.

        Parameters:
        - smart_1 (str): The first reaction in SMARTS notation.
        - smart_2 (str): The second reaction in SMARTS notation.

        Returns:
        - str: A standardized reaction SMILES representing the composition.
        """
        std = Standardize()
        rsmi_1 = std.fit(smart_1)
        rsmi_2 = std.fit(smart_2)
        r1, p1 = rsmi_1.split(">>")
        r2, p2 = rsmi_2.split(">>")
        new_rsmi = std.fit(f"{r1}.{r2}>>{p1}.{p2}")
        return new_rsmi

    def get_rule_comp(self, smart_1: str, smart_2: str) -> Optional[str]:
        """
        Compose two reaction SMARTS strings into a rule (GML format) that reproduces a reference reaction.

        Parameters:
        - smart_1 (str): The first reaction in SMARTS notation.
        - smart_2 (str): The second reaction in SMARTS notation.

        Returns:
        - Optional[str]: The composed rule (in GML) if a valid candidate is found; otherwise, None.
        """
        rule_1 = smart_to_gml(
            smart_1, sanitize=True, explicit_hydrogen=False, reindex=False
        )
        rule_2 = smart_to_gml(
            smart_2, sanitize=True, explicit_hydrogen=False, reindex=False
        )
        reference_rsmi = self._get_comp_reaction(smart_1, smart_2)
        candidate_rules = self._compose(rule_1, rule_2, return_string=True)
        candidate_rules = [_increment_gml_ids(value) for value in candidate_rules]
        initial_smiles = reference_rsmi.split(">>")[0].split(".")
        largest_prod = find_longest_fragment(reference_rsmi.split(">>")[1].split("."))
        cds = []
        for candidate in candidate_rules:
            reactor = MODReactor(initial_smiles, candidate).run()
            inferred_rsmi = reactor.get_reaction_smiles()
            inferred_rsmi = Cleanning.clean_smiles(inferred_rsmi)
            inferred_prod = [i.split(">>")[1].split(".") for i in inferred_rsmi]
            if any(largest_prod in smi for smi in inferred_prod):
                cds.append(candidate)
                # return candidate

        cds = [ruleGMLString(i) for i in cds]
        cds = self.filter_smallest_vertex(cds)
        cds = [i.getGMLString() for i in cds]

        return cds
