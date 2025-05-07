# import warnings
# from rdkit import Chem
# from pathlib import Path
# from typing import List, Union
# from collections import Counter
# from synkit.IO.data_io import load_gml_as_text
# from synkit.Synthesis.reactor_utils import _deduplicateGraphs, _get_connected_subgraphs

# import mod
# from mod import smiles, config, ruleGMLString, DG


# class CoreEngine:
#     """
#     The MØDModeling class encapsulates functionalities for reaction modeling using the MØD
#     toolkit. It provides methods for forward and backward prediction based on templates
#     library.
#     """

#     def __init__(self) -> None:
#         warnings.warn("deprecated", DeprecationWarning)
#         pass

#     @staticmethod
#     def generate_reaction_smiles(
#         temp_results: List[str], base_smiles: str, is_forward: bool = True
#     ) -> List[str]:
#         """
#         Constructs reaction SMILES strings from intermediate results using a base SMILES
#         string, indicating whether the process is a forward or backward reaction. This
#         function iterates over a list of intermediate SMILES strings, combines them with
#         the base SMILES, and formats them into complete reaction SMILES strings.

#         Parameters:
#         - temp_results (List[str]): Intermediate SMILES strings resulting from partial
#         reactions or combinations.
#         - base_smiles (str): The SMILES string representing the starting point of the
#         reaction, either as reactants or products, depending on the reaction direction.
#         - is_forward (bool, optional): Flag to determine the direction of the reaction;
#         'True' for forward reactions where 'base_smiles' are reactants, and 'False' for
#         backward reactions where 'base_smiles' are products. Defaults to True.

#         Returns:
#         - List[str]: A list of complete reaction SMILES strings, formatted according to
#         the specified reaction direction.
#         """
#         results = []
#         for comb in temp_results:
#             joined_smiles = ".".join(comb)
#             reaction_smiles = (
#                 f"{base_smiles}>>{joined_smiles}"
#                 if is_forward
#                 else f"{joined_smiles}>>{base_smiles}"
#             )
#             results.append(reaction_smiles)
#         return results

#     @staticmethod
#     def _prediction_wo_reagent(
#         initial_molecules: List[Union[str, object]],
#         rule: mod.libpymod.Rule,
#         print_results: bool = False,
#         verbosity: int = 0,
#     ) -> List[List[str]]:
#         """
#         Applies the reaction rule to the given molecules without considering reagents.

#         Parameters:
#         - initial_molecules (List[Union[str, object]]): List of initial molecules represented by SMILES or objects.
#         - rule (mod.libpymod.Rule): The reaction rule to apply.
#         - print_results (bool): Whether to print the results.
#         - verbosity (int): Verbosity level for output.

#         Returns:
#         - List[List[str]]: A list of intermediate SMILES strings for the reaction products.
#         """
#         # Initialize the derivation graph and execute the strategy
#         dg = DG(graphDatabase=initial_molecules)
#         config.dg.doRuleIsomorphismDuringBinding = False
#         dg.build().apply(initial_molecules, rule, verbosity=verbosity)
#         if print_results:
#             dg.print()

#         temp_results = []
#         for e in dg.edges:
#             productSmiles = [v.graph.smiles for v in e.targets]
#             temp_results.append(productSmiles)
#         del dg
#         return temp_results

#     @staticmethod
#     def _prediction_with_reagent(
#         initial_smiles: List[str],
#         initial_molecules: List[Union[str, object]],
#         rule: mod.libpymod.Rule,
#         print_results: bool = False,
#         verbosity: int = 0,
#     ) -> List[List[str]]:
#         """
#         Applies the reaction rule to the given molecules considering the reagents.

#         Parameters:
#         - initial_smiles (List[str]): Initial molecules represented as SMILES strings.
#         - initial_molecules (List[Union[str, object]]): List of initial molecules.
#         - rule (mod.libpymod.Rule): The reaction rule to apply.
#         - print_results (bool): Whether to print the results.
#         - verbosity (int): Verbosity level for output.

#         Returns:
#         - List[List[str]]: A list of intermediate SMILES strings with reagents included.
#         """
#         dg = DG(graphDatabase=initial_molecules)
#         config.dg.doRuleIsomorphismDuringBinding = False
#         dg.build().apply(initial_molecules, rule, verbosity=verbosity, onlyProper=False)
#         if print_results:
#             dg.print()
#         temp_results, small_educt = [], []
#         for edge in dg.edges:
#             temp_results.append([vertex.graph.smiles for vertex in edge.targets])
#             small_educt.append([vertex.graph.smiles for vertex in edge.sources])

#         for key, solution in enumerate(temp_results):
#             educt = small_educt[key]
#             small_educt_counts = Counter(
#                 Chem.CanonSmiles(smile) for smile in educt if smile is not None
#             )
#             reagent_counts = Counter([Chem.CanonSmiles(s) for s in initial_smiles])
#             reagent_counts.subtract(small_educt_counts)
#             reagent = [
#                 smile
#                 for smile, count in reagent_counts.items()
#                 for _ in range(count)
#                 if count > 0
#             ]
#             solution.extend(reagent)
#         del dg
#         return temp_results

#     @staticmethod
#     def _inference(
#         rule_file_path: Union[str, Path],
#         initial_smiles: List[str],
#         prediction_type: str = "forward",
#         print_results: bool = False,
#         verbosity: int = 0,
#     ) -> List[str]:
#         """
#         Applies a specified reaction rule to a set of initial molecules represented by SMILES strings.
#         The reaction can be simulated in forward or backward direction.

#         Parameters:
#         - rule_file_path (Union[str, Path]): Path to the GML file containing the reaction rule.
#         - initial_smiles (List[str]): Initial molecules as SMILES strings.
#         - prediction_type (str): Direction of the reaction ('forward' or 'backward').
#         - print_results (bool): Whether to print the results.
#         - verbosity (int): Verbosity level for output.

#         Returns:
#         - List[str]: SMILES strings of the resulting molecules or reactions.
#         """

#         # Determine the rule inversion based on reaction type
#         invert_rule = prediction_type == "backward"
#         # Convert SMILES strings to molecule objects, avoiding duplicate conversions
#         initial_molecules = [smiles(smile, add=False) for smile in (initial_smiles)]

#         initial_molecules = _deduplicateGraphs(initial_molecules)

#         initial_molecules = sorted(
#             initial_molecules, key=lambda molecule: molecule.numVertices, reverse=False
#         )
#         # Load the reaction rule from the GML file
#         rule_path = Path(rule_file_path)

#         try:
#             if rule_path.is_file():
#                 gml_content = load_gml_as_text(rule_file_path)
#             else:
#                 gml_content = rule_file_path
#         except Exception as e:
#             # print(f"An error occurred while loading the GML file: {e}")
#             gml_content = rule_file_path
#         reaction_rule = ruleGMLString(gml_content, invert=invert_rule, add=False)

#         _number_subgraphs = _get_connected_subgraphs(gml_content, invert=invert_rule)
#         if len(initial_molecules) <= _number_subgraphs:
#             temp_results = CoreEngine._prediction_wo_reagent(
#                 initial_molecules, reaction_rule, print_results, verbosity
#             )
#         else:
#             temp_results = CoreEngine._prediction_with_reagent(
#                 initial_smiles,
#                 initial_molecules,
#                 reaction_rule,
#                 print_results,
#                 verbosity,
#             )

#         reaction_processing_map = {
#             "forward": lambda smiles: CoreEngine.generate_reaction_smiles(
#                 temp_results, ".".join(initial_smiles), is_forward=True
#             ),
#             "backward": lambda smiles: CoreEngine.generate_reaction_smiles(
#                 temp_results, ".".join(initial_smiles), is_forward=False
#             ),
#         }

#         # Use the reaction type to select the appropriate processing function and apply it
#         if prediction_type in reaction_processing_map:
#             return reaction_processing_map[prediction_type](initial_smiles)
#         else:
#             return ""
