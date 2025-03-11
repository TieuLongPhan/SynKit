import os
import shutil
import warnings
import subprocess
import networkx as nx
from synkit.IO.nx_to_gml import NXToGML
from synkit.IO.chem_converter import smart_to_gml
from mod import ruleGMLString


class DPOVis:
    """
    A class that handles visualization-related tasks for a given input. The class processes inputs in
    different formats such as strings or tuples of network graphs, and outputs corresponding GML data.

    Optionally, the class can compile the result using an external tool when the 'compile' flag is set to True.
    """

    def __init__(self) -> None:
        """
        Initializes the DPOVis object.
        """
        pass

    def _vis(self, input, compile=False, clean=False):
        """
        This method takes an input, which can either be a string or a tuple of network graphs,
        and processes it accordingly into a GML representation. If the input is not valid,
        a ValueError will be raised. Additionally, the method can trigger the compilation of the output
        into a specific format if the 'compile' flag is set to True.

        Parameters:
        - input (str or tuple): A string that may contain 'rule' or a tuple of nx.Graph objects.
        - compile (bool): If True, attempts to compile the output using an external tool.
        - clean (bool): If True, clean out and summary folder.

        Raises:
        - ValueError: If the input is of an invalid type or format.
        - subprocess.CalledProcessError: If the compilation fails.
        """
        # Process string input
        if isinstance(input, str):
            if "rule" in input:
                gml = input
            else:
                gml = smart_to_gml(input)

        # Process tuple of network graphs
        elif isinstance(input, tuple):
            if all(isinstance(i, nx.Graph) for i in input):
                gml = NXToGML().transform(input, reindex=True)
            else:
                raise ValueError(
                    "Each element in the tuple must be a networkx.Graph object."
                )

        # Handle invalid input types
        else:
            raise ValueError(
                "Input must be a string or a tuple of networkx.Graph objects."
            )

        # Ensure output directory exists
        os.makedirs("out", exist_ok=True)

        # Generate rule from GML string
        rule = ruleGMLString(gml)

        # Print the rule
        rule.print()

        # Optionally compile the result if 'compile' is True
        if compile:
            warnings.warn("Latex Compiler is required to compile the result.")

            try:
                # Assume 'mod_post' is the necessary command for compilation
                subprocess.run("mod_post", check=True, shell=True)
            except subprocess.CalledProcessError as e:
                raise subprocess.CalledProcessError(
                    e.returncode, e.cmd, output=e.output, stderr=e.stderr
                ) from e
        if clean:
            self._clean_output_directory()

    def _clean_output_directory(self):
        """
        Helper function to clean the 'out' and 'summary' directories by deleting the entire directories
        and their contents.
        """
        directories = ["out", "summary"]

        for dir_name in directories:
            if os.path.exists(dir_name):
                print(f"Cleaning up the {dir_name} directory...")

                try:
                    shutil.rmtree(dir_name)  # Remove directory and all its contents
                    print(f"{dir_name} directory cleaned and deleted.")
                except Exception as e:
                    print(f"Error removing {dir_name}: {e}")
            else:
                print(f"No {dir_name} directory found to clean.")
