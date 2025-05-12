import os
import subprocess
import importlib.util
from typing import Union, List

from synkit.IO import load_database, setup_logging
from synkit.Synthesis.reactor_utils import _deduplicateGraphs

logger = setup_logging("INFO")

if importlib.util.find_spec("mod"):
    from mod import Rule, smiles, config, DG, addSubset, repeat, Graph
else:
    Rule = None
    smiles = None
    DG = None
    config = None
    addSubset = None
    repeat = None
    Graph = None
    logger.warning("Optional 'mod' package not found")


class MODCRN:
    """
    MODCRN
    ======

    High-level class for constructing, inspecting, and reporting a chemical reaction
    network using the MØD derivation graph (DG) API.

    Key Features
    ------------
    * Flexible rule loading: accept a JSON database path or a list of GML strings.
    * Initial molecule deduplication via graph isomorphism.
    * Customizable iterative strategy for rule applications.
    * Built-in summary and external report export.

    Parameters
    ----------
    rule_db_path : Union[str, List[str]]
        Path to a JSON rule database or list of GML rule strings.
    initial_smiles : List[str]
        SMILES strings of the seed molecules.
    repeats : int, default=2
        Number of repeat cycles for rule application.

    Properties
    ----------
    rules : List[Rule]
        Loaded Rule objects for network construction.
    graphs : List[Graph]
        Deduplicated initial Graph objects.
    derivation_graph : DG
        The active derivation graph instance.
    num_vertices : int
        Number of molecules in the derivation graph.
    num_edges : int
        Number of reactions in the derivation graph.

    Methods
    -------
    build() -> None
        Populate the derivation graph with the configured strategy.
    print_summary() -> None
        Print and save a concise summary of the derivation graph.
    export_report(path: str) -> None
        Generate an external report via the `mod_post` CLI.
    help() -> None
        Print usage examples and API summarylog.
    """

    def __init__(
        self,
        rule_db_path: Union[str, List[str]],
        initial_smiles: List[str],
        n_repeats: int = 2,
    ):
        # Load rules from path or raw list
        if isinstance(rule_db_path, str):
            entries = load_database(rule_db_path)
            gml_strings = [e["gml"] for e in entries]
        elif isinstance(rule_db_path, list):
            gml_strings = rule_db_path
        else:
            raise TypeError("rule_db_path must be str or list of GML strings")
        self._rules = [Rule.fromGMLString(g) for g in gml_strings]

        # Initialize and deduplicate seed graphs
        seeds = [smiles(s, add=False) for s in initial_smiles]
        self._graphs = _deduplicateGraphs(seeds)

        # Configure DG
        self.repeats = n_repeats
        self._dg = DG(graphDatabase=self._graphs)
        config.dg.doRuleIsomorphismDuringBinding = False

    @property
    def rules(self) -> List[Rule]:
        """Loaded Rule objects for network construction."""
        return self._rules

    @property
    def graphs(self) -> List["Graph"]:
        """Deduplicated initial Graph objects."""
        return self._graphs

    @property
    def derivation_graph(self) -> DG:
        """The active derivation graph instance."""
        return self._dg

    @property
    def num_vertices(self) -> int:
        """Number of molecules in the derivation graph."""
        try:
            return self._dg.numVertices
        except AttributeError:
            return self._dg.graphSize()

    @property
    def num_edges(self) -> int:
        """Number of reactions in the derivation graph."""
        try:
            return self._dg.numEdges
        except AttributeError:
            return self._dg.edgeCount()

    def build(self) -> None:
        """
        Populate the derivation graph using:
        addSubset(initial) >> repeat[repeats](rules)
        """
        strat = addSubset(self._graphs) >> repeat[self.repeats](self._rules)
        builder = self._dg.build()
        builder.execute(strat)

    def print_summary(self) -> None:
        """
        Print and save a concise summary of the derivation graph.
        """
        out_dir = "out"
        os.makedirs(out_dir, exist_ok=True)

        self._dg.print()

    def export_report(self) -> None:
        """
        Generate an external report via the `mod_post` CLI.

        """
        try:
            subprocess.run(["mod_post"], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"mod_post failed with exit code {e.returncode}")

    def help(self) -> None:
        """
        Print usage examples and API summary for MODCRN.
        """
        print(
            "MODCRN Usage:\n"
            "    crn = MODCRN(rule_db_path, initial_smiles, repeats)\n"
            "    crn.build()\n"
            "    crn.print_summary()\n"
            "    crn.export_report('summary')\n"
            "Properties:\n"
            "    rules, graphs, derivation_graph, num_vertices, num_edges"
        )
