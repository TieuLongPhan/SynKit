from __future__ import annotations

from synkit.IO.chem_converter import rsmi_to_graph, graph_to_rsmi, smiles_to_graph
from synkit.Graph.ITS.its_decompose import its_decompose
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Graph.ITS.its_builder import ITSBuilder
from synkit.Chem.Reaction.standardize import Standardize
from synkit.Graph.ITS.its_relabel import ITSRelabel

std = Standardize()


class ITSExpand:
    """Partially expand a reaction SMILES (RSMI) by reconstructing intermediate
    transition states (ITS) and applying transformation rules based on the
    reaction center graph.

    This class identifies the reaction center from an RSMI, builds and
    reconstructs the ITS graph, decomposes it back into reactants and
    products, and standardizes atom mappings to produce a fully mapped
    AAM RSMI.

    :cvar std: Standardize instance for reaction SMILES standardization.
    :type std: Standardize
    """

    def __init__(self) -> None:
        """Initialize ITSExpand.

        No instance-specific attributes are required.
        """
        pass

    @staticmethod
    def expand_aam_with_its(
        rsmi: str,
        relabel: bool = False,
        use_G: bool = True,
    ) -> str:
        """Expand a partial reaction SMILES to a full AAM RSMI using ITS
        reconstruction.

        :param rsmi: Reaction SMILES string in the format 'reactant>>product'.
        :type rsmi: str
        :param use_G: If True, expand using the reactant side; otherwise use the product side.
        :type use_G: bool
        :param light_weight: Flag indicating whether to apply a lighter-weight standardization.
        :type light_weight: bool
        :returns: Fully atom-mapped reaction SMILES after ITS expansion and standardization.
        :rtype: str
        :raises ValueError: If input RSMI format is invalid or ITS reconstruction fails.

        :example:
        >>> expander = ITSExpand()
        >>> expander.expand_aam_with_its("CC[CH2:3][Cl:1].[N:2]>>CC[CH2:3][N:2].[Cl:1]")
        '[CH3:1][CH2:2][CH2:3][Cl:4].[N:5]>>[CH3:1][CH2:2][CH2:3][N:5].[Cl:4]'
        """
        if relabel:
            return ITSRelabel().fit(rsmi)
        # Validate and split reaction SMILES
        try:
            react_smi, prod_smi = rsmi.split(">>")
        except ValueError as e:
            raise ValueError("Input RSMI must be 'reactant>>product'") from e

        # Build graphs for reactants and products
        react_graph, prod_graph = rsmi_to_graph(rsmi)

        # Construct the ITS reaction center graph
        rc_graph = ITSConstruction().ITSGraph(react_graph, prod_graph)

        # Choose which side to expand
        smi_side = react_smi if use_G else prod_smi
        side_graph = smiles_to_graph(
            smi_side, sanitize=True, drop_non_aam=False, use_index_as_atom_map=False
        )

        # Reconstruct the full ITS graph
        its_graph = ITSBuilder().ITSGraph(side_graph, rc_graph)

        # Decompose ITS back into reactant and product graphs
        new_react, new_prod = its_decompose(its_graph)

        # Convert graphs back to RSMI and standardize atom mappings
        expanded_rsmi = graph_to_rsmi(new_react, new_prod, its_graph, True, False)
        return std.fit(expanded_rsmi, remove_aam=False)
