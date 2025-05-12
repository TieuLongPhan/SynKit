import importlib.util
from typing import List
from synkit.IO.chem_converter import gml_to_smart

from synkit.Chem.Reaction.standardize import Standardize
from synkit.Chem.Reaction.rsmi_utils import reverse_reaction

from synkit.Graph.ITS.normalize_aam import NormalizeAAM
from synkit.Graph.ITS.its_expand import ITSExpand

from synkit.Rule.Modify.molecule_rule import MoleculeRule
from synkit.Rule.Compose.rule_compose import RuleCompose
from synkit.Rule.Modify.rule_utils import _increment_gml_ids

from synkit.Synthesis.reactor_utils import (
    _get_unique_aam,
    _add_reagent,
    _get_reagent_rsmi,
)


if importlib.util.find_spec("mod"):
    from mod import ruleGMLString
else:
    ruleGMLString = None
    print("Optional 'mod' package not found")


class ReactorRule:
    """
    Handles the transformation of SMILES strings to reaction SMILES (RSMI) by applying
    chemical reaction rules defined in GML strings. It can optionally reverse the reaction,
    exclude atom mappings, and include unchanged reagents in the output.
    """

    def __init__(self) -> None:
        """
        Initializes the ReactorRule object.
        """
        pass

    def _process(
        self,
        smiles: str,
        gml_rule: str,
        invert: bool = False,
        exclude_aam: bool = False,
        include_reagents: bool = False,
    ) -> List[str]:
        """
        Processes a reaction SMILES (RSMI) to adjust atom mappings, extract reaction centers,
        decompose into separate reactant and product graphs, and generate the corresponding SMILES.

        Parameters:
        - smiles (str): The SMILES string of the molecule to be transformed.
        - gml_rule (str): The GML string representing the transformation rule.
        - invert (bool, optional): Whether to reverse the reaction direction. Defaults to False.
        - exclude_aam (bool, optional): Whether to exclude atomic atom mapping (AAM) numbers
                                        from the final rSMI. Defaults to False.
        - include_reagents (bool, optional): Whether to include unchanged reagents in the output.
                                             Defaults to False.

        Returns:
        List[str]: A list of unique rSMI strings resulting from the applied rule, possibly including
                   reagents and modified by other options.
        """
        new_rsmi = []
        standardizer = Standardize()

        rule = ruleGMLString(gml_rule, invert=invert, add=False)
        mol_rule = MoleculeRule().generate_molecule_rule(smiles)
        comp_rules = RuleCompose()._compose(ruleGMLString(mol_rule, add=False), rule)

        if comp_rules:
            for value in comp_rules:
                gml = _increment_gml_ids(value.getGMLString())
                smart = gml_to_smart(gml, explicit_hydrogen=True)
                new_rsmi.append(smart)

        unique_rsmi = _get_unique_aam(new_rsmi)

        for key, value in enumerate(unique_rsmi):
            if invert:
                value = reverse_reaction(value)
            norm = NormalizeAAM().fit(value, fix_aam_indice=False)
            if include_reagents:
                reagents = _get_reagent_rsmi(value)
                norm = _add_reagent(norm, reagents)
                norm = ITSExpand().expand_aam_with_its(norm)
            unique_rsmi[key] = standardizer.fit(norm, remove_aam=exclude_aam)

        return unique_rsmi
