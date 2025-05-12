import importlib.util
from typing import List
from synkit.Chem.Reaction.standardize import Standardize
from synkit.Chem.Reaction.rsmi_utils import (
    find_longest_fragment,
    merge_reaction,
    remove_common_reagents,
    remove_duplicates,
    reverse_reaction,
)
from synkit.IO.chem_converter import gml_to_smart
from synkit.Rule.Modify.molecule_rule import MoleculeRule
from synkit.Rule.Modify.rule_utils import _increment_gml_ids
from synkit.Rule.Compose.rule_compose import RuleCompose

if importlib.util.find_spec("mod"):
    from mod import ruleGMLString
else:
    ruleGMLString = None
    print("Optional 'mod' package not found")


class RuleRBL:
    def __init__(self) -> None:
        """Initialize the RuleRBL class."""
        pass

    def rbl(self, rsmi: str, gml_rule: str, remove_aam: bool = True) -> List[str]:
        """
        Applies transformation rules to a reaction SMILES string based on GML rules.

        Parameters:
        - rsmi (str): Reaction SMILES string to process.
        - gml_rule (str): GML rule string to apply transformations.

        Returns:
        - List[str]: List of new reaction SMILES strings after applying the rules.
        """
        new_rsmi = []
        raw_rsmi = {0: [], 1: []}
        standardizer = Standardize()

        standardized_rsmi = standardizer.fit(rsmi)
        molecules = standardized_rsmi.split(">>")

        for index, mol in enumerate(molecules):
            rule = ruleGMLString(gml_rule, invert=(index % 2 != 0), add=False)
            mol_rule = MoleculeRule().generate_molecule_rule(mol)
            comp_rules = RuleCompose()._compose(
                ruleGMLString(mol_rule, add=False), rule
            )

            if comp_rules:
                for value in comp_rules:
                    gml = _increment_gml_ids(value.getGMLString())
                    smart = gml_to_smart(gml, explicit_hydrogen=True)
                    if index == 1:
                        smart = reverse_reaction(smart)
                    standardized_smart = standardizer.fit(smart)
                    if standardized_smart != rsmi:
                        raw_rsmi[index].append(standardized_smart)
                        target_index = len(molecules) - index - 1
                        target_molecule = molecules[target_index]
                        if (
                            target_molecule
                            in standardized_smart.split(">>")[target_index]
                        ):
                            if remove_aam:
                                new_rsmi.append(standardized_smart)
                            else:
                                new_rsmi.append(smart)

        if len(new_rsmi) == 0:
            r = remove_duplicates(raw_rsmi[0])
            p = remove_duplicates(raw_rsmi[1])
            for i in r:
                for j in p:
                    product_r = find_longest_fragment(i.split(">>")[1].split("."))
                    reactant_p = find_longest_fragment(j.split(">>")[0].split("."))
                    if product_r == reactant_p:
                        merge = merge_reaction(i, j)
                        clean = remove_common_reagents(merge)
                        new_rsmi.append(clean)
        if new_rsmi:
            new_rsmi = new_rsmi[0]
        return new_rsmi
