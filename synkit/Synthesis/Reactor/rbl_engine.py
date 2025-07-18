from rdkit import Chem
from synkit.IO import its_to_rsmi, rsmi_to_its, smiles_to_graph
from synkit.Chem.utils import remove_explicit_H_from_rsmi
from synkit.Chem.Reaction.radical_wildcard import RadicalWildcardAdder
from synkit.Synthesis.Reactor.syn_reactor import SynReactor
from synkit.Graph.Wildcard.fuse_graph import fuse_wc_graphs, find_wc_graph_isomorphism


class RBLEngine:
    """Reaction-based Learning Engine that takes a reaction SMARTS (rsmi) and a
    transformation template, applies forward and backward synthesis via
    SynReactor, augments with radical wildcards, identifies wildcard-aware
    graph isomorphisms between forward and backward intermediates, and fuses
    matching graphs into new intermediates.

    :param rsmi: Reaction SMARTS string in the form
        "Reactants>>Products".
    :type rsmi: str
    :param template: A reaction template SMARTS string, may include
        explicit H.
    :type template: str
    """

    def __init__(self, rsmi: str, template: str) -> None:
        """Initialize the RBLEngine with a reaction SMARTS and a template.

        Cleans explicit hydrogens in the template, parses the template
        into an ITS (internal template structure), and converts the
        reactant and product SMARTS into graph representations for host
        forward and backward graphs.

        :param rsmi: Reaction SMARTS string "Reactants>>Products".
        :type rsmi: str
        :param template: Reaction template SMARTS, possibly with
            explicit H.
        :type template: str
        """
        self.rsmi = rsmi
        template = remove_explicit_H_from_rsmi(template)

        self.rc = rsmi_to_its(template, core=True)
        r, p = self.rsmi.split(">>")
        self.host_fw = smiles_to_graph(r)
        self.host_bw = smiles_to_graph(p)

    def _fw(self):
        """Generate forward reaction intermediates using SynReactor, then apply
        RadicalWildcardAdder to each reaction SMARTS, and return their ITS
        representations.

        :returns: List of ITS objects for forward intermediates.
        :rtype: List
        """
        reactor = SynReactor(
            self.host_fw, self.rc, partial=True, implicit_temp=True, explicit_h=False
        )
        fw = reactor.smarts_list
        fw = [RadicalWildcardAdder().transform(rxn) for rxn in fw]
        return [rsmi_to_its(rxn) for rxn in fw]

    def _bw(self):
        """Generate backward reaction intermediates by inverting the template
        in SynReactor, apply RadicalWildcardAdder, and return ITS
        representations.

        :returns: List of ITS objects for backward intermediates.
        :rtype: List
        """
        reactor = SynReactor(
            self.host_bw,
            self.rc,
            partial=True,
            implicit_temp=True,
            explicit_h=False,
            invert=True,
        )
        bw = reactor.smarts_list
        bw = [RadicalWildcardAdder().transform(rxn) for rxn in bw]
        return [rsmi_to_its(rxn) for rxn in bw]

    def fit(self):
        """Attempt to fuse forward and backward ITS graphs into new
        intermediates.

        For each forward ITS and backward ITS pair:
        1. Find a wildcard-aware graph isomorphism.
        2. If found, fuse the two graphs via fuse_wc_graphs.

        :returns: A list of new reaction SMARTS strings representing fused
                  intermediates, or None if no fusions succeed.
        :rtype: List[str] | None
        """
        list_fw_its = self._fw()
        list_bw_its = self._bw()
        list_its = []
        for i in list_fw_its:
            for j in list_bw_its:
                mapping = find_wc_graph_isomorphism(i, j)
                if mapping:
                    try:
                        its = fuse_wc_graphs(i, j, mapping)
                        list_its.append(its)
                    except Exception:
                        continue
        if not list_its:
            return None
        list_smart = [its_to_rsmi(its) for its in list_its]
        return [value for value in list_smart if value]

    @staticmethod
    def remove_explicith_rsmi(rsmi: str) -> str:
        """Strip any explicit hydrogens from a reaction SMARTS string.

        :param rsmi: Reaction SMARTS "Reactants>>Products".
        :type rsmi: str
        :returns: Reaction SMARTS with explicit H removed.
        :rtype: str
        """
        r, p = rsmi.split(">>")
        mol_r = Chem.MolFromSmiles(r)
        mol_p = Chem.MolFromSmiles(p)
        return f"{Chem.MolToSmiles(mol_r)}>>{Chem.MolToSmiles(mol_p)}"
