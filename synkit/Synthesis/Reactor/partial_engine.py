from synkit.IO import rsmi_to_its, smiles_to_graph
from synkit.Chem.Reaction.radical_wildcard import RadicalWildcardAdder
from synkit.Synthesis.Reactor.syn_reactor import SynReactor
from synkit.Chem.utils import remove_explicit_H_from_rsmi


class PartialEngine:
    """Partial Reaction Learning Engine that applies a single‐direction
    (forward or backward) template transformation, injects radical wildcards,
    and returns a list of intermediate ITS strings.

    :param smi: A reaction SMARTS (rsmi) string in the form
        "Reactants>>Products" or a simple SMILES string when used for
        one‐sided synthesis.
    :type smi: str
    :param template: A reaction template SMARTS string, which may
        include explicit H.
    :type template: str
    """

    def __init__(self, smi: str, template: str) -> None:
        """Initialize the PartialEngine.

        - Removes explicit hydrogens from the given template SMARTS.
        - Parses the cleaned template into an internal template structure (ITS).
        - Converts the provided SMILES or rsmi string into a NetworkX graph
          to serve as the host for synthesis.

        :param smi: The input SMILES or rsmi string for which to generate intermediates.
        :type smi: str
        :param template: The reaction template SMARTS; explicit H atoms will be stripped.
        :type template: str
        """
        # Remove explicit H atoms from the template
        template = remove_explicit_H_from_rsmi(template)

        # Convert cleaned template to internal structure (core atoms only)
        self.rc = rsmi_to_its(template, core=True)

        # Build host graph from the provided SMILES or rsmi
        self.host = smiles_to_graph(smi)

    def fit(self, invert: bool = False) -> list[str]:
        """Apply the template in one direction to generate radical‐wildcarded
        reaction SMARTS (ITS).

        - Instantiates a SynReactor on the host graph and ITS.
        - Sets partial, implicit‐template, and explicit‐H flags.
        - If `invert=True`, runs the backward direction; otherwise forward.
        - Post‐processes each reaction SMARTS with RadicalWildcardAdder.

        :param invert: If True, apply the template in the reverse direction (Products→Reactants).
                       Default is False (forward direction).
        :type invert: bool
        :returns: A list of ITS‐encoded reaction SMARTS strings,
                  each augmented with radical wildcard notation.
        :rtype: list[str]
        """
        reactor = SynReactor(
            self.host,
            self.rc,
            partial=True,
            implicit_temp=True,
            explicit_h=False,
            invert=invert,
        )
        # Generate SMARTS, then inject radical wildcards
        smarts_list = reactor.smarts_list
        wildcarded = [RadicalWildcardAdder().transform(rxn) for rxn in smarts_list]
        return wildcarded
