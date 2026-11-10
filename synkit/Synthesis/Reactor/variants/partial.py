from synkit.IO import rsmi_to_its, smiles_to_graph
from synkit.Chem.Reaction.radical_wildcard import RadicalWildcardAdder
from ..core.engine import SynReactor
from synkit.Chem.utils import remove_explicit_H_from_rsmi


class PartialEngine:
    """Partial Reaction Learning Engine that applies a single‐direction
    (forward or backward) template transformation, injects radical wildcards,
    and returns a list of intermediate ITS strings.

    This compatibility wrapper intentionally interprets reactant radicals as
    open substituent sites. Use ``WildcardEngine`` for typed attachment ports
    and preservation of genuine radicals.

    :param smi: A single-sided substrate SMILES string.
    :type smi: str
    :param template: A reaction template SMARTS string, which may
        include explicit H.
    :type template: str
    """

    def __init__(
        self,
        smi: str,
        template: str,
        electron_diagnostics: bool = False,
    ) -> None:
        """Initialize the PartialEngine.

        - Removes explicit hydrogens from the given template SMARTS.
        - Parses the cleaned template into an internal template structure (ITS).
        - Converts the provided SMILES string into a NetworkX graph
          to serve as the host for synthesis.

        :param smi: The input substrate SMILES for which to generate intermediates.
        :type smi: str
        :param template: The reaction template SMARTS; explicit H atoms will be stripped.
        :type template: str
        """
        # Remove explicit H atoms from the template
        template = remove_explicit_H_from_rsmi(template)

        # Convert cleaned template to internal structure (core atoms only)
        self.rc = rsmi_to_its(template, core=True)

        # Build host graph from the provided SMILES
        self.host = smiles_to_graph(smi)
        if self.host is None:
            raise ValueError("smi must be a valid single-sided substrate SMILES.")
        self.electron_diagnostics = electron_diagnostics
        self._diagnostics = []

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
        :return: A list of ITS‐encoded reaction SMARTS strings,
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
            electron_diagnostics=self.electron_diagnostics,
        )
        # Generate SMARTS, then inject radical wildcards
        smarts_list = reactor.smarts_list
        self._diagnostics = reactor.diagnostics
        wildcarded = [RadicalWildcardAdder().transform(rxn) for rxn in smarts_list]
        return wildcarded

    @property
    def diagnostics(self) -> list[dict]:
        """Electron diagnostics from the last reactor run."""
        return list(self._diagnostics)
