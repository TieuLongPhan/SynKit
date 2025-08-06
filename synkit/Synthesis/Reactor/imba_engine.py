import networkx as nx
from typing import Union, Optional, List
from synkit.Graph.Canon.canon_graph import GraphCanonicaliser
from synkit.Synthesis.Reactor.syn_reactor import SynReactor, Strategy
from synkit.Graph.syn_graph import SynGraph
from synkit.Rule.syn_rule import SynRule
from synkit.Graph.Wildcard.radwc import RadWC
from synkit.Chem.Reaction.radical_wildcard import clean_wc


class ImbaEngine:
    """
    Reactor for applying a SynKit reaction template to a substrate, with
    options for inversion, canonicalisation, strategy, partial ITS, and
    radical wildcard appending and fragment cleaning in products.

    :param substrate: Input substrate; SMILES string, networkx.Graph, or SynGraph.
    :type substrate: Union[str, nx.Graph, SynGraph]
    :param template: Reaction template; SMARTS (bracketed) string, networkx.Graph, or SynRule.
    :type template: Union[str, nx.Graph, SynRule]
    :param add_wildcard: If True, apply radical wildcard transform to each product SMARTS.
    :type add_wildcard: bool
    :param clean_fragments: If True, remove wildcard fragments and optionally keep max fragment.
    :type clean_fragments: bool
    :param max_frag: If True, force maximal fragment selection when cleaning.
    :type max_frag: bool
    :param invert: If True, apply the template in reverse (product → reactant).
    :type invert: bool
    :param canonicaliser: Optional GraphCanonicaliser for preprocessing or postprocessing.
    :type canonicaliser: Optional[GraphCanonicaliser]
    :param strategy: Enumeration strategy (Strategy enum or string).
    :type strategy: Union[Strategy, str]
    :param partial: If True, perform partial ITS graph construction on results.
    :type partial: bool
    """

    def __init__(
        self,
        substrate: Union[str, nx.Graph, SynGraph],
        template: Union[str, nx.Graph, SynRule],
        add_wildcard: bool = True,
        clean_fragments: bool = False,
        max_frag: bool = False,
        invert: bool = False,
        canonicaliser: Optional[GraphCanonicaliser] = None,
        strategy: Union[Strategy, str] = Strategy.ALL,
        partial: bool = False,
        embed_threshold: float = None,
        embed_pre_filter: bool = False,
    ) -> None:
        # Assign parameters
        self.substrate = substrate
        self.template = template
        self.add_wildcard = add_wildcard
        self.clean_fragments = clean_fragments
        self.max_frag = max_frag
        self.invert = invert
        self.canonicaliser = canonicaliser
        self.strategy = strategy
        self.partial = partial
        self.embed_threshold = embed_threshold
        self.embed_pre_filter = embed_pre_filter
        # Internal state
        self._results: List[str] = []
        # Auto-run fit on init
        self.fit()

    def __repr__(self) -> str:
        return (
            f"<ImbaEngine(substrate={type(self.substrate).__name__}, "
            f"template={type(self.template).__name__}, add_wildcard={self.add_wildcard}, "
            f"clean_fragments={self.clean_fragments}, max_frag={self.max_frag}, "
            f"invert={self.invert}, strategy={self.strategy}, partial={self.partial})>"
        )

    @staticmethod
    def describe() -> None:
        """
        Print class documentation and usage examples.
        """
        print(ImbaEngine.__doc__)

    def fit(self) -> "ImbaEngine":
        """
        Apply the reaction template to the substrate, producing product SMARTS.
        Optionally clean wildcard fragments and add radical wildcards.
        Results are stored internally and self is returned.

        :returns: self
        :rtype: ImbReactor
        :raises ValueError: If substrate cannot be parsed or reaction fails.
        """
        from synkit.IO import graph_to_smi

        # Determine reactant SMILES
        if isinstance(self.substrate, (nx.Graph, SynGraph)):
            react_smiles = graph_to_smi(self.substrate)
        elif isinstance(self.substrate, str):
            react_smiles = self.substrate
        else:
            raise ValueError(f"Unsupported substrate type: {type(self.substrate)}")

        reactor = SynReactor(
            react_smiles,
            template=self.template,
            invert=self.invert,
            strategy=self.strategy,
            partial=self.partial,
            implicit_temp=True,
            explicit_h=False,
            canonicaliser=self.canonicaliser,
            embed_threshold=self.embed_threshold,
            embed_pre_filter=self.embed_pre_filter,
        )
        raw_smarts: List[str] = reactor.smarts_list

        # Add radical wildcards if requested
        if self.add_wildcard:
            wc = []
            for s in raw_smarts:
                try:
                    wc.append(RadWC.transform(s))
                except Exception as e:
                    print(e)
        else:
            wc = raw_smarts
            # Clean fragments if requested
        if self.clean_fragments:
            self._results = [
                clean_wc(s, invert=False, max_frag=self.max_frag, wild_card=True)
                for s in wc
            ]
        else:
            self._results = wc

        return self

    @property
    def smarts_list(self) -> List[str]:
        """
        Product SMARTS results from the last fit() invocation.

        :returns: List of SMARTS strings.
        :rtype: List[str]
        """
        return self._results.copy()

    def __len__(self) -> int:
        """
        Number of product SMARTS results.
        """
        return len(self._results)

    def __getitem__(self, idx: int) -> str:
        """
        Get the product SMARTS at index `idx`.

        :param idx: Index of desired SMARTS.
        :type idx: int
        :returns: SMARTS string at position `idx`.
        :rtype: str
        :raises IndexError: If idx is out of bounds.
        """
        return self._results[idx]

    def to_list(self) -> List[str]:
        """
        Return all product SMARTS as a list.

        :returns: List of SMARTS strings.
        :rtype: List[str]
        """
        return self._results.copy()
