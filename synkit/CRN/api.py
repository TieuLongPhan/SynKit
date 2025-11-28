from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .core import CRNSpecies, CRNReaction, CRNNetwork
from .analysis import CRNAnalysisResult, CRNAnalyzer
from .io import _parse_side
from .thermo import CRNThermoProperties, compute_thermo_properties


@dataclass
class ReactionNetwork(CRNNetwork):
    """
    Convenience wrapper around :class:`CRNNetwork` with higher-level
    constructors and helper methods.

    This class does not add new state; it only provides user-friendly
    constructors such as :meth:`from_strings` and convenience methods
    such as :meth:`run_full_crnt_analysis`.

    Typical usage::

        from synkit.crn import ReactionNetwork

        rxns = [
            "A + E -> AE",
            "AE -> A + E",
            "AE -> B + E",
        ]
        net = ReactionNetwork.from_strings(rxns)
        result = net.run_full_crnt_analysis(kinetics="mass_action")
        print(result.summary)
    """

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_strings(cls, reactions: Sequence[str]) -> "ReactionNetwork":
        """
        Build a reaction network from simple reaction strings.

        Supported formats (examples)::

            "A -> B"
            "A + 2 B -> C"
            "A + E <-> AE"
            "2 X <-> 3 X"

        Reactant / product parsing is delegated to :func:`_parse_side`
        from :mod:`synkit.crn.io`, so any formats supported there will
        work here as well.

        :param reactions: Iterable of reaction strings.
        :type reactions: Sequence[str]
        :returns: Constructed reaction network.
        :rtype: ReactionNetwork
        :raises ValueError: If a reaction string cannot be parsed.
        """
        species_index: Dict[str, int] = {}
        species_list: List[CRNSpecies] = []
        rxn_objs: List[CRNReaction] = []

        for raw in reactions:
            s = raw.strip()
            if not s:
                continue

            reversible = False

            # Prefer explicit <-> if present
            if "<->" in s:
                left, right = s.split("<->", 1)
                reversible = True
            elif "->" in s:
                left, right = s.split("->", 1)
            elif ">>" in s:
                left, right = s.split(">>", 1)
            else:
                raise ValueError(
                    f"Cannot parse reaction string '{raw}': " "expected '->' or '<->'."
                )

            reactants = _parse_side(left, species_index, species_list)
            products = _parse_side(right, species_index, species_list)

            rxn_objs.append(
                CRNReaction(
                    reactants=reactants,
                    products=products,
                    reversible=reversible,
                    metadata={"source": "from_strings"},
                )
            )

        return cls(species=species_list, reactions=rxn_objs)

    # ------------------------------------------------------------------
    # Convenience analysis wrapper
    # ------------------------------------------------------------------
    def run_full_crnt_analysis(
        self,
        *,
        kinetics: str = "mass_action",
    ) -> CRNAnalysisResult:
        """
        Run the full ERNEST-style CRNT analysis on this network.

        This is a thin wrapper around :class:`CRNAnalyzer`::

            analyzer = CRNAnalyzer()
            return analyzer.analyze_model(self, kinetics=kinetics)

        :param kinetics: Kinetic assumption (``'mass_action'`` or
            ``'general'``). Some criteria only apply to mass-action.
        :type kinetics: str
        :returns: Aggregated analysis result.
        :rtype: CRNAnalysisResult
        """
        analyzer = CRNAnalyzer()
        return analyzer.analyze_model(self, kinetics=kinetics)

    def run_thermo_analysis(self) -> CRNThermoProperties:
        """
        Run thermodynamic / mass-conservation analysis on this network.

        Wraps :func:`compute_thermo_properties` from :mod:`synkit.crn.thermo`.
        """
        return compute_thermo_properties(self)
