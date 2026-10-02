"""A curated benchmark set of reaction networks with expected verdicts.

Each entry pairs a network with the structural quantities it must produce.
The expectations are stated as the four CRNT numbers ``n`` (complexes), ``l``
(linkage classes), ``s`` (rank) and ``delta`` (deficiency), plus weak
reversibility, the number of independent conservation laws, and — where it
applies — the structural persistence verdict. Because ``delta = n - l - s`` by
definition, every entry carries its own consistency check: a reader can verify
the deficiency from the other three numbers without trusting the
implementation.

The set covers the network shapes that break naive implementations:

- disconnected networks, where linkage-class counting must not merge blocks;
- networks with non-unit stoichiometry, where integer arithmetic matters;
- irreversible networks, where weak reversibility must come out false;
- networks with a conservation law, where the rank is deficient;
- metabolic motifs (Michaelis-Menten, futile cycle, two-site phosphorylation)
  that a bioinformatics reader will recognise.

Use :mod:`synkit.CRN.Benchmark.validate` to run the set and
:mod:`synkit.CRN.Benchmark.crosschecks` for the independent recomputation of
each quantity.

.. rubric:: Example

.. code-block:: python

    from synkit.CRN.Benchmark import BENCHMARK_NETWORKS, get_network

    print(len(BENCHMARK_NETWORKS))
    entry = get_network("michaelis_menten")
    print(entry.build().to_equations(species="label", include_id=False))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..Structure.syncrn import SynCRN

__all__ = [
    "BENCHMARK_NETWORKS",
    "BenchmarkNetwork",
    "get_network",
    "network_names",
]


@dataclass(frozen=True)
class BenchmarkNetwork:
    """One benchmark network and the verdicts it must reproduce.

    :param name:
        Stable identifier, used to select the entry.
    :type name: str

    :param description:
        What the network is and why it is in the set.
    :type description: str

    :param reactions:
        Reaction strings in :meth:`SynCRN.from_reaction_strings` syntax.
    :type reactions: Tuple[str, ...]

    :param n_complexes:
        Expected number of distinct complexes.
    :type n_complexes: int

    :param n_linkage_classes:
        Expected number of linkage classes.
    :type n_linkage_classes: int

    :param rank:
        Expected rank of the stoichiometric matrix.
    :type rank: int

    :param deficiency:
        Expected deficiency; must equal ``n_complexes - n_linkage_classes - rank``.
    :type deficiency: int

    :param weakly_reversible:
        Expected weak reversibility.
    :type weakly_reversible: bool

    :param n_conservation_laws:
        Expected number of independent conservation laws, i.e.
        ``n_species - rank``.
    :type n_conservation_laws: int

    :param persistent:
        Expected structural persistence verdict, or ``None`` when the test is
        not informative for this network.
    :type persistent: Optional[bool]

    :param source:
        Where the network comes from.
    :type source: str

    :param tags:
        Free-form labels used to select subsets.
    :type tags: Tuple[str, ...]

    :param strict:
        Whether the network is built in strict mode. Open systems with an empty
        reaction side need ``False``.
    :type strict: bool
    """

    name: str
    description: str
    reactions: Tuple[str, ...]
    n_complexes: int
    n_linkage_classes: int
    rank: int
    deficiency: int
    weakly_reversible: bool
    n_conservation_laws: int
    persistent: Optional[bool] = None
    source: str = ""
    tags: Tuple[str, ...] = field(default_factory=tuple)
    strict: bool = True

    def build(self) -> SynCRN:
        """Construct the network object.

        :return:
            The benchmark network as a :class:`SynCRN`.
        :rtype: SynCRN
        """
        return SynCRN.from_reaction_strings(
            list(self.reactions), strict=self.strict
        )

    def expected(self) -> Dict[str, object]:
        """Return the expected verdicts as a mapping.

        :return:
            Expected values keyed by quantity name.
        :rtype: Dict[str, object]
        """
        return {
            "n_complexes": self.n_complexes,
            "n_linkage_classes": self.n_linkage_classes,
            "rank": self.rank,
            "deficiency": self.deficiency,
            "weakly_reversible": self.weakly_reversible,
            "n_conservation_laws": self.n_conservation_laws,
            "persistent": self.persistent,
        }


def _reversible(*pairs: Tuple[str, str]) -> Tuple[str, ...]:
    """Expand ``(lhs, rhs)`` pairs into forward and backward reaction strings.

    :param pairs:
        Reaction sides as ``(lhs, rhs)`` string pairs.
    :type pairs: Tuple[str, str]

    :return:
        Forward and backward reaction strings, in order.
    :rtype: Tuple[str, ...]
    """
    out: List[str] = []
    for lhs, rhs in pairs:
        out.append(f"{lhs}>>{rhs}")
        out.append(f"{rhs}>>{lhs}")
    return tuple(out)


#: The benchmark set.
BENCHMARK_NETWORKS: Tuple[BenchmarkNetwork, ...] = (
    BenchmarkNetwork(
        name="reversible_isomerization",
        description=(
            "A <-> B. The smallest weakly reversible network; deficiency zero, "
            "so the Deficiency Zero Theorem gives a unique stable equilibrium."
        ),
        reactions=("A>>B", "B>>A"),
        n_complexes=2,
        n_linkage_classes=1,
        rank=1,
        deficiency=0,
        weakly_reversible=True,
        n_conservation_laws=1,
        persistent=True,
        source="Feinberg, Foundations of Chemical Reaction Network Theory (2019), Ch. 3",
        tags=("textbook", "deficiency-zero"),
    ),
    BenchmarkNetwork(
        name="irreversible_isomerization",
        description=(
            "A -> B. Deficiency zero but not weakly reversible, so no positive "
            "steady state exists for any rate constants. Not persistent: A is "
            "driven to zero."
        ),
        reactions=("A>>B",),
        n_complexes=2,
        n_linkage_classes=1,
        rank=1,
        deficiency=0,
        weakly_reversible=False,
        n_conservation_laws=1,
        persistent=False,
        source="Feinberg, Foundations of Chemical Reaction Network Theory (2019), Ch. 3",
        tags=("textbook", "deficiency-zero"),
    ),
    BenchmarkNetwork(
        name="isomerization_cycle",
        description=(
            "A -> B -> C -> A. Weakly reversible without any single reaction "
            "being reversible; separates weak reversibility from reversibility."
        ),
        reactions=("A>>B", "B>>C", "C>>A"),
        n_complexes=3,
        n_linkage_classes=1,
        rank=2,
        deficiency=0,
        weakly_reversible=True,
        n_conservation_laws=1,
        persistent=True,
        source="Feinberg, Foundations of Chemical Reaction Network Theory (2019), Ch. 3",
        tags=("textbook", "weak-reversibility"),
    ),
    BenchmarkNetwork(
        name="disjoint_reversible_blocks",
        description=(
            "Three independent reversible pairs. Regression for linkage-class "
            "counting and for minimal-semiflow supports, which must come out as "
            "three disjoint pairs rather than their union."
        ),
        reactions=("A>>B", "B>>A", "C>>D", "D>>C", "E>>F", "F>>E"),
        n_complexes=6,
        n_linkage_classes=3,
        rank=3,
        deficiency=0,
        weakly_reversible=True,
        n_conservation_laws=3,
        persistent=True,
        source="constructed (regression for finding F1)",
        tags=("regression", "disconnected"),
    ),
    BenchmarkNetwork(
        name="disjoint_blocks_with_stoichiometry",
        description=(
            "Disjoint reversible blocks where one uses non-unit stoichiometry "
            "(E <-> 2F). The conservation law on that block is 2E + F, which a "
            "float kernel basis reports incorrectly."
        ),
        reactions=("A>>B", "B>>A", "C>>D", "D>>C", "E>>2F", "2F>>E"),
        n_complexes=6,
        n_linkage_classes=3,
        rank=3,
        deficiency=0,
        weakly_reversible=True,
        n_conservation_laws=3,
        persistent=True,
        source="constructed (regression for finding F1)",
        tags=("regression", "stoichiometry"),
    ),
    BenchmarkNetwork(
        name="michaelis_menten",
        description=(
            "E + S <-> ES -> E + P. The canonical enzyme mechanism. Deficiency "
            "zero and not weakly reversible, so no positive steady state exists; "
            "two conservation laws (total enzyme and total substrate moiety)."
        ),
        reactions=("E+S>>ES", "ES>>E+S", "ES>>E+P"),
        n_complexes=3,
        n_linkage_classes=1,
        rank=2,
        deficiency=0,
        weakly_reversible=False,
        n_conservation_laws=2,
        persistent=False,
        source="Michaelis & Menten (1913); Feinberg (2019), Ch. 3",
        tags=("biology", "enzyme", "deficiency-zero"),
    ),
    BenchmarkNetwork(
        name="reversible_michaelis_menten",
        description=(
            "E + S <-> ES <-> E + P. Making product release reversible makes the "
            "mechanism weakly reversible at unchanged deficiency, flipping the "
            "Deficiency Zero verdict to a unique stable equilibrium."
        ),
        reactions=("E+S>>ES", "ES>>E+S", "ES>>E+P", "E+P>>ES"),
        n_complexes=3,
        n_linkage_classes=1,
        rank=2,
        deficiency=0,
        weakly_reversible=True,
        n_conservation_laws=2,
        persistent=True,
        source="Feinberg (2019), Ch. 3",
        tags=("biology", "enzyme", "deficiency-zero"),
    ),
    BenchmarkNetwork(
        name="edelstein",
        description=(
            "A <-> 2A, A + B <-> C <-> B. The standard bistable motif and the "
            "smallest familiar network of deficiency one: reversible, hence "
            "weakly reversible, yet outside the Deficiency Zero Theorem."
        ),
        reactions=_reversible(("A", "2A"), ("A+B", "C"), ("C", "B")),
        n_complexes=5,
        n_linkage_classes=2,
        rank=2,
        deficiency=1,
        weakly_reversible=True,
        n_conservation_laws=1,
        persistent=True,
        source="Edelstein, J. Theor. Biol. 29:57 (1970)",
        tags=("textbook", "deficiency-one", "bistable"),
    ),
    BenchmarkNetwork(
        name="futile_cycle_1_site",
        description=(
            "One-site phosphorylation-dephosphorylation cycle with distinct "
            "kinase and phosphatase. The core motif of signal transduction; "
            "three conservation laws (kinase, phosphatase, substrate pools)."
        ),
        reactions=(
            "S0+E>>S0E",
            "S0E>>S0+E",
            "S0E>>S1+E",
            "S1+F>>S1F",
            "S1F>>S1+F",
            "S1F>>S0+F",
        ),
        n_complexes=6,
        n_linkage_classes=2,
        rank=3,
        deficiency=1,
        weakly_reversible=False,
        n_conservation_laws=3,
        persistent=True,
        source="Wang & Sontag, J. Math. Biol. 57:29 (2008)",
        tags=("biology", "signalling", "deficiency-one"),
    ),
    BenchmarkNetwork(
        name="futile_cycle_2_site",
        description=(
            "Distributive two-site phosphorylation, the standard multistationarity "
            "example in signal transduction. Larger than the one-site cycle and a "
            "useful size check on complex extraction."
        ),
        reactions=(
            "S0+E>>S0E",
            "S0E>>S0+E",
            "S0E>>S1+E",
            "S1+E>>S1E",
            "S1E>>S1+E",
            "S1E>>S2+E",
            "S2+F>>S2F",
            "S2F>>S2+F",
            "S2F>>S1+F",
            "S1+F>>S1F2",
            "S1F2>>S1+F",
            "S1F2>>S0+F",
        ),
        n_complexes=10,
        n_linkage_classes=2,
        rank=6,
        deficiency=2,
        weakly_reversible=False,
        n_conservation_laws=3,
        persistent=True,
        source="Wang & Sontag, J. Math. Biol. 57:29 (2008)",
        tags=("biology", "signalling", "multistationarity"),
    ),
    BenchmarkNetwork(
        name="shinar_feinberg_acr",
        description=(
            "A + B -> 2B, B -> A. The minimal absolute-concentration-robustness "
            "motif: two complexes in the same linkage class differ in exactly one "
            "species, which is what makes B's steady-state concentration "
            "independent of total mass."
        ),
        reactions=("A+B>>2B", "B>>A"),
        n_complexes=4,
        n_linkage_classes=2,
        rank=1,
        deficiency=1,
        weakly_reversible=False,
        n_conservation_laws=1,
        persistent=False,
        source="Shinar & Feinberg, Science 327:1389 (2010)",
        tags=("biology", "robustness", "deficiency-one"),
    ),
    BenchmarkNetwork(
        name="open_inflow_outflow",
        description=(
            "An open reactor: 0 -> A -> B -> 0. Exercises the zero complex, which "
            "must be treated as a complex like any other. Counting it is what "
            "makes the complex graph a cycle 0 -> A -> B -> 0, so the network is "
            "weakly reversible; dropping it would give the wrong verdict. There "
            "is no conservation law, and no siphon at all, so persistence holds "
            "vacuously."
        ),
        reactions=(">>A", "A>>B", "B>>"),
        n_complexes=3,
        n_linkage_classes=1,
        rank=2,
        deficiency=0,
        weakly_reversible=True,
        n_conservation_laws=0,
        persistent=True,
        source="constructed (zero-complex handling)",
        tags=("regression", "open-system"),
        strict=False,
    ),
)


def network_names() -> List[str]:
    """Return the names of every benchmark network.

    :return:
        Benchmark network names in registry order.
    :rtype: List[str]

    .. rubric:: Example

    .. code-block:: python

        print(network_names()[:3])
    """
    return [entry.name for entry in BENCHMARK_NETWORKS]


def get_network(name: str) -> BenchmarkNetwork:
    """Look up one benchmark network by name.

    :param name:
        Benchmark network name.
    :type name: str

    :return:
        The matching entry.
    :rtype: BenchmarkNetwork

    :raises KeyError:
        If no benchmark network carries that name.

    .. rubric:: Example

    .. code-block:: python

        entry = get_network("michaelis_menten")
        print(entry.deficiency)
        # 0
    """
    for entry in BENCHMARK_NETWORKS:
        if entry.name == name:
            return entry
    raise KeyError(
        f"Unknown benchmark network {name!r}. Available: {', '.join(network_names())}"
    )
