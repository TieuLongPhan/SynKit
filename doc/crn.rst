.. _crn:

CRN
===

The ``synkit.CRN`` package provides SynKit’s **chemical reaction network** layer.
It covers the whole chain from chemistry or curated pathway data to a structural
verdict: construction, normalized representation through
:py:class:`~synkit.CRN.Structure.syncrn.SynCRN`, Chemical Reaction Network Theory
(complexes, linkage classes, deficiency), the Petri-net layer (minimal semiflows,
siphons, persistence), pathway reachability and realizability, symmetry-aware
comparison, and SBML interchange with the wider CRN ecosystem.

Key CRN submodules include:

- **Construct** — expand rule systems into reaction-network digraphs
- **Query** — retrieve and curate KEGG-derived reaction collections, and build
  a ``SynCRN`` straight from a KEGG module
- **Structure** — represent CRNs as normalized ``SynCRN`` objects
- **Props** — stoichiometry, exact conservation laws, conserved moieties, and
  CRNT (deficiency, linkage classes, weak reversibility, the Deficiency Zero
  and Deficiency One theorems)
- **Petrinet** — minimal semiflows, siphons, traps and structural persistence
- **Pathway** — reachability, path-finding, and realizability analysis
- **Symmetry** — canonicalization and isomorphism for CRN comparison
- **IO** — SBML import and export
- **Benchmark** — the shipped validation set, scaling benchmark and KEGG case
  study

Everything in the list above is importable from the top level::

    from synkit.CRN import SynCRN, crnt_summary, conserved_moieties, crn_to_sbml

.. raw:: html

   <style>
     .admonition.synkit-example-output { border-left-width: 6px; }
     .admonition.synkit-example-output .admonition-title { font-weight: 700; letter-spacing: 0.2px; }
     .admonition.synkit-example-output .admonition-title::before { content: "⟡ "; }
     .admonition .highlight pre { border-radius: 8px; }
     .sd-card { border-radius: 10px; }
     .sd-card-title { font-weight: 700; }
   </style>

.. grid:: 1 1 2 3
   :gutter: 2

   .. grid-item-card:: :octicon:`gear` Construct
      :class-card: sd-shadow-sm

      Expand rule sets into reaction-network digraphs and convert them into
      normalized ``SynCRN`` objects.

   .. grid-item-card:: :octicon:`database` Query
      :class-card: sd-shadow-sm

      Retrieve KEGG pathway content, curate incomplete entries, and prepare
      pathway-derived CRN inputs.

   .. grid-item-card:: :octicon:`mirror` Symmetry
      :class-card: sd-shadow-sm

      Canonicalize and compare CRNs through isomorphism and symmetry-aware
      structural analysis.

Construct
---------

The ``synkit.CRN.Construct`` package expands reaction rules into a reaction-network
digraph, which can then be converted into a normalized
:py:class:`~synkit.CRN.Structure.syncrn.SynCRN` object.

This layer generates a CRN from:

- a seed pool of starting molecules
- a rule set
- repeated rule applications
- frontier-based exploration with deduplication

.. code-block:: python
   :caption: Expand a CRN from seeds and rules, then convert to SynCRN
   :linenos:

   from synkit.CRN.Construct.builder import CRNExpand
   from synkit.CRN.Structure.syncrn import SynCRN

   dg = CRNExpand(
       rules=RULES,
       repeats=repeats,
       explicit_h=False,
       implicit_temp=False,
       keep_aam=False,
       use_frontier=True,
       dedup_delta=True,
   )

   g = dg.build(seeds=SEEDS, parallel=False)
   syn = SynCRN.from_digraph(g)

   print(syn.n_species)
   print(syn.n_reactions)

.. admonition:: Example output
   :class: note synkit-example-output

   .. code-block:: text

      42
      67

Query
-----

The ``synkit.CRN.Query`` package provides KEGG-oriented retrieval and curation
utilities. It is the natural entry point for CRNs derived from a biological
pathway rather than from rule-based expansion.

Key helpers include:

- :py:class:`~synkit.CRN.Query.kegg_extract.KEGGExtractor`
- :py:class:`~synkit.CRN.Query.kegg_impute.KEGGImputer`
- :py:class:`~synkit.CRN.Query.kegg_api.KEGGClient`

Example: retrieve a KEGG pathway as structured JSON
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: Extract KEGG pathway content for downstream CRN work
   :linenos:

   from synkit.CRN.Query.kegg_extract import KEGGExtractor

   pathway_data = KEGGExtractor().build_pathway_json(
       "hsa00010",
       with_compounds=True,
       with_atom_maps=True,
       save_as="Data/KEGG/hsa00010_raw.json",
   )

   pathway_data

The query layer supports the following operations:

- retrieve pathway reaction entries
- keep compound metadata together with reactions
- atom-map pathway reactions when possible
- patch incomplete records before building a final CRN

Structure
---------

The ``synkit.CRN.Structure`` package provides the main normalized CRN representation
used throughout SynKit.

The core object is:

- :py:class:`~synkit.CRN.Structure.syncrn.SynCRN`

with supporting structural components:

- :py:class:`~synkit.CRN.Structure.species.Species`
- :py:class:`~synkit.CRN.Structure.reaction.Reaction`
- :py:class:`~synkit.CRN.Structure.rule.Rule`

A ``SynCRN`` object is the preferred representation for:

- stable species and reaction ordering
- table-like and graph-like interoperability
- conversion to digraph or Petri-net views
- downstream symmetry, property, and pathway analysis

.. code-block:: python
   :caption: Build and inspect a SynCRN object
   :linenos:

   from synkit.CRN.Structure.syncrn import SynCRN

   syn = SynCRN.from_reaction_strings([
       "A + B >> C",
       "C >> D",
   ])

   print(syn)
   print(syn.n_species)
   print(syn.n_reactions)

Symmetry
--------

The ``synkit.CRN.Symmetry`` package supports CRN **isomorphism** and **canonicalization**.
This layer compares networks independently of their original labeling.

Relevant modules include:

- :py:mod:`~synkit.CRN.Symmetry.isomorphism`
- :py:mod:`~synkit.CRN.Symmetry.canon`
- :py:mod:`~synkit.CRN.Symmetry.automorphism`
- :py:mod:`~synkit.CRN.Symmetry.wl_canon`

Typical use cases:

- test whether two CRNs are structurally equivalent
- compute deterministic canonical forms
- compare rule-expanded and pathway-curated networks
- perform symmetry-aware downstream analysis

.. code-block:: python
   :caption: Compare two CRNs by isomorphism and canonicalization
   :linenos:

   from synkit.CRN.Structure.syncrn import SynCRN
   from synkit.CRN.Symmetry.isomorphism import crn_isomorphic
   from synkit.CRN.Symmetry.canon import canonicalize_crn

   syn1 = SynCRN.from_reaction_strings([
       "A + B >> C",
       "C >> D",
   ])

   syn2 = SynCRN.from_reaction_strings([
       "X + Y >> Z",
       "Z >> W",
   ])

   print(crn_isomorphic(syn1, syn2))

   can1 = canonicalize_crn(syn1)
   can2 = canonicalize_crn(syn2)

   print(can1 == can2)

.. admonition:: Example output
   :class: note synkit-example-output

   .. code-block:: text

      True
      True

Props
-----

The ``synkit.CRN.Props`` package provides quantitative summaries and analysis layers
on top of a normalized CRN.

Important modules include:

- :py:mod:`~synkit.CRN.Props.stoich`
- :py:mod:`~synkit.CRN.Props.thermo`
- :py:mod:`~synkit.CRN.Props.dynamics`
- :py:mod:`~synkit.CRN.Props.helper`

This layer is useful for computing:

- stoichiometric summaries and matrices
- thermodynamic annotations or aggregate summaries
- symbolic dynamical quantities such as Jacobian structure
- helper statistics over species and reactions

Example: stoichiometric analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: Compute stoichiometric summaries from a SynCRN
   :linenos:

   from synkit.CRN.Structure.syncrn import SynCRN
   from synkit.CRN.Props.stoich import summary

   syn = SynCRN.from_reaction_strings([
       "A + B >> C",
       "C >> D",
   ])

   stoich = summary(syn)
   print(stoich)

Example: thermodynamic summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: Compute thermodynamic summary information
   :linenos:

   from synkit.CRN.Props.thermo import compute_thermo_summary

   thermo = compute_thermo_summary(syn)
   print(thermo)

Example: Jacobian or dynamical structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: Compute symbolic dynamical structure
   :linenos:

   from synkit.CRN.Props.dynamics import symbolic_jacobian

   species_order, rule_order, J = symbolic_jacobian(syn)
   print(J)

Pathway
-------

The ``synkit.CRN.Pathway`` package provides pathway-level reasoning on top of a CRN.
This layer distinguishes graph connectivity from reachability and realizability
under a specified starting pool.

Relevant modules include:

- :py:mod:`~synkit.CRN.Pathway.reachability`
- :py:mod:`~synkit.CRN.Pathway.realizability`
- :py:mod:`~synkit.CRN.Pathway.pathfinder`

Typical questions addressed here are:

- which species are reachable from a given seed set?
- can a target route actually be realized under mass-conserving progression?
- which subset of reactions supports a target product?

Example: reachability
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: Reachable species from a starting set
   :linenos:

   from synkit.CRN.Structure.syncrn import SynCRN
   from synkit.CRN.Pathway.reachability import PathwayReachability

   syn = SynCRN.from_reaction_strings([
       "A + B >> C",
       "C >> D",
       "E >> F",
   ])

   result = PathwayReachability().load_syncrn(syn).compute_layers_set(["A", "B"])
   print(sorted(result.species_first_depth))

Example: realizability
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: Test whether a target can be realized from an initial pool
   :linenos:

   from synkit.CRN.Pathway.realizability import PathwayRealizability

   checker = PathwayRealizability().load_hypergraph_and_flow(
       vertices=["A", "B", "C", "D"],
       edges={
           "r1": ({"A": 1, "B": 1}, {"C": 1}),
           "r2": ({"C": 1}, {"D": 1}),
       },
       flow={"r1": 1, "r2": 1},
       initial_marking={"A": 1, "B": 1},
   )
   checker.build_petri_net_from_flow()
   ok, certificate = checker.is_realizable()
   print(ok)

.. admonition:: Example output
   :class: note synkit-example-output

   .. code-block:: text

      True

Chemical Reaction Network Theory
--------------------------------

:py:mod:`synkit.CRN.Props.deficiency` supplies the structural quantities CRNT is
built on, and the two classical theorems that turn them into dynamical verdicts.
Writing ``n`` for the number of complexes, ``l`` for the number of linkage
classes and ``s`` for the rank of the stoichiometric matrix, the **deficiency**
is ``delta = n - l - s``. All ranks are computed exactly over rationals, so an
integer network never gets a floating-point deficiency.

Example: deficiency and the Deficiency Zero Theorem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: Structural verdicts for a reversible enzyme mechanism
   :linenos:

   from synkit.CRN import SynCRN, crnt_summary, deficiency_zero_verdict

   syn = SynCRN.from_reaction_strings([
       "E + S >> ES", "ES >> E + S", "ES >> E + P", "E + P >> ES",
   ])

   print(crnt_summary(syn))
   print(deficiency_zero_verdict(syn)["conclusion"])

.. admonition:: Example output
   :class: note synkit-example-output

   .. code-block:: text

      unique_stable_equilibrium

Example: conservation laws and conserved moieties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``integer_conservation_laws`` returns an exact integer basis of ``ker(S^T)``;
``conserved_moieties`` returns the non-negative, inclusion-minimal laws — the
ones with a chemical reading such as "total enzyme".

.. code-block:: python
   :caption: Moiety pools of a phosphorylation cycle
   :linenos:

   from synkit.CRN import SynCRN, conserved_moieties

   syn = SynCRN.from_reaction_strings([
       "S0 + E >> S0E", "S0E >> S0 + E", "S0E >> S1 + E",
       "S1 + F >> S1F", "S1F >> S1 + F", "S1F >> S0 + F",
   ])
   print(conserved_moieties(syn))

Petri net
---------

The ``synkit.CRN.Petrinet`` package gives the Petri-net view of a network:
minimal **non-negative integer** semiflows (Farkas / Colom-Silva), minimal
siphons and traps found by a closure operator with branch-and-bound rather than
subset enumeration, and structural persistence via the
Angeli-De Leenheer-Sontag condition.

.. code-block:: python
   :caption: Siphons and structural persistence
   :linenos:

   from synkit.CRN import SynCRN, find_siphons, siphon_persistence_details

   syn = SynCRN.from_reaction_strings(["A >> B", "B >> A", "C >> D", "D >> C"])
   print(find_siphons(syn))
   print(siphon_persistence_details(syn).persistence_ok)

.. admonition:: Example output
   :class: note synkit-example-output

   .. code-block:: text

      True

SBML interchange
----------------

:py:mod:`synkit.CRN.IO.sbml` reads and writes SBML Level 3 Version 2, so a
network built here can be analysed by ``crnpy``, CRNT4SBML, CoNtRol or COPASI,
and a published SBML model can be analysed here. The adapter is self-contained
and needs no ``libsbml`` installation.

.. code-block:: python
   :caption: Round-trip a network through SBML
   :linenos:

   from synkit.CRN import SynCRN, crn_from_sbml, crn_to_sbml

   syn = SynCRN.from_reaction_strings(["2A >> B", "B >> 2A"])
   back = crn_from_sbml(crn_to_sbml(syn))
   print(back.to_equations(species="label", include_id=False))

Benchmark, validation and case study
------------------------------------

``synkit.CRN.Benchmark`` ships the material behind the package's published
results, all reproducible offline:

- a **validation set** of literature and regression networks, where every
  quantity is additionally recomputed by an independent route (deficiency via a
  rank identity that never counts linkage classes, siphons via brute force,
  semiflows via their defining equations);
- a **scaling benchmark** over four network families;
- a **KEGG case study** over four cached metabolic modules.

.. code-block:: python
   :caption: Reproduce the validation table and the case study
   :linenos:

   from synkit.CRN.Benchmark import (
       analyze_kegg_module, run_validation, validation_table,
   )

   print(validation_table(run_validation()))
   print(analyze_kegg_module("M00001"))

The same artifacts are produced from the command line by::

    python Experiment/CRN/run_all.py

Recommended workflow
--------------------

A practical CRN workflow in SynKit is:

1. **Construct** a CRN from rules, or **Query** a curated pathway source such as KEGG
2. Convert the result into :py:class:`~synkit.CRN.Structure.syncrn.SynCRN`
3. Use **Props** to inspect stoichiometry, conservation laws and CRNT verdicts
4. Use **Petrinet** for semiflows, siphons and structural persistence
5. Use **Pathway** to analyze reachability and realizability
6. Use **Symmetry** for canonicalization or structural comparison
7. Use **IO** to exchange the network with other CRN tools through SBML

See Also
--------

- :doc:`synthesis` — rule application and multi-step generation workflows
- :doc:`graph` — graph matching, ITS, MTG, and canonicalization utilities
- :doc:`chem` — chemical validation, standardization, and balancing
- :doc:`io` — conversion between graph and chemical representations
