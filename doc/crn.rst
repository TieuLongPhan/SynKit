.. _crn:

CRN
===

The ``synkit.CRN`` package provides SynKit’s **chemical reaction network** layer.
It supports CRN construction from rules or curated pathway data, normalized network
representation through :py:class:`~synkit.CRN.Structure.syncrn.SynCRN`, symmetry-aware
comparison, stoichiometric and thermodynamic summaries, and pathway-level analysis such
as reachability and realizability.

Key CRN submodules include:

- **Construct** — expand rule systems into reaction-network digraphs
- **Query** — retrieve and curate KEGG-derived reaction collections
- **Structure** — represent CRNs as normalized ``SynCRN`` objects
- **Symmetry** — canonicalization and isomorphism for CRN comparison
- **Props** — stoichiometric, thermodynamic, and dynamical summaries
- **Pathway** — reachability, path-finding, and realizability analysis

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

This layer is useful when you want to generate a CRN from:

- a seed pool of starting molecules
- a rule set
- repeated rule applications
- frontier-based exploration with deduplication

.. code-block:: python
   :caption: Expand a CRN from seeds and rules, then convert to SynCRN
   :linenos:

   from synkit.CRN.Construct.DAG.crn import CRNExpand
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
utilities. It is the natural entry point when your CRN comes from a biological
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

This query layer is especially useful when you want to:

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

A ``SynCRN`` object is the preferred representation when you want:

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
This is the right layer when you want to compare two networks independently of their
original labeling.

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
   from synkit.CRN.Props.stoich import compute_stoich_summary

   syn = SynCRN.from_reaction_strings([
       "A + B >> C",
       "C >> D",
   ])

   stoich = compute_stoich_summary(syn)
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

   from synkit.CRN.Props.dynamics import compute_jacobian_structure

   J = compute_jacobian_structure(syn)
   print(J)

.. note::
   Exact function names may vary slightly across versions, but the ``Props`` package
   is the correct place for stoichiometric, thermodynamic, and Jacobian-style CRN
   analysis.

Pathway
-------

The ``synkit.CRN.Pathway`` package provides pathway-level reasoning on top of a CRN.
This is the right layer when you want to analyze whether a target is merely connected
in the graph or actually reachable and realizable from a specified starting pool.

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
   from synkit.CRN.Pathway.reachability import reachable_species

   syn = SynCRN.from_reaction_strings([
       "A + B >> C",
       "C >> D",
       "E >> F",
   ])

   reached = reachable_species(syn, seeds={"A", "B"})
   print(reached)

Example: realizability
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: Test whether a target can be realized from an initial pool
   :linenos:

   from synkit.CRN.Pathway.realizability import is_realizable

   ok = is_realizable(syn, seeds={"A", "B"}, targets={"D"})
   print(ok)

.. admonition:: Example output
   :class: note synkit-example-output

   .. code-block:: text

      True

Recommended workflow
--------------------

A practical CRN workflow in SynKit is:

1. **Construct** a CRN from rules, or **Query** a curated pathway source such as KEGG
2. Convert the result into :py:class:`~synkit.CRN.Structure.syncrn.SynCRN`
3. Use **Props** to inspect stoichiometric, thermodynamic, or dynamical features
4. Use **Pathway** to analyze reachability and realizability
5. Use **Symmetry** when you need canonicalization or structural comparison

See Also
--------

- :doc:`synthesis` — rule application and multi-step generation workflows
- :doc:`graph` — graph matching, ITS, MTG, and canonicalization utilities
- :doc:`chem` — chemical validation, standardization, and balancing
- :doc:`io` — conversion between graph and chemical representations