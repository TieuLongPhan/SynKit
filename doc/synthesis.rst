.. _synkit-synthesis:

Synthesis
=========

The ``synkit.Synthesis`` package provides a unified interface for **reaction prediction**
and **chemical reaction network (CRN) exploration**. It applies rule-based graph rewriting
to molecular structures, allowing you to enumerate candidate products (forward mode) or
candidate precursors (backward mode) from reaction templates.

.. raw:: html

   <style>
     /* Optional: makes "Example output" boxes look a bit more like callouts */
     .synkit-admonition-title {
       font-weight: 700;
       letter-spacing: 0.2px;
     }
     /* Slightly soften code blocks inside admonitions */
     .admonition .highlight pre {
       border-radius: 8px;
     }
   </style>

Reaction Prediction: Reactor
----------------------------

The ``synkit.Synthesis.Reactor`` submodule applies a reaction **template** (SMARTS / rule)
to an input **substrate** (SMILES) and enumerates all valid transformations under a chosen
graph-matching strategy.

Reaction rewriting uses the native
:py:class:`~synkit.Synthesis.Reactor.syn_reactor.SynReactor`, with NetworkX
graphs and direct integration with SynKit Lewis-state and stereo models.

Reactor parameters
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 10 10 65

   * - **Name**
     - **Type**
     - **Default**
     - **Description**
   * - ``invert``
     - bool
     - ``False``
     - Direction of application.
       Use ``False`` for **forward** prediction (substrate → products) and
       ``True`` for **backward** prediction (target → precursors).
   * - ``explicit_h``
     - bool
     - ``False``
     - When ``True``, hydrogens in the **reaction center** are rendered explicitly
       in the output SMARTS. This is useful for debugging, auditing rule scope, and
       disambiguating closely related matches.
   * - ``strategy``
     - str
     - ``'bt'``
     - Graph-matching strategy used to enumerate transformations:

       - ``'comp'``: component-aware matching (fastest; recommended for multi-component SMILES)
       - ``'all'``: exhaustive arbitrary subgraph search (most expensive)
       - ``'bt'``: fallback strategy (tries ``comp`` first, then ``all`` if no match is found)
   * - ``template_format``
     - str
     - ``'typesGH'``
     - ITS representation used when the template is a reaction string.
       Use ``'tuple'`` for the Lewis-labelled graph representation.
   * - ``electron_diagnostics``
     - bool
     - ``False``
     - When ``True``, keep Lewis-state accounting diagnostics on generated ITS
       objects. This is useful when inspecting charge, lone-pair, or radical
       recomputation. The option name remains ``electron_diagnostics`` for API
       compatibility.
   * - ``automorphism``
     - bool
     - ``True``
     - Deduplicate symmetry-equivalent matches before rewriting.
   * - ``dedup_its``
     - bool
     - ``True``
     - Consolidate equivalent post-rewrite ITS graphs. Set this to ``False``
       to retain deterministic raw mapping and stereo-branch multiplicity.
       Electron finalization and stereo validation still run. This option is
       available only on ``SynReactor``; ``BatchReactor`` always uses the
       default consolidated behavior.

Raw ITS applications and multiplicity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``automorphism`` and ``dedup_its`` control different stages. The first prunes
equivalent mappings before rewriting; the second consolidates equivalent ITS
graphs afterward. Disable both to inspect every raw application:

.. code-block:: python

   reactor = SynReactor(
       substrate="CC",
       template="[CH2:1]([H:3])[CH2:2]([H:4])>>[CH2:1]=[CH2:2].[H:3][H:4]",
       template_format="tuple",
       explicit_h=False,
       automorphism=False,
       dedup_its=False,
   )

   for its in reactor.its_list:
       print(its.graph["application_provenance"])

In raw mode, ``its_list``, ``smarts_list``, and optional diagnostics remain
aligned one-to-one. Raw multiplicity counts graph applications and stereo
branches; it is not a kinetic weight or predicted product distribution.

Example: Forward Prediction (NetworkX)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: Forward prediction with explicit H and backtracking strategy
   :linenos:

   from synkit.Synthesis.Reactor.syn_reactor import SynReactor

   input_fw = 'CC=O.CC=O'
   template = '[C:2]=[O:3].[C:4]([H:7])[H:8]>>[C:2]=[C:4].[O:3]([H:7])[H:8]'

   reactor = SynReactor(
       substrate=input_fw,
       template=template,
       invert=False,
       explicit_h=True,
       strategy='bt'
   )

   smarts_list = reactor.smarts_list
   print(smarts_list)

.. admonition:: Example output
   :class: note synkit-example-output

   .. code-block:: text

      [
        '[CH3:1][CH:2]=[O:3].[CH:4]([CH:5]=[O:6])([H:7])[H:8]>>[CH3:1][CH:2]=[CH:4][CH:5]=[O:6].[O:3]([H:7])[H:8]',
        '[CH3:4][CH:5]=[O:6].[CH:1]([CH:2]=[O:3])([H:7])[H:8]>>[CH:1]([CH:2]=[O:3])=[CH:5][CH3:4].[O:6]([H:7])[H:8]'
      ]

Example: Backward Prediction (NetworkX)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: Backward prediction targeting product to precursors
   :linenos:

   from synkit.Synthesis.Reactor.syn_reactor import SynReactor

   target = 'CC=CC=O.O'
   template = '[C:2]=[O:3].[C:4]([H:7])[H:8]>>[C:2]=[C:4].[O:3]([H:7])[H:8]'

   reactor_bw = SynReactor(
       substrate=target,
       template=template,
       invert=True,
       explicit_h=False,
       strategy='comp'
   )

   precursors = reactor_bw.smarts_list
   print(precursors)

.. admonition:: Example output
   :class: note synkit-example-output

   .. code-block:: text

      [
        '[CH3:1][CH:2]=[O:6].[CH3:3][CH:4]=[O:5]>>[CH3:1][CH:2]=[CH:3][CH:4]=[O:5].[OH2:6]',
        '[CH3:1][CH3:2].[CH:3]([CH:4]=[O:5])=[O:6]>>[CH3:1][CH:2]=[CH:3][CH:4]=[O:5].[OH2:6]'
      ]

Example: Implicit-H Template (NetworkX)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your template is written in an **implicit-H** form, enable it via ``implicit_temp=True``
while keeping ``explicit_h=False``.

.. code-block:: python
   :caption: Backward prediction with an implicit-H template
   :linenos:

   from synkit.Synthesis.Reactor.syn_reactor import SynReactor

   target = 'CC=CC=O.O'
   template = '[C:2]=[O:3].[CH2:4]>>[C:2]=[C:4].[OH2:3]'

   reactor_imp = SynReactor(
       substrate=target,
       template=template,
       invert=True,
       explicit_h=False,
       strategy='comp',
       implicit_temp=True
   )

   precursors = reactor_imp.smarts_list
   print(precursors)

.. admonition:: Example output
   :class: note synkit-example-output

   .. code-block:: text

      [
        '[CH3:1][CH:2]=[O:6].[CH3:3][CH:4]=[O:5]>>[CH3:1][CH:2]=[CH:3][CH:4]=[O:5].[OH2:6]',
        '[CH3:1][CH3:2].[CH:3]([CH:4]=[O:5])=[O:6]>>[CH3:1][CH:2]=[CH:3][CH:4]=[O:5].[OH2:6]'
      ]

Lewis-labelled graph templates
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The NetworkX reactor can consume Lewis-labelled graph (LLG) templates. This is
the SynKit-native path for transformations where valence-state information
matters: lone pairs, radicals, valence electrons, and sigma/pi bond components
are stored in the template and used during matching/rewrite. In the current API
LLG construction is requested with ``format="tuple"``.

There are two common entry points:

.. code-block:: python
   :caption: Build the LLG template explicitly
   :linenos:

   from synkit.IO import rsmi_to_its
   from synkit.Synthesis.Reactor.syn_reactor import SynReactor

   smart = "[NH3:1].[CH3:2][Cl:3]>>[NH3+:1][CH3:2].[Cl-:3]"
   substrate = "CCl.N"
   template = rsmi_to_its(smart, core=False, format="tuple")

   reactor = SynReactor(
       substrate=substrate,
       template=template,
       implicit_temp=True,
       explicit_h=False,
       electron_diagnostics=True,
   )

   print(reactor.smarts)

.. code-block:: python
   :caption: Let SynReactor build an LLG template from a reaction string
   :linenos:

   reactor = SynReactor(
       substrate="CCl.N",
       template="[NH3:1].[CH3:2][Cl:3]>>[NH3+:1][CH3:2].[Cl-:3]",
       template_format="tuple",
       implicit_temp=True,
       explicit_h=False,
       electron_diagnostics=True,
   )

LLG rewrite policy:

.. list-table::
   :header-rows: 1

   * - Concept
     - Policy
   * - Bond truth
     - ``sigma_order`` and ``pi_order`` are authoritative in new mode.
   * - Product reconstruction
     - ``kekule_order`` is computed from ``sigma_order + pi_order`` before
       conversion through RDKit.
   * - Charge
     - Charge is recomputed from valence electrons, lone pairs, hydrogen count,
       radical count, and Kekule bond-order sum.
   * - Aromaticity
     - Aromatic flags are still useful for matching and display, but aromatic
       ``order=1.5`` is not used as the LLG-authoritative rewrite value.

Radical-based linking
---------------------

``RBLEngine`` links forward and backward template applications through a
wildcard-aware reaction-centre overlap. It is useful when a direct reactor
application is insufficient and the two sides need to be fused through a
shared core.

Choose the execution mode according to the required recall and cost:

- ``"fast_track"`` performs only a cheap reactor round-trip.
- ``"early_stop"`` (the default) also constructs ITS candidates but stops
  before maximum-common-subgraph (MCS) fusion.
- ``"full"`` performs wildcard-aware MCS fusion and returns all collected
  unique candidates; it is the most expensive mode.

.. code-block:: python
   :caption: Run the RBL engine with its default exact MCS matcher

   from synkit.Synthesis.Reactor.rbl_engine import RBLEngine

   engine = RBLEngine(mode="early_stop")
   result = engine.process(reaction_rsmi, template)
   candidates = result.fused_rsmis

Use ``mode="full"`` only when the early path does not provide enough
candidates. ``matcher_cls`` accepts ``ApproxMCSMatcher`` for a faster,
heuristic alternative on large or highly symmetric ITS graphs.

See Also
--------

- :mod:`synkit.IO` — format conversion utilities (SMILES/SMARTS/GML and related helpers)
- :mod:`synkit.Graph` — graph data structures, matching, and transformations
