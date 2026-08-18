.. _synkit-synthesis:

Synthesis
=========

The ``synkit.Synthesis`` package provides a unified interface for **reaction prediction**
and **chemical reaction network (CRN) exploration**. It applies rule-based graph rewriting
to molecular structures and enumerates candidate products (forward mode) or
candidate precursors (backward mode) from reaction templates.

.. raw:: html

   <style>
     .synkit-admonition-title {
       font-weight: 700;
       letter-spacing: 0.2px;
     }
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
:py:class:`~synkit.Synthesis.Reactor.SynReactor`, with NetworkX graphs and
direct integration with SynKit Lewis-state and stereo models. Public classes
are imported from :mod:`synkit.Synthesis.Reactor`; implementation modules are
organized by responsibility:

- ``core`` owns orchestration, graph rewriting, and product-state perception.
- ``matching`` owns match policy and symmetry quotients.
- ``stereo`` owns stereo assignment limits and product-branch propagation.
- ``output`` owns exact deduplication and serialization.
- ``workflow`` owns batch application, filtering, benchmarking, and postprocessing.
- ``variants`` contains specialized engines built on ``SynReactor``.

For example:

.. code-block:: python

   from synkit.Synthesis.Reactor import BatchReactor, Strategy, SynReactor

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
   * - ``serialization_errors``
     - str
     - ``'raise'``
     - Raw ITS serialization policy. ``'raise'`` preserves compatibility and
       aborts ``smarts_list`` if any raw application cannot be serialized.
       ``'skip'`` retains every serializable result in its original order and
       reports the omitted indices. This affects only ``dedup_its=False``.

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
aligned one-to-one when every ITS graph is serializable. Raw multiplicity
counts graph applications and stereo branches; it is not a kinetic weight or
predicted product distribution.

For exploratory batches where valid serializations should survive isolated
failures, select the skip policy:

.. code-block:: python

   import warnings

   from synkit.Synthesis.Reactor import RawITSApplicationSerializationWarning

   reactor = SynReactor(
       substrate,
       rule,
       explicit_h=False,
       dedup_its=False,
       serialization_errors="skip",
   )
   with warnings.catch_warnings(record=True) as caught:
       warnings.simplefilter("always")
       serializable_smarts = reactor.smarts_list

   print(reactor.serialization_failure_indices)

Exactly one ``RawITSApplicationSerializationWarning`` is emitted for a
computed batch with failures. Its ``indices`` attribute and
``serialization_failure_indices`` both contain every omitted zero-based raw
application index. ``its_list`` remains the complete raw graph batch, whereas
``smarts_list`` contains only the valid serializations. Exceptions raised by
matching, rewriting, or unrelated code are never converted into skips.

Example: Forward Prediction (NetworkX)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: Forward prediction with explicit H and backtracking strategy
   :linenos:

   from synkit.Synthesis.Reactor import SynReactor

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

   from synkit.Synthesis.Reactor import SynReactor

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

For templates written in **implicit-H** form, enable ``implicit_temp=True``
while keeping ``explicit_h=False``.

.. code-block:: python
   :caption: Backward prediction with an implicit-H template
   :linenos:

   from synkit.Synthesis.Reactor import SynReactor

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
   from synkit.Synthesis.Reactor import SynReactor

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
   * - Lone-pair rule values
     - Tuple endpoint counts are normalized to resource demand and supply.
       ``(2, 1)`` therefore becomes ``(1, 0)`` and consumes one lone pair
       from the matched host instead of assigning an absolute product count.
   * - Aromaticity
     - Electron LLG matching normalizes phase-equivalent Kekulé placement
       inside aromatic systems while retaining aromatic node state and local
       pi-electron valence. Stored sigma/pi values remain authoritative during
       rewriting; partial aromatic-system morphisms remain phase-sensitive.

Radical-based linking
---------------------

The RBL subsystem lives in :mod:`synkit.Synthesis.RBL`. Generic rule
application remains in :mod:`synkit.Synthesis.Reactor`; RBL types are not
duplicated or re-exported from the Reactor package.

``RBLEngine`` links forward and backward template applications through a
wildcard-aware reaction-centre overlap. It is useful when a direct reactor
application is insufficient and the two sides need to be fused through a
shared core.

Choose the execution mode according to the required recall and cost:

- ``"fast_track"`` performs only a cheap reactor round-trip.
- ``"fast_fusion"`` adds a WL-ranked, bounded categorical-fusion fallback and
  always reports an incomplete search when that fallback is used.
- ``"early_stop"`` (the default) tries both cheap paths first, then performs
  MCS fusion and stops at the first validated candidate.
- ``"full"`` retains candidates from both cheap paths and continues with the
  compatibility maximum-MCS generator. Its historical mapping cap,
  component assignment, and uncertified automorphism quotient are always
  exposed as incomplete-search reasons when active.
- ``"verified"`` enumerates all admitted typed partial overlaps, with no
  automorphism quotient or mapping cap, uses explicit mapped hydrogen and
  categorical pushouts, and requires a replayable end-to-end proof plus the
  strict reconstruction acceptance relation.

The search policy has independent candidate-scope, termination,
overlap-scope, proof-level, and acceptance-task axes. Search-scope monotonicity
therefore applies only while the acceptance task is fixed: strict
reconstruction can correctly reject a candidate accepted by compatibility
mode. Explicit pair, mapping, typed-overlap state, result-count, and wall-time
limits are never interpreted as proof that no candidate exists.

Verified overlap semantics
~~~~~~~~~~~~~~~~~~~~~~~~~~

For forward graph :math:`F` and backward graph :math:`B`, verified search
enumerates every non-empty injective partial map
:math:`f:S\subseteq V(F)\rightarrow V(B)` admitted by the typed
``FusionInterface`` contract. Unique compatible atom-map identities are
mandatory provenance anchors. Node state, isotope, Lewis resources, bond
state, wildcard role/domain, owner incidence, and stereo constraints must all
agree. The constructed candidate is the audited graph pushout
:math:`F\sqcup_f B`; maximum cardinality is an ordering preference, not an
admissibility restriction. Typed leaf-port assignments are exact bipartite
matching enumerations with maximum-matching upper-bound pruning.

The strict acceptance relation treats observed endpoints as component
multisets. For each side and canonical molecular component :math:`m`, it
requires :math:`O_s(m)\leq C_s(m)`. A closed boundary additionally requires
zero element/isotope and formal-charge delta. An open boundary accepts only an
exact, explicitly declared environment delta. Every material atom must retain
an atom-map provenance identity.

Each verified output has a ``synkit.rbl-proof/2`` document. Replay restores
the raw forward and backward application witnesses, rebuilds the typed
interface and pushout, certifies the narrow wildcard-to-hydrogen
post-processing transformation, deterministically reserializes the final
graph, and reruns strict acceptance. Outcome identity and derivation identity
are separate: one molecular result can list multiple proof digests.

Results use four search statuses: ``FOUND``, ``PROVED_NONE``, ``INCOMPLETE``,
and ``ERROR``. ``PROVED_NONE`` is emitted only after the declared overlap
universe is exhausted without limits or operational failures. Deterministic
candidate ranking is lexicographic over off-rule edits, resource imbalance,
unresolved wildcards, added heavy material, total additions, charge
separation, radical electrons, and finally stable graph/proof digests.

.. code-block:: python
   :caption: Run the RBL engine with its default exact MCS matcher

   from synkit.Synthesis.RBL import RBLEngine, RBL_RESULT_SCHEMA

   engine = RBLEngine(mode="early_stop")
   result = engine.process(reaction_rsmi, template)
   candidates = result.fused_rsmis
   assert result.result["schema"] == RBL_RESULT_SCHEMA

Use ``mode="full"`` only when compatibility recall is the objective, and
``mode="verified"`` when every returned fusion must carry a replayable proof
and strict conservation semantics. ``matcher_cls`` accepts
``ApproxMCSMatcher`` for a faster heuristic alternative in compatibility
profiles; it cannot support the verified completeness claim.
The serializable ``result`` mapping uses the versioned
``synkit.rbl-result/2`` contract.

See Also
--------

- :mod:`synkit.IO` — format conversion utilities (SMILES/SMARTS/GML and related helpers)
- :mod:`synkit.Graph` — graph data structures, matching, and transformations
