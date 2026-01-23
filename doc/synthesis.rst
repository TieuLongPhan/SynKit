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

Two interchangeable backends are available:

- **NetworkX-based reactor**
  :py:class:`~synkit.Synthesis.Reactor.syn_reactor.SynReactor`
  (lightweight, pure-Python workflow and tight integration with ``synkit`` graphs)

- **MØD-based reactor**
  :py:class:`~synkit.Synthesis.Reactor.mod_reactor.MODReactor` :cite:`andersen2016software`
  (graph-grammar engine backend, suitable for robust rewriting and larger workloads)

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

Example: Forward Prediction (MØD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: Forward prediction using the MØD backend
   :linenos:

   from synkit.Synthesis.Reactor.mod_reactor import MODReactor

   input_fw = 'CC=O.CC=O'
   template = '[C:2]=[O:3].[C:4]([H:7])[H:8]>>[C:2]=[C:4].[O:3]([H:7])[H:8]'

   reactor_mod = MODReactor(
       substrate=input_fw,
       rule_file=template,
       invert=False,
       strategy='bt'
   )

   reaction_list = reactor_mod.reaction_smiles
   print(reaction_list)

.. admonition:: Example output
   :class: note synkit-example-output

   .. code-block:: text

      ['CC=O.CC=O>>CC=CC=O.O']

Example: Backward Prediction with AAM (MØD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When atom mapping must be retained end-to-end, use the AAM-aware variant (e.g., ``MODAAM``)
together with a GML rule representation.

.. code-block:: python
   :caption: Backward prediction with atom-map preservation
   :linenos:

   from synkit.Synthesis.Reactor.mod_aam import MODAAM
   from synkit.IO import smart_to_gml

   input_bw = 'CC=CC=O.O'
   rule_gml = smart_to_gml(
       '[C:2]=[O:3].[C:4]([H:7])[H:8]>>[C:2]=[C:4].[O:3]([H:7])[H:8]',
       core=True
   )

   reactor_aam = MODAAM(
       substrate=input_bw,
       rule_file=rule_gml,
       invert=True,
       strategy='bt'
   )

   smarts_list = reactor_aam.get_smarts()
   print(smarts_list)

.. admonition:: Example output
   :class: note synkit-example-output

   .. code-block:: text

      [
        '[CH3:1][CH:2]=[O:3].[CH:4]([CH:5]=[O:6])([H:7])[H:8]>>[CH3:1][CH:2]=[CH:4][CH:5]=[O:6].[O:3]([H:7])[H:8]',
        '[CH3:1][CH:2]([H:3])[H:4].[CH:5]([CH:6]=[O:7])=[O:8]>>[CH3:1][CH:2]=[CH:5][CH:6]=[O:7].[H:3][O:8][H:4]'
      ]

See Also
--------

- :mod:`synkit.IO` — format conversion utilities (SMILES/SMARTS/GML and related helpers)
- :mod:`synkit.Graph` — graph data structures, matching, and transformations
