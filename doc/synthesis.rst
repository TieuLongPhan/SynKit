Synthesis Module
================

The ``synkit.Synthesis`` package offers tools for both **reaction prediction** and **chemical reaction network (CRN) exploration**.  

.. contents::
   :local:
   :depth: 1

Reaction Prediction: Reactor
----------------------------

The ``synkit.Synthesis.Reactor`` submodule supports two backends:

- **NetworkX-based** reactor  
  :py:class:`~synkit.Synthesis.Reactor.syn_reactor.SynReactor`  
- **MØD-based** reactor  
  :py:class:`~synkit.Synthesis.Reactor.mod_reactor.MODReactor` :cite:`andersen2016software`. 

**Reactor parameters**  

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
     - Use ``False`` for **forward** prediction (substrate → products);  
       ``True`` for **backward** prediction (target → precursors).
   * - ``explicit_h``
     - bool
     - ``False``
     - When ``True``, all hydrogens within the reaction center  
       are rendered explicitly in the resulting SMARTS.
   * - ``strategy``
     - str
     - ``'bt'``
     - Graph-matching strategy to enumerate transformations:  
       - ``'comp'``: component-aware subgraph search (fastest, preferred)  
       - ``'all'``: exhaustive arbitrary subgraph search  
       - ``'bt'``: backtracking fallback (tries ``comp`` first, then ``all`` if no match)

Example: Forward Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: Forward reaction prediction with explicit H and backtracking strategy
   :linenos:

   from synkit.Synthesis.Reactor.syn_reactor import SynReactor

   input_fw = 'CC=O.CC=O'
   template = '[C:2]=[O:3].[C:4]([H:7])[H:8]>>[C:2]=[C:4].[O:3]([H:7])[H:8]'

   reactor = SynReactor(
       substrate=input_fw,
       template=template,
       invert=False,        # forward prediction
       explicit_h=True,     # show H in reaction center
       strategy='bt'        # try component search, then exhaustive
   )

   smarts_list = reactor.smarts_list
   print(smarts_list)
   # >> ['[CH3:1][CH:2]=[O:3].[CH:4]([CH:5]=[O:6])([H:7])[H:8]>>[CH3:1][CH:2]=[CH:4][CH:5]=[O:6].[O:3]([H:7])[H:8]', '[CH3:4][CH:5]=[O:6].[CH:1]([CH:2]=[O:3])([H:7])[H:8]>>[CH:1]([CH:2]=[O:3])=[CH:5][CH3:4].[O:6]([H:7])[H:8]']

Example: Backward Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: Backward reaction prediction targeting product to precursors
   :linenos:

   from synkit.Synthesis.Reactor.syn_reactor import SynReactor

   target = 'CC=CC=O.O'
   template = '[C:2]=[O:3].[C:4]([H:7])[H:8]>>[C:2]=[C:4].[O:3]([H:7])[H:8]'

   reactor_bw = SynReactor(
       substrate=target,
       template=template,
       invert=True,         # backward prediction
       explicit_h=False,    # hydrogens implicit
       strategy='comp'      # component-aware search
   )

   precursors = reactor_bw.smarts_list
   print(precursors)
   # >> ['[CH3:1][CH:2]=[O:6].[CH3:3][CH:4]=[O:5]>>[CH3:1][CH:2]=[CH:3][CH:4]=[O:5].[OH2:6]', '[CH3:1][CH3:2].[CH:3]([CH:4]=[O:5])=[O:6]>>[CH3:1][CH:2]=[CH:3][CH:4]=[O:5].[OH2:6]']


Example: Implicit Hydrogen (NetworkX)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some cases you may want to use an **implicit-H** template. Set `implicit_temp=True` and `explicit_h=False`:

.. code-block:: python
   :caption: Implicit-H template for backward prediction
   :linenos:

   from synkit.Synthesis.Reactor.syn_reactor import SynReactor

   target = 'CC=CC=O.O'
   template = '[C:2]=[O:3].[CH2:4]>>[C:2]=[C:4].[OH2:3]'

   reactor_imp = SynReactor(
       substrate=target,
       template=template,
       invert=True,          # backward prediction
       explicit_h=False,     # hydrogens implicit
       strategy='comp',      # component-aware search
       implicit_temp=True    # use implicit-H template
   )

   precursors = reactor_imp.smarts_list
   print(precursors)
   # >> ['[CH3:1][CH:2]=[O:6].[CH3:3][CH:4]=[O:5]>>[CH3:1][CH:2]=[CH:3][CH:4]=[O:5].[OH2:6]', '[CH3:1][CH3:2].[CH:3]([CH:4]=[O:5])=[O:6]>>[CH3:1][CH:2]=[CH:3][CH:4]=[O:5].[OH2:6]']

Example: Forward Prediction (MØD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: Forward prediction (no atom-map reservation)
   :linenos:

   from synkit.Synthesis.Reactor.mod_reactor import MODReactor

   input_fw = 'CC=O.CC=O'
   template = '[C:2]=[O:3].[C:4]([H:7])[H:8]>>[C:2]=[C:4].[O:3]([H:7])[H:8]'

   reactor_mod = MODReactor(
       substrate=input_fw,
       rule_file=template,
       invert=False,         # forward direction
       strategy='bt'         # component-aware, then exhaustive
   )

   reaction_list = reactor_mod.reaction_smiles
   print(reaction_list)
   # >> ['CC=O.CC=O>>CC=CC=O.O']


Example: Backward Prediction with AAM (MØD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: Backward prediction (with atom-map preservation)
   :linenos:

   from synkit.Synthesis.Reactor.mod_reactor import MODReactor
   from synkit.Synthesis.Reactor.mod_aam      import MODAAM
   from synkit.IO                            import smart_to_gml

   input_bw = 'CC=CC=O.O'
   rule_gml = smart_to_gml(
       '[C:2]=[O:3].[C:4]([H:7])[H:8]>>[C:2]=[C:4].[O:3]([H:7])[H:8]',
       core=True
   )

   reactor_bw = MODAAM(
       substrate=input_bw,
       rule_file=rule_gml,
       invert=True,
       strategy='bt'
   )

   smarts_list = reactor_bw.get_smarts()
   print(smarts_list)
   # >> [
   #   '[CH3:1][CH:2]=[O:3].[CH:4]([CH:5]=[O:6])([H:7])[H:8]>>[CH3:1][CH:2]=[CH:4][CH:5]=[O:6].[O:3]([H:7])[H:8]',
   #   '[CH3:1][CH:2]([H:3])[H:4].[CH:5]([CH:6]=[O:7])=[O:8]>>[CH3:1][CH:2]=[CH:5][CH:6]=[O:7].[H:3][O:8][H:4]'
   # ]


Chemical Reaction Networks: CRN
-------------------------------

The ``synkit.Synthesis.CRN`` submodule provides classes for constructing,
analyzing, and exploring chemical reaction networks derived from reaction rules.

.. automodule:: synkit.Synthesis.CRN.mod_crn
   :members:
   :undoc-members:
   :show-inheritance:

See Also
--------

- :mod:`synkit.IO` — format conversion utilities  
- :mod:`synkit.Graph` — graph data structures and transformations  
