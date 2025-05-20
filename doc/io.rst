IO
====================

The ``synkit.IO`` module provides tools for converting chemical reaction representations between various formats:

- **Reaction SMILES** / SMARTS  
- **ITS** (Internal Transformation Sequence) Graph  
- **GML** (Graph Modeling Language)

.. contents::
   :local:
   :depth: 2

Aldol Reaction Example
----------------------

Below is the aldol condensation between an aldehyde and a ketone:

.. container:: figure

   .. image:: ./figures/aldol.png
      :alt: Aldol condensation scheme
      :align: center
      :width: 500px

   *Figure:* Aldol condensation between an aldehyde and a ketone.

Conversion to Reaction SMARTS
-----------------------------

Use ``rsmi_to_rsmarts`` to transform a reaction SMILES into a reaction SMARTS template:

.. code-block:: python
   :caption: Converting Reaction SMILES to SMARTS
   :linenos:

   from synkit.IO import rsmi_to_rsmarts

   template = (
       '[C:2]=[O:3].[C:4]([H:7])[H:8]'
       '>>'
       '[C:2]=[C:4].[O:3]([H:7])[H:8]'
   )

   smart = rsmi_to_rsmarts(template)
   print("Reaction SMARTS:", smart)
   # Reaction SMARTS: "[#6:2]=[#8:3].[#6:4](-[H:7])-[H:8]>>[#6:2]=[#6:4].[#8:3](-[H:7])-[H:8]"

Conversion to ITS Graph
-----------------------

Use ``rsmi_to_its`` to convert a reaction SMILES/SMARTS string into an ITS graph.  
Set ``core=True`` to restrict to the **reaction center** only.

.. code-block:: python
   :caption: Generating and Visualizing an ITS Graph
   :linenos:

   from synkit.IO import rsmi_to_its
   from synkit.Vis import GraphVisualizer

   rsmi = (
       '[CH3:1][CH:2]=[O:3].'
       '[CH:4]([H:7])([H:8])[CH:5]=[O:6]'
       '>>'
       '[CH3:1][CH:2]=[CH:4][CH:5]=[O:6].'
       '[O:3]([H:7])([H:8])'
   )

   # Full ITS graph
   full_graph = rsmi_to_its(rsmi, core=False)
   viz = GraphVisualizer()
   viz.visualize_its(full_graph, use_edge_color=True)
   # >> Figure A: Full ITS graph

   # Reaction-center-only ITS graph
   core_graph = rsmi_to_its(rsmi, core=True)
   viz.visualize_its(core_graph, use_edge_color=True)
   # >> Figure B: Reaction-center ITS graph

.. container:: figure

   .. image:: ./figures/aldol_its.png
      :alt: ITS graph and reaction-center of aldol condensation
      :align: center
      :width: 600px

   *Figure:* (A) Full ITS graph and (B) reaction-center-only ITS graph for the aldol condensation.

Conversion to DPO Rule (GML)
----------------------------

Convert a reaction SMARTS (or SMILES) template into a **DPO rule** in GML format:

- ``smart_to_gml(react_template, core=False, useSmile=False)``  
- ``its_to_gml(its_graph, core=False)``

Set ``core=True`` to include only the **reaction center**, and ``useSmile=True`` to treat the input as SMILES.

.. code-block:: python
   :caption: Generating, Saving, and Loading a DPO Rule in GML
   :linenos:

   from synkit.IO import (
      rsmi_to_its,
      smart_to_gml,
      its_to_gml,
      save_text_as_gml,
      load_gml_as_text,
   )

   # Define the aldol reaction template
   reaction = (
      '[CH3:1][CH:2]=[O:3].'
      '[CH:4]([H:7])([H:8])[CH:5]=[O:6]'
      '>>'
      '[CH3:1][CH:2]=[CH:4][CH:5]=[O:6].'
      '[O:3]([H:7])([H:8])'
   )

   # Option 1: Direct SMARTS → GML
   gml_rule_1 = smart_to_gml(reaction, core=True, useSmiles=False)

   # Option 2: SMILES → ITS → GML
   its_graph = rsmi_to_its(reaction, core=True)
   gml_rule_2 = its_to_gml(its_graph, core=True)

   # Save to disk
   save_text_as_gml(gml_text=gml_rule_2, file_path="aldol_rule.gml")

   # Load back into text
   loaded_rule = load_gml_as_text("aldol_rule.gml")
   print(loaded_rule)

.. code-block:: none
   :caption: Example DPO Rule (GML)

   rule [
     ruleID "aldol_rule"
     left [
       edge [ source 2 target 3 label "=" ]
       edge [ source 4 target 7 label "-" ]
       edge [ source 4 target 8 label "-" ]
     ]
     context [
       node [ id 2 label "C" ]
       node [ id 3 label "O" ]
       node [ id 4 label "C" ]
       node [ id 7 label "H" ]
       node [ id 8 label "H" ]
     ]
     right [
       edge [ source 2 target 4 label "=" ]
       edge [ source 3 target 7 label "-" ]
       edge [ source 3 target 8 label "-" ]
     ]
   ]

See Also
--------

- :mod:`synkit.Vis` — visualization utilities  
- :mod:`synkit.Graph` — graph data structures and transformations  
