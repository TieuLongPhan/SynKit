.. _synkit-io:

IO
==

The ``synkit.IO`` module provides format-conversion utilities for reaction informatics.
It helps you move between string-based representations (SMILES/SMARTS) and graph-based
representations used throughout SynKit, including **ITS** graphs and **DPO rules** in **GML**.

Supported conversions include:

- **Reaction SMILES / Reaction SMARTS** (string templates)
- **ITS** (Imaginary Transition State) graphs for reaction-center analysis
- **GML** (Graph Modeling Language) rules for DPO-style rewriting workflows

.. raw:: html

   <style>
     /* Optional: callout styling for consistent "Example output" blocks */
     .admonition.synkit-example-output { border-left-width: 6px; }
     .admonition.synkit-example-output .admonition-title { font-weight: 700; letter-spacing: 0.2px; }
     .admonition.synkit-example-output .admonition-title::before { content: "⟡ "; }
     .admonition .highlight pre { border-radius: 8px; }
   </style>

Aldol Reaction Example
----------------------

Below is an aldol condensation between an aldehyde and a ketone.

.. container:: figure

   .. image:: ./figures/aldol.png
      :alt: Aldol condensation scheme
      :align: center
      :width: 500px

   *Figure:* Aldol condensation between an aldehyde and a ketone.

Conversion to Reaction SMARTS
-----------------------------

Use :py:func:`~synkit.IO.rsmi_to_rsmarts` to transform a reaction SMILES/SMARTS string
into a **reaction SMARTS template**. This step is useful when you want a normalized,
atom-typed SMARTS representation for matching and rule construction.

.. code-block:: python
   :caption: Converting reaction SMILES/SMARTS to a reaction SMARTS template
   :linenos:

   from synkit.IO import rsmi_to_rsmarts

   template = (
       '[C:2]=[O:3].[C:4]([H:7])[H:8]'
       '>>'
       '[C:2]=[C:4].[O:3]([H:7])[H:8]'
   )

   smart = rsmi_to_rsmarts(template)
   print("Reaction SMARTS:", smart)

.. admonition:: Example output
   :class: note synkit-example-output

   .. code-block:: text

      Reaction SMARTS: "[#6:2]=[#8:3].[#6:4](-[H:7])-[H:8]>>[#6:2]=[#6:4].[#8:3](-[H:7])-[H:8]"

Conversion to ITS Graph
-----------------------

Use :py:func:`~synkit.IO.rsmi_to_its` to convert a reaction SMILES/SMARTS string into
an **ITS graph**. Set ``core=True`` to restrict the output to the **reaction center**
only (a compact view that highlights changed bonds and directly participating atoms).

.. code-block:: python
   :caption: Generating and visualizing ITS graphs (full vs. reaction center)
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

   viz = GraphVisualizer()

   # Full ITS graph
   full_graph = rsmi_to_its(rsmi, core=False)
   viz.visualize_its(full_graph, use_edge_color=True)

   # Reaction-center-only ITS graph
   core_graph = rsmi_to_its(rsmi, core=True)
   viz.visualize_its(core_graph, use_edge_color=True)

.. admonition:: Example output
   :class: note synkit-example-output

   .. code-block:: text

      Figure A: Full ITS graph
      Figure B: Reaction-center ITS graph

.. container:: figure

   .. image:: ./figures/aldol_its.png
      :alt: ITS graph and reaction-center of aldol condensation
      :align: center
      :width: 600px

   *Figure:* (A) Full ITS graph and (B) reaction-center-only ITS graph for the aldol condensation.

Conversion to DPO Rule (GML)
----------------------------

Convert reaction templates or ITS graphs into **DPO rules** encoded in **GML**. Two common
paths are supported:

- :py:func:`~synkit.IO.smart_to_gml` — convert a reaction SMARTS/SMILES template to GML
- :py:func:`~synkit.IO.its_to_gml` — convert an ITS graph to GML

Key options:

- ``core=True`` includes only the **reaction center** (recommended for compact rules)
- ``useSmiles=True`` treats the input string as SMILES (instead of SMARTS)

.. code-block:: python
   :caption: Generating, saving, and loading a DPO rule in GML
   :linenos:

   from synkit.IO import (
      rsmi_to_its,
      smart_to_gml,
      its_to_gml,
      save_text_as_gml,
      load_gml_as_text,
   )

   reaction = (
      '[CH3:1][CH:2]=[O:3].'
      '[CH:4]([H:7])([H:8])[CH:5]=[O:6]'
      '>>'
      '[CH3:1][CH:2]=[CH:4][CH:5]=[O:6].'
      '[O:3]([H:7])([H:8])'
   )

   # Option 1: Direct template → GML
   gml_rule_1 = smart_to_gml(reaction, core=True, useSmiles=False)

   # Option 2: Template → ITS → GML
   its_graph = rsmi_to_its(reaction, core=True)
   gml_rule_2 = its_to_gml(its_graph, core=True)

   # Save to disk
   save_text_as_gml(gml_text=gml_rule_2, file_path="aldol_rule.gml")

   # Load back
   loaded_rule = load_gml_as_text("aldol_rule.gml")
   print(loaded_rule)

.. admonition:: Example output
   :class: note synkit-example-output

   .. code-block:: text

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
