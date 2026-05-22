.. _vis:

Visualization
=============

SynKit's visualization layer has two roles:

* chemistry-first drawings for molecular graphs, reaction panels, and ITS graphs;
* diagnostic graph drawings for raw NetworkX objects, adapters, and future MTG
  development.

For normal chemistry work, prefer the molecule, reaction, and ITS helpers from
``synkit.Vis``. The generic graph drawer is useful when debugging attributes or
new graph representations, but it is intentionally less polished.

Molecular Graphs From SMILES
----------------------------

Start from a real SMILES, convert it to a SynKit molecular graph, and draw it
with atom-map-aware labels.

.. code-block:: python

   from synkit.IO.chem_converter import smiles_to_graph
   from synkit.Vis import draw_molecule_graph

   smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
   graph = smiles_to_graph(
       smiles,
       sanitize=True,
       use_index_as_atom_map=True,
   )

   ax = draw_molecule_graph(
       graph,
       title=smiles,
       label_mode="hetero",
       show_atom_map=True,
       aromatic_style="circle",
   )
   ax.figure

Useful molecule options:

.. list-table::
   :header-rows: 1

   * - Option
     - Use
   * - ``label_mode="hetero"``
     - Keep carbon labels compact while showing hetero atoms explicitly.
   * - ``show_atom_map=True``
     - Show atom-map numbers when present.
   * - ``aromatic_style="circle"``
     - Draw aromatic rings with a compact ring marker instead of cluttered edge labels.

Reaction Panels
---------------

Reaction drawings show reactants and products side by side and highlight atoms
and bonds that change.

.. code-block:: python

   from synkit.Vis import draw_reaction_graph

   rsmi = "[CH3:1][Cl:2].[NH3:3]>>[CH3:1][NH3+:3].[Cl-:2]"

   fig, axes = draw_reaction_graph(
       rsmi,
       title="SN2 reaction",
       show_atom_map=True,
   )

ITS Graphs
----------

ITS visualization is centered on the transformation graph itself, not the full
reactant/product panels. By default, changed bonds are shown as compact
``kekule_order`` transitions such as ``1->0`` or ``0->1``.

.. code-block:: python

   from synkit.Vis import draw_its_from_rsmi

   rsmi = (
       "[Cl:1][Cl:2].[H:9][c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1"
       ">>"
       "[Cl:1][H:9].[Cl:2][c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1"
   )

   fig, axes = draw_its_from_rsmi(
       rsmi,
       format="tuple",
       core=False,
       title="ITS: chlorine transfer to arene",
       edge_label_mode="kekule",
   )

ITS edge-label modes:

.. list-table::
   :header-rows: 1

   * - Mode
     - Meaning
   * - ``edge_label_mode="kekule"``
     - Show changed ``kekule_order`` only. This is the recommended compact view.
   * - ``edge_label_mode="sigma_pi"``
     - Show changed sigma/pi components. Unchanged components are suppressed.
   * - ``edge_label_mode="none"``
     - Hide edge labels and use only edge color/style.

Electron Labels
---------------

For tuple/electron-aware ITS graphs, node labels can show charge, lone-pair, or
radical changes. Use one signal at a time for readable figures.

.. code-block:: python

   from synkit.Vis import draw_its_from_rsmi

   sn2 = "[CH3:1][Cl:2].[NH3:3]>>[CH3:1][NH3+:3].[Cl-:2]"

   fig, axes = draw_its_from_rsmi(
       sn2,
       format="tuple",
       core=False,
       title="ITS: SN2 lone-pair changes",
       show_electron_labels=True,
       electron_label_mode="lone_pair",
   )

Electron-label modes:

.. list-table::
   :header-rows: 1

   * - Mode
     - Meaning
   * - ``electron_label_mode="charge"``
     - Show charge changes, for example ``q0->+1``.
   * - ``electron_label_mode="lone_pair"``
     - Show lone-pair changes, for example ``lambda1->0``.
   * - ``electron_label_mode="radical"``
     - Show radical changes.
   * - ``electron_label_mode="all"``
     - Show every changed electron attribute. This is useful for debugging but can be busy.

Reactant/Product Projections
----------------------------

ITS helpers can also render reactant and product projections when needed for
debugging.

.. code-block:: python

   fig, axes = draw_its_from_rsmi(
       rsmi,
       format="tuple",
       core=False,
       projection=True,
       title="ITS with reactant/product projections",
   )

Use ``projection=True`` when you need to inspect how an ITS decomposes back into
left and right molecular graphs. Use the default ITS-only view for reports and
notebooks.

Diagnostic Graph View
---------------------

The visual model adapter normalizes molecule, reaction, ITS, and MTG-like graph
objects into a common drawing model. This layer is mainly for development and
debugging.

.. code-block:: python

   from synkit.Vis import detect_visual_kind, summarize_visual_graph, to_visual_graph
   from synkit.Vis import draw_graph

   kind = detect_visual_kind(graph)
   visual_graph = to_visual_graph(graph)
   summary = summarize_visual_graph(visual_graph)

   ax = draw_graph(graph, title=f"{kind}: {summary.kind}")

Legacy Helpers
--------------

The older visualization classes are still exported for compatibility:

* ``RXNVis`` for reaction image grids;
* ``RuleVis`` for rule/ITS style drawings;
* ``GraphVisualizer`` for general NetworkX graph visualization.

New code should use ``draw_molecule_graph``, ``draw_reaction_graph``, and
``draw_its_from_rsmi`` unless a legacy workflow specifically depends on the
class-based API.

