Graph Module
============

The ``synkit.Graph`` package provides core graph-based utilities in three submodules:

- **Matcher**: graph comparison and subgraph search  
- **ITS**: Internal Transition State graph construction and decomposition  
- **MTG**: Mechanistic Transition Graph generation and exploration  

.. contents::
   :local:
   :depth: 2

Graph Canonicalization
----------------------

The class :py:class:`~synkit.Graph.canon_graph.GraphCanonicaliser` canonicalises a graph by computing a canonical relabeling of node indices. It employs a Weisfeiler–Lehman colour-refinement backend (default: **3** iterations) to ensure that each atom-map assignment is uniquely and consistently ordered across isomorphic reactions :cite:`weisfeiler1968reduction`.

.. code-block:: python
   :caption: Canonicalising an ITS graph
   :linenos:

   from synkit.IO import rsmi_to_its
   from synkit.Graph.canon_graph import GraphCanonicaliser
   from synkit.Graph.Matcher.graph_matcher import GraphMatcherEngine

   canon = GraphCanonicaliser(backend='wl', wl_iterations=3)
   rsmi = (
       '[CH3:1][CH:2]=[O:3].'
       '[CH:4]([H:7])([H:8])[CH:5]=[O:6]>>'
       '[CH3:1][CH:2]=[CH:4][CH:5]=[O:6].'
       '[O:3]([H:7])([H:8])'
   )
   its_graph      = rsmi_to_its(rsmi)
   canon_graph    = canon.canonicalise_graph(its_graph).canonical_graph
   print(its_graph == canon_graph)               # False

   gm = GraphMatcherEngine(backend='nx')
   print(gm.isomorphic(its_graph, canon_graph))   # True

Matcher
-------

The ``synkit.Graph.Matcher`` submodule offers:

- :py:class:`~synkit.Graph.Matcher.graph_matcher.GraphMatcherEngine` — generic graph-isomorphism and subgraph matching  
- :py:class:`~synkit.Graph.Matcher.subgraph_matcher.SubgraphMatch` — subgraph search  

Example: Graph Isomorphism
~~~~~~~~~~~~~~~~~~~~~~~~~~

Check whether two ITS graphs—derived from reaction SMILES differing only by atom‐map ordering—are truly isomorphic:

.. code-block:: python
   :caption: Full-graph isomorphism check with GraphMatcherEngine
   :linenos:

   from synkit.IO import rsmi_to_its
   from synkit.Graph.Matcher.graph_matcher import GraphMatcherEngine

   # Two reaction SMILES with permuted atom-map labels
   rsmi_1 = (
       '[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][OH:6]'
       '>>'
       '[CH3:1][C:2](=[O:3])[O:6][CH3:5].[OH2:4]'
   )
   rsmi_2 = (
       '[CH3:5][C:1](=[O:2])[OH:3].[CH3:6][OH:4]'
       '>>'
       '[CH3:5][C:1](=[O:2])[O:4][CH3:6].[OH2:3]'
   )

   # Build ITS graphs
   its_1 = rsmi_to_its(rsmi_1)
   its_2 = rsmi_to_its(rsmi_2)

   # Initialize the matcher, comparing element, charge, and bond order
   gm = GraphMatcherEngine(
       backend='nx',
       node_attrs=['element', 'charge'],
       edge_attrs=['order']
   )

   # Test isomorphism
   are_isomorphic = gm.isomorphic(its_1, its_2)
   print(are_isomorphic)  # True — they differ only by map labels


Example: Subgraph Search
~~~~~~~~~~~~~~~~~~~~~~~~

Locate a smaller “reaction-center” ITS graph as a subgraph within a larger ITS graph:

.. code-block:: python
   :caption: Reaction-center subgraph isomorphism with SubgraphMatch
   :linenos:

   from synkit.IO import rsmi_to_its
   from synkit.Graph.Matcher.subgraph_matcher import SubgraphMatch

   # Core ITS graph of the first reaction
   core_its = rsmi_to_its(
      '[CH3:1][C:2](=[O:3])[OH:4]>>[CH3:1][C:2](=[O:3])[O:6][CH3:5]',
      core=True
   )

   # Full ITS graph of a second reaction
   full_its = rsmi_to_its(
      '[CH3:5][C:1](=[O:2])[OH:3]>>[CH3:5][C:1](=[O:2])[O:4][CH3:6]'
   )

   # Initialize subgraph search engine
   sub_search = SubgraphMatch(
      
   )

   # Check if core_its is contained within full_its
   found = sub_search.subgraph_isomorphism(core_its, full_its)
   print(found)  # True — the reaction center is present as a subgraph


ITS
---

The ``synkit.Graph.ITS`` package provides tools for constructing and decomposing Internal Transition State (ITS) graphs:

- **ITS construction**  
  :py:class:`~synkit.Graph.ITS.its_construction.ITSConstructor` — build an ITS graph from reactant/product NetworkX graphs  
- **Reaction-center extraction**  
  :py:func:`~synkit.Graph.ITS.its_decompose.get_rc` — extract the minimal reaction-center subgraph from an ITS  
- **Graph decomposition**  
  :py:func:`~synkit.Graph.ITS.its_decompose.its_decompose` — split an ITS graph back into reactant and product graphs  

Example: Construct and Visualize an ITS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: Building, extracting the center, and plotting an ITS graph
   :linenos:

   from synkit.IO.chem_converter import rsmi_to_graph
   from synkit.Graph.ITS.its_construction import ITSConstruction
   from synkit.Graph.ITS.its_decompose import get_rc, its_decompose
   from synkit.Vis import GraphVisualizer
   import matplotlib.pyplot as plt

   # Parse the reaction SMILES into reactant and product graphs
   rsmi = (
       '[CH3:1][CH:2]=[O:3].'
       '[CH:4]([H:7])([H:8])[CH:5]=[O:6]'
       '>>'
       '[CH3:1][CH:2]=[CH:4][CH:5]=[O:6].'
       '[O:3]([H:7])([H:8])'
   )
   react_graph, prod_graph = rsmi_to_graph(rsmi)

   # Build the full ITS graph
   its_graph = ITSConstruction().ITSGraph(react_graph, prod_graph)

   # Extract the reaction-center subgraph
   reaction_center = get_rc(its_graph)

   # Visualize both side by side
   vis = GraphVisualizer()
   fig, axes = plt.subplots(1, 2, figsize=(14, 6))
   vis.plot_its(its_graph, axes[0], use_edge_color=True, title='A. Full ITS Graph')
   vis.plot_its(reaction_center, axes[1], use_edge_color=True, title='B. Reaction Center')
   plt.show()

.. container:: figure

   .. image:: ./figures/aldol_its.png
      :alt: ITS graph and reaction-center of aldol condensation
      :align: center
      :width: 600px

   *Figure:* (A) Full ITS graph and (B) reaction-center-only ITS graph for the aldol condensation.


MTG Submodule
-------------

The ``synkit.Graph.MTG`` package provides tools for constructing and analyzing Mechanistic Transition Graphs (MTGs) from ITS reaction-center graphs:

- :py:class:`~synkit.Graph.MTG.mcs_matcher.MCSMatcher`  
  Compute maximum common substructure (MCS) mappings between two reaction-center ITS graphs  
- :py:class:`~synkit.Graph.MTG.mtg.MTG`  
  Build a step-by-step MTG from a pair of ITS graphs and an MCS mapping  

Example: Generate an MTG (with Composite Reaction Visualization)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example builds two reaction-center ITS graphs, computes their MCS mapping, constructs the MTG, and then visualizes:

1. Each individual reaction center  
2. The composite ITS for the overall mechanism  
3. The final MTG  

.. code-block:: python
   :caption: Building and visualizing an MTG with composite ITS
   :linenos:

   from synkit.Graph.MTG.mtg import MTG
   from synkit.Graph.ITS.its_decompose import get_rc
   from synkit.examples import list_examples, load_example
   import matplotlib.pyplot as plt
   from synkit.Vis.graph_visualizer import GraphVisualizer


   data = load_example("aldol")  

   mech_neutral = data[0]['mechanisms'][1]['steps']
   smart_neutral = [i['smart_string'] for i in mech_neutral]

   mech_acid = data[0]['mechanisms'][2]['steps']
   smart_acid = [i['smart_string'] for i in mech_acid]

   # neutral
   mtg = MTG(smart_neutral, mcs_mol=True)
   mtg_its_neutral = mtg.get_compose_its()
   mtg_rc_neutral = get_rc(mtg_its_neutral, keep_mtg=True)
   rc_neutral = get_rc(mtg_its_neutral, keep_mtg=False)

   # acid
   mtg = MTG(smart_acid, mcs_mol=True)
   mtg_its_acid = mtg.get_compose_its()
   mtg_rc_acid = get_rc(mtg_its_acid, keep_mtg=True)
   rc_acid = get_rc(mtg_its_acid, keep_mtg=False)

   # Visualize
   fig, ax = plt.subplots(2, 2, figsize=(16, 8))
   vis = GraphVisualizer()

   vis.plot_its(
      mtg_rc_neutral,
      ax=ax[0, 0],
      use_edge_color=True,
      og=True,
      title='A. MTG for aldol addition (neutral)',
      title_font_size=20,
      title_font_weight='medium',
      title_font_style='normal'
   )
   vis.plot_its(
      rc_neutral,
      ax=ax[0, 1],
      use_edge_color=True,
      og=True,
      title='B. Reaction center (neutral)',
      title_font_size=20,
      title_font_weight='medium',
      title_font_style='normal'
   )
   vis.plot_its(
      mtg_rc_acid,
      ax=ax[1, 0],
      use_edge_color=True,
      og=True,
      title='C. MTG for aldol addition (acid)',
      title_font_size=20,
      title_font_weight='medium',
      title_font_style='normal'
   )
   vis.plot_its(
      rc_acid,
      ax=ax[1, 1],
      use_edge_color=True,
      og=True,
      title='D. Reaction center (acid)',
      title_font_size=20,
      title_font_weight='medium',
      title_font_style='normal'
   )

   plt.tight_layout()
   plt.show()


.. container:: figure

   .. image:: ./figures/mtg_mechanism.png
      :alt: Composite ITS and MTG visualization
      :align: center
      :width: 1000px

   *Figure:*  
   Composition of the mechanistic sequences for aldol addition under neutral and acidic conditions, showing the composite MTG (left column) and the reaction center (right column).

Context graph
-------------

The ``synkit.Graph.Context`` submodule provides tools for expanding reaction center graphs to include nearest neighbors, enabling context‑aware analysis of reaction networks.

.. code-block:: python
   :caption: Context graph expansion example
   :linenos:

   from synkit.IO import rsmi_to_its
   from synkit.Graph.Context.radius_expand import RadiusExpand
   from synkit.Vis.graph_visualizer import GraphVisualizer

   smart = (
       '[CH3:1][O:2][C:3](=[O:4])[CH:5]([CH2:6][CH2:7][CH2:8][CH2:9]'
       '[NH:10][C:11](=[O:12])[O:13][CH2:14][c:15]1[cH:16][cH:17]'
       '[cH:18][cH:19][cH:20]1)[NH:21][C:22](=[O:23])[NH:24][c:25]1'
       '[cH:26][c:27]([O:28][CH3:29])[cH:30][c:31]([C:32]([CH3:33])'
       '([CH3:34])[CH3:35])[c:36]1[OH:37].[OH:38][H:39]>>'
       '[C:11](=[O:12])([O:13][CH2:14][c:15]1[cH:16][cH:17][cH:18]'
       '[cH:19][cH:20]1)[OH:38].[CH3:1][O:2][C:3](=[O:4])[CH:5]'
       '([CH2:6][CH2:7][CH2:8][CH2:9][NH:10][H:39])[NH:21][C:22]'
       '(=[O:23])[NH:24][c:25]1[cH:26][c:27]([O:28][CH3:29])[cH:30]'
       '[c:31]([C:32]([CH3:33])([CH3:34])[CH3:35])[c:36]1[OH:37]'
   )
   its = rsmi_to_its(smart)
   rc  = rsmi_to_its(smart, core=True)
   exp = RadiusExpand()
   k1  = exp.extract_k(its, n_knn=1)

   gv = GraphVisualizer()
   gv.visualize_its_grid([rc, k1])

.. container:: figure

   .. image:: ./figures/context.png
      :alt: Context graph expansion example
      :align: center
      :width: 1000px

   *Figure:*  
   (A) Minimal reaction center subgraph obtained by contracting all atoms that participate directly in bond‑order changes.  
   Nodes are colour‑coded by element; edges in **red** indicate bonds being broken, while edges in **blue** mark bonds being formed.  
   (B) First shell ($k=1$) context expansion: every reaction center atom is augmented with all of its immediate neighbours.


See Also
--------

- :mod:`synkit.IO` — format conversion utilities  
- :mod:`synkit.Synthesis` — reaction prediction & network exploration  

