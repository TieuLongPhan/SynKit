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
   from synkit.Graph.matcher.graph_matcher import GraphMatcherEngine

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

   from synkit.IO.chem_converter import rsmi_to_its
   from synkit.Graph.MTG.mcs_matcher import MCSMatcher
   from synkit.Graph.MTG.mtg import MTG
   from synkit.Graph import clean_graph_keep_largest_component
   from synkit.Vis import GraphVisualizer
   import matplotlib.pyplot as plt

   # 1) Define two related reaction SMILES and build their reaction-center ITS graphs
   rsmi_list = [
      '[CH:4]([H:7])([H:8])[CH:5]=[O:6]>>[CH:4]([H:8])=[CH:5][O:6]([H:7])',  # tautomerization
      '[CH3:1][C:2]=[O:3].[CH:4]([H:8])=[CH:5][O:6]([H:7])>>'
      '[CH3:1][C:2]([O:3][H:7])[CH:4]([H:8])[CH:5]=[O:6]'                     # nucleophilic addition
   ]
   rc_graphs = [rsmi_to_its(r, core=True) for r in rsmi_list]

   # 2) Find MCS mapping between the two ITS graphs
   mcs = MCSMatcher(node_label_names=['element', 'charge'], edge_attribute='order')
   mcs.find_rc_mapping(rc_graphs[0], rc_graphs[1], mcs=True)
   mapping = mcs.get_mappings()[0]

   # 3) Build the Mechanistic Transition Graph (MTG)
   mtg = MTG(rc_graphs[0], rc_graphs[1], mapping)
   mtg_graph = mtg.get_graph()

   # 4) Also build the composite ITS by directly gluing the two RC graphs
   its_composite = clean_graph_keep_largest_component(mtg_graph)


   # 5) Visualize all four graphs: two RCs, the composite ITS, and the MTG
   fig, axes = plt.subplots(2, 2, figsize=(14, 6))
   vis = GraphVisualizer()

   vis.plot_its(rc_graphs[0], axes[0, 0], use_edge_color=True, title='A. Tautomerization RC')
   vis.plot_its(rc_graphs[1], axes[0, 1], use_edge_color=True, title='B. Nucleophilic Addition RC')
   vis.plot_its(its_composite, axes[1, 0], use_edge_color=True, title='C. Composite ITS')
   vis.plot_its(mtg_graph, axes[1, 1], use_edge_color=True, title='D. Mechanistic TG', og=True)

   plt.tight_layout()
   plt.show()


.. container:: figure

   .. image:: ./figures/mtg.png
      :alt: Composite ITS and MTG visualization
      :align: center
      :width: 1000px

   *Figure:*  
   (A) Reaction‐center graph for the tautomerization step  
   (B) Reaction‐center graph for the nucleophilic addition step  
   (C) Composite ITS graph "gluing" both transformations  
   (D) Mechanistic Transition Graph (MTG) showing step-wise mechanism  



See Also
--------

- :mod:`synkit.IO` — format conversion utilities  
- :mod:`synkit.Synthesis` — reaction prediction & network exploration  

