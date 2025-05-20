====
API
====

Chem Module
===========
The `Chem` module provides tools for handling input and output operations related to the chemical converter. It allows seamless interaction with various chemical data formats.

.. automodule:: synkit.Chem.Reaction.canon_rsmi
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synkit.Chem.Reaction.standardize
   :members:
   :undoc-members:
   :show-inheritance:  

.. automodule:: synkit.Chem.Reaction.aam_validator
   :members:
   :undoc-members:
   :show-inheritance:  

Synthesis Module
================

.. automodule:: synkit.Synthesis.Reactor.syn_reactor
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synkit.Synthesis.Reactor.mod_reactor
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synkit.Synthesis.Reactor.mod_aam
   :members:
   :undoc-members:
   :show-inheritance:

Graph Module
============

ITS Submodule
-------------
The `ITS` submodule provides tools for constructing, decomposing, and validating ITS (input-transformation-output) graphs.

- **its_construction**: Functions for constructing an ITS graph.
- **its_decompose**: Functions for decomposing an ITS graph and extracting reaction center.

.. automodule:: synkit.Graph.ITS.its_construction
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synkit.Graph.ITS.its_decompose
   :members:
   :undoc-members:
   :show-inheritance:


Matcher Submodule
-----------------

The ``synkit.Graph.Matcher`` package provides comprehensive tools for graph comparison, subgraph search, and clustering. It is organized into four main areas:

- **Matching Engines**  
  Perform graph‐to‐graph and subgraph isomorphism checks:  
  - :py:class:`~synkit.Graph.Matcher.graph_matcher.GraphMatcherEngine`  
  - :py:class:`~synkit.Graph.Matcher.subgraph_matcher.SubgraphSearchEngine`

- **Single-Graph Clustering**  
  Cluster a single graph’s nodes or components:  
  - :py:mod:`~synkit.Graph.Matcher.graph_cluster`

- **Batch Clustering**  
  Process and cluster multiple graphs in parallel:  
  - :py:mod:`~synkit.Graph.Matcher.batch_cluster`

- **High-Throughput Isomorphism**  
  Specialized routines for multi-pattern searches in a host graph:  
  - :py:mod:`~synkit.Graph.Matcher.sing`  
  - :py:mod:`~synkit.Graph.Matcher.turbo_iso`


Matching Engines
~~~~~~~~~~~~~~~~

.. automodule:: synkit.Graph.Matcher.graph_matcher
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synkit.Graph.Matcher.subgraph_matcher
   :members:
   :undoc-members:
   :show-inheritance:


Clustering
~~~~~~~~~~

.. automodule:: synkit.Graph.Matcher.graph_cluster
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synkit.Graph.Matcher.batch_cluster
   :members:
   :undoc-members:
   :show-inheritance:


High-Throughput Isomorphism
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: synkit.Graph.Matcher.sing
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synkit.Graph.Matcher.turbo_iso
   :members:
   :undoc-members:
   :show-inheritance:

MTG Submodule
-------------

.. automodule:: synkit.Graph.MTG.mtg
   :members:
   :undoc-members:
   :show-inheritance:



Rule Module
===========

The ``synkit.Rule`` package provides a flexible framework for **reaction rule** manipulation, composition, and application in retrosynthesis and forward‐prediction workflows. It is organized into three main subpackages:

- **Compose**  
  Build new reaction rules by composing existing ones, supporting both SMARTS‐based and GML workflows.  
- **Apply**  
  Apply rules to molecule or reaction graphs for retro‐prediction or forward‐simulation (e.g., in reactor contexts).  
- **Modify**  
  Generate artificial rule, edit and adjust rule templates—add or remove explicit hydrogens, adjust contexts, and fine‐tune matching behavior.

.. automodule:: synkit.Rule.Compose.rule_compose
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synkit.Rule.Apply.reactor_rule
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synkit.Rule.Modify.molecule_rule
   :members:
   :undoc-members:
   :show-inheritance:


Vis Module
==========

The ``synkit.Vis`` package offers a suite of **visualization utilities** for both chemical reactions and graph structures, enabling clear interpretation of mechanisms, templates, and network architectures:

- **RXNVis** (:py:class:`~synkit.Vis.rxn_vis.RXNVis`)  
  Render full reaction schemes with mapped atom‐colors, curved arrows, and publication‐quality layouts.  
- **RuleVis** (:py:class:`~synkit.Vis.rule_vis.RuleVis`)  
  Display rule templates (SMARTS/GML) as annotated graph transformations, highlighting bond changes.  
- **GraphVisualizer** (:py:class:`~synkit.Vis.graph_visualizer.GraphVisualizer`)  
  General‐purpose NetworkX graph plotting, with support for ITS, MTG, and custom node/edge styling.

.. automodule:: synkit.Vis.rxn_vis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synkit.Vis.rule_vis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synkit.Vis.graph_visualizer
   :members:
   :undoc-members:
   :show-inheritance:

IO Module
=========
The `IO` module provides tools for handling input and output operations related to the chemical converter. It allows seamless interaction with various chemical data formats.

Chemical Conversion
-------------------
.. automodule:: synkit.IO.chem_converter
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synkit.IO.mol_to_graph
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synkit.IO.graph_to_mol
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synkit.IO.nx_to_gml
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synkit.IO.gml_to_nx
   :members:
   :undoc-members:
   :show-inheritance:

IO Functions
------------

.. automodule:: synkit.IO.data_io
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synkit.IO.data_io
   :members:
   :undoc-members:
   :show-inheritance:
