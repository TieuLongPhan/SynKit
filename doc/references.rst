==========
References
==========

IO Module
=========
The `IO` module provides tools for handling input and output operations related to the chemical converter. It allows seamless interaction with various chemical data formats.

.. automodule:: synkit.IO.chem_converter
   :members:
   :undoc-members:
   :show-inheritance:

Graph Module
============
The `Graph` module includes several submodules focused on graph construction, decomposition, and analysis, which are essential for visualizing and processing graph-based data.

ITS Submodule
-------------
The `ITS` submodule provides tools for constructing, decomposing, and validating ITS (input-transformation-output) graphs.

- **its_construction**: Functions for constructing an ITS graph.
- **its_decompose**: Functions for decomposing an ITS graph.
- **aam_validator**: Functions for comparing ITS graph or AAM.

.. automodule:: synkit.Graph.ITS.its_construction
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synkit.Graph.ITS.its_decompose
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synkit.Graph.ITS.aam_validator
   :members:
   :undoc-members:
   :show-inheritance:

Cluster Submodule
-----------------
The `Cluster` submodule provides functions for clustering graph data into groups for analysis, including both single and batch clustering techniques.

- **graph_cluster**: Functions for clustering a graph into distinct groups.
- **batch_cluster**: Functions for batch processing and clustering multiple graphs.

.. automodule:: synkit.Graph.Cluster.graph_cluster
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synkit.Graph.Cluster.batch_cluster
   :members:
   :undoc-members:
   :show-inheritance:

Reactor Module
==============
The `Reactor` module includes functions for forward and backward prediction in chemical processes, with or without AAM (atomic action model) results.

This module supports both single-step and multi-step predictions based on graph structures.

- **core_engine**: The core engine for forward and backward prediction.
- **reactor_engine**: High-level engine for reaction prediction.
- **crn**: Functions for working with Chemical Reaction Networks.
- **path_finder**: Functions to identify possible reaction paths.

.. automodule:: synkit.Reactor.core_engine
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synkit.Reactor.reactor_engine
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synkit.Reactor.crn
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synkit.Reactor.path_finder
   :members:
   :undoc-members:
   :show-inheritance:

Rule Module
===========
The `Rule` module includes functions for rule modification and retrosynthesis, essential for transforming and optimizing chemical reaction pathways.

- **rule_compose**: Functions for composing new reaction rules.
- **reactor_rule**: Functions for applying rules to reactor systems.

.. automodule:: synkit.Rule.rule_compose
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synkit.Rule.reactor_rule
   :members:
   :undoc-members:
   :show-inheritance:

Vis Module
===========
The `Vis` module provides visualization tools for graph-based data and chemical reaction pathways, helping users interpret complex relationships and structures.

.. automodule:: synkit.Vis
   :members:
   :undoc-members:
   :show-inheritance:
