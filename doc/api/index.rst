API
===

The API reference is organized by the main SynKit packages.
Each page groups the most relevant public modules for a practical, package-level view of the library.

.. raw:: html

   <style>
     .sd-card { border-radius: 10px; }
     .sd-card-title { font-weight: 700; }
     .api-lead { margin-bottom: 0.9rem; }
   </style>

.. raw:: html

   <div class="api-lead">
     Explore SynKit by package: chemistry, graph methods, reaction rules, synthesis workflows,
     chemical reaction networks, visualization, and input/output utilities.
   </div>

.. grid:: 1 1 2 3
   :gutter: 2

   .. grid-item-card:: :octicon:`beaker` Chem
      :link: chem
      :link-type: doc
      :class-card: sd-shadow-sm

      Reaction standardization, validation, fingerprints, clustering, and molecular utilities.

   .. grid-item-card:: :octicon:`share-android` Graph
      :link: graph
      :link-type: doc
      :class-card: sd-shadow-sm

      ITS, MTG, graph matching, canonicalization, context expansion, and structural analysis.

   .. grid-item-card:: :octicon:`sync` IO
      :link: io
      :link-type: doc
      :class-card: sd-shadow-sm

      Conversion between SMILES, graphs, GML, SMARTS, and other SynKit-ready representations.

   .. grid-item-card:: :octicon:`gear` Rule
      :link: rule
      :link-type: doc
      :class-card: sd-shadow-sm

      Rule composition, transformation templates, matching, and rule modification utilities.

   .. grid-item-card:: :octicon:`git-branch` Synthesis
      :link: synthesis
      :link-type: doc
      :class-card: sd-shadow-sm

      Reactor engines, multi-step workflows, scoring, benchmarking, and synthesis utilities.

   .. grid-item-card:: :octicon:`graph` CRN
      :link: crn
      :link-type: doc
      :class-card: sd-shadow-sm

      Chemical reaction network construction, querying, pathway analysis, symmetry, and properties.

   .. grid-item-card:: :octicon:`eye` Vis
      :link: vis
      :link-type: doc
      :class-card: sd-shadow-sm

      Visualization tools for reactions, rules, graphs, embeddings, and CRN structures.


.. toctree::
   :maxdepth: 1
   :hidden:

   chem
   graph
   io
   rule
   synthesis
   crn
   vis