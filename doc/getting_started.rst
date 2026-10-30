.. _getting-started-synkit:

Getting Started
===============

.. raw:: html

   <div style="margin-bottom: 0.75rem;">
     <a href="https://pypi.org/project/synkit/"><img alt="PyPI" src="https://img.shields.io/pypi/v/synkit.svg"></a>
     <a href="https://github.com/TieuLongPhan/SynKit"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-SynKit-181717?logo=github"></a>
   </div>

This guide covers installation and a minimal SynKit workflow.

Scope
-----

SynKit is a modular toolkit for **reaction informatics** and **graph-first chemistry**:

- reaction / AAM canonicalization
- atom mapping with a uniform ``Mapper(method=...)`` interface
- balance checks and standardization
- graph construction (ITS/MTG), hashing, and matching
- reaction rules (compose / apply / modify)
- lightweight **CRN exploration** (:doc:`CRN <crn>`) 

Install
-------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: :octicon:`package` pip (recommended)

      .. code-block:: bash

         python -m pip install -U pip
         pip install synkit

   .. grid-item-card:: :octicon:`container` Docker

      .. code-block:: bash

         docker pull tieulongphan/synkit:latest
         docker run --rm tieulongphan/synkit:latest \
           python -c "import importlib.metadata as m; print(m.version('synkit'))"

SynKit requires Python 3.10 or later. Install optional general dependencies,
including the kernel MILP solver, with ``python -m pip install 'synkit[all]'``.
For the MILP solver alone, use ``python -m pip install 'synkit[milp]'``.

Platform wheels include the C++ atom mapper. See :doc:`mapper` for native
solver availability, hydrogen models and source builds. Optional Python
extras do not enable the C++ solver SDKs.

For development, create and activate the project environment from the
repository root:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate synkit

The environment installs SynKit in editable mode, documentation dependencies,
and the lint, test, and package build tools.

Verify the installation
-----------------------

.. code-block:: bash

   python -c "import importlib.metadata as m; print(m.version('synkit'))"

A printed version confirms that the package is importable.

Quick tour
----------

Map an unmapped reaction
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from synkit.Chem.Mapper import map_reaction_implicit

   result = map_reaction_implicit("CCO>>COC", time_limit_seconds=10)
   print(result.mapped_reactions)
   print(result.minimum_cost, result.complete)

This uses C++ PABS with SLAP starting mappings and minimum implicit-H Lewis scoring. The
:doc:`mapper` guide explains PMCD proofs, enumeration at a supplied CD,
and the separate Python and explicit-H interfaces.

Inspect changed bonds
~~~~~~~~~~~~~~~~~~~~~

Convert an atom-mapped substitution into an imaginary transition-state graph:

.. code-block:: python

   from synkit.IO import rsmi_to_its

   reaction = "[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:3].[Br-:2]"
   its = rsmi_to_its(reaction, core=False, format="tuple")
   changed_bonds = [
       (source, target, data["order"])
       for source, target, data in its.edges(data=True)
       if data["order"][0] != data["order"][1]
   ]
   print(changed_bonds)

The broken C-Br bond has order pair ``(1.0, 0.0)`` and the formed C-O bond
has ``(0.0, 1.0)``. Continue with the :doc:`graph`, :doc:`rule`, and
:doc:`synthesis` guides.

Build a reaction network
~~~~~~~~~~~~~~~~~~~~~~~~

Construct a small CRN and convert it to a NetworkX digraph:

.. code-block:: python

   from synkit.CRN.Structure.syncrn import SynCRN

   rxns = ["A+B>>C", "C>>D"]
   crn = SynCRN.from_reaction_strings(rxns)

   print(crn.n_species, crn.n_reactions)

   graph = crn.to_digraph()
   print(graph.number_of_nodes(), graph.number_of_edges())

For an offline biochemical example, load the cached KEGG glycolysis module,
compute CRNT and Petri-net structure, obtain a replayable pathway certificate,
and export the network through SBML:

.. code-block:: python

   from pathlib import Path

   from synkit.CRN import crnt_summary, conserved_moieties, find_siphons, write_sbml
   from synkit.CRN.Benchmark import build_kegg_crn, check_glycolysis_flux

   crn = build_kegg_crn("M00001")
   summary = crnt_summary(crn)
   assert (summary.n_species, summary.n_reactions, summary.deficiency) == (14, 15, 1)
   assert len(conserved_moieties(crn)) == 3
   assert len(find_siphons(crn)) == 3

   outcome = check_glycolysis_flux()
   assert outcome["status"] == "realizable"
   assert outcome["explored_states"] == 30
   assert len(outcome["certificate"]) == 15

   write_sbml(crn, Path("glycolysis.xml"))

The `executable glycolysis notebook
<https://github.com/TieuLongPhan/SynKit/blob/main/Experiment/CRN/notebooks/crn_glycolysis_case_study.ipynb>`_
explains and independently replays the certificate. See :doc:`crn` for the
network API and benchmark workflow.

Next steps
----------

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: :octicon:`graph` CRN
      :link: crn
      :link-type: doc

      Canonicalize and analyze reaction networks.

   .. grid-item-card:: :octicon:`share-android` Graph Module
      :link: graph
      :link-type: doc

      ITS/MTG, matching, clustering, WL hashing.

   .. grid-item-card:: :octicon:`beaker` Chem Module
      :link: chem
      :link-type: doc

      Atom mapping, standardization, balance checking, AAM validation.

Support
-------

- Report issues: https://github.com/TieuLongPhan/SynKit/issues
- Releases / changelog: https://github.com/tieulongphan/synkit/releases
