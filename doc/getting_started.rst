.. _getting-started-synkit:

Getting Started
===============

.. raw:: html

   <div style="margin-bottom: 0.75rem;">
     <a href="https://pypi.org/project/synkit/"><img alt="PyPI" src="https://img.shields.io/pypi/v/synkit.svg"></a>
     <a href="https://github.com/TieuLongPhan/SynKit"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-SynKit-181717?logo=github"></a>
   </div>

Welcome! This guide gets you from **zero → first SynKit workflow** in minutes.

What you get
------------

SynKit is a modular toolkit for **reaction informatics** and **graph-first chemistry**:

- reaction / AAM canonicalization
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

Optional: MØD backend
---------------------

Some legacy workflows (e.g., MØD-backed CRN construction) require the external package **mod**.

.. note::
   On Linux, you can install **mod** via Conda:

   .. code-block:: bash

      conda install -c jakobandersen -c conda-forge "mod>=0.17" -y

Verify your install
-------------------

.. code-block:: bash

   python -c "import importlib.metadata as m; print(m.version('synkit'))"

If this prints a version, you are ready.

Quick tour
----------

Try one small end-to-end snippet that touches the most common building blocks:

.. code-block:: python

   # 1) Build a tiny CRN
   from synkit.CRN.DAG.syncrn import SynCRN

   rxns = ["A+B>>C", "C>>D"]
   crn = SynCRN(rxns).build()

   # 2) Inspect basic counts
   print(crn.n_species, crn.n_reactions)

   # 3) Convert to hypergraph (useful for canonicalization)
   from synkit.CRN.Hypergraph.conversion import rxns_to_hypergraph

   H = rxns_to_hypergraph(rxns)
   print(H)

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

      Standardization, balance checking, AAM validation.

Support
-------

- Report issues: https://github.com/TieuLongPhan/SynKit/issues
- Releases / changelog: https://github.com/tieulongphan/synkit/releases
