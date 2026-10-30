.. _chem:

Chem
====

``synkit.Chem`` provides atom mapping, reaction processing and molecular
properties.

.. list-table:: Chemistry modules
   :header-rows: 1
   :widths: 20 80

   * - Module
     - Use
   * - :doc:`Mapper <mapper>`
     - Map reaction atoms, prove PMCD and enumerate classes at a supplied CD.
   * - :doc:`Reaction <chem/reaction>`
     - Standardize reaction SMILES, relabel atom maps and compare transformations.
   * - :doc:`Molecule <chem/molecule>`
     - Calculate formulae and masses. Inspect molecular stereochemistry.

The guides below contain examples. Function and class signatures are in
:doc:`api/chem`.

.. toctree::
   :maxdepth: 1
   :hidden:

   Mapper <mapper>
   Reaction <chem/reaction>
   Molecule <chem/molecule>
