Chem
====

Chemical utilities for reactions, molecules, fingerprints, clustering, and related helpers.

Reaction
--------

.. automodule:: synkit.Chem.Reaction.canon_rsmi
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Reaction.standardize
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Reaction.aam_validator
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Reaction.balance_check
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Reaction.cleaning
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Reaction.deionize
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Reaction.fix_aam
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Reaction.neutralize
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Reaction.radical_wildcard
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Reaction.tautomerize
   :members:
   :show-inheritance:

Atom-to-atom mapping
---------------------

The mapper is split into a public chemistry front end, the WL/SLAP matching
engine, and optional exact refinement tools. Most applications should start
with :class:`synkit.Chem.Reaction.Mapper.AAMapper`; the lower-level modules
are useful for inspecting mappings, resolving symmetric reaction centres, or
obtaining an optimality certificate. The exact helpers are available from
``synkit.Chem.Reaction.Mapper.exact``.

.. automodule:: synkit.Chem.Reaction.Mapper.chem.aam
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Reaction.Mapper.chem.its
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Reaction.Mapper.chem.smiles
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Reaction.Mapper.slap.sequential
   :members:
   :show-inheritance:

Molecule
--------

.. automodule:: synkit.Chem.Molecule.atom_features
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Molecule.descriptors
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Molecule.formula
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Molecule.graph_annotator
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Molecule.standardize
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Molecule.valence
   :members:
   :show-inheritance:

Fingerprint
-----------

.. automodule:: synkit.Chem.Fingerprint.fp_calculator
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Fingerprint.smiles_featurizer
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Fingerprint.transformation_fp
   :members:
   :show-inheritance:

Cluster
-------

.. automodule:: synkit.Chem.Cluster.butina
   :members:
   :show-inheritance:

Utilities
---------

.. automodule:: synkit.Chem.utils
   :members:
   :show-inheritance:
