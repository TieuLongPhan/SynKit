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

.. automodule:: synkit.Chem.Mapper.validation.aam_validator
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

Use :class:`synkit.Chem.Mapper.Mapper` for uniform method selection through
``Mapper(method='propagate').map(reaction_or_graph_pair)``. Its eight names are
``slap``, ``kernel``, ``milp``, ``distance``, ``propagate``, ``native``, ``edits``
and ``exhaustive``. All return :class:`synkit.Chem.Mapper.MappingResult`, which
reports the chosen method, search scope, proof status and completion.
See :ref:`atom-to-atom-mapping` for method comparisons, runnable examples,
result semantics, and supported resource limits.

Use :func:`synkit.Chem.Mapper.map_reaction` for the established exact reaction-SMILES API
or :class:`synkit.Chem.Mapper.AAMapper` for the WL/SLAP workflow. ``map_reaction``
defaults to Python propagation and assignment-bounded search (PABS).
``AAMapper.map_smiles`` defaults to heuristic sequential linear assignment
(SLAP); ``enumerate_exact=True`` adds refinement of a reduced uncertainty
kernel, and ``certify="exact"`` requests optional kernel MILP certification.
``AAMapper.enumerate_smiles`` selects the full-distance solver, with an optional
SLAP seed for a minimum query. Kernel optimality is scoped to its supplied
fixed assignments and candidate domains. Exact methods
are grouped under ``exact.distance_search``, ``exact.kernel_solvers``,
``exact.pabs``, and ``exact.native``; their package-level entry points remain
available from ``synkit.Chem.Mapper.exact``. Analysis, validation, and adapters
live in ``reporting``, ``validation``, and ``adapters`` respectively.

.. automodule:: synkit.Chem.Mapper.mapper
   :members:

.. automodule:: synkit.Chem.Mapper.exact.search
   :members:

.. automodule:: synkit.Chem.Mapper.exact.native.search
   :members:

.. automodule:: synkit.Chem.Mapper.reaction_search
   :members:

.. automodule:: synkit.Chem.Mapper.chem.aam
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Mapper.chem.its
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Mapper.chem.smiles
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Mapper.slap.sequential
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Mapper.exact.distance_search.search
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Mapper.exact.pabs.propagation
   :members:
   :undoc-members:

.. automodule:: synkit.Chem.Mapper.exact.hybrid
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Mapper.exact.edit_support
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Mapper.reporting.analysis
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Mapper.reporting.spectrum
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Mapper.reporting.alternatives
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Mapper.adapters.prediction
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Mapper.adapters.template
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Mapper.reporting.native_analysis
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Mapper.exact.native.build
   :members: build_native

.. automodule:: synkit.Chem.Mapper.exact.native.candidates
   :members: enumerate_native_candidates

Molecule
--------

.. automodule:: synkit.Chem.Molecule.stereo_perception
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Molecule.chirality
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Molecule.global_stereo
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Molecule.cip_ranking
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Molecule.cip_assignment
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Molecule.coordinate_stereo
   :members:
   :show-inheritance:

.. automodule:: synkit.Chem.Molecule.derived_stereo_naming
   :members:
   :show-inheritance:

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
