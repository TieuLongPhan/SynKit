.. _chem:

Chem
====

The ``synkit.Chem`` module provides utilities for **reaction SMILES processing**,
covering atom-map canonicalization, atom-map equivalence validation, and configurable
SMILES standardization. These tools are designed to make reactions comparable across
datasets and pipelines by enforcing consistent labeling and normalized string forms.
For unmapped reactions, see :ref:`atom-to-atom-mapping` for the WL/SLAP-based
``AAMapper`` workflow and a complete runnable example.

.. raw:: html

   <style>
     /* Optional: consistent styling for "Example output" boxes in HTML builds */
     .admonition.synkit-example-output { border-left-width: 6px; }
     .admonition.synkit-example-output .admonition-title { font-weight: 700; letter-spacing: 0.2px; }
     .admonition.synkit-example-output .admonition-title::before { content: "⟡ "; }
     .admonition .highlight pre { border-radius: 8px; }
   </style>

Canonicalization
----------------

The class :py:class:`~synkit.Chem.Reaction.canon_rsmi.CanonRSMI` standardizes reaction
SMILES and **atom-map indices** by computing a canonical relabeling of mapped atoms.
By default it employs a Weisfeiler–Lehman (WL) colour-refinement backend (``wl_iterations=3``)
to obtain a deterministic ordering that is consistent across isomorphic reactions
:cite:`weisfeiler1968reduction`.

.. code-block:: python
   :caption: Canonicalizing a mapped reaction SMILES with WL refinement
   :linenos:

   from synkit.Chem.Reaction import CanonRSMI

   canon = CanonRSMI(backend='wl', wl_iterations=3)
   canon.canonicalise(
       '[CH3:1][CH:2]=[O:3].[CH:4]([H:7])([H:8])[CH:5]=[O:6]'
       '>>'
       '[CH3:1][CH:2]=[CH:4][CH:5]=[O:6].[O:3]([H:7])([H:8])'
   )
   print(canon.canonical_rsmi)

.. admonition:: Example output
   :class: note synkit-example-output

   .. code-block:: text

      '[CH:3]([CH3:7])=[O:8].[H:1][CH:4]([H:2])[CH:6]=[O:5]>>[CH:3](=[CH:4][CH:6]=[O:5])[CH3:7].[H:1][O:8][H:2]'

AAM comparison
--------------

The class :py:class:`~synkit.Chem.Reaction.aam_validator.AAMValidator` verifies atom-map
equivalence by constructing an **Imaginary Transition State (ITS)** graph for each reaction
and testing graph isomorphism via NetworkX’s VF2 algorithm. This ensures that two mapped
reactions induce the same ITS topology, i.e., they represent the same transformation under
different atom-map assignments :cite:`phan2025syntemp`.

.. code-block:: python
   :caption: Checking whether two mapped reactions are atom-map equivalent
   :linenos:

   from synkit.Chem.Reaction import AAMValidator

   validator = AAMValidator()
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

   is_eq = validator.smiles_check(rsmi_1, rsmi_2, check_method='ITS')
   print(is_eq)

.. admonition:: Example output
   :class: note synkit-example-output

   .. code-block:: text

      True

Standardization
---------------

The class :py:class:`~synkit.Chem.Reaction.standardize.Standardize` cleans and normalizes
reaction SMILES by applying RDKit sanitization and optional post-processing steps such as:

- removing atom-map annotations (``remove_aam=True``)
- stripping stereochemical labels (``ignore_stereo=True``)

This produces a minimal, consistent representation suitable for indexing, deduplication,
and downstream CRN construction.

.. code-block:: python
   :caption: Standardizing a reaction SMILES (remove atom maps and ignore stereo)
   :linenos:

   from synkit.Chem.Reaction.standardize import Standardize

   std = Standardize()
   rsmi = (
       '[CH3:1][CH:2]=[O:3].[CH:4]([H:7])([H:8])[CH:5]=[O:6]'
       '>>'
       '[CH3:1][CH:2]=[CH:4][CH:5]=[O:6].[O:3]([H:7])([H:8])'
   )

   std_rsmi = std.fit(rsmi, remove_aam=True, ignore_stereo=True)
   print(std_rsmi)

.. admonition:: Example output
   :class: note synkit-example-output

   .. code-block:: text

      'CC=O.CC=O>>CC=CC=O.O'

Tautomerization and functional-group support
--------------------------------------------

``Tautomerize`` now uses SynKit's native functional-group detector instead of
an external FG utility. The detector works on the same molecular graph
representation used elsewhere in SynKit, so tautomer targets and graph-indexed
functional-group labels stay aligned.

.. code-block:: python
   :caption: Detecting tautomer-relevant functional groups
   :linenos:

   from synkit.Graph.FG import smiles_to_graph_and_functional_groups

   graph, groups = smiles_to_graph_and_functional_groups("C=C(O)C")
   print(groups)

The tautomerization helper still keeps a small local compatibility rule for
geminal diols. Those are treated as hydrated-carbonyl repair targets, not as a
general public functional-group label.

.. _atom-to-atom-mapping:

Atom-to-atom mapping
--------------------

``synkit.Chem.Reaction.Mapper`` provides the current atom-to-atom mapping
(AAM) workflow. ``AAMapper`` combines WL label refinement with sequential
linear-assignment matching (SLAP). It can optionally enumerate
symmetry-distinct exact optima, attach a certificate, and prefer
electron-balanced mapped reactions. The mapper replaces the former
``wl_mapper`` module; import from the package-level API shown below.

.. code-block:: python
   :caption: Map an unmapped reaction SMILES
   :linenos:

   from synkit.Chem.Reaction.Mapper import AAMapper

   mapper = AAMapper(binary=True)
   mapper.map_smiles(
       "CC(=O)O.CO>>CC(=O)OC",
       unique=True,
       electron_balance=True,
   )

   for result in mapper.results:
       print(result["smiles"], "chemical distance:", result["cd"])

``mapper.results`` is a list of mapping records. Each record includes the
mapped reaction in ``smiles``, the mapped ITS form in ``its_smiles``, and the
chemical-distance score ``cd``. Request ``enumerate_exact=True`` when a
reaction centre is symmetric and all distinct optimal mappings are needed;
``certify=True`` attaches corresponding certificate metadata.

The mapper can represent hydrogens at three levels through ``add_Hs``:

- ``False`` keeps hydrogens implicit.
- ``True`` maps fully explicit hydrogens.
- ``"reaction_center"`` exposes only hydrogens involved in the reaction
  centre, which is generally the clearest output for inspection.

For comparison against a reference mapping, use ``AAMValidator``. Its
``smiles_check`` method accepts either ``"RC"`` (reaction centre) or
``"ITS"`` matching and can be called from an instance when you need to set a
default policy for unbalanced atom maps.

.. code-block:: python

   from synkit.Chem.Reaction.Mapper import AAMValidator

   validator = AAMValidator(strip_unbalanced_maps=True)
   equivalent = validator.smiles_check(candidate, reference, check_method="ITS")

Reaction utilities
------------------

``remove_explicit_H_from_rsmi`` is also exported from
``synkit.Chem.Reaction``. Use it when a mapped reaction contains explicit
hydrogen atoms that should be folded back into normal implicit-hydrogen SMILES
before a downstream workflow that does not require them.

.. code-block:: python

   from synkit.Chem.Reaction import remove_explicit_H_from_rsmi

   compact_rsmi = remove_explicit_H_from_rsmi(explicit_h_rsmi)

See Also
--------

- :mod:`synkit.Graph` — graph modeling and matching utilities
- :doc:`Synthesis <synthesis>` — applying templates after mapping or rule extraction
