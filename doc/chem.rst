.. _chem:

Chem
====

The ``synkit.Chem`` module provides utilities for **reaction SMILES processing**,
covering atom-map canonicalization, atom-map equivalence validation, and configurable
SMILES standardization. These tools are designed to make reactions comparable across
datasets and pipelines by enforcing consistent labeling and normalized string forms.
For unmapped reactions, see :ref:`atom-to-atom-mapping` for the WL/SLAP-based
``AAMapper`` workflow and a complete runnable example.

Whole-molecule chirality
------------------------

Stereo-element perception is a separate, earlier step. The typed detector
reports whether atom, bond, or axis support can carry stereochemistry and
whether the input supplies configuration. It does not assign a descriptor,
CIP label, stability state, or molecular chiral/achiral verdict.

For tetrahedral atoms, broad carrier detection comes first. A second operation
canonically partitions the four ligand slots under exact molecular
automorphisms that fix the center. Four singleton ligand classes confirm an
ordinary constitutionally distinct stereogenic center. Repeated classes remain
``symmetry_related`` rather than being declared permanently nonstereogenic,
because the configuration-neutral detector cannot confirm them. Supplied
configuration and CIP are attached later and never split or reorder these
constitutional neighbor classes.

.. code-block:: python
   :caption: Perceiving stereo elements without promoting configuration

   from rdkit import Chem
   from synkit.Chem.Molecule.stereo_perception import (
       canonicalize_tetrahedral_constitution,
       detect_potential_stereo_elements,
       detect_tetrahedral_carriers,
   )

   molecule = Chem.MolFromSmiles("FC(Cl)Br.FC=CCl.ClC=C=CCl")
   elements = detect_potential_stereo_elements(molecule)
   print([(item.element_type.value, item.configuration_state.value)
          for item in elements])

   carrier = Chem.MolFromSmiles("FC(Cl)Br")
   support = detect_tetrahedral_carriers(carrier)[0]
   symmetry = canonicalize_tetrahedral_constitution(carrier, support.center)
   print(symmetry.status.value)

Cumulated double bonds are not emitted as independent local ``E/Z`` elements.
Even-bond cumulenes are represented once as an axis, while odd-bond extended
cumulenes are represented once by ``ExtendedCisTransStereo`` over the complete
path. Bare helical connectivity likewise never creates a configured path
descriptor.

``classify_molecular_chirality`` determines whether a molecule is identical to
its mirror image. This is a molecule-level global automorphism calculation,
not a reaction-rule operation. The classifier completes eligible sp3 topology
before reflection, so it also handles globally chiral cages whose local atom
tags are removed by RDKit.

.. code-block:: python
   :caption: Classifying whole-molecule chirality
   :linenos:

   from rdkit import Chem
   from synkit.Chem.Molecule.chirality import (
       assess_molecular_chirality,
       classify_molecular_chirality,
       detect_potential_stereo_loci,
   )

   molecule = Chem.MolFromSmiles("F[C@](Cl)(Br)I")
   result = classify_molecular_chirality(molecule)
   print(result.classification.value)

   unspecified = Chem.MolFromSmiles("FC(Cl)C(Br)I")
   assessment = assess_molecular_chirality(unspecified, max_isomers=256)
   print(assessment.outcome.value)

   cumulene = Chem.MolFromSmiles("ClC=C=CCl")
   loci = detect_potential_stereo_loci(cumulene)
   print(loci[0].orientation_state.value)

.. admonition:: Example output
   :class: note synkit-example-output

   .. code-block:: text

      Chiral
      necessarily_chiral
      unspecified

Removing ``@``, ``@@``, slash, or backslash stereo markers is lossy. The
classifier's ``stereo_complete`` option can probe whether an unlabelled
topology supports chirality, but it cannot recover an erased relative
configuration. Distinct meso and chiral stereoisomers may have the same
stereo-free SMILES; such input should be treated as stereochemically
underspecified rather than assigned an exact stereoisomer-level label. The
result exposes ``input_stereo_status`` and ``unspecified_stereo_loci`` so a
caller can enforce that distinction instead of silently accepting the
provisional binary classification.

Potential cumulene and biaryl axes are returned as typed
``PotentialStereoLocus`` values. Their atom support, terminal references,
2D-connectivity provenance, unspecified orientation, and unassessed stability
remain evidence—not configured descriptors. They do not participate in binary
mirror comparison until configuration comes from explicit input, validated
geometry, or exhaustive enumeration.

Configured cumulene and helical descriptors can be supplied through a typed
``MolecularStereoConfiguration``. Descriptor references at this boundary are
zero-based RDKit atom indices. Orientation, evidence source, stability, and
population remain separate values; a descriptor whose parity is unspecified
is rejected as configuration evidence.

.. code-block:: python
   :caption: Classifying a declared cumulene configuration

   from synkit.Chem.Molecule.stereo_evidence import (
       MolecularStereoConfiguration,
       StereoEvidenceSource,
   )
   from synkit.Graph.Stereo import CumuleneAxisStereo

   cumulene = Chem.MolFromSmiles("FC=C=C(Cl)Br")
   axis = CumuleneAxisStereo(
       (1, 2, 3),
       ((0, "@H:1"), (4, 5)),
       1,
       "declared_sidecar",
   )
   evidence = MolecularStereoConfiguration(
       (axis,),
       StereoEvidenceSource.DECLARED_SIDECAR,
   )
   configured = classify_molecular_chirality(
       cumulene,
       stereo_configuration=evidence,
       require_specified=True,
   )

Derived CIP labels
------------------

``assign_cip_label`` projects a configured descriptor to a local label without
changing the molecule or descriptor. Tetrahedral and trigonal-pyramidal
environments use ``R/S`` (with the owner-scoped virtual ligand participating at
atomic number zero), planar bonds use ``E/Z``, and configured axes or helices
use ``M/P``. Labels are report values only: they never enter stereo descriptor
IDs, hashes, serialization, reaction rules, or graph identity.

``derive_rdkit_stereo_names`` accepts one exact Version 2
``StereoAssignment`` produced by stereograph enumeration, verifies its
certificate digest and complete fixed configuration, and binds the local
assignment reports to that source certificate. This is a one-way dependency:
the resulting names cannot change the assignment or its canonical code.

.. code-block:: python
   :caption: Deriving a witnessed local label

   from synkit.Chem.Molecule.cip_assignment import assign_cip_label
   from synkit.Graph.Stereo import TetrahedralStereo

   molecule = Chem.MolFromSmiles("F[C@](Cl)(Br)I")
   descriptor = TetrahedralStereo((1, 0, 2, 3, 4), -1)
   assignment = assign_cip_label(molecule, descriptor)
   print(assignment.label, assignment.status.value)

.. admonition:: Example output
   :class: note synkit-example-output

   .. code-block:: text

      S assigned

The independent ranking kernel currently implements witnessed Sequence Rules
1a, 1b, and 2, including exact nuclide masses for isotope comparisons. Ties
that require Rules 3--5, a complete aromatic duplicate-node
model, or a relabel-invariant axis direction return a typed unsupported or
unresolved result instead of an atom-index tiebreak. Coordination geometries
also return structured unsupported results. Unspecified configurations never
emit a label.

For supported unassigned tetrahedral atoms and double bonds,
``assess_molecular_chirality`` enumerates unique configurations and returns one
of ``necessarily_chiral``, ``necessarily_achiral``,
``configuration_dependent``, or ``unsupported_or_incomplete``. Its explicit
``max_isomers`` bound never promotes a truncated one-sided sample to a
necessary conclusion; discovering both chiral and achiral completions is
already a definitive configuration-dependent result. Unresolved square-planar,
TBP, octahedral, cumulene, helical, and atropisomeric input fails closed unless
the configured extended locus is covered by authorized evidence.
Call ``classify_molecular_chirality(..., require_specified=True)`` when a
binary-only consumer should reject every underspecified input.

External molecule-stereo datasets are registered under
``Experiment/Stereo/Data`` with task and license metadata. The ACS 258-case set
supports both binary supplied-stereo and four-state stereo-stripped protocols.
ChiralFinder RotA is instead a positive axial-locus dataset, and the CIP
Validation Suite is a local descriptor-assignment suite; neither contributes
to global chiral/achiral accuracy. RotA is vendored under MIT. The CIP fixture
remains external-only because its repository did not provide a redistribution
license at the audited revision. At the pinned 300-record revision, SynKit's
independent incremental label layer exactly reproduces 155 complete record
sets and 843 of 1,252 individual reference labels (67.33% recall, 97.80%
precision). The 145 non-exact records remain explicitly classified as 15
unsupported-class, 27 missing-orientation, 101 ranking-defect, and 2
label-projection primary limitations. This is native local-label validation,
not global molecular-chirality accuracy.

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
