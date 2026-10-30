Molecule
========

``synkit.Chem.Molecule`` provides molecular formula, mass and stereochemistry
utilities. The formula and mass methods accept SMILES strings:

.. code-block:: python

   from synkit.Chem.Molecule.formula import Formula

   formula = Formula()
   print(formula.hill_formula("CCO"))
   print(formula.mol_weight("CCO"))

See :doc:`/api/chem` for the API reference.

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

Coupled framework evidence follows the same information rule. Use
``analyze_global_stereo_support`` to obtain a map-independent
``GlobalStereoCertificate``. Two-dimensional topology may prove a
``necessarily_chiral`` molecular outcome, but its ``FrameworkStereo`` remains
orientation-unspecified. ``configured_framework_from_certificate`` requires
explicit positive/negative orientation and non-empty provenance; it never
derives handedness from the ACS label, atom maps, CIP, or a common arbitrary
probe parity.

The analyzer fails closed when coupled frames span disconnected components,
when support exceeds 256 atoms, or when more than 64 frames would be coupled.
Those bounds limit exact auxiliary-graph growth; they are unsupported outcomes,
not achiral classifications.

.. code-block:: python
   :caption: Separating necessary chirality from configured orientation

   from rdkit import Chem
   from synkit.Chem.Molecule.global_stereo import (
       analyze_global_stereo_support,
       configured_framework_from_certificate,
   )

   cage = Chem.MolFromSmiles("C1C2(OCC1)OCCC2")
   certificate = analyze_global_stereo_support(cage)
   assert certificate.necessarily_chiral
   assert certificate.descriptor.orientation is None

   configured_framework = configured_framework_from_certificate(
       certificate,
       1,
       provenance="declared_sidecar",
   )

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

``derive_rdkit_stereo_names`` accepts one exact configured-stereograph
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
1a, 1b, and 2, including exact nuclide masses for isotope comparisons, simple
mancude-ring averaging, Rule 3, and the reference-independent one-pair subset
of Rules 4c/5. Multi-unit Rule 4b, the complete hierarchical ring digraph, and
coordination geometries return typed unsupported or unresolved results instead
of an atom-index tiebreak. Unspecified configurations never emit a label.

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

Molecule-stereo dataset registrations, task and license metadata, and
experimental data are maintained locally until release. The ACS 258-case set
supports both binary supplied-stereo and four-state stereo-stripped protocols.
ChiralFinder RotA is instead a positive axial-locus dataset, and the CIP
Validation Suite is a local descriptor-assignment suite; neither contributes
to global chiral/achiral accuracy. RotA uses the MIT license. The CIP fixture
remains external-only because its repository did not provide a redistribution
license at the audited revision. At the pinned 300-record revision, SynKit's
independent incremental label layer, with the pinned 3D orientation input,
exactly reproduces 187 complete record sets and 926 of 1,252 individual
reference labels while emitting 937 (73.96% recall, 98.83% precision). The 113
non-exact records remain explicitly classified as 9 unsupported-class, 21
missing-orientation, and 83 ranking-defect primary limitations. The
SMILES-only baseline remains 175/300. This is native local-label validation,
not global molecular-chirality accuracy.

The supplied SMILES contract has a proven ceiling of 299 exact records:
``VS010`` and ``VS011`` have byte-identical input but opposite helical labels.
The pinned external 3D file distinguishes the pair by a reflection-sensitive
signed-coordinate witness, so a 300-record input contract is identifiable only
when coordinates or an equivalent declared orientation sidecar are supplied.
That witness removes the information obstruction. The coordinate protocol now
completes both helical records, all five CT4 records, four of seven AT records,
and the fully material-framed VS144 cumulene; it does not complete the
remaining CIP sequence rules.

RotA's positive-only source is supplemented by 24 synthetic constitutional
negative controls. Thirteen axis-like controls carry exact axis-fixed
automorphisms that exchange a terminal ligand pair, and eleven have no
supported axis topology. All 24 are atom-renumbering invariant and none is a
confirmed-axis false positive. These controls validate constitutional
specificity only, not rotational barriers, isolation timescales, stability, or
handedness.

.. raw:: html

   <style>
     /* Optional: consistent styling for "Example output" boxes in HTML builds */
     .admonition.synkit-example-output { border-left-width: 6px; }
     .admonition.synkit-example-output .admonition-title { font-weight: 700; letter-spacing: 0.2px; }
     .admonition.synkit-example-output .admonition-title::before { content: "⟡ "; }
     .admonition .highlight pre { border-radius: 8px; }
   </style>

