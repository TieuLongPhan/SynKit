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

External molecule-stereo datasets are registered under
``Experiment/Stereo/Data`` with task and license metadata. The ACS 258-case set
supports both binary supplied-stereo and four-state stereo-stripped protocols.
ChiralFinder RotA is instead a positive axial-locus dataset, and the CIP
Validation Suite is a local descriptor-assignment suite; neither contributes
to global chiral/achiral accuracy. RotA is vendored under MIT. The CIP fixture
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

The class :py:class:`~synkit.Chem.Mapper.AAMValidator` verifies atom-map
equivalence by constructing an **Imaginary Transition State (ITS)** graph for each reaction
and testing graph isomorphism via NetworkX’s VF2 algorithm. This ensures that two mapped
reactions induce the same ITS topology, i.e., they represent the same transformation under
different atom-map assignments :cite:`phan2025syntemp`.

.. code-block:: python
   :caption: Checking whether two mapped reactions are atom-map equivalent
   :linenos:

   from synkit.Chem.Mapper import AAMValidator

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

``synkit.Chem.Mapper`` provides the current atom-to-atom mapping
(AAM) workflow. ``AAMapper`` combines WL label refinement with sequential
linear-assignment matching (SLAP). It can optionally enumerate
symmetry-distinct exact optima, attach a certificate, and prefer
electron-balanced mapped reactions. The mapper replaces the former
``wl_mapper`` module; import from the package-level API shown below.

.. code-block:: python
   :caption: Map an unmapped reaction SMILES
   :linenos:

   from synkit.Chem.Mapper import AAMapper

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

For a complete search over every atom-compatible assignment, use
``enumerate_smiles``. ``CD="minimal"`` returns every globally minimal mapping;
any finite non-negative numeric value selects that exact distance shell.

.. code-block:: python

   search = mapper.enumerate_smiles(
       "CCC>>CCC",
       CD=8,
       add_Hs=False,
       time_limit_seconds=30,
       certify=True,
   )

   if search.status == "no_solutions":
       print("the complete exact-CD shell is empty")
   elif search.status == "timeout":
       print("partial search; absence of further mappings is not proven")

``search.complete`` is true for both a fully enumerated non-empty shell and a
proven-empty ``no_solutions`` shell. A deadline instead returns
``status="timeout"`` and ``complete=False``. ``max_bijections`` is checked
before search and raises explicitly when the atom-compatible assignment space
is beyond the requested cap. Empty numeric shells above
``search.maximum_cost_upper_bound`` are certified immediately by that bound.
The older ``enumerate_exact=True`` mode remains the faster uncertainty-kernel
enumeration and has a narrower proof scope.

By default, numeric shells use a memory-constant optimization pass first, so
``search.minimum_cost`` is the proven global minimum rather than merely the
lowest mapping encountered below the requested shell. Set
``compute_minimum_cost=False`` when only the requested numeric shell matters;
this avoids the separate optimization search and leaves ``minimum_cost`` as
``None``. The enumeration pass uses element-block assignment bounds on both
sides of the target: a subtree is rejected when it can neither stay below nor
rise high enough to reach the requested CD.

When a shell may contain many mappings, stream it without retaining the output
list. The callback receives only final mappings from the proven shell; minimal
queries therefore optimize first and never emit provisional incumbents.

.. code-block:: python

   def consume(mapping, distance):
       print(mapping, distance)

   search = mapper.enumerate_smiles(
       "CCC>>CCC",
       CD="minimal",
       add_Hs=False,
       collect_mappings=False,
       mapping_callback=consume,
   )
   print("exact mapping count:", search.selected_mapping_count)

In this mode ``search.mappings`` remains empty. The current prefix-certificate
format requires ``collect_mappings=True`` because its digest binds the complete
sorted mapping set.

With ``certify=True``, ``search.certificate`` is a JSON-serializable terminal
prefix cover. Its verifier reconstructs the atom-compatible decision tree,
checks that the prefixes are disjoint and exhaustive, independently recomputes
every partial-CD pruning bound, and binds the selected mappings by SHA-256.

.. code-block:: python

   import json
   from synkit.Chem.Mapper import verify_distance_enumeration_certificate
   from synkit.Chem.Mapper.chem.smiles import smiles2lgp

   record = json.loads(json.dumps(search.certificate.as_dict()))
   original_lgp = smiles2lgp("CCC>>CCC", add_Hs=False)
   verified = verify_distance_enumeration_certificate(
       original_lgp,
       record,
       mappings=search.mappings,
   )

Timeout results carry an incomplete certificate whose disjoint
``frontier_prefixes`` cover every unresolved assignment subtree. The same
verifier checks the union of terminal and frontier prefixes, but the nonempty
frontier forces ``status="timeout"`` and cannot establish shell completeness.

Set ``symmetry_pruning=True`` to retain certified lex leaders under a bounded
verified subgroup of product automorphisms. At each prefix, a Schreier
stabilizer fixes the images already selected and branching visits one candidate
per resulting orbit. Generator and work limits are memory bounded; exceeding a
limit disables deeper symmetry reduction instead of making the result
incomplete. Bounded discovery may still leave multiple representatives from a
full product-automorphism orbit. SynKit's internal exact canonicalizer supplies
concrete automorphism witnesses; no external nauty binding is required. Every
symmetry-pruned prefix is stored with an exact transporter, and the verifier
checks atom types and the complete directed product matrix before accepting the
reduction. Remaining-assignment lower and upper bounds are replayed
independently as well.

For the complete labeled shell with symmetry acceleration, also set
``expand_symmetry=True``. SynKit then restricts orbital pruning to one fully
enumerated cyclic subgroup generated by a verified product automorphism and
streams every group image of each representative. The subgroup action on
bijections is free, so expansion contains every labeled mapping exactly once
even when the subgroup is smaller than the full automorphism group. This mode
is currently incompatible with prefix certificates.

For an auditable automatic choice between exact algorithms, call
``enumerate_hybrid_distance_mappings``. A numeric binary labeled shell uses
the edit-support isomorphism backend only when its exact support-pair estimate
is below the configured limit. Minimal, weighted, fixed-map, certificate, and
symmetry-quotient requests stay on assignment branch-and-bound. The returned
``backend`` and ``backend_statistics`` fields record the decision and never
change the requested output scope.

Exact alternative ITS classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``enumerate_mapped_reaction_its_alternatives`` turns a completed shell into an
auditable reference-relative application. It returns one deterministic AAM
representative for every exact canonical ITS class that differs from the
supplied reference ITS. The result reports the number of retained
representative-scope mappings and, when the verified subgroup order is known,
the corresponding labeled mappings in each class.

.. code-block:: python

   from synkit.Chem.Mapper import (
       GlobalShellConfig,
       enumerate_mapped_reaction_its_alternatives,
   )

   mapped_reaction = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][OH:3]"
   alternatives = enumerate_mapped_reaction_its_alternatives(
       mapped_reaction,
       CD="reference",  # also "minimal" or any non-negative numeric CD
       seed_mode="reference",
       config=GlobalShellConfig(
           time_limit_seconds=300,
           max_bijections=None,
       ),
   )

   if alternatives.shell.complete:
       for candidate in alternatives.as_dict()["shell"]["alternatives"]:
           print(candidate["its_class_id"], candidate["atom_map_correspondence"])

``CD="reference"`` uses the reference's scalar distance but still searches all
atom-compatible bijections at that distance. ``seed_mode="reference"`` may use
the map as an incumbent and ordering hint; it never fixes atoms or removes
candidates. A supplied seed is not guaranteed to reduce runtime because exact
shell enumeration remains output-sensitive. ``complete`` is true only when
both the global shell and every exact ITS classification have completed.
The exact class identifiers are seed-invariant; a labeled representative may
differ between seed modes while remaining in the same certified class.
Alternative classes can be used as controlled contrastive or hard-negative
examples, but they are not automatically chemically impossible mechanisms.

The mapper can represent hydrogens at three levels through ``add_Hs``:

- ``False`` keeps hydrogens implicit.
- ``True`` maps fully explicit hydrogens.
- ``"reaction_center"`` exposes only hydrogens involved in the reaction
  centre, which is generally the clearest output for inspection.

For a fixed heavy-atom mapping, use
``enumerate_lgp_hydrogen_transfers`` to enumerate every exact minimum
donor--acceptor hydrogen flow from the per-atom ``hcounts`` metadata. It also
computes the labeled explicit-hydrogen multiplicity without constructing
hydrogen nodes. Unequal total hydrogen inventories are rejected, and a plan
cap returns ``complete=False``.

For comparison against a reference mapping, use ``AAMValidator``. Its
``smiles_check`` method accepts either ``"RC"`` (reaction centre) or
``"ITS"`` matching and can be called from an instance to set a default policy
for unbalanced atom maps.

.. code-block:: python

   from synkit.Chem.Mapper import AAMValidator

   validator = AAMValidator(strip_unbalanced_maps=True)
   equivalent = validator.smiles_check(candidate, reference, check_method="ITS")

Reference-blinded Synister campaign
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The resumable Synister runner independently permutes reactant and product atom
orders, uses a reference-free SLAP candidate only as an incumbent, and queries
either a supplied reference-distance shell, a globally minimal shell, or both.
The FlowER cohort is not distributed with SynKit; an explicit CSV with
``source_line``, ``reaction_id``, and ``mapped_reaction`` columns is required::

   python scripts/run_synister_global_shells.py \
       --dataset /path/to/flower_test_10000_v252.csv.gz \
       --output benchmark_results/synister_global_shells_v4 \
       --mode both --time-limit-per-shell 300 --memory-limit-gib 6

The runner uses one process and one numerical thread. It writes digest-bound
gzip JSON records atomically and resumes only records matching the dataset,
implementation, schema, and options. ``reference_cd`` permits the held-out
reference's scalar CD to define the shell but never supplies the mapping to
search. ``minimal`` hides both mapping and CD until the global optimum shell
has been explored. Every timeout, output cap, canonicalization bound, or error
remains explicitly incomplete.

Complete records include labeled and verified-subgroup mapping multiplicities,
mapping Hartley entropy, exact canonical ITS/template richness, reaction-centre
intersection/union/frequencies, and post-search reference-class checks. Verify
case digests and regenerate censored manuscript statistics with
``scripts/summarize_synister_evidence.py``.

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
