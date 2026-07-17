Relation-based reaction stereochemistry
=======================================

Boundary
--------

Reaction-induced stereo is separate from structural ``GraphMorphism`` values
and match-preserving ``StereoMorphism`` proofs.  A ``StereoChange`` consists
of two ordered operations:

#. align the reactant and product reference frames by replacing leaving
   ligands with entering ligands; and
#. apply a witnessed permutation relation to the aligned frame.

This order is authoritative.  It proves SN2 inversion without consulting CIP
labels and allows the same rule to act covariantly on either substrate
enantiomer.

Reference alignment
-------------------

``StereoReferenceAlignment`` records a deterministic replacement mapping and
one of ``identity``, ``inferred``, ``explicit``, or ``refused`` status.  SynKit
infers exactly one removed/added ligand pair.  Two or more replacements require
an explicit bijection; an absent, incomplete, or non-bijective declaration
produces a structured ``STEREO_ALIGNMENT_*`` refusal and is never guessed.

``ITSConstruction.construct(..., stereo_reference_mappings=...)`` accepts
explicit maps keyed by descriptor target.  ``compare_stereo_registries`` and
``annotate_its_stereo`` expose the same lower-level boundary.

Relations and compatibility labels
----------------------------------

The aligned endpoint configurations are classified by the Sprint 16 orbit
kernel as ``equivalent``, ``opposite``, ``reconfigured``, ``unspecified``, or
``unrelated``.  Accepted fixed relations carry a stable double-coset class ID
and a concrete permutation witness.  ``StereoChange.relation_evidence()`` and
``SynRule.stereo_summary()`` expose the alignment, class, and witness.

The existing ``RETAINED``, ``INVERTED``, ``FORMED``, ``BROKEN``, ``FLEETING``,
and ``UNSPECIFIED`` strings remain compatibility projections.  In particular,
legacy TBP and octahedral parity inversions keep the ``INVERTED`` projection
while their authoritative evidence can be
``reconfigured:<double-coset-class>``.  This avoids pretending that every
non-tetrahedral change is one binary inversion operation.

Application and reversal
------------------------

``StereoChange.apply_to()`` first replaces references in the concrete matched
substrate frame and then replays the stored witness.  An unspecified substrate
therefore yields an unspecified product rather than acquiring the template's
orientation.  A refused alignment raises ``StereoAlignmentError``.  Reactor
``propagate`` mode catches that refusal and fails closed to unknown orientation;
strict matching can still apply an exact endpoint template.

Reversal inverts the reference mapping and reclassifies the endpoint relation.
Known-to-unknown ``UNSPECIFIED`` changes remain non-invertible and continue to
raise ``NonInvertibleStereoEffectError`` before reverse-rule construction.
Branch outcomes and vicinal ``SYN``/``ANTI`` couplings retain their existing
independent semantics.

Migration comparison
--------------------

Classification, application, and reversal accept ``orbit``, ``legacy``, and
``compare`` semantics.  Orbit results are authoritative.  Compare mode emits
``StereoSemanticComparison`` records and never falls back.  TBP/octahedral
non-binary relation classifications are registered as
``nonbinary_orbit_reconfiguration``; every other disagreement remains
unregistered and sprint-blocking.

``SynReactor(..., stereo_semantics="compare")`` threads the same audit through
stereo-sensitive mapping acceptance, relation application, and reverse-rule
construction.  Accumulated records are available through
``stereo_semantic_diagnostics``.  ``stereo_semantics="legacy"`` is an explicit
diagnostic execution mode; no error or missing witness silently selects it.
