Stereo-preserving graph morphisms
=================================

Boundary
--------

``GraphMorphism`` remains SynKit's immutable, total, injective material-node
embedding with its typed wildcard substitution environment.  A
``StereoMorphism`` refines that value with endpoint-local stereo evidence.  It
does not retain either mutable NetworkX graph.

This is a match-morphism contract.  It proves that declared stereo information
is preserved, specialized by an explicit query policy, ignored explicitly, or
left as an explicit propagation obligation.  It does not classify chemical
change or authorize reaction-induced reconfiguration; those are separate
reaction semantics.

Local certificates
------------------

For every source descriptor, ``LocalStereoCertificate`` records:

* its registry layer;
* source and optional target ``StereoConfiguration`` values;
* the relative ``StereoRelation``;
* a concrete ``PermutationWitness``;
* the descriptor-local information policy; and
* ``matched``, ``ignored``, or ``propagate`` status.

Configurations use endpoint-local node tokens.  Material references are
transported through the underlying injective node map.  ``@H`` and ``@LP``
tokens retain their typed owner, and a degree-one explicit hydrogen is
projected onto the same owner-scoped virtual-H token unless that descriptor
already has a distinct virtual-H slot.  A mapped H removed by graph
normalization is recovered only when the owner retains explicit-H neighbor
evidence or when all remaining material ligands are owner-incident and its
``hcount`` supplies the missing slot.  Wildcard ports remain material positions
in the frame and their incidence is included in the canonical morphism
signature.

Every matched certificate is checked twice: its witness must replay the
transported concrete frame exactly, and its stored relation kind and
double-coset class must agree with direct endpoint classification.  Missing
references, invalid owners, non-injective material maps, unmatched required
descriptors, strict target extras, and forged witnesses are structured
failures.

Information and presence policies
---------------------------------

The two policy axes are deliberately separate.

``StereoInformationPolicy`` controls orientation information:

``exact``
   Fixed matches equivalent fixed; unspecified matches unspecified only.

``wildcard``
   An unspecified query may specialize to any admissible fixed or unspecified
   configuration on the same frame support.  A fixed query remains exact.

``either``
   An unspecified query behaves as a wildcard.  A fixed query accepts its
   exact compatibility encoding or the descriptor's explicitly declared
   alternative encoding; it does not accept arbitrary reconfiguration.

``StereoPresenceMode`` controls descriptor presence:

``require``
   Every source descriptor needs one distinct policy-valid target; target-only
   descriptors are allowed.

``strict``
   As ``require``, and no unmatched target descriptor may touch the material
   image.

``ignore``
   Structural acceptance is explicit and every source descriptor receives an
   ``ignored`` certificate.

``propagate``
   A descriptor without a valid target becomes a visible propagation
   obligation rather than a silent wildcard.

The default information policy is morphism-wide, while a source registry key
may override it.  The effective choice is stored on each local certificate.

Identity, composition, and signatures
-------------------------------------

An identity stereo morphism contains identity node maps and identity
permutation witnesses.  For composable morphisms ``f: A -> B`` and
``g: B -> C``, SynKit:

#. composes ``GraphMorphism`` values;
#. pairs the concrete intermediate configurations;
#. composes the replayable permutation witnesses;
#. checks the composed witness directly on the transported ``A`` frame; and
#. reclassifies the direct ``A``-to-``C`` relation.

Consequently, witness composition is operational while relation double-coset
identifiers remain classifications and are never composed as operations.

``StereoMorphism.canonical_signature()`` extends the numbering-independent
graph-morphism signature with shape, information levels, status, effective
policy, relation class, witness permutation, virtual-reference kinds, and
wildcard-frame positions.  Consistent endpoint renumbering preserves it.

Mapping APIs and comparison
---------------------------

``candidate_mapping_stereo_morphism()`` returns the proof for one structural
candidate.  ``stereo_isomorphism_morphisms()`` returns proofs for the complete
accepted isomorphism set, while ``stereo_isomorphism_mappings()`` returns the
corresponding node maps.

The boolean compatibility predicates remain available.  Their ``orbit``,
``legacy``, and ``compare`` modes return the orbit result.  Compare mode audits
the complete accepted mapping set against the frozen predicate and records a
structured result; it never falls back to the legacy decision.  The ordinary
first-mapping path remains lazy, while complete enumeration is performed only
when requested or required for comparison.
