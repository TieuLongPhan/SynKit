Permutation-orbit stereochemistry
=================================

Status and boundary
-------------------

SynKit represents local stereochemical identity as an ordered frame modulo a
finite geometry-preserving permutation group.  This follows the ordered-list
model described by Andersen, Flamm, Merkle, and Stadler in *Chemical Graph
Transformation with Stereo-Information* (ICGT 2017).  Sprint 16 implements
the finite algebra.  Sprint 17 makes it authoritative for descriptor identity
and descriptor-query matching.  Rules, reactors, and fusion remain on the
Beta-2 semantics until their later migration sprints.

This is a MØD-compatible local-configuration model, not a claim that SynKit's
operational NetworkX rewriting is a complete implementation of MØD's
monomorphic Double-Pushout rule system.

Frames and actions
------------------

A permutation ``p`` uses the single convention

.. code-block:: text

   p.apply(frame)[i] = frame[p.image[i]]

and ``p.then(q)`` denotes ``q ∘ p``.  Atom-centered frames contain the locus
at position zero followed by the ordered ligand references.  Bond-centered
frames retain the existing SynKit order
``(left_1, left_2, left_center, right_center, right_1, right_2)``.

For shape ``S`` with preserving group ``G_S``, a fixed configuration is the
orbit ``[N]_(G_S)``.  An unspecified configuration uses a larger,
shape-specific information group; it is not a query wildcard.  The registered
group orders are:

=======================  =================  =================
Shape                    Fixed group        Unspecified group
=======================  =================  =================
tetrahedral              ``A4`` (12)        ``S4`` (24)
square-planar            ``D4`` (8)         ``S4`` (24)
trigonal-bipyramidal     ``D3`` (6)         ``S5`` (120)
octahedral               proper ``O`` (24)  ``S6`` (720)
planar bond              ``V4`` (4)         bond-frame (8)
atrop bond               ``V4`` (4)         bond-frame (8)
=======================  =================  =================

Planar and atrop bond frames deliberately use distinct order-four actions.
They have the same abstract group but different bond-reversal permutations.

Canonical identity and relations
--------------------------------

The canonical representative is the minimum deterministically encoded member
of the orbit.  Two configurations are equal exactly when shape,
specification, and canonical frame agree.

For fixed configurations with a relative witness ``pi``, the
representative-independent relation class is the double coset
``G_S pi G_S``.  SynKit stores both its stable minimum permutation identifier
and the concrete witness.  Witnesses compose; double-coset identifiers do not.
The public relation kinds are ``EQUIVALENT``, ``OPPOSITE``, ``RECONFIGURED``,
``UNRELATED``, and ``UNSPECIFIED``.

Only tetrahedral, planar-bond, and atrop-bond definitions currently declare a
binary opposite operation.  Square-planar, trigonal-bipyramidal, and
octahedral nonidentity relations are general reconfigurations.

Descriptor compatibility boundary
---------------------------------

All six public descriptors expose ``configuration``, ``specification``,
``same_configuration()``, ``relation_to()``, ``replace_reference()``, and
``replace_references()``.  Equality and hashing use the orbit configuration.
The existing ``atoms``, ``parity``, and ``provenance`` constructor fields are
unchanged: parity decodes a compatibility representation into a concrete
frame and is not the mathematical identity.

``canonical_form()`` remains the exact Beta-2 compatibility encoding because
existing RDKit and interchange adapters consume it.  Likewise, ``to_dict()``
and JSON/GML payloads add no field.  Thus old data round-trips byte-for-byte at
the descriptor payload level while new code can use the formal configuration
and replayable relation witness directly.

Stored unspecified orientation is distinct from both fixed orientation and
descriptor absence.  Query wildcard behavior remains an explicit matching
policy and does not mutate stored information.  Query and identity boundaries
accept ``orbit``, ``legacy``, and ``compare`` semantics; compare mode always
returns the orbit decision and records the independent legacy decision.

Executable obligations
----------------------

Sprint 16 tests establish by finite enumeration:

* identity, closure, and inverse laws for every registered group;
* canonical equality if and only if frames share one declared orbit;
* the 24 tetrahedral frames split into two 12-member orbits;
* fixed atom-frame orbit counts are 2, 3, 20, and 30 for tetrahedral,
  square-planar, trigonal-bipyramidal, and octahedral geometry;
* relation classes do not depend on the selected representatives;
* every stored witness replays and witness composition matches direct
  transport;
* relabeling and reference replacement commute with the group action;
* the binary opposite operation is an involution where it is declared.

Sprint 17 additionally proves exhaustively that every supported legacy
``atoms/parity`` encoding induces exactly the same identity partition under
the orbit adapter, including unspecified atom and bond frames.  It also checks
unchanged serialization, reference-locus protection, geometry-specific
relations, and zero unregistered identity/query divergences.  Full symmetric
unspecified groups use the equivalent sorted-position canonical form instead
of enumerating every group element; an executable law checks that shortcut
against the exact orbit minimum.

Legacy comparison
-----------------

The Beta-2 canonicalization, inversion, query matching, and change classifier
are frozen as independent pure functions.  They do not call the production
descriptor methods they audit.  The migration modes are ``orbit``, ``legacy``,
and ``compare``.  Comparison returns the orbit result plus structured evidence
and never silently falls back to the legacy result.  Later sprints must
register every intentional divergence; an unregistered divergence is a hard
migration failure.

References
----------

* J. L. Andersen, C. Flamm, D. Merkle, and P. F. Stadler, *Chemical Graph
  Transformation with Stereo-Information*, ICGT 2017,
  https://doi.org/10.1007/978-3-319-61470-0_4.
* MØD graph and rule model, https://jakobandersen.github.io/mod/graphModel/.
