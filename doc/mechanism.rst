Supplied mechanism verification (v2 draft)
==========================================

SynKit verifies explicit annotations. It does not rank or predict mechanisms.
The versioned record keeps electron events separate from declared stereo
effects, and replay commits every event group atomically.

Polar curved-arrow tutorial
---------------------------

Construct two-electron moves with ``electron_count=2`` and
``arrow_type="curved"``. Sources and targets use canonical loci ``lp``, ``σ``,
``π``, and ``∙``. Put simultaneous moves in one ``ElectronMoveGroup``, then
call ``record.verify(electron="strict")``.

Radical fishhook tutorial
-------------------------

Fishhooks use ``electron_count=1``. Homolysis and recombination require paired
moves with a shared ``coupling_id``. The six reviewed macro names are
``HOMOLYSIS``, ``RECOMBINATION``, ``RADICAL_ADDITION``, ``BETA_SCISSION``,
``H_ABSTRACTION``, and ``RADICAL_RESONANCE``. An incomplete coupled group is
rejected before any graph mutation.

Stereo-aware multi-step tutorial
--------------------------------

Attach ``StereoEffect`` values to a ``MechanisticStep`` and verify with
``stereo="stepwise"``. Supported effects are preserve, invert, break, form,
fleeting, and explicitly unspecified. ``record.to_mtg()`` stores a descriptor
timeline, while ``record.draw(path="mechanism.svg")`` provides a headless
static audit view.

Stereo rule application
-----------------------

``SynRule`` keeps reactant constraints, descriptor transformations, and
product distributions independent. ``stereo_guards`` are enforced by
``SynReactor(stereo_mode="require"|"strict")``. An unknown-parity guard is
exact by default; ``stereo_query_mode="wildcard"`` makes it a query wildcard.
Per-rule ``either`` guards are generated when a two-enantiomer outcome is
reversed.

``stereo_effects`` store endpoint descriptors and optional ITS transition
stereo. Retained, inverted, broken, formed, and fleeting are derived chemical
labels. ``stereo_outcomes`` then decide product cardinality: ``SINGLE`` emits
one product, ``RACEMIC`` emits the reference orientation and its inverse at
0.5/0.5, and ``ENANTIOMERIC_MIXTURE`` requires explicitly unequal normalized
weights. Missing or unknown stereo never implies a mixture.

Stereo-sensitive rules use descriptor-position-aware pattern symmetry and
defer host-orbit collapse until product structural/stereo deduplication. This
keeps enantiotopic embeddings available without disabling ordinary symmetry
handling for atoms outside the descriptor.

For molecule-level comparison, ``stereo_isomorphic(left, right)`` enumerates
structural isomorphisms and retains only mappings that preserve the complete
relative descriptor registry. ``stereo_isomorphism_mapping`` returns one such
mapping for audit or relabeling workflows.

Typed stereo ligand ports
-------------------------

A wildcard used as an ordered stereo ligand can declare
``wildcard_role="stereo_ligand_port"`` together with an ``owner`` and
``stereo_slot``. The slot is zero-based within the owner's peripheral ligand
sequence in the authoritative orbit configuration; locus positions are not
counted. For bond-centred descriptors each endpoint has its own owner-local
slot sequence.

Typed ports are enforced predicates, not proof annotations. Reactor preserves
their rule metadata, filters materialization candidates, and passes the
constraints into ``StereoMorphism``. Accepted proofs validate owner incidence,
ordered slot, element, charge, radical, bond order, endpoint side, mapped
identity, capacity, materialization, and optional virtual ``H``/``LP`` kind.
Unknown orientation and wildcard-ligand matching remain independent policies.
An invalid typed port fails closed and never falls back to the legacy untyped
wildcard matcher.

Migration from v1.5
-------------------

Legacy EPD and EF-SMIRKS enter through adapters. ASCII/legacy locus names are
normalized at the boundary; public v2 JSON always emits canonical Unicode.
The public Lewis-state graph acronym is LSG; LWG remains a compatibility name.
Conversions to formats that cannot carry grouping, fishhook coupling, or
stereo effects return ``ConversionLossReport`` instead of silently dropping
those fields. The schema is ``2.0.0-draft1`` until all release-owner gates are
complete.

Stereo attribution
-------------------

The relative descriptor model independently implements ideas from
StereoMolGraph (Papusha and Leonhard, 2026). Cross-validation targets upstream
commit ``2189f610f23eaaf992e2e01a12ea4d0532496601`` and compares connectivity
and relative tetrahedral, square-planar, trigonal-bipyramidal, octahedral,
planar-bond, and atrop-bond descriptor identity. Non-tetrahedral permutation
groups are adapted under StereoMolGraph's MIT license. All six classes support
SynKit graph storage, relabeling, rule matching/rewriting, and JSON/GML
serialization. RDKit conversion covers tetrahedral, square-planar,
trigonal-bipyramidal, octahedral, planar-bond, and assigned atrop-bond stereo.
Atrop conversion transports supplied orientation but never infers physical
rotational stability. Rigid-bond variants and 3D coordinate inference remain
deferred.

Run the optional pinned development oracle with
``python tools/stereo_conformance.py /path/to/StereoMolGraph``. StereoMolGraph
is not a runtime dependency. Its equality between unknown and specified parity
is recorded as an intentional difference because SynKit reserves wildcard
behavior for explicit rule-query policy.
