.. _stereo-reactions:

Reaction stereochemistry
========================

SynKit represents reaction stereochemistry as information-bearing values,
not as a comparison of ``@`` and ``@@`` tokens. The primary aggregate is
:py:class:`synkit.Graph.Stereo.StereoReactionValue`; it separates reactant
guards, endpoint/transition effects, product outcomes, correlated couplings,
semantic assertions, and typed refusals.

Information model
-----------------

The lifecycle axis is ``RETAINED``, ``INVERTED``, ``FORMED``, ``BROKEN``,
``FLEETING``, or ``UNSPECIFIED``. Product population is independent:
``SINGLE``, ``RACEMIC``, ``ENANTIOMERIC_MIXTURE``, ``DIASTEREOMER_SET``,
``MESO``, ``ACHIRAL``, or ``UNKNOWN``. Unknown orientation or population is
never interpreted as racemic.

Rejected assertions use :py:class:`synkit.Graph.Stereo.StereoRefusal` and a
stable :py:class:`synkit.Graph.Stereo.StereoRefusalCode`. Invalid references,
unsupported geometry, ambiguous alignment, lossy projection, missing context,
assignment/branch exhaustion, irreversible loss, contradictions, and invalid
couplings therefore remain visible to callers.

Rules and execution
-------------------

Exact and generic rule extraction both replay the mapped source before
returning a certificate. Generic ports are limited to eligible peripheral
references and require explicit domains when class or corpus generalization is
requested.

.. literalinclude:: ../Experiment/StereoReaction/workflows.py
   :language: python
   :pyobject: exact_and_generic_rules

``SynReactor`` exposes four stereo modes:

``ignore``
   Structural application without a stereo guard claim.
``require``
   Require matching input stereo but do not propagate a rule effect.
``propagate``
   Apply declared effects; absent orientation remains one unknown result.
``strict``
   Require guards, supports, effects, couplings, and reconstruction to agree.

``stereo_assignment_limit`` and ``stereo_branch_limit`` fail before returning
an incomplete search. ``StereoWildcardAssignmentLimitError`` and
``StereoBranchLimitError`` expose both the permitted and requested-at-least
counts.

Outcomes and couplings
----------------------

An outcome expands one configured product descriptor. A coupling represents a
single correlated event, so two vicinal centers produce two paired face
branches rather than a four-way Cartesian product.

.. literalinclude:: ../Experiment/StereoReaction/workflows.py
   :language: python
   :pyobject: branching_and_coupling

Composition and fusion evidence
-------------------------------

:py:func:`synkit.Graph.Stereo.compose_reaction_stereo` composes reference
transport, effects, guards, branch measures, and multiplicity. Intermediate
mismatch, conflicting populations/couplings, or broken-then-formed information
loss refuses. Verified graph fusion uses ``synkit.fusion-proof/2`` and binds
the entire serialized document with a digest; readers recheck every local
stereo witness.

.. literalinclude:: ../Experiment/StereoReaction/workflows.py
   :language: python
   :pyobject: composition_and_interchange

Mechanism replay and the Figure 11 rule
---------------------------------------

Mechanism verification has explicit ``off``, ``endpoint``, and ``stepwise``
modes. Only ``stepwise`` validates every supplied local effect and correlated
motion. :py:class:`synkit.Mechanism.ElectrocyclicStereoMotion` represents one
conrotatory/disrotatory terminal motion with ring direction, mapped termini,
tracked substituents, 4π/6π electron count, thermal/photochemical
activation, and provenance.

The termini are not encoded as strict molecular stereo descriptors. During
stepwise replay, SynKit derives a relative frame from the shared inward
neighbor and tracked substituent at each terminus. It canonicalizes that pair
with a map-independent local-chemistry key and stores the resulting
``synkit.relative-neighbor-change/1`` witnesses in
``certificate.final_match["canonical_neighbor_changes"]`` and on the
mechanism-transition edge. Each witness retains the canonical neighbor order,
the frame-permutation parity, the canonical rotation, and both neighbors'
before-to-after bond states. The parity transports the stored rotation back to
the physical terminal frame before conrotatory/disrotatory comparison.

For a supplied annotation, thermal 4π and 6π paths require
conrotatory and disrotatory motion, respectively; photochemical activation
reverses that matrix. SynKit verifies this supplied Woodward--Hoffmann context.
It does not select the pathway, predict activation conditions, or estimate
torquoselectivity.

.. literalinclude:: ../Experiment/StereoReaction/workflows.py
   :language: python
   :pyobject: electrocyclic_figure_11

Interchange
-----------

Canonical dict/JSON and authenticated internal-graph/GML sidecars preserve the
complete reaction-stereo value. Reaction SMILES, CXSMILES, and MOL V3000 are
endpoint carriers: populated guards, effects, outcomes, couplings, assertions,
or refusals produce a machine-readable loss report and strict-mode refusal.
Enhanced ABS/AND/OR groups are not silently equated with reaction product
populations.

Compatibility, migration, and limitations
-----------------------------------------

Schema ``synkit.reaction-stereo/2`` is canonical; v1 remains readable, while a
lossy downgrade refuses. Fusion proof v1 remains readable but cannot acquire
v2 evidence retroactively. Mechanism schema ``2.0.0-draft1`` is independently
versioned from package ``1.6.2``.

Upgrade existing values with ``StereoReactionValue.from_dict`` or
``from_json``; both accept v1 and normalize to
``synkit.reaction-stereo/2``. Use ``to_legacy_dict`` only when the value has
no assertions or refusals. A lossy downgrade raises
``ReactionStereoSchemaError`` with ``LOSSY_PROJECTION``.

Treat ``StereoChange.relation`` and its alignment witness as authoritative.
The compatibility strings ``RETAINED`` and ``INVERTED`` cannot express every
non-binary coordination-geometry relation. Likewise, never infer ``RACEMIC``
from an unknown descriptor or unspecified product.

Stricter validation during upgrade
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ITS construction defaults remain compatible, but
  ``stereo_validation="strict"`` rejects dangling paths, absent ligands,
  stale virtual H/LP resources, and wrong registry ownership.
- Generic extraction validates descriptor support before creating a rule.
- Strict reactor mode requires matching guards and valid coupling context.
- Assignment and product branching are bounded; a limit error means no
  complete result was returned.
- Mechanism ``stepwise`` mode validates local effects and electrocyclic
  machinery. ``endpoint`` is intentionally weaker, while ``off`` records
  that stereo verification was not performed.

Interchange and proof upgrades
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``project_reaction_stereo`` for complete values. Canonical JSON/dict and
authenticated graph/GML routes are lossless. Endpoint line notations require
a sidecar; strict mode raises ``ReactionStereoInterchangeError`` when a
populated reaction-semantic axis would be lost. The lower-level
``stereo_graph_to_gml`` route reports graph metadata that it omits.

Fusion proof v2 includes a whole-document digest. Older v2 documents without
that field remain readable but are reported as not whole-document verified.
Proof v1 also remains readable; new writers emit the complete v2 digest.

Minimal upgrade check
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from synkit.Graph.Stereo import StereoReactionValue

   upgraded = StereoReactionValue.from_dict(old_payload)
   assert upgraded.schema == "synkit.reaction-stereo/2"
   normalized = upgraded.normalized_json()

Run the maintained reaction gates after upgrading:

.. code-block:: console

   python -m pytest -q Test/Graph/Stereo Test/Graph/ITS
   python -m pytest -q Test/Rule Test/Synthesis/Reactor Test/Mechanism

Configured transport covers tetrahedral, planar/atrop bond, square-planar,
trigonal-bipyramidal, octahedral, cumulene/extended axes, helical, planar
chirality, and framework descriptors at their documented graph boundary.
Isotope/pseudoasymmetry ranking, enhanced-group population semantics,
ring-relative locked elimination, facial pericyclic selectivity, and advanced
allene/spiro/coordination execution remain explicit deferred claims.
