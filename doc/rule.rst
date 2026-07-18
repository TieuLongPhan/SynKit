.. _rule:

Rule
====

The :py:mod:`synkit.Rule` package treats reaction transformations as **first-class objects**.
It provides a focused toolkit to **identify**, **apply**, and **modify** reaction rules for
forward prediction, retrosynthesis, and rule-database workflows. In practice, rules serve as
portable graph-rewriting operators that can be analyzed, normalized, and reused across datasets.

.. raw:: html

   <style>
     /* Optional: keep the cards visually consistent and slightly more compact */
     .sd-card { border-radius: 10px; }
     .sd-card-title { font-weight: 700; }
   </style>

.. grid:: 1 1 2 3
   :gutter: 2

   .. grid-item-card:: :octicon:`diff` Identity
      :class-card: sd-shadow-sm

      Compare and deduplicate native rule and graph objects with exact,
      stereo-aware identity semantics.

   .. grid-item-card:: :octicon:`play` Apply
      :class-card: sd-shadow-sm

      Apply rules to molecule/reaction graphs for forward or backward inference.
      Often used together with :py:mod:`synkit.Graph.Matcher` strategies to enumerate
      all valid matches and transformations.

   .. grid-item-card:: :octicon:`tools` Modify
      :class-card: sd-shadow-sm

      Edit and normalize rule templates: handle hydrogens, tune context,
      and adjust matching behavior to improve robustness across heterogeneous data.

Package layout
--------------

Rule utilities are organized into native graph-based subpackages:

- :py:mod:`synkit.Rule.Compose` — native rule identity and clustering helpers
- :py:mod:`synkit.Rule.Apply` — native rule matching
- :py:mod:`synkit.Rule.Modify` — editing / normalization (e.g., hydrogen handling, context tuning)

Proof-bearing generic stereo rules
----------------------------------

Ordinary :class:`~synkit.Rule.SynRule` construction remains exact. To
generalize unchanged peripheral ligands of a mapped stereo reaction, use the
additive :class:`~synkit.Rule.GenericStereoRuleExtractor` API::

   from synkit.Rule import GenericStereoRuleExtractor, GenericStereoRulePolicy

   policy = GenericStereoRulePolicy(
       domain_source="class",
       explicit_domains={
           1: {"elements": {"C"}},
           3: {"elements": {"F", "Br"}},
       },
   )
   result = GenericStereoRuleExtractor(policy).extract(mapped_reaction)
   rule = result.rule
   certificate = result.certificate

The default ``observed`` policy admits only the chemistry present in the
source. Broader ``class`` and ``corpus`` domains must be supplied explicitly;
``exact`` performs no generalization. Accepted extraction requires exact
forward replay and, for invertible effects, exact reverse replay. Reaction
loci, entering and leaving ligands, virtual H/LP identities, descriptor orbit
relations, and witnesses are retained rather than replaced by unrestricted
wildcards.

Typed stereo ports are assigned exhaustively during application. Set
``SynReactor(stereo_assignment_limit=N)`` to bound that search; exceeding the
cap raises ``StereoWildcardAssignmentLimitError`` and never returns a partial
population. Exact stereo-aware product quotienting records
``application_orbit`` multiplicity and all contributing mapping, port, and
local-morphism provenance for certified generic rules.

Common patterns
---------------

Context and hydrogen handling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Real-world datasets often mix implicit/explicit hydrogens, inconsistent aromaticity flags,
and different atom-mapping conventions. Utilities in :py:mod:`synkit.Rule.Modify` (together with
related helpers in :py:mod:`synkit.Chem` and :py:mod:`synkit.Graph`) help normalize templates and
improve matching robustness.

Typical operations include:

- switching between implicit-H and explicit-H conventions in the reaction center
- tightening or relaxing context around the reaction center to control rule specificity
- aligning template conventions with standardization/canonicalization steps

Where to go next
----------------

- :doc:`Graph Module <graph>` — matching engines and isomorphism strategies
- :doc:`Chem Module <chem>` — reaction standardization and mapping validation
- :doc:`API Reference <api/index>` — auto-generated API docs for :py:mod:`synkit.Rule`
