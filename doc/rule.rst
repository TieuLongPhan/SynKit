.. _rule:

Rule
====

The :py:mod:`synkit.Rule` package treats reaction transformations as **first-class objects**.
It provides a focused toolkit to **compose**, **apply**, and **modify** reaction rules for
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

   .. grid-item-card:: :octicon:`git-merge` Compose
      :class-card: sd-shadow-sm

      Build new rules by composing smaller transformations.
      Useful for mechanism-inspired template construction, rule closure,
      and assembling multi-step edits into a single reusable template.

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

Rule utilities are typically organized into three subpackages:

- :py:mod:`synkit.Rule.Compose` — rule composition and combination
- :py:mod:`synkit.Rule.Apply` — applying rules (often via reactor workflows)
- :py:mod:`synkit.Rule.Modify` — editing / normalization (e.g., hydrogen handling, context tuning)

Common patterns
---------------

Compose then apply
^^^^^^^^^^^^^^^^^^

Compose a new rule from existing templates and apply it via a reactor workflow. This is a
common pattern for building “macro-templates” that capture a richer transformation while
remaining directly applicable to substrates.

.. code-block:: python
   :caption: Compose rules, then apply the composed rule via a reactor
   :linenos:

   # Pseudocode (exact function names may vary by version)
   from synkit.Rule.Compose.rule_compose import compose_rules
   from synkit.Synthesis.Reactor.syn_reactor import SynReactor

   r_new = compose_rules(rule_a, rule_b)

   reactor = SynReactor(
       substrate="CC=O.CC=O",
       template=r_new,
       invert=False,
       strategy="comp",
   )
   products = reactor.smarts_list
   print(products)

.. admonition:: Notes
   :class: tip

   - Composition is useful when you want a single template that “summarizes” multiple edits.
   - Applying a composed rule typically benefits from ``strategy='comp'`` when substrates contain
     multiple disconnected components.

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
- :doc:`API Reference <api>` — auto-generated API docs for :py:mod:`synkit.Rule`
