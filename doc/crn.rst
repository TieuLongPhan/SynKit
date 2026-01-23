.. _crn:

CRN
===

SynKit provides a lightweight, pure-Python CRN layer for **building**, **checking**, and **canonicalizing**
chemical reaction networks. In SynKit **v1.1.0+**, this functionality lives under :py:mod:`synkit.CRN`
(and no longer under the synthesis stack).

.. tip::
   If you only need fast prototyping (no MØD), the :py:mod:`synkit.CRN` module is designed to work with
   NetworkX out-of-the-box.

What SynKit considers a CRN
---------------------------

A CRN is represented as a **directed bipartite graph**:

- **Species nodes**: molecules (SMILES) or abstract species labels.
- **Reaction nodes**: reaction events (rule applications), often annotated with rule/step/app indices.

Edges are directed:

- ``species → reaction`` for reactants
- ``reaction → species`` for products

.. graphviz::

   digraph crn {
     rankdir=LR;
     node [shape=box, fontsize=10];
     A [label="species: A"]; B [label="species: B"]; C [label="species: C"]; D [label="species: D"];
     r0 [shape=ellipse, label="rxn: r0@1@0"];
     r1 [shape=ellipse, label="rxn: r1@2@0"];
     A -> r0; B -> r0; r0 -> C;
     C -> r1; r1 -> D;
   }

Quickstart
----------

Build a CRN from reaction SMILES (or abstract labels):

.. code-block:: python

   from synkit.CRN.DAG.syncrn import SynCRN

   rxns = [
       "A+B>>C",
       "C>>D",
   ]

   crn = SynCRN(rxns).build()
   G = crn.graph  # a directed NetworkX graph

   # convenience summaries (names may differ slightly across versions)
   print(crn.n_species, crn.n_reactions)

Recommended node conventions
----------------------------

For interoperability (visualization, exports, canonicalization), SynKit uses a practical node schema:

- Species nodes include:

  - ``kind='species'``
  - ``smiles`` and/or ``label``

- Reaction nodes include:

  - ``kind='rxn'``
  - ``rule_name``, ``step``, and an application index

.. note::
   The CRN layer is designed to be flexible: you can keep species as abstract labels in early-stage analysis,
   then “upgrade” to SMILES when you connect to :py:mod:`synkit.Chem` and :py:mod:`synkit.IO` workflows.

CRN diagnostics
---------------

Typical checks and summaries you can run on a built network:

- **Stoichiometric summaries** per reaction node (in/out degree, multiplicities).
- **Structural checks** (e.g., isolated nodes, unreachable products, repeated reaction events).
- **Balancing support** by delegating to :py:mod:`synkit.Chem.Reaction.balance_check`.

Canonicalization and symmetry
-----------------------------

CRNs get large quickly, and network comparison is hard unless you can canonicalize structure.
SynKit provides two complementary strategies:

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: CRNCanonicalizer (Bliss-style)

      A fast, structure-driven canonicalizer inspired by *Bliss* strategies.
      Good when you want deterministic labels / stable hashes for CRN comparison.

   .. grid-item-card:: WLCanonicalizer (approximate)

      Weisfeiler–Lehman refinement for fast, approximate canonicalization.
      Useful on very large networks or when exact symmetry detection is too expensive.

.. warning::
   Exact CRN automorphism groups may be imperfect in some cases.
   Use WL-based orbits when you need fast, “good enough” symmetry breaking.

Hypergraph view
---------------

Many CRN operations are easier when you treat a reaction as a **hyperedge** (multi-reactant → multi-product).
SynKit exposes a hypergraph representation (see :py:mod:`synkit.CRN.Hypergraph`) that is designed to be:

- consistent with CRN DAGs,
- compatible with canonicalizers,
- friendly for downstream analysis.

A common pattern:

.. code-block:: python

   from synkit.CRN.Hypergraph.conversion import rxns_to_hypergraph

   H = rxns_to_hypergraph(["A+B>>C", "C>>D"])
   # H is a CRNHyperGraph-like object (backend may vary)

Interoperability and export
---------------------------

Because the CRN layer is NetworkX-friendly, you can easily:

- serialize graphs (GraphML / JSON) for external tools,
- visualize with :py:mod:`synkit.Vis` utilities,
- attach custom attributes to nodes/edges without breaking canonicalization.

See also
--------

- :doc:`Synthesis Module <synthesis>` (reaction prediction + multi-step utilities)
- :doc:`Graph Module <graph>` (matching, WL hashing, ITS/MTG construction)
- :doc:`Chemical Modeling <chem>` (balance checks, standardization, AAM validation)
