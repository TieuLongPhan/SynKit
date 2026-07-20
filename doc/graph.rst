.. _graph:

Graph
=====

The ``synkit.Graph`` package provides the core **graph-based infrastructure** used across SynKit.
It supports graph construction, matching, canonicalization, and reaction-specific graph formalisms.
Most workflows in rule application, mapping validation, and CRN exploration rely on these utilities.

Key submodules include:

- **Matcher** — graph isomorphism and subgraph search engines
- **ITS** — Internal Transition State (ITS) graph construction and decomposition
- **MTG** — Mechanistic Transition Graph generation and exploration
- **FG** — graph-native functional-group detection and audit tooling
- **Context** — reaction-center expansion for context-aware matching and analysis

.. raw:: html

   <style>
     /* Optional: consistent styling for "Example output" blocks in HTML builds */
     .admonition.synkit-example-output { border-left-width: 6px; }
     .admonition.synkit-example-output .admonition-title { font-weight: 700; letter-spacing: 0.2px; }
     .admonition.synkit-example-output .admonition-title::before { content: "⟡ "; }
     .admonition .highlight pre { border-radius: 8px; }

     /* Optional: slightly nicer cards (requires sphinx-design) */
     .sd-card { border-radius: 10px; }
     .sd-card-title { font-weight: 700; }
   </style>

.. grid:: 1 1 2 3
   :gutter: 2

   .. grid-item-card:: :octicon:`search` Matcher
      :class-card: sd-shadow-sm

      Isomorphism, subgraph search, and match enumeration for labeled molecular graphs.
      Powers rule application and equivalence checks.

   .. grid-item-card:: :octicon:`beaker` ITS
      :class-card: sd-shadow-sm

      Construct and decompose **Internal Transition State** graphs to isolate reaction centers
      and represent bond-order changes explicitly.

   .. grid-item-card:: :octicon:`share-android` MTG
      :class-card: sd-shadow-sm

      Build **Mechanistic Transition Graphs** from reaction-center ITS graphs to represent
      stepwise mechanisms and compare pathways.

   .. grid-item-card:: :octicon:`filter` FG
      :class-card: sd-shadow-sm

      Detect functional groups directly on SynKit molecular graphs, with
      hierarchical labels and aromatic ring-system reporting.

Graph Canonicalization
----------------------

The class :py:class:`~synkit.Graph.canon_graph.GraphCanonicaliser` canonicalises a graph by
computing a deterministic relabeling of node indices. By default it employs a Weisfeiler–Lehman
(WL) colour-refinement backend (``wl_iterations=3``) to obtain a consistent canonical form across
isomorphic graphs :cite:`weisfeiler1968reduction`.

.. code-block:: python
   :caption: Canonicalising an ITS graph and verifying isomorphism
   :linenos:

   from synkit.IO import rsmi_to_its
   from synkit.Graph.canon_graph import GraphCanonicaliser
   from synkit.Graph.Matcher.graph_matcher import GraphMatcherEngine

   canon = GraphCanonicaliser(backend='wl', wl_iterations=3)

   rsmi = (
       '[CH3:1][CH:2]=[O:3].'
       '[CH:4]([H:7])([H:8])[CH:5]=[O:6]>>'
       '[CH3:1][CH:2]=[CH:4][CH:5]=[O:6].'
       '[O:3]([H:7])([H:8])'
   )

   its_graph = rsmi_to_its(rsmi)
   canon_graph = canon.canonicalise_graph(its_graph).canonical_graph

   print(its_graph == canon_graph)  # structural relabeling differs

   gm = GraphMatcherEngine(backend='nx')
   print(gm.isomorphic(its_graph, canon_graph))  # graph structure is preserved

.. admonition:: Example output
   :class: note synkit-example-output

   .. code-block:: text

      False
      True

Matcher
-------

The ``synkit.Graph.Matcher`` submodule provides matching engines for labeled graphs:

- :py:class:`~synkit.Graph.Matcher.graph_matcher.GraphMatcherEngine` — generic graph isomorphism / subgraph checks
- :py:class:`~synkit.Graph.Matcher.subgraph_matcher.SubgraphMatch` — subgraph search and containment tests

Example: Graph Isomorphism
~~~~~~~~~~~~~~~~~~~~~~~~~~

Check whether two ITS graphs—derived from reaction SMILES differing only by atom-map ordering—are isomorphic.

.. code-block:: python
   :caption: Full-graph isomorphism check with GraphMatcherEngine
   :linenos:

   from synkit.IO import rsmi_to_its
   from synkit.Graph.Matcher.graph_matcher import GraphMatcherEngine

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

   its_1 = rsmi_to_its(rsmi_1)
   its_2 = rsmi_to_its(rsmi_2)

   gm = GraphMatcherEngine(
       backend='nx',
       node_attrs=['element', 'charge'],
       edge_attrs=['order'],
   )

   are_isomorphic = gm.isomorphic(its_1, its_2)
   print(are_isomorphic)

.. admonition:: Example output
   :class: note synkit-example-output

   .. code-block:: text

      True

Example: Subgraph Search
~~~~~~~~~~~~~~~~~~~~~~~~

Locate a smaller “reaction-center” ITS graph as a subgraph within a larger ITS graph.

.. code-block:: python
   :caption: Reaction-center subgraph isomorphism with SubgraphMatch
   :linenos:

   from synkit.IO import rsmi_to_its
   from synkit.Graph.Matcher.subgraph_matcher import SubgraphMatch

   core_its = rsmi_to_its(
      '[CH3:1][C:2](=[O:3])[OH:4]>>[CH3:1][C:2](=[O:3])[O:6][CH3:5]',
      core=True
   )

   full_its = rsmi_to_its(
      '[CH3:5][C:1](=[O:2])[OH:3]>>[CH3:5][C:1](=[O:2])[O:4][CH3:6]'
   )

   sub_search = SubgraphMatch()
   found = sub_search.subgraph_isomorphism(core_its, full_its)
   print(found)

.. admonition:: Example output
   :class: note synkit-example-output

   .. code-block:: text

      True

ITS
---

The ``synkit.Graph.ITS`` package supports the construction and decomposition of
**Internal Transition State (ITS)** graphs:

- :py:class:`~synkit.Graph.ITS.its_construction.ITSConstruction` — build ITS graphs from reactant/product graphs
- :py:func:`~synkit.Graph.ITS.its_decompose.get_rc` — extract the minimal reaction-center subgraph
- :py:func:`~synkit.Graph.ITS.its_decompose.its_decompose` — split an ITS graph into reactant/product graphs

Lewis-labelled graph fields
~~~~~~~~~~~~~~~~~~~~~~~~

SynKit 1.4 introduced the representation now called the Lewis-labelled graph
(LLG) framework for the
pure-Python reactor and new mechanistic work. Legacy ITS remains available,
but LLG is the preferred representation when valence-state information must be
explicit. In the current API this representation is requested with
``format="tuple"``.

Important LLG fields:

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
   * - ``sigma_order`` / ``pi_order``
     - Authoritative bond components for Lewis-state rewriting.
   * - ``kekule_order``
     - Integer-like bond order used for product reconstruction; normally
       ``sigma_order + pi_order``.
   * - ``lone_pairs`` / ``radical``
     - Valence-state fields used by LLG matching and product accounting.
   * - ``valence_electrons``
     - Element valence-shell reference used when recomputing charge.
   * - ``order``
     - Legacy or presentation order. Aromatic ``1.5`` values are useful for
       matching and visualization, but not the LLG-authoritative rewrite
       source.

.. code-block:: python
   :caption: Building an LLG/ITS graph with Lewis-state fields
   :linenos:

   from synkit.IO import rsmi_to_its

   rsmi = "[CH3:1][Cl:2].[NH3:3]>>[CH3:1][NH3+:3].[Cl-:2]"
   its = rsmi_to_its(rsmi, format="tuple", core=False)

   print(its.nodes[2]["lone_pairs"])
   print(its.edges[1, 2]["sigma_order"])

.. note::

   Aromatic LLG matching is intentionally conservative. Aromaticity is still
   useful for presentation and pruning, but full aromatic-system relabeling is
   tracked as ongoing work.

Example: Construct and Visualize an ITS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: Build an ITS, extract the reaction center, and visualize
   :linenos:

   from synkit.IO.chem_converter import rsmi_to_graph
   from synkit.Graph.ITS.its_construction import ITSConstruction
   from synkit.Graph.ITS.its_decompose import get_rc
   from synkit.Vis import GraphVisualizer
   import matplotlib.pyplot as plt

   rsmi = (
       '[CH3:1][CH:2]=[O:3].'
       '[CH:4]([H:7])([H:8])[CH:5]=[O:6]'
       '>>'
       '[CH3:1][CH:2]=[CH:4][CH:5]=[O:6].'
       '[O:3]([H:7])([H:8])'
   )

   react_graph, prod_graph = rsmi_to_graph(rsmi)

   its_graph = ITSConstruction().ITSGraph(react_graph, prod_graph)
   rc_graph = get_rc(its_graph)

   vis = GraphVisualizer()
   fig, axes = plt.subplots(1, 2, figsize=(14, 6))
   vis.plot_its(its_graph, axes[0], use_edge_color=True, title='A. Full ITS Graph')
   vis.plot_its(rc_graph, axes[1], use_edge_color=True, title='B. Reaction Center')
   plt.show()

.. container:: figure

   .. image:: ./figures/aldol_its.png
      :alt: ITS graph and reaction-center of aldol condensation
      :align: center
      :width: 600px

   *Figure:* (A) Full ITS graph and (B) reaction-center-only ITS graph for the aldol condensation.

MTG Submodule
-------------

The ``synkit.Graph.MTG`` package provides tools for constructing and analyzing
**Mechanistic Transition Graphs (MTGs)** from reaction-center ITS graphs:

- :py:class:`~synkit.Graph.MTG.mcs_matcher.MCSMatcher` — maximum common substructure mappings
- :py:class:`~synkit.Graph.MTG.mtg.MTG` — MTG construction from ITS graphs and MCS mapping

The current MTG direction is aligned with LLG/ITS. Invariant atom data such
as ``element`` and ``atom_map`` should be stored once, while temporal fields
such as ``charge``, ``hcount``, ``lone_pairs``, ``radical``,
``sigma_order``, and ``pi_order`` store compact histories across snapshots.
This avoids redundant ``*_step_history`` attributes and makes MTG-to-ITS
round trips easier to inspect.

.. code-block:: python
   :caption: MTG to ordered ITS steps
   :linenos:

   from synkit.Graph.MTG.mtg import MTG

   mtg = MTG([step_1_its, step_2_its])
   step_its = mtg.get_its_steps()
   composed = mtg.get_compose_its()

When an MTG is built from RSMI strings, SynKit 1.4.0 converts those strings
to Lewis-labelled graph ITS by default:

.. code-block:: python
   :caption: RSMI sequence to LLG MTG
   :linenos:

   mtg = MTG(step_rsmis, mcs_mol=True)

Legacy string conversion is still available for compatibility:

.. code-block:: python
   :caption: Legacy MTG from RSMI strings
   :linenos:

   mtg = MTG(step_rsmis, mcs_mol=True, its_format="typesGH")

Compact MTG data model
~~~~~~~~~~~~~~~~~~~~~~

An LLG-backed MTG is a normal ``networkx.Graph``. Node attributes split into
two categories:

.. list-table::
   :header-rows: 1

   * - Attribute type
     - Examples
     - Meaning
   * - Invariant atom fields
     - ``element``, ``atom_map``, ``valence_electrons``
     - Stored once because the atom identity does not change across the
       mechanism.
   * - State timelines
     - ``hcount``, ``charge``, ``radical``, ``lone_pairs``, ``present``
     - Tuples with one value per mechanism state. For ``n`` elementary
       steps, these timelines have length ``n + 1``.
   * - Bond timelines
     - ``kekule_order``, ``sigma_order``, ``pi_order``
     - Tuples with one bond state per mechanism state. ``None`` means the
       bond or one endpoint is outside that state; ``0`` means both atoms are
       present but no bond exists.

This compact form intentionally avoids legacy ``typesGH`` and redundant
``*_step_history`` attributes in the new Lewis-labelled graph path.

Example: LLG MTG changed core
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example reads a stepwise aldol mechanism, constructs an LLG-backed MTG
directly from the RSMI strings, and visualizes the changed core. The default
MTG string conversion uses ``format="tuple"`` internally, so the result stores
Lewis-state timelines rather than legacy ``typesGH`` fields.

.. code-block:: python
   :caption: Building and visualizing a compact LLG MTG
   :linenos:

   from synkit.IO import load_database
   from synkit.Graph.MTG.mtg import MTG
   from synkit.Vis import draw_mtg_graph

   data = load_database("Data/Testcase/mech.json.gz")[0]
   neutral = data["mechanisms"][1]
   steps = [step["smart_string"] for step in neutral["steps"]]

   mtg = MTG(steps, mcs_mol=True)
   graph = mtg.get_mtg()

   assert mtg._tuple_its
   assert not any("typesGH" in attrs for _, attrs in graph.nodes(data=True))

   fig, ax = draw_mtg_graph(
       mtg,
       title=f"{neutral['mech_name']} - changed core",
       changed_only=True,
       show_edge_labels=True,
       compress=True,
   )

``compress=True`` labels only the first and final state of each changed edge.
Use ``compress=False`` when debugging the full mechanism-state sequence.

.. container:: figure

   .. image:: ./figures/mtg_lsg_changed_core.png
      :alt: Compact LLG MTG changed-core visualization
      :align: center
      :width: 760px

   *Figure:* LLG MTG changed-core view for the neutral aldol mechanism.
   Green edges are net formed, red edges are net broken, and pink dashed edges
   are transient timelines that change internally but have the same compressed
   first/final state.

Round-trip helpers
~~~~~~~~~~~~~~~~~~

MTGs can be projected back to their ordered ITS steps or to a composed
outer-state ITS:

.. code-block:: python
   :caption: MTG projections
   :linenos:

   step_its = mtg.get_its_steps()
   step_rsmi = mtg.get_rsmi_steps()
   composed = mtg.get_compose_its()

Use ``get_its_steps()`` when validating temporal history. Use
``get_compose_its()`` when you need the net start/end reaction encoded as a
single ITS graph.

Functional Groups
-----------------

The ``synkit.Graph.FG`` package detects functional groups directly on SynKit
molecular ``networkx`` graphs. It avoids an external FG representation and
returns labels in graph/node-index space.

Core APIs:

- :py:class:`~synkit.Graph.FG.detector.FunctionalGroupDetector`
- :py:func:`~synkit.Graph.FG.api.smiles_to_graph_and_functional_groups`
- :py:class:`~synkit.Graph.FG.audit.FunctionalGroupAudit`

.. code-block:: python
   :caption: Functional groups from SMILES
   :linenos:

   from synkit.Graph.FG import smiles_to_graph_and_functional_groups

   graph, groups = smiles_to_graph_and_functional_groups(
       "CC(=O)OC1=CC=CC=C1C(=O)O"
   )

   print(groups)

.. admonition:: Example output
   :class: note synkit-example-output

   .. code-block:: text

      [('ester', (2, 3, 4)), ('carboxylic_acid', (11, 12, 13))]

Detection is hierarchical: specific labels such as ``carboxylic_acid`` can
suppress generic nested labels such as ``carbonyl`` when the broader label
would be less useful. Public labels cover common carbonyl/acyl, oxygen,
nitrogen/C=N, sulfur, boron, silicon, phosphorus, and heteroaromatic families.

Context graph
-------------

The ``synkit.Graph.Context`` submodule expands reaction centers to include local neighborhoods,
enabling context-aware matching and analysis.

.. code-block:: python
   :caption: Context graph expansion example
   :linenos:

   from synkit.IO import rsmi_to_its
   from synkit.Graph.Context.radius_expand import RadiusExpand
   from synkit.Vis.graph_visualizer import GraphVisualizer

   smart = (
       '[CH3:1][O:2][C:3](=[O:4])[CH:5]([CH2:6][CH2:7][CH2:8][CH2:9]'
       '[NH:10][C:11](=[O:12])[O:13][CH2:14][c:15]1[cH:16][cH:17]'
       '[cH:18][cH:19][cH:20]1)[NH:21][C:22](=[O:23])[NH:24][c:25]1'
       '[cH:26][c:27]([O:28][CH3:29])[cH:30][c:31]([C:32]([CH3:33])'
       '([CH3:34])[CH3:35])[c:36]1[OH:37].[OH:38][H:39]>>'
       '[C:11](=[O:12])([O:13][CH2:14][c:15]1[cH:16][cH:17][cH:18]'
       '[cH:19][cH:20]1)[OH:38].[CH3:1][O:2][C:3](=[O:4])[CH:5]'
       '([CH2:6][CH2:7][CH2:8][CH2:9][NH:10][H:39])[NH:21][C:22]'
       '(=[O:23])[NH:24][c:25]1[cH:26][c:27]([O:28][CH3:29])[cH:30]'
       '[c:31]([C:32]([CH3:33])([CH3:34])[CH3:35])[c:36]1[OH:37]'
   )

   its = rsmi_to_its(smart)
   rc = rsmi_to_its(smart, core=True)

   exp = RadiusExpand()
   k1 = exp.extract_k(its, n_knn=1)

   gv = GraphVisualizer()
   gv.visualize_its_grid([rc, k1])

.. container:: figure

   .. image:: ./figures/context.png
      :alt: Context graph expansion example
      :align: center
      :width: 1000px

   *Figure:* (A) Minimal reaction center. (B) First-shell context expansion (k=1).

See Also
--------

- :mod:`synkit.IO` — format conversion utilities
- :mod:`synkit.Synthesis` — reaction prediction and network exploration
