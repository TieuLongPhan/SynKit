Changelog
=========

.. _implementation-plan:

Implementation plan
-------------------

.. list-table:: Planned production releases
   :header-rows: 1
   :widths: 14 18 68

   * - Release
     - Status
     - Primary outcome
   * - ``1.7.0``
     - Planned
     - Production molecular stereochemistry.
   * - ``1.8.0``
     - Planned
     - Complete, reproducible, and efficient CRN exploration.
   * - ``1.9.0``
     - Planned
     - Stable mechanism trajectory graph model and interchange.
   * - ``2.0.0``
     - Planned
     - Unified stereo-aware reaction, rewriting, and verification APIs.

Unreleased
----------

**Atom-to-atom mapping**

- Promoted the mapper from ``synkit.Chem.Reaction.Mapper`` to the peer package
  ``synkit.Chem.Mapper``. The former namespace remains available as a
  compatibility alias, including its lower-level submodules.
- Added complete atom-compatible enumeration for ``CD="minimal"`` and any
  finite non-negative exact CD, with explicit ``complete``, ``no_solutions``,
  and ``timeout`` outcomes.
- Added JSON-serializable exact-CD terminal-prefix certificates and an
  independent verifier for input binding, exhaustive tree coverage, pruning
  bounds, and selected-mapping digests.
- Added memory-bounded product-automorphism orbital branching backed by
  verified generators and prefix stabilizers. Symmetry-pruned subtrees carry
  exact transporter witnesses in the exact-CD certificate.
- Added replayable element-block assignment lower and upper bounds, seeded
  two-pass shell search, streaming/count-only enumeration, and an auditable
  binary edit-support/assignment hybrid selector.
- Added an OOM-safe serial, reference-blinded campaign runner for supplied-CD
  and globally minimal shells, with digest-bound resumable records and a
  censored-statistics verifier.
- Vectorized exact prefix-cost updates, added reference-free reaction-centre
  ordering and admissible atom-profile filtering, and added duplicate-free
  labeled expansion from a verified cyclic product-symmetry subgroup.
- Added exact canonical ITS/template spectra and a second-pass implicit-H flow
  enumerator with exact labeled-hydrogen multiplicities.
- Added candidate-complete alternative-ITS generation at the reference CD,
  any supplied numeric CD, or the proven global minimum. The API exports one
  AAM per non-reference exact ITS class, discloses reference seeding as an
  ordering/incumbent hint, and fails closed on shell or canonicalization caps.
- Removed unsound fingerprint/orbit sorting from the older exhaustive and
  uncertainty-kernel solvers; equal fingerprints or global orbit membership do
  not prove that arbitrary atom transpositions are valid automorphisms.

**CRN — correctness**

- Replaced the float-kernel basis behind ``integer_conservation_laws`` with an
  exact integer basis of ``ker(S^T)`` computed by rational row reduction. The
  previous implementation returned vectors that were not conservation laws at
  all on networks such as the two-site phosphorylation cycle.
- Added ``conserved_moieties`` for the non-negative, inclusion-minimal
  conservation laws — the moiety pools with a chemical reading.
- Fixed ``CRNVis``: its default palette named a palette that does not exist, so
  constructing it with defaults always raised, and every domain layout rejected
  the keyword arguments ``compute_layout`` passes, so no CRN could be drawn.
- Made ``synkit.CRN.Visualize.validation.node_sort_key`` recognise reaction
  nodes through the shared node-kind vocabulary, so ``kind="reaction"`` and
  ``kind="rule"`` graphs draw identically.
- Made node ordering natural (digit-aware), so ``r_2`` precedes ``r_10`` and a
  graph round-trip no longer permutes a network's reactions.

**CRN — new capabilities**

- Added ``synkit.CRN.Props.deficiency``: complexes, linkage classes, strong and
  terminal strong linkage classes, weak reversibility, reversibility,
  deficiency, per-linkage-class deficiencies, and the Deficiency Zero and
  Deficiency One theorems. Ranks are exact over rationals.
- Added ``synkit.CRN.IO``: self-contained SBML Level 3 Version 2 import and
  export, for interoperability with ``crnpy``, CRNT4SBML, CoNtRol and COPASI.
  No ``libsbml`` installation is required.
- Added ``synkit.CRN.Query.to_syncrn``: build a ``SynCRN`` directly from KEGG
  equations or a KEGG module, with reversible-equation expansion and optional
  currency-metabolite removal.
- Added ``synkit.CRN.Benchmark``: a validation set with independent
  cross-checks, a scaling benchmark, and a reproducible KEGG case study over
  four cached metabolic modules. ``Experiment/CRN/run_all.py`` regenerates
  every reported result.

**CRN — API**

- ``SynCRN.from_reaction_strings`` now mints the same prefixed ids as
  ``from_digraph`` (``s_1`` / ``r_1`` / ``rule_1``). The legacy shared-namespace
  scheme remains available as ``id_style="numeric"``. **Breaking** for code that
  hard-coded the old numeric ids.
- ``SynCRN.to_stoichiometric_matrices`` now returns ``numpy`` arrays instead of
  nested Python lists, and accepts ``sparse=True`` for ``scipy.sparse`` output
  and ``dtype=`` for the element type. **Breaking** for code relying on list
  semantics.
- Removed ``synkit.Vis.crn.CRNVisualizer``, which plotted a hypergraph object
  that no longer exists. ``synkit.Vis.crn`` now re-exports the single CRN
  visualizer from ``synkit.CRN.Visualize``.
- Moved ``numpy`` from the ``all`` extra to the core dependencies, where the
  code has always assumed it.
- Split ``synkit/CRN/Structure/syncrn.py`` into ``_parse``, ``_graph_io`` and
  ``_matrices`` alongside the ``SynCRN`` class.

Version 1.6.2
-------------

- Made mechanism replay charges authoritative from committed electron-resource
  deltas and separated exact Lewis-state validity from invariant-residual
  ``DELTA_CONSISTENT`` verification.
- Added explicit, atom-selected ``closed_shell_pair`` endpoint comparison with
  provenance-bearing ``NORMALIZED`` certificates; strict resource comparison
  remains the default.
- Deprecated ``LWGEditor.matches_product`` as exact verification because it is
  only a projected charge/canonical-SMILES match.

Version 1.6.1
-------------

- Fixed relative lone-pair updates in tuple rules and added replay regressions.
- Reduced duplicate work in hydrogen extension, matching, and graph rewriting.
- Added retained stereo, MechanismBench, and FLOWER benchmark metadata.
- Made validation and replay diagnostics portable across supported platforms.

Version 1.6.0
-------------

**Synthesis**

- Added an opt-in ``serialization_errors="skip"`` policy to
  ``SynReactor`` raw ITS mode. Valid serializations retain their application
  order, while one structured warning and a cached diagnostic expose every
  omitted raw-application index. The compatibility default remains
  ``"raise"``.

**Lewis-labelled graphs**

- Added Lewis-labelled graph states with explicit lone-pair, radical,
  sigma-bond, and pi-bond resources and derived charge and bond-order fields.
- Added resource-aware matching and rewriting with explicit availability and
  policy-validity checks.
- Standardized tuple-rule lone-pair changes as relative resource edits.
  For example, an extracted ``S(lp2) -> S(lp1)`` endpoint change is stored as
  ``S(lp1) -> S(lp0)`` (consume one), so matching a host with one, two, or
  three lone pairs produces a host with zero, one, or two, respectively.

**Arrow-pushing grammar**

- Added locus-sorted two-electron curved-arrow moves and atomically coupled
  one-electron fishhook groups.
- Added executable polar transition classes and radical macros with
  integrality, locality, resource-availability, and endpoint replay checks.
- Added deterministic mechanism verification with structured diagnostics for
  invalid or inconsistent supplied electron-flow annotations.

**Native graph backend**

- Removed the external MØD-backed reactor, rule-composition, CRN,
  visualization, and persistence adapters.
- Standardized supported execution paths on the native NetworkX graph stack.
- Retained GML as a supported serialization format through the native readers
  and writers.

Version 1.5.0
-------------

**Atom-to-atom mapping**

- Replaced the former monolithic ``wl_mapper`` module with the modular
  ``synkit.Chem.Reaction.Mapper`` package. The public entry point is
  ``AAMapper``, which combines WL/SLAP mapping with optional exact
  reaction-centre refinement, symmetry-distinct enumeration, and certificates.
- Added mapped-reaction ITS hashing, mapped-reaction deduplication, and
  electron-balance checks to the mapper chemistry layer.
- Added hydrogen-count-aware ranking and reaction-centre-only explicit-H
  output for mapped reactions.
- Made ``scipy>=1.14.0`` a required dependency because the WL/SLAP mapper and
  exact refinement use SciPy's linear-assignment solver.

**Synthesis**

- Expanded ``RBLEngine`` with explicit ``fast_track``, ``fast_fusion``,
  ``early_stop``, ``full``, and proof-bearing ``verified`` execution modes,
  pluggable exact or approximate MCS matching, and wildcard-aware ITS fusion.
- Hardened fusion identity for isotopes and Lewis state, required
  side-symmetric mapped atoms, and separated coarse matcher labels from the
  stricter categorical-interface contract; hydrogen presentation remains an
  explicit completed-graph proof obligation.

**EF-SMIRKS conversion**

- Added ``ef_smirks_to_epd`` and ``epd_to_ef_smirks``. The forward helper
  preserves flow-code maps, completes AAM, and returns generic EPD plus typed
  ``epd_lw`` records; the reverse helper reconstructs EF-SMIRKS from complete
  AAM and either EPD representation.
- Exposed these helpers from both ``synkit.IO`` and ``synkit.IO.conversion``.

**Electron-pushing visualization**

- Added an EF-SMIRKS-to-EPD visualization workflow using
  ``MechanismVisualizer``. It renders the completed AAM, typed ``epd_lw``
  flow, product, and ITS changes in one trajectory figure.
- Refined trajectory layout, arrow styling, spacing, and legends. Step-number
  bubbles, ITS bond-pair labels, and electron-state badges are now opt-in so
  report figures remain compact by default.

**Documentation and user interface**

- Added a versioned EF-SMIRKS/EPD figure and API examples for direct use from
  ``synkit.IO``.
- Improved dark-mode navigation contrast and added a dedicated dark-theme logo.

**Highlights**

- Added the Lewis State Graph (LSG) reactor representation, graph-native
  functional-group detection, compact MTG timelines, and modern molecule,
  reaction, ITS, and MTG visualization helpers.

**Compatibility**

- ``AAMValidator`` remains available from ``synkit.Chem.Reaction`` as a
  backward-compatible import path. New mapper code should import public
  classes from ``synkit.Chem.Reaction.Mapper``.

Version 1.4.0
-------------

**Highlights**

- Added the Lewis State Graph (LSG) framework for ``SynReactor``. LSG
  templates carry ``lone_pairs``, ``radical``, ``valence_electrons``,
  ``sigma_order``, ``pi_order``, and ``kekule_order`` so the NetworkX reactor
  can rewrite from explicit valence-state information while keeping the legacy
  ``typesGH`` path available.
- Added graph-native functional-group detection under ``synkit.Graph.FG``.
  The detector works directly on SynKit molecular ``networkx`` graphs and
  provides a SMILES convenience API returning both the graph and detected
  ``(name, atom_indices)`` labels.
- Added compact MTG and visualization helpers for LSG/ITS and MTG timeline
  inspection. The modern Vis API now covers molecule graphs, reaction panels,
  ITS-only drawings, Lewis-state labels, and MTG step/timeline panels.

**Lewis State Graph reactor**

- LSG matching now uses explicit valence-state fields for new-mode templates:
  element, charge, lone-pair count, radical count, and bond changes represented
  by ``sigma_order`` / ``pi_order`` / ``kekule_order``.
- Product charge recomputation is driven from Lewis-state accounting in
  new-mode rewrites, with ``kekule_order = sigma_order + pi_order`` used
  instead of aromatic ``order`` values.
- Hydrogen handling was tightened for explicit-H reaction centers, implicit-H
  templates, and simple ``H-H`` transfer cases.
- Atom-map preservation for LSG-reactor SMARTS output was fixed by using graph
  node identity where the template does not carry original AAM.
- Real-case regression tooling was added around the first smart-database
  fixture, batch round trips, and previously failing LSG rewrite examples.

**Functional groups**

- Added ``FunctionalGroupDetector``, ``FunctionalGroupRegistry``,
  ``FunctionalGroupAudit``, and
  ``smiles_to_graph_and_functional_groups``.
- Added hierarchical family handling so more specific labels such as
  ``carboxylic_acid`` suppress generic nested labels such as ``carbonyl`` when
  appropriate.
- Added aromatic ring-system detection, selected fused heteroaromatic public
  names, and transform-relevant families across carbonyl/acyl, oxygen,
  nitrogen/C=N, sulfur, boron, silicon, and phosphorus chemistry.
- Replaced the previous ``fgutils`` usage in tautomerization support with the
  SynKit-native functional-group API.

**MTG**

- MTG construction from RSMI strings now defaults to Lewis State Graph ITS,
  producing compact atom and bond timelines without ``typesGH``. Use
  ``its_format="typesGH"`` to request legacy string conversion.
- Reworked the MTG plan around LSG/ITS representation: invariant atom fields
  are stored once, while temporal fields store compact histories across
  mechanism snapshots.
- Added round-trip coverage for converting reaction sequences to MTG and back
  to ordered ITS steps / composed ITS views.
- Marked aromatic relabeling and partial-order mechanism DAGs as active design
  areas rather than solved MTG semantics.

**Visualization**

- Added ``draw_molecule_graph``, ``draw_reaction_graph``,
  ``draw_its_from_rsmi``, ``draw_its_only``, ``draw_mtg_graph``, and
  ``draw_mtg_steps`` as the preferred modern rendering helpers.
- Added compact LSG/ITS labels for ``kekule_order`` transitions and optional
  ``sigma/pi`` labels that suppress unchanged components.
- Added selectable Lewis-state labels for charge, lone-pair, and radical
  changes.
- Added Matplotlib ``Agg`` smoke tests for molecule, reaction, ITS, visual
  adapter, and MTG drawing paths.

**Compatibility and known limits**

- Legacy ITS / ``typesGH`` behavior remains available for existing workflows.
- Aromatic LSG matching is still conservative. Some aromatic false-positive
  or false-negative cases require a future aromatic-system relabeling policy
  rather than a local matcher tweak.
- Functional-group fused positional isomers such as quinoline vs isoquinoline
  are not fully distinguished yet.

**Infrastructure**

- Added ``networkx>=3.3`` as a direct dependency for graph algorithms.


Version 1.1.1
-------------

**Bug fixes**

- Automorphism and AutoEst now prefer orbits with the largest anchor set.
- Added ``deduplicate_matches_with_anchor``: anchor a connected component and deduplicate remaining components.

**New features**

- **CRNCanonicalizer (Bliss-style)**: reimplemented canonicalization using a Bliss-inspired strategy; ~10× faster on large CRNs.
- **WLCanonicalizer**: Weisfeiler–Lehman–based approximate canonicalization for CRNs; fast orbit approximations for large/noisy networks.

**Known issues**

- ``CRNAutomorphism`` may not return fully correct automorphism groups in all cases.


Version 1.1.0
-------------

**Features**

- Lightweight CRN exploration (pure Python).
- CRN property analysis utilities (stoichiometric summaries and structural checks).
- CRN promoted to a dedicated submodule (:py:mod:`synkit.CRN`).
- Approximate automorphism + MCS to speed up symmetry-aware computations.


Version 0.0.7
-------------

**Highlights**

- Refactored source-code structure into six primary submodules at the root level:  
  `IO`, `Chem`, `Graph`, `Rule`, `Synthesis`, and `Vis`.  

IO Module
^^^^^^^^^

- Exposed core I/O utilities directly in `synkit.IO`:  
  `chemical_converter.py`, `data_io.py`, and `debug.py`.  

Chem Module
^^^^^^^^^^^

- Introduced **`CanonRSMI`** for atom–atom mapping (AAM) canonicalization.  
- Moved **`AAMValidator`** into `synkit.Chem.Reaction` for consistency.  

Graph Module
^^^^^^^^^^^^

- Added **`SynGraph`** wrapper for reaction and molecule graphs.  
- New canonicalisation backends:  
  - **node‐type sort**  
  - **Morgan‐prime hashing**  
  - **Weisfeiler–Lehman refinement**  
- Renamed “Cluster” to **Matcher**; enhanced **`GraphMatcher`** and **`SubgraphMatch`**.  
- Added **`SubgraphSearchEngine`** with three strategies:  
  - `component‐aware`  
  - `arbitrary`  
  - `backtracking`  
- Introduced **`SING`** and **`TURBOIS`** for mapping multiple patterns in a single host graph.  
- Extended **`GraphCluster`** and **`BatchClustering`** native graph support.
- Enhanced **`WLHash`** to hash lists of node/edge attributes.  
- Added **`MTG`** submodule for Mechanistic Transition Graphs (direct rule composition).  
- New **`Hydrogen`** submodule for reaction-center H-completion and **`Context`** for radius-based expansion.  

Rule Module
^^^^^^^^^^^

- Introduced **`SynRule`** wrapper supporting NetworkX graphs and GML.  
- Reorganized into three packages:  
  - **Apply** (retro-prediction via partial composition)  
  - **Compose** (rule composition)  
  - **Modify** (rule editing and H-handling)  

Synthesis Module
^^^^^^^^^^^^^^^^

- Divided into native reactor, CRN, and multi-step pathfinder submodules.
- **`SynReactor`** now supports implicit‐H templates.  

Vis Module
^^^^^^^^^^

- Visualization tools organized under **`synkit.Vis`**:  
  - **`RXNVis`** (reaction visualisation)  
  - **`RuleVis`** (template/rule visualisation)  
  - **`GraphVisualizer`** (generic graph editing & display)  

Documentation
^^^^^^^^^^^^^

- Added comprehensive examples for each submodule.  
- Scaffolding for an API Reference page.  
