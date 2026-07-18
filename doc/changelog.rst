Changelog
=========

Version 2.0.0
-------------

**Beta 2 release checkpoint**

- Prepared the ``2.0.0b2`` prerelease with typed wildcard-aware graph
  morphisms and verified, proof-bearing graph fusion.
- Added deterministic exhaustive fusion search, exact structural/stereo
  deduplication, endpoint certificates, and explicit completeness reporting.
- Optimized interface discovery and proof reuse while retaining the Beta 1
  compatibility projections.
- Enforced typed stereo-ligand ports through owner-local orbit-frame slots,
  chemical/virtual domains, Reactor candidate expansion, and immutable
  morphism proof replay; legacy untyped wildcard rules remain explicit
  compatibility inputs.
- Added a standalone whole-molecule chiral/achiral classifier under
  ``synkit.Chem.Molecule``. It completes eligible sp3 stereo topology and tests
  exact equality with the molecular mirror using element, hydrogen-count, and
  connectivity identity; it is independent of reaction and rule semantics.
- Added bounded, cached configuration-aware molecular chirality assessment for
  stereo-underspecified input, with necessary/configuration-dependent/incomplete
  outcomes and an opt-in strict binary mode that rejects unresolved stereo.
- Consolidated molecule-stereo datasets under a task-aware benchmark registry:
  the ACS whole-molecule set and MIT-licensed ChiralFinder RotA are vendored
  with integrity metadata, while the unlicensed CIP Validation Suite remains
  an external-only descriptor-benchmark reference.

**Beta 1 release checkpoint**

- Prepared the ``2.0.0b1`` prerelease after completing the native-only
  migration, the 80/80/80 MechanismBench data boundary, and stereo rule
  soundness for intentionally non-invertible outcomes.
- Restored RBL explicit-hydrogen/AAM compatibility and made every accepted
  result pass one structured fusion-validation contract.
- Added proof-gated product preservation through SynKit's component-aware
  subgraph matcher and relative-stereo registry, with explicit search scope,
  termination policies, and typed wildcard-role conflicts.

**Native graph stack**

- Removed the external legacy graph-grammar backend and its reactor,
  derivation-graph, rule-composition, CRN, visualization, and persistence
  adapters.
- Standardized reaction rewriting on ``SynReactor``, CRN construction on the
  native builders/``SynCRN``, and matching on NetworkX graph identities.
- Kept GML as a supported serialization format through native parsers and
  graph-isomorphism tests.

**SynReactor execution policy and performance**

- Added the SynReactor-only ``dedup_its`` policy. The default retains exact
  post-rewrite consolidation; raw mode preserves deterministic mapping and
  stereo-branch multiplicity with application provenance.
- Kept mapping-level ``automorphism`` pruning independent from ITS-level
  consolidation and left ``BatchReactor`` on the default consolidated path
  without exposing the new option.
- Separated deferred tuple product-electron finalization from ITS clustering,
  so raw results still carry current electron fields and validated stereo
  registries.
- Reduced tuple rewrite overhead through shallow graph copies, direct common
  electron refresh, native WL hashing, cached stereo preparation, and lazy
  product serialization.

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

- Expanded ``RBLEngine`` with explicit ``fast_track``, ``early_stop``, and
  ``full`` execution modes, pluggable exact or approximate MCS matching, and
  wildcard-aware ITS fusion.

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
