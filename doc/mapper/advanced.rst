.. _legacy-mapping-apis:
.. _python-mapping-apis:

Python interfaces
=================

See :doc:`/mapper` for installation, C++ PABS with SLAP starting mappings,
implicit-H scoring, PMCD and exact-CD enumeration. This page covers the Python
interfaces, graph workflows, certificates and reference analysis.

Python mapper APIs
~~~~~~~~~~~~~~~~~~

The Python interfaces accept reaction SMILES.
``map_reaction(...)`` defaults to full-space exact Python propagation search.
``backend="cpp"`` selects the optional C++ full-space solver.
``AAMapper().map_smiles(...)`` defaults to approximate WL refinement and
sequential linear-assignment matching (SLAP). Select the entry point according
to the workflow you need.

LAP means linear assignment problem, a matching subproblem used as a building
block. SLAP uses successive LAP decisions to construct heuristic mappings.
PABS means Propagation and Assignment-Bounded Search, this package's descriptive
name for the exact-search interface used by ``map_reaction``. A kernel is a
reduced uncertainty problem extracted from supplied SLAP results, with fixed
assignments and candidate domains. An exact solver can refine that problem.

On ``AAMapper``, ``enumerate_exact=True`` selects SLAP followed by exact kernel
refinement. ``certify="exact"`` requests optional kernel MILP certification.
``enumerate_smiles`` or a supplied ``CD`` selects full-distance exact search,
where SLAP may supply a seed for a minimum query. An exact kernel optimum is
scoped to the supplied restrictions. A complete full-space search establishes
the global result under the declared distance objective.

.. code-block:: python
   :caption: Map an unmapped reaction SMILES
   :linenos:

   from synkit.Chem.Mapper import AAMapper

   mapper = AAMapper(binary=True)
   mapper.map_smiles(
       "CC(=O)O.CO>>CC(=O)OC",
       unique=True,
       electron_balance=True,
   )

   for result in mapper.results:
       print(result["smiles"], "chemical distance:", result["cd"])

``mapper.results`` is a list of mapping records. Each record includes the
mapped reaction in ``smiles``, the mapped ITS form in ``its_smiles``, and the
chemical-distance score ``cd``. Request ``enumerate_exact=True`` when a
reaction centre is symmetric and distinct optima within the supplied kernel
are needed.
``certify=True`` attaches corresponding certificate metadata.

For a complete search over every atom-compatible assignment, use
``enumerate_smiles``. ``CD="minimal"`` returns every globally minimal mapping.
Any finite non-negative numeric value selects that exact distance shell.

.. code-block:: python

   search = mapper.enumerate_smiles(
       "CCC>>CCC",
       CD=8,
       add_Hs=False,
       time_limit_seconds=30,
       certify=True,
   )

   if search.status == "no_solutions":
       print("the complete exact-CD shell is empty")
   elif search.status == "timeout":
       print("partial search. Absence of further mappings is not proven")

``search.complete`` is true for both a fully enumerated non-empty shell and a
proven-empty ``no_solutions`` shell. A deadline instead returns
``status="timeout"`` and ``complete=False``. ``max_bijections`` is checked
before search and raises explicitly when the atom-compatible assignment space
is beyond the requested cap. Empty numeric shells above
``search.maximum_cost_upper_bound`` are certified immediately by that bound.
The ``enumerate_exact=True`` mode enumerates the uncertainty kernel and has a
narrower proof scope.

By default, numeric shells use a memory-constant optimization pass first, so
``search.minimum_cost`` is the proven global minimum rather than merely the
lowest mapping encountered below the requested shell. Set
``compute_minimum_cost=False`` when only the requested numeric shell matters.
This avoids the separate optimization search and leaves ``minimum_cost`` as
``None``. The enumeration pass uses element-block assignment bounds on both
sides of the target: a subtree is rejected when it can neither stay below nor
rise high enough to reach the requested CD.

When a shell may contain many mappings, stream it without retaining the output
list. The callback receives only final mappings from the proven shell. Minimal
queries therefore optimize first and never emit provisional incumbents.

.. code-block:: python

   def consume(mapping, distance):
       print(mapping, distance)

   search = mapper.enumerate_smiles(
       "CCC>>CCC",
       CD="minimal",
       add_Hs=False,
       collect_mappings=False,
       mapping_callback=consume,
   )
   print("exact mapping count:", search.selected_mapping_count)

In this mode ``search.mappings`` remains empty. The current prefix-certificate
format requires ``collect_mappings=True`` because its digest binds the complete
sorted mapping set.

With ``certify=True``, ``search.certificate`` is a JSON-serializable terminal
prefix cover. Its verifier reconstructs the atom-compatible decision tree,
checks that the prefixes are disjoint and exhaustive, independently recomputes
every partial-CD pruning bound, and binds the selected mappings by SHA-256.

.. code-block:: python

   import json
   from synkit.Chem.Mapper import verify_distance_enumeration_certificate
   from synkit.Chem.Mapper.chem.smiles import smiles2lgp

   record = json.loads(json.dumps(search.certificate.as_dict()))
   original_lgp = smiles2lgp("CCC>>CCC", add_Hs=False)
   verified = verify_distance_enumeration_certificate(
       original_lgp,
       record,
       mappings=search.mappings,
   )

Timeout results carry an incomplete certificate whose disjoint
``frontier_prefixes`` cover every unresolved assignment subtree. The same
verifier checks the union of terminal and frontier prefixes, but the nonempty
frontier forces ``status="timeout"`` and cannot establish shell completeness.

Set ``symmetry_pruning=True`` to retain certified lex leaders under a bounded
verified subgroup of product automorphisms. At each prefix, a Schreier
stabilizer fixes the images already selected and branching visits one candidate
per resulting orbit. Generator and work limits are memory bounded. Exceeding a
limit disables deeper symmetry reduction instead of making the result
incomplete. Bounded discovery may still leave multiple representatives from a
full product-automorphism orbit. SynKit's internal exact canonicalizer supplies
concrete automorphism witnesses. No external nauty binding is required. Every
symmetry-pruned prefix is stored with an exact transporter, and the verifier
checks atom types and the complete directed product matrix before accepting the
reduction. Remaining-assignment lower and upper bounds are replayed
independently as well.

For the complete labeled shell with symmetry acceleration, also set
``expand_symmetry=True``. SynKit then restricts orbital pruning to one fully
enumerated cyclic subgroup generated by a verified product automorphism and
streams every group image of each representative. The subgroup action on
bijections is free, so expansion contains every labeled mapping exactly once
even when the subgroup is smaller than the full automorphism group. This mode
is currently incompatible with prefix certificates.

For an auditable automatic choice between exact algorithms, call
``enumerate_hybrid_distance_mappings``. A numeric binary labeled shell uses
the edit-support isomorphism backend only when its exact support-pair estimate
is below the configured limit. Minimal, weighted, fixed-map, certificate, and
symmetry-quotient requests stay on assignment branch-and-bound. The returned
``backend`` and ``backend_statistics`` fields record the decision and never
change the requested output scope.

Python exact propagation search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``enumerate_pabs_mappings(..., backend="python")`` selects the same Python
engine used by ``map_reaction`` by default. It propagates cost-filtered
assignment domains, removes edges unsupported by any perfect matching, and
reuses repaired integer
assignment certificates for adaptive bounds.
``enumerate_distance_mappings`` provides Python assignment search.

.. code-block:: python

   from synkit.Chem.Mapper import enumerate_pabs_mappings

   result = enumerate_pabs_mappings(
       [reactant_graph, product_graph], backend="python", CD="minimal",
       binary=False,
       initial_mapping=feasible_seed, symmetry_pruning=True,
       expand_symmetry=True, time_limit_seconds=60,
   )

The Python engine supports undirected, loop-free half-integer bond matrices
with at most 256 atoms and a bounded integer cost range. Other graph inputs
and requests for tree-cover certificates use an explicit fallback,
recorded in ``backend`` and ``backend_statistics``. ``PropagationConfig``
allows controlled comparisons of domain propagation, adaptive bounds, and
incremental assignments, cached support filtering, assignment blocks, typed
bond-mass bounds, forced-column bounds, and alternating-cycle cost bounds.
The experimental ``branch_order="pagerank"`` option changes only row tie
breaks. It does not remove candidates.
Symmetry discovery and subgroup closure share a bounded setup budget. Any
subgroup retained after a deadline or size limit is fully closed and valid.
Each assignment bound retains an independently checked integer primal/dual
certificate. The verifier checks every supported-edge dual inequality within
an explicit range that prevents integer overflow. Independent atom-type blocks
can have different dual offsets. Unsupported cross-type edges are excluded
from the certified assignment domain.

The alternating-cycle bound computes the cheapest complete assignment that
contains each candidate edge. It reuses a certified matching and completes
the shortest-path calculation before applying any rejection. Deadline checks
cover cyclic-subgroup selection and Hungarian augmentations as well as search
nodes. An unfinished bound reports interruption rather than infeasibility.

Completion and proved minimum are separate fields.
An interrupted enumeration can retain a proved minimum and partial output.

Mapper source layout and C++ graph search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mapper's Python source separates mapping methods from their consumers.
``chem/``, ``graph/``, ``solvers/slap/``, and ``io/`` provide chemical conversion,
graph representation, approximate assignment, and mapping serialization.
``reporting/`` contains shell analysis, ITS spectra, and structural alternatives.
``validation/`` contains AAM checks and annotation evaluation.
``adapters/`` handles prediction seeds and executable templates.

Inside ``solvers/exact/``, ``distance_search/`` contains full-space distance enumeration,
``kernel_solvers/`` contains exact refinement of a supplied WL/SLAP uncertainty
kernel, ``pabs/`` contains Python propagation search, and ``native/`` contains
the optional C++ implementation and its bindings. ``assignment/`` provides
shared LAP domains and proofs. ``seeds/`` provides feasible-incumbent heuristics.
Kernel optimality applies to the supplied restrictions. A complete full-space
search establishes the global result under its declared distance objective.
The source guide at ``synkit/Chem/Mapper/README.rst`` compares the entry points.

Package-level imports and historical flat module imports continue to work.
New implementation code uses the grouped canonical paths.

The existing C++17 source is
``synkit/Chem/Mapper/solvers/exact/native/native_distance.cpp``. It implements native
atom-assignment search, distance-bound pruning, symmetry handling, and
resumable subtree enumeration. Python loads the compiled shared library
through ``ctypes`` in ``solvers/exact/native/candidates.py``.
``reporting/native_analysis.py`` coordinates native shell analysis. This graph
interface requires an explicit library path. The bundled
shared core is selected through ``Mapper("native")`` or
``map_reaction_implicit``. See :doc:`/mapper`.
``enumerate_distance_mappings`` runs the Python engine.

Build the optional engine from a checkout or installed package:

.. code-block:: console

   python -m synkit.Chem.Mapper.solvers.exact.native.build --output-dir /tmp/synkit-native

The command prints the library path and records source, compiler, flags,
and binary hashes beside it.

Tests follow the same component layout under ``Test/Chem/Mapper/``:
``api/``, ``chem/``, ``graph/``, ``slap/``, ``exact/``, ``io/``, and
``studies/``. Native tests share a session fixture in ``conftest.py`` and
compile once. ``SYNKIT_TEST_NATIVE_LIBRARY`` can select a prebuilt library.

.. code-block:: console

   python -m pytest Test/Chem/Mapper

Exact alternative ITS classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``enumerate_mapped_reaction_its_alternatives`` turns a completed shell into an
auditable reference-relative application. It returns one deterministic AAM
representative for every exact canonical ITS class. The ``classes`` export
includes the reference class. ``alternatives`` contains the classes that
differ from the supplied reference ITS. The result reports the number of retained
representative-scope mappings and, when the verified subgroup order is known,
the corresponding labeled mappings in each class.

.. code-block:: python

   from synkit.Chem.Mapper import (
       GlobalShellConfig,
       enumerate_mapped_reaction_its_alternatives,
   )

   mapped_reaction = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][OH:3]"
   alternatives = enumerate_mapped_reaction_its_alternatives(
       mapped_reaction,
       CD="reference",  # also "minimal" or any non-negative numeric CD
       seed_mode="reference",
       config=GlobalShellConfig(
           time_limit_seconds=300,
           max_bijections=None,
       ),
   )

   if alternatives.shell.complete:
       for candidate in alternatives.as_dict()["shell"]["alternatives"]:
           print(candidate["its_class_id"], candidate["atom_map_correspondence"])

``CD="reference"`` uses the reference's scalar distance but still searches all
atom-compatible bijections at that distance. ``seed_mode="reference"`` may use
the map as an incumbent and ordering hint. It never fixes atoms or removes
candidates. A supplied seed is not guaranteed to reduce runtime because exact
shell enumeration remains output-sensitive. ``complete`` is true only when
both the global shell and every exact ITS classification have completed.
The exact class identifiers are seed-invariant. A labeled representative may
differ between seed modes while remaining in the same certified class.
Alternative classes can be used as controlled contrastive or hard-negative
examples, but they are not automatically chemically impossible mechanisms.

The mapper can represent hydrogens at three levels through ``add_Hs``:

- ``False`` keeps hydrogens implicit.
- ``True`` maps fully explicit hydrogens.
- ``"reaction_center"`` exposes only hydrogens involved in the reaction
  centre, which is generally the clearest output for inspection.

For a fixed heavy-atom mapping, use
``enumerate_lgp_hydrogen_transfers`` to enumerate every exact minimum
hydrogen flow between donors and acceptors from the per-atom ``hcounts`` metadata. It also
computes the labeled explicit-hydrogen multiplicity without constructing
hydrogen nodes. Unequal total hydrogen inventories are rejected, and a plan
cap returns ``complete=False``.

For comparison against a reference mapping, use ``AAMValidator``. Its
``smiles_check`` method accepts either ``"RC"`` (reaction centre) or
``"ITS"`` matching and can be called from an instance to set a default policy
for unbalanced atom maps.

.. code-block:: python

   from synkit.Chem.Mapper import AAMValidator

   validator = AAMValidator(strip_unbalanced_maps=True)
   equivalent = validator.smiles_check(candidate, reference, check_method="ITS")

PABS graph backends
-------------------

PABS (Propagation and Assignment-Bounded Search) is this package's descriptive
name for the exact graph-assignment search interface. The default Python solver
uses reversible one-to-one assignment domains and LAP lower bounds. The C++
implementation uses integer-shell search with a distinct traversal. Both
search compatible assignments without extracting the reduced SLAP uncertainty
kernel or invoking its MILP solver. Historical API names remain supported.

.. code-block:: python

   from synkit.Chem.Mapper import enumerate_pabs_mappings
   from synkit.Chem.Mapper.solvers.exact.native.build import build_native

   result = enumerate_pabs_mappings(lgp, backend="python", CD="minimal")
   library = build_native("/tmp/synkit-native")  # requires a C++17 compiler
   native = enumerate_pabs_mappings(
       lgp, backend="cpp", library_path=library, CD="minimal",
       time_limit_seconds=10,
   )
   shell = enumerate_pabs_mappings(
       lgp, backend="cpp", library_path=library, CD=4,
       compute_minimum_cost=False, time_limit_seconds=10,
   )

The C++ source is ``synkit/Chem/Mapper/solvers/exact/native/native_distance.cpp``.
Python bindings and shell runners live in ``solvers/exact/native/candidates.py``
and ``solvers/exact/native/search.py``. Builds are explicit, hashed and accompanied by a
compiler/source manifest. C++ never silently falls back to Python.

The shared options include binary or weighted distance, minimum or numeric CD,
seeds, time and output caps, and streaming callbacks. C++ returns labeled
mappings without symmetry reduction and supports 1 to 256 atoms, half-integer
weights, at most sixteen distinct bond levels and bounded integer distance.
Fixed assignments, symmetry controls, propagation configuration and replayable
tree-cover certificates currently require the Python backend. Unsupported C++
options raise an error. Python's existing out-of-scope fallback is recorded in
result metadata. Inspect ``complete``, ``minimum_cost`` and ``truncation_reason``
separately: proving a minimum does not imply completing its mapping shell.

C++ proves a minimum by searching half-unit shells in increasing order. This
can be slower than the Python optimization pass for a high minimum. No speedup
is promised without comparing the same cases, budgets and output scopes.

Unmapped reaction SMILES
------------------------

``map_reaction`` connects PABS search to unmapped reaction SMILES and returns
one mapped reaction per exact colored ITS symmetry class. Several inequivalent
classes can share the same minimum. Both Python and C++ return the same result
structure. The C++ choice needs an explicitly built library.

.. code-block:: python

   from synkit.Chem.Mapper import map_reaction
   from synkit.Chem.Mapper import PropagationConfig
   from synkit.Chem.Mapper.solvers.exact.native.build import build_native

   result = map_reaction("CCC>>CCC", hydrogen="heavy")
   print(result.mapped_reactions)
   print(result.complete, result.minimum_cost)

   # Experimental exact separator bound and selected-CD spectrum search.
   result = map_reaction(
       "CCO>>CC=O",
       search_config=PropagationConfig(separator_spectrum=True),
   )

   # Joint optimization over heavy atoms and all explicit hydrogens.
   result = map_reaction("CO>>C=O", hydrogen="explicit", balance="dummy")

   # Lewis-style distance with explicit H and opt-in missing-atom padding.
   library = build_native("/tmp/synkit-native")
   result = map_reaction(
       "O>>[OH-]", backend="cpp", library_path=library,
       hydrogen="explicit", objective="lewis", balance="dummy",
       time_limit_seconds=10,
   )
   print(result.mapped_reactions)
   print(result.reactions[0].reactant_only_maps)

The default objective is weighted bond-order distance on heavy atoms. Set
``binary=True`` to count bond presence only. Hydrogen options are:

* ``heavy``: H stays implicit. It contributes to class attributes but not cost.
* ``explicit``: include all H vertices and minimize their bond changes jointly
  with heavy-atom changes. Isotopic and free hydrogens are supported here.
* ``compressed``: enumerate minimum-H parent flows for each selected heavy
  mapping, then deduplicate the resulting explicit-H representatives. This is
  conditional optimization, not the global explicit-H objective.

``search_config`` accepts a ``PropagationConfig`` for the Python backend. Its
``separator_bounds`` option is experimental, disabled by default, and has not
shown a general runtime improvement. Use it for controlled comparisons only.
``separator_spectrum`` enables exact cost-support decomposition for eligible
small residuals and reconstructs mappings only in the requested CD shell. It
also remains opt-in and falls back to PABS DFS when its state budget is reached.
``factor_spectrum_bounds`` applies a cheaper factor-wise bitset relaxation to
reject residual CD intervals only when their relaxed support is empty. It is
sound but inconclusive when support remains, is disabled by default, and did
not prune in the initial five-reaction diagnostic pilot.
``suffix_spectrum`` builds an exact bounded decision diagram for eligible
small residuals. Its states merge prefixes only when the used product atoms
and every accumulated future row/image cost agree. The stored cost support can
serve repeated shell queries on the same diagram. State-cap exhaustion falls
back to PABS DFS before any mappings are emitted. This option is experimental
and disabled by default. Paired development benchmarks did not justify
enabling the reward-frontier variant by default. Preparation is limited to
two calls per search, 20,000 states per call, and 0.002 seconds per call by
default. Exceeding a cap abandons that spectrum and resumes DFS.

The optional ``suffix_spectrum_representation="reward_frontier"`` stores used
product atoms and the assigned images incident to future source bonds. It
uses the same exact suffix cost objective and bounded fallback contract.
``suffix_spectrum_orbit_pruning`` also remains disabled by default.
Ordinary Python PABS includes the verified free product-group emission
improvement. This changes output overhead without changing search bounds or
the requested mapping shell.

For several numeric CDs on the same small reaction pair, call
``enumerate_pabs_shells(lgp, CDs)`` to prepare one exact suffix spectrum and
query all requested shells from it. The result maps each normalized CD to a
``DistanceEnumerationResult``. If the residual is too large or preparation
reaches a resource cap, each shell uses the ordinary exact PABS search under
the remaining shared time budget. This batch interface currently covers
numeric weighted/binary heavy-atom CDs. It does not prove a minimum or perform
symmetry quotienting. For example:

.. code-block:: python

   from synkit.Chem.Mapper.solvers.exact import enumerate_pabs_shells

   shells = enumerate_pabs_shells([reactant_graph, product_graph], [0, 1, 2])
   for cd, result in shells.items():
       print(cd, result.status, len(result.mappings))

For compressed mode, ``minimum_cost`` refers to the heavy-atom search. Each
representative's ``distance`` includes its additional ``hydrogen_distance``.
The compression assumes balanced indistinguishable ordinary H attached by unit
bonds to heavy parents. Parent-flow matrices avoid enumerating factorially many
permutations of equivalent H atoms, and exact full-ITS canonicalization also
identifies equivalent heavy-parent choices.

``objective="lewis"`` requires ``hydrogen="explicit"`` and ``binary=False``.
Its declared cost is
``sum(i<j, abs(A[i,j]-B[f(i),f(j)])) + sum(i, abs(e[i]-e'[f(i)])/2)``,
where ``e = 2*estimated_lone_pairs + radical_electrons`` uses SynKit's existing
RDKit lone-pair estimator. A uniquely typed anchor represents the unary terms
as weighted edges, making this objective available to both existing backends.
This is a defined Lewis-state objective for the supplied structures, not an
assertion about unique physical electron arrangements or reaction mechanisms.

``balance="strict"`` rejects unequal element/isotope inventories in the searched
representation. ``balance="dummy"`` adds isolated, zero-electron placeholders
for missing atoms. Reported optima apply to that augmented problem. Exported
SMILES contains only real input molecules. ``reactant_only_maps`` and
``product_only_maps`` identify missing counterparts. No reagent is inferred.
Compressed H requires strict heavy-atom and H balance.

Deduplication uses exact full colored ITS codes, equivalent to the two-sided
reactant/product automorphism action. Node colors retain isotope, charge,
radical, H count, CIP annotations and placeholder status. Edge colors retain
endpoint E/Z annotations. This equivalence is defined on these colored graphs.
A class receives an identifier only after exact canonicalization succeeds.

The wrapper accepts both ``R>>P`` and ``R>agents>P``. Agents are returned separately
and excluded from search. Existing atom maps are cleared. They do not constrain
mapping. Atom-index mappings refer to parsed atom order, followed by added H,
missing-atom placeholders and, for Lewis mode, the internal anchor. The map
numbers in the exported reaction are the convenient public correspondence.

``complete`` requires search, hydrogen lifting and classification completion.
Inspect their separate flags and ``incomplete_reason`` when a deadline or cap
is reached. ``max_mappings`` limits labeled search output, not unique classes.
``max_hydrogen_plans`` limits flows per heavy mapping. All stages share the
wall-time budget. ``max_bijections=None`` explicitly disables the factorial
preflight cap for large searches. The objective describes graph changes and
does not guarantee a mechanistically preferred mapping.
