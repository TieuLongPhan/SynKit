.. _atom-to-atom-mapping:

Atom mapping
============

SynKit maps atoms between reaction endpoints and enumerates mappings at a
specified chemical distance (CD). PABS is the exact search engine. SLAP supplies
feasible starting mappings. A SLAP proposal can improve the upper bound, but
does not establish optimality or restrict the exact search.

Install
-------

.. code-block:: console

   python -m pip install synkit

Platform wheels include the C++ core. The release workflow builds wheels for
Linux x86-64 and ARM64, macOS Intel and Apple Silicon, and Windows x86-64.
Installing a matching wheel needs no C++ compiler. These bundled-library
examples require a release containing the native wheels. Other releases may
require an explicit build.

Source installations need CMake and a C++17 compiler to build the wheel.
For an editable checkout, see :ref:`mapper-custom-build`.

Find the solvers available in the installed library:

.. code-block:: python

   from synkit.Chem.Mapper import Mapper

   print(Mapper.native_methods(available_only=True))

Standard wheels contain ten native configurations. CP-SAT, SCIP/MILP, KMILP
and MaxSAT need additional native dependencies. Installing ``synkit[milp]``
enables the Python PuLP interface. It does not enable these native
adapters or the PABS LP bounds.

PMCD and all minimum classes
----------------------------

Use ``map_reaction_implicit`` to search heavy-atom mappings while including
hydrogen changes in the Lewis score:

.. code-block:: python

   from synkit.Chem.Mapper import map_reaction_implicit

   reaction = "CCO>>COC"
   result = map_reaction_implicit(
       reaction, balance="strict", time_limit_seconds=10,
   )

   print(result.minimum_cost)
   print(result.complete, result.incomplete_reason)
   for mapped in result.mapped_reactions:
       print(mapped)

Omitting CD is equivalent to ``CD="minimal"``. The solver proves the minimum
and enumerates one representative per minimum-distance heavy-atom ITS class.
``complete=True`` confirms that search, classification and rendering completed
under the selected objective.

For a minimum proof without class enumeration:

.. code-block:: python

   proof = map_reaction_implicit(
       reaction, proof_only=True, balance="strict", time_limit_seconds=1,
   )
   print(proof.minimum_cost)

This mode does not return mapped reaction classes. A proved scalar minimum
does not establish that its complete solution set has been enumerated.

Exact CD and distance ranges
----------------------------

A numeric CD selects exactly that distance by default:

.. code-block:: python

   if result.minimum_cost is not None:
       shell = map_reaction_implicit(
           reaction, CD=result.minimum_cost, CD_mode="exact",
           balance="strict", time_limit_seconds=10,
       )
       print(shell.complete, len(shell.mapped_reactions))

Set ``CD_mode="at_most"`` to enumerate classes at distance less than or equal
to the supplied CD. Numeric queries do not run a separate minimum proof.
An exact shell can be empty and complete.

Results and limits
------------------

.. list-table:: Reaction result fields
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``minimum_cost``
     - Proved minimum for a minimal query. Otherwise ``None``.
   * - ``mapped_reactions``
     - Returned class representatives as atom-mapped reaction SMILES.
   * - ``complete``
     - All stages requested by this call completed.
   * - ``search_complete``
     - The requested search finished.
   * - ``classification_complete``
     - The required exact class comparisons finished.
   * - ``incomplete_reason``
     - The deadline, output cap, unsupported model or other limit encountered.
   * - ``minimum_cost_scope``
     - The graph and hydrogen model to which the minimum applies.
   * - ``elapsed_seconds``
     - Time spent in the reaction call, including preparation and output handling.

``map_reaction_implicit`` accepts ``time_limit_seconds``, ``max_states``,
``max_mappings``, ``max_classes``, ``max_frontier``, ``memory_limit_bytes``
and ``classification_max_search_nodes``. A supported limit can be disabled
with ``None``. Its default deadline is ``None`` and its default mapping cap
is 10,000. Set a deadline explicitly for batch work.

The deadline covers preparation, search, classification and rendering.
An interrupted call can retain a proved minimum or some mapped classes.
Check completion before treating the returned list as exhaustive.

Chemical model and hydrogen
---------------------------

The implicit-H interface scores bond orders, nonbonding-electron changes and
the minimum hydrogen completion for each heavy-atom mapping. It does not add
hydrogen vertices to the search. Literal H atoms in the input are retained in
rendered molecules without mapped H identities. Supported inputs include
molecular hydrogen, for example:

.. code-block:: python

   hydrogenation = map_reaction_implicit(
       "[H][H].C=C>>CC", balance="strict", time_limit_seconds=10,
   )

The returned classes describe heavy-atom correspondences. They do not
enumerate individual hydrogen-transfer mappings. Expanding a heavy map into
hydrogen-resolved classes is a separate operation. In particular, an exact
shell of minimum H-completion scores differs from projecting every mapping
in a full explicit-H exact shell onto its heavy atoms.

``balance="strict"`` rejects incompatible endpoint inventories.
``balance="dummy"`` allows typed zero-state placeholders for omitted atoms.
The implicit-H interface uses this mode by default. Dummy balancing defines
a different mapping problem and does not infer missing chemical species.

Both ``reactants>>products`` and ``reactants>agents>products`` are accepted.
Agents are stored separately and excluded from mapping. Existing atom-map
numbers are cleared before search. Supplied stereochemistry contributes to
class comparison. Omit stereo annotations when comparing a stereo-free model.
The Lewis objective uses the prepared bond representation. It does not
minimize over resonance drawings or tautomer families.

Explicit-H and graph interfaces
-------------------------------

``Mapper`` accepts reaction SMILES or two ``LabeledGraph`` objects. For
reaction SMILES it defaults to Lewis scoring with explicit hydrogens and
strict balancing:

.. code-block:: python

   from synkit.Chem.Mapper import Mapper

   mapper = Mapper("native", solver="pabs")
   explicit = mapper.map("CCO>>COC", time_limit_seconds=10)
   print(explicit.minimum_cost, explicit.optimality_proven, explicit.complete)

``hydrogen_quotient=True`` enables native H reduction for supported cases
while retaining the explicit-H problem's output semantics. This is separate
from ``map_reaction_implicit`` and its minimum-completion heavy classes.

``Mapper()`` continues to select the Python ``propagate`` implementation.
``Mapper("slap", backend="cpp")`` selects the standalone C++ heuristic.
Graph pairs use their supplied weighted bonds and need equal atom-type
inventories, symmetric adjacency and no loops. A graph result contains atom
index mappings rather than rendered reaction SMILES.

Eight public methods
--------------------

``Mapper.methods()`` returns these names:

.. list-table:: Public mapping methods
   :header-rows: 1
   :widths: 20 35 45

   * - Method
     - Implementation
     - Scope
   * - ``slap``
     - Python or C++ sequential linear assignment
     - Heuristic candidates without a minimum proof.
   * - ``kernel``
     - Python branch-and-bound
     - Exact within the SLAP-derived candidate kernel.
   * - ``milp``
     - Python PuLP model with CBC
     - One optimum within the candidate kernel.
   * - ``distance``
     - Python assignment branch-and-bound
     - Full-space minimum or numeric shell.
   * - ``propagate``
     - Python PABS
     - Full-space minimum or numeric shell.
   * - ``native``
     - Selected C++ solver
     - Depends on ``solver``. PABS searches the full compatible space.
   * - ``edits``
     - Python bond-edit-support enumeration
     - Numeric binary-CD shells.
   * - ``exhaustive``
     - Python exhaustive DFS
     - One full-space optimum for small inputs.

``optimality_proven`` on a ``Mapper`` result applies to its reported
``search_scope``. Kernel optimality is conditional on its fixed assignments
and domains. Heuristic and single-optimum methods do not claim complete
class enumeration.

Fourteen native configurations
------------------------------

Select these with ``Mapper("native", solver=...)``. The implicit-H entry
point uses the keyword ``backend`` instead of ``solver``.

.. list-table:: Native solver configurations
   :header-rows: 1
   :widths: 18 52 30

   * - Name
     - Approach
     - Dependency or restriction
   * - ``pabs``
     - Propagation and assignment-bounded search with SLAP feasible starts
     - Standard wheel
   * - ``cpabs``
     - PABS variant with canonical-parent traversal
     - Standard wheel
   * - ``slap``
     - WL refinement and sequential linear assignment
     - Standard wheel, heuristic only
   * - ``astar``
     - Best-first search with admissible bounds
     - Standard wheel
   * - ``abb``
     - Assignment branch-and-bound
     - Standard wheel
   * - ``kbb``
     - Kernel branch-and-bound
     - Standard wheel, restricted domains
   * - ``edit``
     - Bond-edit-support enumeration
     - Standard wheel, binary objective
   * - ``dfs``
     - Exhaustive DFS validation baseline
     - Standard wheel
   * - ``mces``
     - Weighted common-edge formulation
     - Standard wheel
   * - ``sepdp``
     - Separator decomposition and dynamic programming
     - Standard wheel
   * - ``cp_sat``
     - Constraint programming and SAT
     - OR-Tools C++ SDK
   * - ``scip``
     - Full-space MILP
     - OR-Tools C++ SDK with SCIP
   * - ``kmilp``
     - Kernel MILP
     - OR-Tools C++ SDK with SCIP, restricted domains
   * - ``maxsat``
     - Weighted MaxSAT
     - Qualified native MaxSAT executable

These are configurations, including related variants and validation
baselines. Each solver has query restrictions. A listed solver may not support
every objective or enumeration mode.

The standard wheel omits optional PDLP bounds. A custom OR-Tools build can
use LP proposals after checking an integer lower-bound certificate. Floating
LP status alone does not prove PMCD. Record the native configuration when
comparing timings.

Reference recovery
------------------

Report three results separately:

1. Was PMCD proved within the budget?
2. Was enumeration at the supplied CD complete?
3. Did an enumerated class match the reference?

The second result does not imply the third. Reference recovery compares
classes under the same objective, hydrogen representation and chemical
normalization. Atom-map numbering alone is not a class comparison. The
``AAMValidator`` interface supports ITS and reaction-centre comparisons.
The comparison policy must match the experiment.

A reference can have CD greater than PMCD. Minimizing graph changes does
not establish a mechanism, including for a single reaction step.

.. _mapper-custom-build:

Custom builds
-------------

For a checkout, editable install or optional SDK build:

.. code-block:: python

   from synkit.Chem.Mapper import Mapper, map_reaction_implicit
   from synkit.Chem.Mapper.solvers.native.build import build_mapper_native

   library = build_mapper_native("build/mapper-core")
   mapper = Mapper("native", solver="pabs", library_path=library)
   result = map_reaction_implicit("CCO>>COC", library_path=library)

An explicit ``library_path`` overrides the bundled library. Imports and
mapping calls never compile native code automatically. The explicit builder
records compiler and source provenance beside the library.

To build a wheel with an existing OR-Tools C++ SDK, set
``SYNKIT_MAPPER_ORTOOLS_DIR`` to the SDK prefix before running
``python -m build --wheel``. Shared SDK dependencies must be included or
resolved when distributing that wheel. Installing the Python ``ortools``
package alone does not enable this native build.

API details are in :doc:`api/chem`. Additional graph-search and certificate
interfaces are described in :ref:`python-mapping-apis`.

Additional interfaces
---------------------

.. toctree::
   :maxdepth: 1

   mapper/advanced
   mapper/implementation
