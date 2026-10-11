Mapper methods and source layout
================================

Use one adapter to select the mapping method::

   from synkit.Chem.Mapper import Mapper

   mapper = Mapper(method="propagate")
   result = mapper.map("CCO>>CC=O")
   print(result.mapped_reactions)
   print(result.method, result.search_scope, result.minimum_cost)
   print(result.optimality_proven, result.complete)

``Mapper`` accepts reaction SMILES or a pair of ``LabeledGraph`` objects and
returns ``MappingResult`` for every method. Constructor options are reusable
defaults; options passed to ``map`` override them for that call. The default
uses weighted bonds and heavy atoms. ``Mapper.methods()`` lists these eight
concise names:

.. list-table:: Unified methods
   :header-rows: 1
   :widths: 15 39 23 23

   * - Method
     - Implementation
     - Search scope
     - Query/output
   * - ``slap``
     - WL refinement and sequential LAP
     - Heuristic candidates
     - Best candidates reached
   * - ``kernel``
     - SLAP, then kernel branch-and-bound
     - Restricted uncertainty kernel
     - Kernel minimum representatives
   * - ``milp``
     - SLAP, then kernel MILP using PuLP/CBC
     - Restricted uncertainty kernel
     - One kernel optimum
   * - ``distance``
     - Assignment branch-and-bound
     - Full compatible assignment space
     - Minimum or numeric CD shell
   * - ``propagate``
     - Python domain propagation and assignment bounds (PABS)
     - Full compatible assignment space
     - Minimum or numeric CD shell
   * - ``native``
     - Optional C++ integer-shell traversal
     - Full compatible assignment space
     - Minimum or numeric CD shell
   * - ``edits``
     - Binary bond-edit-support enumeration
     - Full compatible assignment space
     - Numeric CD shell; requires ``binary=True``
   * - ``exhaustive``
     - Independent small-graph DFS
     - Full compatible assignment space
     - One global optimum

Every result reports ``method`` and ``search_scope``. ``optimality_proven``
and ``minimum_cost`` refer to that scope: a kernel minimum is conditional on
its fixed assignments and candidate domains. ``slap`` never claims a proven
minimum. Numeric shells do not run a minimum proof. ``complete`` reports
enumeration and chemical classification; heuristic and single-optimum methods
return ``complete=False`` with ``heuristic`` or ``optimum_only`` as the reason.
A proven minimum can coexist with incomplete enumeration. For compressed H,
the minimum is for heavy bonds and returned distances include conditional H
lifts; inspect ``reaction_result.minimum_cost_scope``.

Reaction results contain symmetry class representatives; graph results contain
the solver's mappings. ``mapped_reactions`` is empty for graph input. The
original solver result is available through ``search_result``; chemistry and
hydrogen metadata are available through ``reaction_result``.

``distance``, ``propagate`` and ``native`` default to a 10-second deadline,
100,000 mappings and a 1,000,000-bijection admission cap. ``edits`` uses the
same deadline/output cap and a 50,000-support-pair admission cap. Pass ``None``
to disable a supported limit. ``slap``, ``kernel``, ``milp`` and ``exhaustive``
have no deadline/output-cap support and reject supplied limits. Graphs must be
symmetric and loop-free, with equal atom-type inventories.

Native compilation remains explicit::

   from synkit.Chem.Mapper.exact.native.build import build_native

   library = build_native("/tmp/synkit-native")
   result = Mapper("native", library_path=library).map("CO>>CO")
   shell = Mapper("edits", binary=True).map("CCC>>CCC", CD=2)

Install the optional MILP dependency with ``pip install 'synkit[milp]'``.
Specialized options remain available through the existing solver APIs.
``map_reaction(...)`` retains its Python/C++ exact-search interface, and
``AAMapper`` retains its SLAP-based workflow. The tables below explain those
established entry points.

What LAP, SLAP, PABS, and kernel mean
-------------------------------------

**LAP: linear assignment problem.** Given a matrix of pair costs, assign each
reactant atom to one product atom with minimum total pair cost. This is a
building block used by several methods. A graph's bond-edit cost also depends
on pairs of assignments, so one LAP alone generally does not solve exact AAM.

**SLAP: sequential linear-assignment matching.** ``GraphMatcher`` combines
WL graph-label refinement with successive LAP decisions to construct heuristic
graph mappings. ``AAMapper`` supplies the reaction-SMILES frontend. A mapping's
``cd`` is its scored chemical distance; an accurately scored mapping can still
be suboptimal.

**Kernel: a reduced uncertainty problem.** Supplied SLAP results determine
assignments to hold fixed and candidate images for uncertain atoms. The
``Kernel`` dataclass stores that restricted problem; branch-and-bound or MILP
solves it. Agreement among heuristic results does not prove global certainty.
An exact kernel optimum is scoped to its fixed assignments and candidate sets.

**PABS: Propagation and Assignment-Bounded Search.** This is the package's
descriptive name for its full-space exact-search interface. Python maintains
compatible one-to-one assignment domains, propagates constraints, and uses
LAP lower bounds to prune branches that cannot reach the requested distance.
The optional C++ backend implements a separate integer-shell traversal under
the same interface. These routes search without extracting a SLAP kernel or
invoking its MILP solver. A Python arithmetic-domain or certificate fallback
uses the established full-distance solver and records that choice.

Several exact modules import ``slap.lap`` for matrix conversion or distance
scoring. The imported utility does not determine which mapper is running.

Which workflow runs
-------------------

.. list-table:: Current call routes with other options left at their defaults
   :header-rows: 1
   :widths: 38 29 21 12

   * - Call
     - Solver
     - SLAP's role
     - Reduced kernel?
   * - ``map_reaction(...)`` or ``enumerate_pabs_mappings(...)``
     - Python exact propagation and assignment bounds
     - None
     - No
   * - Either entry point with ``backend="cpp"``
     - C++ full-space integer-shell search
     - None
     - No
   * - ``AAMapper().map_smiles(...)``
     - SLAP heuristic
     - Main mapper
     - No
   * - ``map_smiles(..., enumerate_exact=True)``
     - SLAP followed by exact kernel refinement
     - Supplies the initial mappings
     - Yes
   * - ``map_smiles(..., certify="exact")``
     - SLAP followed by optional kernel MILP certification; LAP fallback
     - Supplies the mappings to certify
     - Yes, if MILP is available
   * - ``enumerate_smiles(CD="minimal")`` or
       ``map_smiles(..., CD="minimal")``
     - Full-distance exact branch-and-bound
     - May supply a feasible seed
     - No
   * - Either SMILES method with numeric ``CD``
     - Full-distance exact branch-and-bound
     - None
     - No

``map_smiles`` and ``enumerate_smiles`` in the last four rows are methods of
``AAMapper``. A seed is an initial feasible guess for full-space search; a
kernel imposes fixed assignments and candidate restrictions. These have
different effects on the search space.

Other graph-level entry points
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 27 35 38

   * - Entry point
     - Purpose
     - Search scope
   * - ``enumerate_distance_mappings``
     - Established full-distance assignment branch-and-bound engine.
     - Global minimum or a selected shell, with optional replay certificates.
       The same solver backs ``AAMapper.enumerate_smiles`` and recorded PABS fallbacks.
   * - ``enumerate_hybrid_distance_mappings``
     - Choose distance search or binary edit-support enumeration.
     - Preserves shell semantics; records the selected method and reason.
       Edit-support search requires a numeric binary shell.
   * - ``solve_kernel`` / ``solve_kernel_milp`` /
       ``enumerate_kernel_optima``
     - Exact refinement of a supplied SLAP uncertainty kernel.
     - Optimality is within the kernel's fixed assignments and candidate sets.
       It does not by itself establish a full-space global optimum.
   * - ``ExactMapper``
     - Independent exhaustive DFS for small graphs and solver checks.
     - Complete bijective graph matching with its own cost convention;
       ``ExactResult.cost`` should not be treated as another API's CD field.

``CD="minimal"`` asks for the minimum shell; a numeric ``CD`` asks for a
particular shell. Resource-limited results must be interpreted through their
completion fields. Symmetry representatives and labeled mappings have
different count units; use the result's scope metadata.

Source responsibilities
-----------------------

::

   Mapper/
     __init__.py             Stable public entry points
     mapper.py               Uniform method selection and result adapter
     reaction_search.py      Reaction-SMILES PABS interface
     chem/                   SMILES, hydrogen representation, ITS chemistry
     graph/                  Labeled graphs, refinement, automorphisms
     slap/                   Approximate sequential-LAP matching
     io/                     Index-mapping serialization
     reporting/              Blinded shells, ITS spectra, alternatives
     validation/             AAM checks, identifiability, evaluation metrics
     adapters/               Prediction seeds and executable templates
     exact/
       search.py             Python/C++ PABS dispatcher
       hybrid.py             Assignment/edit-support selector
       edit_support.py       Binary bond-edit-support search
       distance_search/      Full-distance search, bounds, records, verifier,
                             independent small-graph exhaustive solver
       kernel_solvers/       Kernel model, branching, MILP, kernel enumeration
       pabs/                 Python propagation, bounds, residual spectra
       native/               C++ source/build, bindings, canonicalization,
                             parallel/frontier/disk orbit scheduling
       assignment/           Shared LAP domains and integer certificates
       seeds/                Swap, relaxation, and fragment seed heuristics
       core.py               Reference-ITS component reduction
       hydrogen.py           Hydrogen transfers and lifts
       symmetry.py           Shared permutation-group operations
       certificate.py        Optimality-gap reports
     _compat.py              Historical module aliases

The propagation bounds and residual spectra are components of PABS, not
additional user-facing mapping methods. Seed heuristics improve feasible
incumbents; they neither prove an optimum nor restrict the exact search space.
``reporting.analysis`` and ``reporting.native_analysis`` share analysis records
while coordinating different execution routes. ``validation.aam_validator``
checks mapped reactions; the other validation modules evaluate changed-bond
labels, their symmetry orbits, annotations, or cohort bounds.

Imports and optional native build
---------------------------------

Application imports stay concise::

   from synkit.Chem.Mapper import map_reaction, enumerate_pabs_mappings

   result = map_reaction("CO>>CO", CD="minimal", time_limit_seconds=10)

Use canonical grouped modules when working on an implementation::

   from synkit.Chem.Mapper.exact.pabs.propagation import PropagationConfig
   from synkit.Chem.Mapper.exact.native.build import build_native
   from synkit.Chem.Mapper.reporting.analysis import GlobalShellConfig
   from synkit.Chem.Mapper.adapters.template import ClassCorrespondence

Historical flat paths, including ``exact.propagation``, ``exact.distance``,
``analysis``, and ``template_adapter``, alias the same module objects. The
older ``synkit.Chem.Reaction.Mapper`` namespace also remains supported.
This preserves class identity, old pickle references, and module monkeypatches
without maintaining duplicate implementations.

Native compilation is explicit::

   python -m synkit.Chem.Mapper.exact.native.build --output-dir /tmp/synkit-native

The historical ``python -m synkit.Chem.Mapper.exact.native_build`` command
remains available. Both use the source in ``exact/native/native_distance.cpp``.

Run the mapper regression suite with::

   python -m pytest Test/Chem/Mapper -q
