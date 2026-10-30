# Biochemical model analysis

`CRNAnalysis` owns a detached snapshot. Editing the source model does not change
an existing analysis. Build a new analysis to inspect edits. Existing lower-level
CRNT, Petri-net, pathway and construction APIs remain available.

```python
from synkit.CRN import CRNAnalysis, Scenario, read_sbml

model = read_sbml('model.xml')
a = CRNAnalysis(model, Scenario.from_model(model))
print(a.audit().to_dict())
print(a.conservation().to_dict())
print(a.structure().to_dict())
print(a.accessibility(['glucose']).to_dict())
```

Selectors accept stable ids, source ids or unique labels. Ambiguous names raise.
SBML boundary flags, compartments and initial values are retained in species
`source_attrs`; initial concentrations are **not** converted to integer tokens.
Nonintegral stoichiometry is rejected rather than silently truncated. Explicit
normalization is required before using discrete reaction semantics.

`Scenario(chemostats=('ATP',))` declares an unlimited external reservoir.
`Scenario(drop_species=('ATP',))` changes the reaction equations by deleting
incidences. Both are explicit and operate on a copy. Conservation results cover
internal species under the selected reservoir policy. CRNT structure includes
the retained equations; closed-system theorem verdicts do not automatically
apply to chemostatted dynamics.

```python
outcome = a.verify_pathway(flow, {'glucose': 1, 'ATP': 2}, max_states=100000)
rescue = a.minimum_supplements(
    flow, initial_marking={'glucose': 1}, allowed=['ATP', 'NAD+'],
    max_added=4, max_candidates=10000,
)
```

Every successful certificate is replayed against reaction records. Rescue
minimizes weighted added tokens **within the declared per-species box**; it
never asserts global optimality. State/depth limits propagate to inconclusive
results, including unresolved cheaper supplementation candidates.

`flux_feasibility(bounds=..., required=...)` solves a different question:
steady-state flux on internal species. All stored reactions are directed;
default flux bounds are nonnegative. Require positive reaction throughput to
exclude the zero solution. `blocked_reactions` uses an explicit positive
threshold and reports unknowns when its total time budget expires. LP results
are numerical and carry the method; they are not exact rational proofs.

Use `a.compare(other_analysis)` for aligned incidence and structural changes.
Comparisons require corresponding stable identities; unrelated imported models
must first be aligned. `audit()` checks elemental and charge balance when
SMILES exist and otherwise records unknown balance, not success. Open exchange
reactions can naturally have nonzero elemental balance.

## Optional exact C++ analysis

Platform wheels include the full CRN backend. `native_backend()` selects the
installed library without compiling, downloading, or requiring local Boost
headers. `backend='cpp'` also uses the bundled library when no explicit path is
supplied. The default `backend='auto'` retains Python unless a native context
or library path is selected.

```python
from synkit.CRN import native_backend, crnt_summary

with native_backend():
    report = crnt_summary(model)
```

For source development, the top-level CMake project builds all SynKit native
components together. The explicit `build_native` helper remains available for
focused CRN builds with a GCC/Clang-compatible compiler:

```python
from synkit.CRN import (build_native, conserved_moieties, enumerate_semiflows,
                       enumerate_structural_sets, PetriAnalyzer)

library = build_native('/tmp/synkit-crn-native', with_semiflows=True)
pools = conserved_moieties(model, library_path=library)
analyzer = PetriAnalyzer(model, library_path=library).compute_all()

# Structured results for explicit resource budgets:
result = enumerate_semiflows(
    model.to_stoichiometric_matrices()['S'], kind='p', library_path=library,
    time_limit=30, max_candidates=1_000_000, max_rows=100_000,
    max_results=10_000, max_coefficient_bits=4096,
)
siphons = enumerate_structural_sets(model, library_path=library,
                                   time_limit=30, max_nodes=200_000)
```

The C++ backend implements the **entire exact minimal P/T-semiflow enumeration**:
Farkas elimination, primitive integer normalization, bitset support filtering and
deduplication. Coefficients use Boost.Multiprecision arbitrary-precision integers.
All returned native semiflows are independently checked against the original
integer matrix in Python. Matrix input is checked int64 without float rounding;
output coefficients may exceed int64. A modular full-rank certificate provides
an exact early exit when the relevant kernel is empty.

The **entire minimal siphon/trap search** also runs in C++, using implication
propagation over selected and excluded species, disjoint branching, and closure
minimization. Each reported minimal set blocks further supersets. Least-species
roots cover all nonempty models, and each implication branch covers every
remaining choice. Returned sets use stable species ids. `max_nodes` counts
search states plus closure-minimization calls. Set `max_nodes=None`,
`time_limit=None`, or `max_results=None` to disable that individual limit on
structural enumeration; this requires a newly built structural ABI 2 library.

`backend='auto'` uses C++ when an explicit `library_path` is supplied; otherwise
it retains Python. `backend='python'` selects the reference implementation.
Importing never compiles or downloads anything. Build manifests record source
hashes, compiler versions, capabilities and library integrity metadata. CMake
builds also record the Boost version. Explicit development builds fingerprint
the transitive header dependencies. Installed wheels verify library bytes
against the wheel's final `RECORD`, including any changes made by wheel repair.

Source native builds require a C++17 compiler and installed Boost headers. The
builder checks standard include locations, `BOOST_INCLUDE_DIR`, and existing
Conda header caches. Use `boost_include_dir='/path/to/include'` for an explicit
location. `with_semiflows=False` builds native structural enumeration using only
the C++ standard library; `with_semiflows=True` fails clearly if headers are absent.

Deadlines, candidate/row/result caps and coefficient-bit caps return
`complete=False` with the termination reason. Intermediate cone tableau rows are
never returned as completed semiflows. The list-returning `minimal_semiflows`
and `conserved_moieties` APIs raise `SemiflowSearchLimit` with the structured
result when native search is incomplete. Timing reports include preparation and
witness checking; the native deadline bounds search, not later serialization.

`PetriAnalyzer` reuses computed semiflows and complete siphon enumeration for
persistence instead of repeating those searches. Partial enumeration cannot
certify persistence. Legacy persistence details return `None` when node or size
limits prevent a sound conclusion.

`has_stoichiometric_dependencies` tests linear dependence; the historical
`has_irreversible_futile_cycles` now tests nonnegative structural circulation.
Neither substitutes for thermodynamic analysis with measured/estimated energies
and declared concentrations and conditions.

`pathways(seeds, targets, initial_marking=...)` suggests bounded qualitative
candidates and verifies each against the inventory. Its candidate list is never
advertised as exhaustive. Use prescribed-flow verification when reaction
multiplicities, including repeated firings, matter.

The optional `to_cobra(bounds=..., exchange_bounds=...)` bridge requires a
separate COBRApy installation and explicit bounds for every directed reaction
and selected chemostat exchange. The exported model can then use COBRApy FVA,
blocked-reaction and deletion analyses. No physiological capacities are inferred.

## Research code and migration

Benchmarks, reference networks, independent cross-checks and biochemical case
studies belong to `Experiment/CRN/`, not the runtime analysis package:

- `Experiment.CRN.benchmarks`: reference fixtures, independent checks, validation,
  KEGG case studies, synthetic generators and timing harnesses;
- `Experiment/CRN/data/kegg_modules.json`: the reviewed four-module dataset.

These are included in the source distribution, not the wheel. The deprecated
`synkit.CRN.Benchmark` namespace contains only forwarding shims for existing
source-checkout consumers. It imports no research code until accessed; consumers
of these old names must migrate and have the experiment source available.

For a reusable offline KEGG workflow, supply your own cache explicitly:

```python
from synkit.CRN.Query import load_kegg_cache, build_cached_kegg_crn

cache = load_kegg_cache('my_modules.json')
model = build_cached_kegg_crn(cache, 'M00001')
```

The generic converter retains all participants by default; currency removal is
an explicit `drop_compounds` choice. It never imports experiment helpers or
implicitly loads the experimental dataset.


## Native CRN analysis coverage

Exact production analysis is available through
Python interfaces to native kernels:

| Computation | Native implementation | Python interface |
| --- | --- | --- |
| Exact rank and rational conservation basis | `cpp/algebra.cpp` | `matrix_analysis`, `integer_conservation_laws` |
| Complexes, linkage/SCC structure, deficiency and theorem inputs | `cpp/algebra.cpp` | `crnt_summary` |
| Minimal P/T-semiflows and conserved pools | `cpp/semiflows.cpp` | `minimal_semiflows`, `conserved_moieties` |
| Minimal siphons, traps and the paired exhaustive baseline | `cpp/structural.cpp` | `find_siphons`, `find_traps`, native exhaustive helper |
| Prescribed-flux execution and qualitative target search | `cpp/pathway.cpp` | `PathwayRealizability`, `qualitative_path` |
| Exact topological canonicalization and automorphism groups | `cpp/symmetry.cpp` | `canonical`, `CRNAutomorphism.summary`, `graph_symmetry` |
| Positive-kernel feasibility | HiGHS C++ via SciPy | `is_conservative`, `is_consistent` |
| Formose SMARTS application | RDKit C++ | `CRNExpand(rule_backend='rdkit', keep_aam=False)` |

```python
from synkit.CRN import build_native, native_backend, crnt_summary

library = build_native('/tmp/crn-native', with_semiflows=True)
with native_backend(library):
    report = crnt_summary(model)
    # Other supported analysis entry points share this task-local backend.
```

The native symmetry interfaces currently cover the complete topological
species/reaction graph used by the studies; unsupported semantic matching options
raise instead of silently running another implementation. Exact and qualitative
pathway certificates are replayed in Python. Native incomplete algebra or
canonicalization never becomes a claimed exact result.

Experiment CLI entry points default to C++. `--backend python` selects the
reference workflow; `--native-library` reuses an explicit build and
`--native-build-dir` controls a requested build. Imports never compile anything.
The runner propagates the backend into subprocesses and timing workers. Evidence
records the library/build digest, capabilities and native operation counts.

```bash
python Experiment/CRN/run_all.py --backend cpp --quick --offline \
  --with-biomodels \
  --output-dir /tmp/crn-native-evidence --summary /tmp/crn-native-evidence/summary.json
python -m pytest Test/CRN/test_native_analysis.py Test/CRN/test_native_semiflows.py -q
```

Python remains responsible for input/output, experiment orchestration, theorem
wording, molecular canonicalization calls and independent certificate checks.
The numerical SVD and CRNT4SBML comparisons deliberately retain their independent
reference implementations. The paired closure/exhaustive timing uses C++ for
both production algorithms, so its ratio does not compare C++ with Python.
Timing values depend on the workload, backend and recorded machine. Use newly
generated evidence for performance comparisons.
