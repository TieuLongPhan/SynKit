# Native graph canonicalization and direct stereo matching

This is an opt-in C++17 prepared-graph core. It uses a versioned C ABI and
ctypes, following the Mapper build and ownership pattern without depending on
the Mapper library, RDKit headers, Python headers, nauty, or solver SDKs.
The existing Python exact canonicalizer and persisted stereo `/2` keys are
unchanged; the existing stereo APIs continue to use the Python engine.
The current platform-wheel build bundles this library, and the native Python
entry points discover it when `library_path` is omitted. Wheel packaging and
installed-wheel smoke checks are implemented; published artifacts and remote
CI results have not been verified in this local review.

## Direct stereo comparison in C++ (no Python)

`synkit_canon/stereo.hpp` provides a separate exact pairwise matcher. Graph
refinement, atom mapping, stereo pruning and witness verification run in C++17.
It does not build a canonical key or a faithful auxiliary-graph expansion.
It has no Python, NetworkX, RDKit, MØD or StereoMolGraph runtime dependency.
The existing Python stereo matcher is not redirected to this new API.

### Group derivation and source provenance

The native matcher implements SynKit's mapping search. Its local groups are
generated in `src/stereo_groups.h`; no enumerated StereoMolGraph permutation
table is embedded or loaded. Atom-frame groups are derived from integer
coordinates: a candidate slot permutation must preserve the Gram matrix and,
for a three-dimensional span, the sign of a basis determinant. This gives
the 12 tetrahedral, 8 square-planar, 6 trigonal-bipyramidal and 24 octahedral
actions. The equilateral triangle uses an exact oblique metric, avoiding
floating-point tolerances. For a planar span, either plane isometry extends
to a proper 3D rotation by choosing the normal's sign.

Bond-frame groups are derived from the eight combinations of endpoint
exchange and the two local reference-pair flips. Planar relative orientation
requires an even number of pair flips. The oriented axial-frame convention
also counts endpoint reversal in its sign, so axial configurations require
`left_flip XOR right_flip XOR endpoint_exchange == 0`. These are transport
rules for the declared reference frames, not a claim that both bond groups
are literal rotation groups of one fixed six-point geometry. The groups are
generated once and cached in fixed C++ arrays.

The initial native implementation ported enumerated groups through SynKit's
older Python tables; those tables recorded StereoMolGraph adaptation. Both
native and compatibility Python tables have now been replaced by generated
actions. The adapted example/fixture tests were replaced by SynKit-generated
geometry and chemistry properties. Their original sources and MIT notice are
preserved together in the local manuscript's historical adaptation archive,
outside the active package. Optional interoperability adapters and independent
external-comparator observations remain. This is not a claim of clean-room
development or new mathematical symmetry groups. Historical releases and
separately sourced chemical datasets retain their original provenance/terms.

```cpp
#include <synkit_canon/stereo.hpp>
#include <iostream>

int main() {
  using namespace synkit::stereo;
  Graph a;
  a.colors = {6, 1, 9, 17, 35}; // one shared exact palette: C, H, F, Cl, Br
  a.edges = {{0,1,1}, {0,2,1}, {0,3,1}, {0,4,1}};
  a.frames = {{SYNKIT_TETRAHEDRAL, 1, {0,1,2,3,4}}};
  Graph b = a;
  std::swap(b.frames[0].vertices[1], b.frames[0].vertices[2]);
  Options options;
  options.seconds = 5;
  options.workspace_bytes = 64 * 1024 * 1024;
  const auto result = isomorphism(a, b, options);
  if (!result.complete()) return 2; // statistics.reason explains the stop
  std::cout << result.isomorphic() << '\n'; // 0: opposite configurations
  // A successful comparison returns mapping[source_vertex] = target_vertex.
}
```

Build and run the three-center example and independent native tests:

```sh
cmake -S synkit/Graph/Canon/cpp -B /tmp/stereo-build \
  -DCMAKE_BUILD_TYPE=Release -DSYNKIT_CANON_BUILD_EXAMPLES=ON \
  -DSYNKIT_CANON_BUILD_TESTS=ON
cmake --build /tmp/stereo-build --config Release --parallel 2
ctest --test-dir /tmp/stereo-build -C Release --output-on-failure
/tmp/stereo-build/stereo_compare
```

On Windows, the executable is `Release/stereo_compare.exe`. To use this library
from another CMake project, add this directory with `add_subdirectory(...)` and
link `synkit_canon`. A standalone CMake install also installs both public stereo
headers. The C++ convenience API prepares CSR from an undirected edge list;
`synkit_canon/stereo.h` exposes the versioned C ABI for caller-prepared CSR.

The example supplies explicit-H graphs, with three fixed Fischer frames. It
prints both complete 8-by-8 comparison matrices and these decisions:

```text
CH2OH-(CHOH)3-CH2OH: LRL vs RLR = same
  site mapping: A1 -> B3, A2 -> B2, A3 -> B1
CHO-(CHOH)3-CH2OH: LRL vs RLR = different
CHO-(CHOH)3-CH2OH: LRL vs LLL (middle change) = different
```

Here L/R records the OH side in a Fischer projection, not a CIP label. In the
symmetric case, the *same global mapping* reverses the chain and transports all
three frames. Replacing the upper end by CHO removes that constitutional
symmetry; the middle frame must then agree under an end-preserving mapping.

### Exact matching algorithm and supported preparation

Both graphs use the same nonnegative integer palettes for vertex and edge
attributes. Equal ranks must mean equal prepared attributes across both inputs;
do not independently rank two unrelated palettes. The core validates sorted,
symmetric, loop-free, simple undirected CSR and each frame's incidence.

Joint color refinement uses exact sorted neighbor signatures and frame-role
incidence. Equal refinement cells are only candidate domains. Backtracking
constructs a single bijection `f` preserving vertex colors, edge colors, edges
and nonedges. For normalized source frame `a`, target frame `b`, and preserving
group `G`, a descriptor pair is compatible precisely when some `p` in `G`
satisfies `f(a[i]) == b[p[i]]` at every slot. Each candidate descriptor pair
retains a bit mask of its possible group elements. An atom assignment filters
only pairs incident to that source or target atom, using both forward and
inverse mappings. An empty candidate set for any descriptor rejects the branch;
backtracking restores masks. Every accepted complete mapping rechecks all
stereo frames from the full groups, independently of the cached masks. Unique
loci and equal descriptor inventory make the induced descriptor mapping bijective.

This tests consistency across entire graphs. Local groups constrain an actual
global atom mapping, so identical one-hop ligand colors do not justify a local
match or a local mismatch. Canonicalization and recursive CIP ranking are not
needed. Refinement and local pruning can leave ambiguous domains; exhaustive
backtracking resolves them or reports an explicit resource stop.

The direct matcher currently accepts six fixed conventional geometries, with
distinct explicit vertex references and at most one descriptor per atom/bond
locus:

| Geometry | Frame slots | Parity | Preserving group size |
| --- | --- | --- | --- |
| Tetrahedral | center, four ligands | +1 / -1 | 12 |
| Square planar | center, four cyclic ligands | 0 | 8 |
| Trigonal bipyramidal | center, two axial, three cyclic equatorial | +1 / -1 | 6 |
| Octahedral | center, ligands; opposite pairs (1,2), (3,5), (4,6) | +1 / -1 | 24 |
| Planar bond | two left refs, left owner, right owner, two right refs | 0 | 4 |
| Atrop bond | two left refs, left owner, right owner, two right refs | +1 / -1 | 4 |

Slot indices above are zero based. The octahedral positive reference axes are
slot 3 = +x, slot 4 = +y, slot 1 = +z. Negative parity is absorbed by a fixed
frame reversal before comparing proper-group orbits. These conventions match
the corresponding six SynKit configured descriptor families. Frame preparation
declares the chemistry: the matcher does not perceive stereocenters, infer
unlisted configurations, validate chemical valence, normalize resonance, or
compare an implicit H token with an explicit H vertex. Virtual references,
unknown configurations and the five extended families need preparation or a
future API extension; they must not be represented as a negative identity
verdict. Unsupported geometry/parity or malformed input raises in the C++ API
and returns `-1` with `synkit_stereo_error()` in the C ABI.

Deadline, node, workspace, output and assignment-depth limits are explicit.
Incomplete work has `isomorphic == -1` and no mapping; the C++ boolean accessor
throws for such a result. Workspace accounting covers core PMR allocations,
including masks, trails and output; it excludes caller buffers, stack storage,
allocator overhead and C++ convenience preparation. The deadline starts in the
C core after convenience preparation. The output cap counts the returned
`int32_t` mapping (four bytes per source vertex). Tokens from `synkit_canon_cancel_create`
can cancel either native API and must remain alive until its calls finish.
The default depth cap is 1024 assigned vertices. Worst-case exponential search
remains possible; this is not a new polynomial graph-isomorphism algorithm.

`Options::defer_stereo` (C flag `SYNKIT_STEREO_DEFER_CHECKS`) retains the same
structural refinement but delays stereo rejection until complete mappings. It
is a diagnostic ablation. Native tests compare both modes, check the full
three-site partitions including the middle-only registry, exhaust local
permutations against independent geometric predicates, compare small graphs
against an independent full-bijection oracle, replay returned witnesses, and
exercise renumbering, stops, malformed inputs and concurrent calls. The search
counts printed by the tests illustrate avoided branches, not a measured
speedup over another package.

## All eleven configured families in native C++

`synkit_canon/configured.hpp` supplies a native `Stereograph` model with exact
attribute tokens, typed owner-linked virtual H/LP, conventional frames, cumulene
paths, helical paths, planar supports/pilots and coupled frameworks. `expand`,
`canonicalize`, `compare` and `classify_mirror` perform their entire configured
operation in C++. `compare` uses incremental direct matching for local frames
and compares faithful native expansions for extended/coupled supports.

`canonicalize` returns full keys under the new convention
`synkit.cpp-configured-stereograph/1`, including profile label and complete
palette tokens. These keys do not share bytes with the Python-expansion native
convention. `atom_order` supplies a projected witness. Unknown configurations,
duplicate/overlapping loci and malformed supports are refused. Resource stops
return no usable key/mapping. `symmetry` uses exact stabilizer-chain orbit
transversals to return a complete **atom** automorphism generating set and
decimal group order; optional full closure is deadline/output bounded. This
output contract is distinct from auxiliary-graph witness enumeration.

Configured preparation has vertex/edge expansion caps and a shared cooperative
deadline. `workspace_bytes` still bounds the graph search's PMR workspace,
not the configured model/expansion or process RSS. Chemical normalization,
carrier perception, completeness inference and CIP are separate from this
prepared-input API; naming an identity profile does not perform its preparation.

The optional `stereo_experiments` executable imports frozen graph/state JSON,
runs native canonicalization/mirror checks and performs native representative-
based registration versus exact in-memory key indexing. Build it with
`-DSYNKIT_STEREO_BUILD_EXPERIMENTS=ON` and
`-DSYNKIT_JSON_INCLUDE_DIR=/path/to/include` containing `nlohmann/json.hpp`.
The JSON SDK is an executable-only dependency and is not vendored into SynKit.

```sh
/tmp/stereo-build/stereo_experiments \
  --input Experiment/Stereo/data/paper/core/inputs/registration_inputs.json \
  --output /tmp/new-native-registration.json --mode registration --repetitions 3
```

The runner refuses to overwrite an output. It explicitly reports chemical
SMILES/resonance specifications requiring preparation. Its in-memory index is
a new protocol rather than a matched rerun of the old disk-index experiment.
Independent expected assignment partitions may be frozen separately; the
native executable performs no Python calls.

## Use from Python

With a platform wheel, select native stereo analysis explicitly:

```python
import networkx as nx
from synkit.Graph.Stereo import TetrahedralStereo
from synkit.Graph.Stereo.native import classify_native_stereograph_mirror

graph = nx.star_graph(4)
nx.set_node_attributes(graph, {v: str(v) for v in graph}, "color")
configured_registry = (TetrahedralStereo((0, 1, 2, 3, 4), 1),)
result = classify_native_stereograph_mirror(
    graph, configured_registry, atom_color="color",
    timeout_seconds=5, max_search_nodes=100_000,
    max_workspace_bytes=64 * 1024 * 1024,
    max_output_bytes=8 * 1024 * 1024,
)
if result.is_definitive:
    print(result.status, result.atom_mirror_isomorphism)
else:
    print(result.reason)
```

The example supplies a prepared colored constitution and one fixed descriptor.
For identity certificates, use
`synkit.Graph.Stereo.native.canonicalize_native_stereograph`.

## Build from a source checkout

Source and editable installations can build the standalone library explicitly
with CMake and a C++17 compiler:

```sh
cmake -S synkit/Graph/Canon/cpp -B /tmp/canon-build \
  -DCMAKE_BUILD_TYPE=Release -DSYNKIT_CANON_BUILD_EXAMPLES=ON
cmake --build /tmp/canon-build --config Release --parallel 2
/tmp/canon-build/canonical_graph
```

The example resides under `Release/` on Windows. For the Python example above,
the explicit, content-addressed builder supplies `library_path`:

```python
from synkit.Graph.Canon._native import build_native_canon
from synkit.Graph.Stereo.native import classify_native_stereograph_mirror

library = build_native_canon("/tmp/canon-libraries")
result = classify_native_stereograph_mirror(
    graph, configured_registry, library_path=library, atom_color="color",
    timeout_seconds=5, max_search_nodes=100_000,
    max_workspace_bytes=64 * 1024 * 1024,
    max_output_bytes=8 * 1024 * 1024,
)
if result.is_definitive:
    print(result.status, result.atom_mirror_isomorphism)
else:
    print(result.reason)
```

Platform-wheel builds instead use the root `CMakeLists.txt` and `hatch_build.py`
to install the library under `synkit/Graph/Canon/_native/lib`. The release
workflow runs `scripts/smoke_native_wheel.py` against an installed wheel outside
the checkout, checking graph identity, stopped searches and stereo mirror
agreement without invoking a compiler.

The stereo adapter accepts an already prepared attributed constitution and
source-declared registry. It uses the existing full configuration-orbit
expansion and all eleven configured families. It does not infer missing
carriers, select a framework orientation, normalize resonance, or replace
strict completeness checking in the existing chemical assessment API.

## Exactness and convention

Initial color cells are ordered by exact palette rank. Refinement repeatedly
splits each cell by sorted neighbor-cell sequences, using lexicographic
signature order. Branching selects the first smallest nonsingleton cell,
individualizes its members, and retains the minimum full sparse leaf record.
Branch order is not semantic. Verified graph automorphisms stabilizing the
current ordered partition and exact twin transpositions may remove equivalent
branches. A terminal partition of twin cells needs only one leaf because
every within-cell permutation preserves the entire colored graph.

Exact open/closed-neighborhood keys cache false/true twin blocks. Their
symmetric factors contribute orbits implicitly, without enumerating factorial
groups. Setup is deferred until refinement leaves an unresolved cell. Direct
generator orbits are tried first. If they leave an unexplored
branch, bounded Schreier calculations can recover compositions fixing the
individualized path; resulting generators are also checked against the current
ordered partition. A discovery work/generator cutoff retains the direct verified
subgroup, so it weakens pruning without invalidating a complete key. This
adapts Mapper's stabilizer and implicit-factor approach, motivated by SURGE's
generation methods; it does not add a canonical-parent assignment enumerator.

Optional `synkit_canon_pruning_statistics` diagnostics report twin factors,
retained generators, skipped branches and stabilizer attempts/fallbacks. The
Python adapter also accepts earlier ABI-1 libraries lacking this additive
accessor. Independent C++ group-oracle checks can be enabled with
`-DSYNKIT_CANON_BUILD_TESTS=ON` and run with `ctest --test-dir BUILD -C Release`.

The sparse leaf record contains every vertex color and every undirected edge,
so equal records lift to an actual color-preserving graph isomorphism. All
automorphisms used for general orbit pruning are replayed against the input
colors and edges. Generators are internal pruning aids; the API does not claim
to return generators, full groups, or group orders.

`synkit.native-sparse-canonical-graph/1` is a **new canonical convention**.
It does not promise byte equality with the reference JSON or Mapper ITS
certificates. Native stereo keys have the separate namespace
`synkit.native-sparse-stereograph/1` and include the declared prepared identity
profile. Comparison methods reject cross-convention and cross-profile uses.
Within one convention, equality compares full bytes, never digests alone.

Graph keys contain the ASCII schema, a zero delimiter, a little-endian uint64
palette byte length, the exact JSON typed palette, then the native graph
record. The latter is `SG01`, little-endian uint32 vertex/edge counts, vertex
color ranks in canonical order, and lexicographically sorted uint32 endpoint
pairs. Stereo keys prepend their schema and a length-delimited UTF-8 profile.
Isolated vertices and exact palette tokens are retained. This format uses
linear space in graph incidence and palette payload, rather than storing
quadratically many nonedges.

## Scope, resource limits and ownership

- The graph ABI accepts simple undirected, loop-free, vertex-colored CSR.
  Explicit edge-color selectors, directed graphs and multigraphs are refused.
  Encode typed edge attributes as incidence vertices first. The stereo
  expansion already preserves its selected bond attributes that way.
- Every result is complete or carries a stop reason without certificate bytes
  or canonical order. Comparing incomplete results raises instead of deciding
  graph inequality. Both mirror legs must complete before a binary verdict.
- Python preparation/expansion uses cooperative deadline checks; it is not a
  hard wall-time sandbox. The native loop checks cancellation/deadlines during
  validation, refinement, branching and output. Node and depth limits are
  explicit; the implementation currently accepts depth limits up to 1024.
- `max_workspace_bytes` caps allocations routed through the core's tracked
  C++ memory resource, including canonical output buffers. It does not bound
  stack storage, Python allocations, allocator overhead or process RSS.
  `max_output_bytes` counts full key bytes and the returned integer order.
  Mirror output/node/workspace caps apply per leg; its deadline is shared.
- C requests borrow their arrays until return. Opaque results are freed by
  `synkit_canon_free`; Python copies outputs before freeing them. Exceptions
  never cross the C ABI. Cancellation tokens cannot be closed while borrowed.
- No build runs at import, no hidden backend fallback occurs, and no original
  certificate namespace is reassigned. Build manifests retain source hashes,
  compiler/environment information and the compiled library hash.

The maintained scientific workloads and independent references are documented in
`Experiment/Stereo/README.md`. Run `python -m Experiment.Stereo.run verify --replay`
to verify the retained paper evidence, or use its `run` command with a new output
directory to execute the native suite. Development pilots have been removed.

## Native scientific pipeline (spatial model 2)

The optional `synkit_science` C++ library and `science_experiments` executable
implement constitutional carrier perception, retained-neighbor orientation
constraints, SynKit's own ligand sequence rules and label projections, SDF
coordinate transport, local coordinate orientation, independent octahedral
pose fitting, source identity, topology completion, and coupled-framework
certificates. The prepared core remains C++17 with no SDK dependency. The
scientific targets use the RDKit C++ SDK, nlohmann JSON and Eigen;
the installed RDKit SDK used for validation requires C++20. RDKit supplies
parsing and chemical attributes; SynKit's CIP implementation does not read
`_CIPCode` or `_CIPRank` or call RDKit's CIP labeler.

```sh
cmake -S synkit/Graph/Canon/cpp -B /tmp/synkit-native -DCMAKE_BUILD_TYPE=Release \
  -DSYNKIT_CANON_BUILD_TESTS=ON -DSYNKIT_STEREO_BUILD_EXPERIMENTS=ON \
  -DSYNKIT_STEREO_WITH_RDKIT=ON -DCMAKE_PREFIX_PATH=/path/to/rdkit-sdk \
  -DSYNKIT_JSON_INCLUDE_DIR=/path/to/json/include
cmake --build /tmp/synkit-native --parallel 3
ctest --test-dir /tmp/synkit-native --output-on-failure
cmake -DRUNNER_DIR=/tmp/synkit-native \
  -DDATA_DIR=Experiment/Stereo/data/paper/core/inputs \
  -DOUTPUT_DIR=/tmp/synkit-native-rerun \
  -P synkit/Graph/Canon/cpp/examples/run_scientific_pipeline.cmake
```

The 15-stage execution uses no Python runtime. Independent truth preparation
is a separate, untimed operation; its frozen inputs and reference checkpoints
are retained. Corpus local-order validation checks complete raw partitions,
not just class counts. The suite includes the 63 designed/public fixtures
(the public fixtures cover formal A/B/C), 940 internal raw orders, 60 RotA
loci, 1,208 corpus records, both 300-record CIP protocols, 40 coordination
inputs, 48 structural negatives, 40 conformance cases, all 165,960 atom
renumberings of the 53 coordination positions, registration and atom-group
cost trials. This consolidates workloads; it is not a claim that all 35 legacy
Python report files or the external StereoMolGraph comparator were rerun.

Spatial model 2 corrects a historical axial error. Exchanging ends reverses
both the axis vector and the order in the cross product, leaving axial
orientation unchanged. The proper group is V4. The former C4 convention
incorrectly classified `FC(Cl)=C=C(F)Cl` as achiral. Tests now derive actions
from actual integer coordinates and check all eight terminal orders against
one coordinate configuration. Axial CIP projection likewise requires no
preferred viewing end. Corrected keys use `synkit.cpp-configured-stereograph/2`;
old keys must not be compared across conventions. The initial native run is
retained separately with an explicit historical-convention audit.

CIP coverage is bounded: multi-unit sequence-rule cases remain explicit
unsupported results. Coordinate-assisted CIP is a family-conditioned input
protocol. Carrier perception does not establish rotational barriers or
experimental persistence. Cost trials return source-atom groups and registration
uses an in-memory exact-key index; these differ from the historical auxiliary
group and disk-index protocols, so their timings are not a measured speedup
against the old paper tables.

No StereoMolGraph matcher or permutation table is used by the native pipeline.
All remaining Python descriptor tables are generated from geometry too.
Adapted legacy tests/source snapshots retain their original notices in the
historical archive; chemical dataset provenance remains separate.
