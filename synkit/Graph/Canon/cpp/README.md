# Native sparse graph canonicalization

This is an opt-in C++17 prepared-graph core. It uses a versioned C ABI and
ctypes, following the Mapper build and ownership pattern without depending on
the Mapper library, RDKit headers, Python headers, nauty, or solver SDKs.
The existing Python exact canonicalizer and persisted stereo `/2` keys are
unchanged. This prototype is not yet bundled in platform wheels.

```sh
cmake -S synkit/Graph/Canon/cpp -B /tmp/canon-build \
  -DCMAKE_BUILD_TYPE=Release -DSYNKIT_CANON_BUILD_EXAMPLES=ON
cmake --build /tmp/canon-build --config Release --parallel 2
/tmp/canon-build/canonical_graph
```

The example resides under `Release/` on Windows. A developer can also call
the explicit, content-addressed CMake builder:

```python
from synkit.Graph.Canon._native import build_native_canon
from synkit.Graph.Stereo.native import classify_native_stereograph_mirror

library = build_native_canon("/tmp/canon-libraries")
result = classify_native_stereograph_mirror(
    graph, configured_registry, library_path=library,
    timeout_seconds=5, max_search_nodes=100_000,
    max_workspace_bytes=64 * 1024 * 1024,
    max_output_bytes=8 * 1024 * 1024,
)
if result.is_definitive:
    print(result.status, result.atom_mirror_isomorphism)
else:
    print(result.reason)
```

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

The first scientific pilot is in
`Experiment/Stereo/Canonicalization/native_pilot.py`. It compares complete
assignment partitions with the independent oracle and reference engine,
checks mirror decisions and relabeling, and keeps all correctness work outside
production timers. Its timings change language, convention, serialization and
symmetry-output workload together; they are development measurements, not an
isolated C++ speedup or a chemical benchmark.
