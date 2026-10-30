# Exact separator-table mapping DP

This is a native separator/path-decomposition dynamic program for the complete
weighted compatible-bijection problem. It is not the earlier bounded residual
oracle, and it does not borrow a polynomial-time or SR-GED complexity claim.
All arbitrary weighted endpoint graphs are supported within this implementation's
explicit `n <= 63` target-bitmask domain. Larger inputs return unsupported.

Let `M` be the sum of all unordered edge weights in A2 and B2. For a bijection f,

    D2(f) = M - 2 * sum_{i<k, A2[i,k]>0} min(A2[i,k], B2[f(i),f(k)]).

This follows from `|a-b| = a+b-2min(a,b)` and from a bijection preserving the
sum of all target edge weights. It is why missing source edges do not require
pair interactions in the DP: their contribution is already in the global
constant, not dropped. Unary Lewis electron anchors remain ordinary forced
vertices, included exactly once by this identity.

Fix a deterministic source order. After each prefix, its separator consists
of processed source vertices with a positive A2 edge to an unprocessed source
vertex. A state records:

1. the entire set of already used target vertices, as a global bitmask;
2. the images of every current separator vertex;
3. the accumulated overlap of source edges with both ends already processed.

A transition assigns the next source vertex to an unused compatible target,
and adds its overlap with earlier separator vertices. Every earlier neighbor
must be in the separator by definition. Every source edge is therefore counted
once at its later endpoint, including edges crossing any intermediate separator.
Forgotten target images remain present in the global mask, so target injectivity
is enforced across components, separators and forgotten parts of the graph.
There is no independent-subproblem matching that could accidentally reuse a
target vertex across components.

Two prefixes with equal state keys admit exactly the same future compatible
targets and edge-overlap increments. Merging them is therefore exact. All
predecessor arcs are retained, including multiple histories for the same state.
Accumulated overlap is part of the key even for numeric shells: retaining only
an optimum prefix would lose legitimate higher-cost exact/at-most mappings.
Every compatible full mapping gives one path in this layered DAG, and every
reconstructed path gives one compatible mapping. This proves complete labeled
minimum, exact and at-most output after all selected final states are traversed.

The optimum is proved only after all DP layers are exhausted. A completed final
state reached just before a limit supplies an incumbent but no minimum proof.
Interruption during predecessor reconstruction preserves an already proved
minimum while enumeration remains incomplete. Numeric thresholds never shrink.
Proof-only queries skip reconstruction and do not pretend to enumerate classes.
The shared class pipeline supplies exact ITS classes from the labeled mappings.

Complexity is exponential in the global target-subset space, separator width,
and retained overlap levels; predecessor reconstruction is output-sensitive.
Dense source graphs can retain almost all processed vertices in the separator.
The method consequently has no general polynomial scaling guarantee. State,
frontier, candidate and soft deadline/cancellation limits are enforced. The
memory cap and `peak_frontier_bytes` measure a conservative accounted retained
table/key/predecessor payload, including released per-layer keys conservatively;
allocator/block/hash-bucket overhead and transient scratch are excluded. This
is not process RSS or an operating-system memory limit. The additional profile
reports width, state/arc/transition counts and accounted table bytes.
