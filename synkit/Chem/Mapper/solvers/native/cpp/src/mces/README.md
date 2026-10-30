# Weighted full-bijection common-edge mapper

This module constructs an explicit association graph and enumerates size-n
cliques. It does not invoke or relabel the existing assignment B&B/PABS kernel.
It is an objective-correct weighted common-edge architecture, not a claim that
ordinary binary MCES automatically solves the Lewis objective.

For nonnegative symmetric loop-free integer matrices A2 and B2, let
C = sum(i<k) A2[i,k] + sum(j<l) B2[j,l]. The common doubled chemical distance of
a compatible bijection p is

    D2(p) = sum(i<k) abs(A2[i,k] - B2[p(i),p(k)])
          = C - 2 * sum(i<k) min(A2[i,k], B2[p(i),p(k)]).

The equality follows term by term from abs(a-b)=a+b-2*min(a,b). C is independent
of p because p is a full bijection. Therefore maximizing weighted common-edge
overlap is exactly equivalent to minimizing the registered weighted distance.
Forced electron anchors, explicit H and typed dummy padding are already in
these matrices/types; the proof includes them without a separate electron term.
Half-integer chemical weights are exact after the common doubling conversion.
Charge, radical, dummy and stereo class attributes do not restrict compatibility.

Association vertices are all type-compatible correspondences (i,j). Two are
adjacent exactly when both source and target indices differ; the edge weight is
min(A2[i,k],B2[j,l]). Every n-clique uses every source and target once and hence
represents a compatible bijection. Conversely each compatible bijection yields
one n-clique. This remains true for zero-weight common edges, which are retained
as compatibility edges rather than deleted. Deleting them would incorrectly
restrict the feasible mapping space.

Traversal intersects the remaining ordered vertices with the selected vertex's
neighbors and retains only later vertices. Thus every clique has one unique
increasing-ID construction path. Source indices form a proper coloring of the
association graph. A full clique needs every missing source color: absence of
one color safely prunes that branch. Bounds sum independent maxima of weights
from each missing color into the current clique, plus an independent maximum
for each pair of missing colors. Dropping mutual consistency only increases
that overlap upper bound. The analogous sum of minimum current-clique
contributions is a lower bound because all future edge weights are nonnegative.
These bounds support maximum-overlap proof, fixed-overlap shells and overlap
threshold balls. No node/class symmetry equivalence is assumed for pruning.

The minimum request first proves maximum overlap, then separately traverses its
exact shell. Proof-only is separately named. Exact D2 queries whose C-target is
odd are empty; at-most queries use ceil((C-target)/2). Every emitted permutation
is independently rescored against the original matrices. States, candidate cap,
shared deadline/cancellation, and accounted graph/candidate memory controls keep
minimum proof separate from enumeration completion. Memory accounting is not
process RSS and excludes allocator overhead and transient per-node bound arrays;
its scope is reported explicitly in metadata. Frontier and native-class caps are
unsupported in this raw module. The common class pipeline can process outputs.

The explicit graph is limited to n<=256 and at most 8192 compatible association
vertices. This is an exploratory exact implementation with tiny-instance and
chemical conformance evidence, not a published-engine reproduction or a speed
claim. Model construction is quadratic in association vertices; molecular H
multiplicity can make its search expensive.
