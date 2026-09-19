# Exactness review for V20

The changes preserve the mapping domain, chemical objective, symmetry action,
class equivalence, labeled multiplicities, coordinate frequencies and public
class identifiers. Timing-based interruption continues to mean incomplete.

## Assignment hints are repaired before use

For a current finite-edge cost matrix C and arbitrary inherited column hints v,
clip the hints to the documented integer range and compute

    u_i = min_{j: C_ij finite} (C_ij - v_j).

Then u_i + v_j <= C_ij on every finite edge. An empty row is infeasible.
Replacing each column potential by min_i(C_ij - u_i) preserves feasibility.
The inherited matching contributes only in-range, finite, distinct edges with
C_ij = u_i + v_j. Other hinted edges are ignored. The existing Hungarian
augmentation completes this complementary partial matching or proves that no
perfect matching exists. A completed matching has primal cost equal to the
feasible dual sum, hence is optimal. The existing forced-edge alternating-path
proof applies to any optimal matching and feasible complementary duals.

Reuse is restricted to residual dimensions at least 16; smaller assignments use
the original fresh row-minimum reduction. This selector affects work only.
No monotonicity of parent and child costs is assumed. Deleted columns, changed
row order, stricter symmetry domains, skipped singleton levels and both increases
and decreases in costs are permitted. Parent state is indexed by original atom
coordinates and belongs to a live synchronous ancestor frame. Hints cannot
justify a prune until the current assignment has been solved. Clipping hints
does not affect correctness because the initial hints are arbitrary.

Exhaustive permutation controls exercise stale, duplicate, forbidden, missing
and extreme integer hints. An independent SciPy diagnostic solves 200 larger
problems and 6,400 forced edges after row/column deletion, including optima above
50 million. Tests check the objective, permutation, dual feasibility,
primal/dual equality and every exhaustive forced-edge optimum.

## Stop the singleton check after its second witness

The cheap domain guard distinguishes cardinalities zero, one, and at least two.
Once it finds two allowed images, the third outcome is proved. It need not count
further images. The exact cost matrix and forced-edge domains still inspect all
allowed edges when required. No candidate list or lower bound is truncated.

## Sparse canonical certificates

At a discrete canonical partition, iterate present neighbors, translate each to
its canonical column, retain the upper triangle including loops, and sort each
row by canonical column. This produces the identical sequence of present
(position, color) entries as the previous dense upper-triangle scan. The existing
implicit-zero ordering and the full node-color prefix are unchanged. Thus every
leaf comparison and resulting canonical certificate are unchanged.

## Local twin group factors

Exact equal-row twin transpositions are already verified and seeded. They connect
an entire twin class in the generator-support graph and generate its symmetric
group. If a support component lies entirely in that twin class, its order is m!:
it contains S_m and no permutation group on those m points can be larger.
Components crossing twin classes retain the general exact stabilizer-chain
calculation. Multiplication retains checked uint64 overflow handling. Controls
cover a twin factor beside a non-twin cycle and exchanges of equal components.

## Encoding and scratch storage

Template encoding cache keys include the exact ordered token tuple and the used
edge-token IDs. Within an instance the edge palette is fixed. These determine
exactly the same sorted palette and numeric translation arrays. Template and full
caches are separate and each is bounded to 256 entries. They cache only encoding
tables, not canonical results, mappings, classes or reference answers.

Instance-local native order/result buffers are synchronous scratch, as were the
existing edge and boundary buffers. Every published key copies immutable bytes;
no output retains a view of scratch. Cached ctypes pointers own references to
live arrays. Alternating different palettes, cache eviction and repeated buffer
writes retain byte-identical certificates in the controls.

## Shared deadline and mapping limit

The parent sets the absolute deadline before releasing the readiness barrier and
does not change it during search. Each slice reads this value once and checks
that same deadline at every callback and native invocation. This removes repeated
process-lock acquisition without extending the deadline. The mapping counter is
still read and updated inside the same shared lock; accessing its raw object
inside that lock avoids redundant reentrant lock acquisitions. Mapping-limit,
zero-time, slow-callback and frontier-coverage controls remain mandatory.

## Evidence boundaries

The V16, V17 and V18 60-second pilots on line 30474 are retained as incomplete runs.
A slower contiguous-refinement experiment was removed. Microbenchmarks are not
whole-case speed or completion claims. V20 uses a fresh frozen source for all
1,200 original tasks with 16 workers per case, two disjoint CPU partitions,
pattern cache disabled, portable C++17 -O3, the existing mapping cap and the
existing 60-second protocol including separately measured output completion.
No earlier successful result substitutes for a V20 cohort record.
