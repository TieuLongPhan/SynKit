# Native supporting algorithms

`dfs.cpp` is a deliberately exhaustive compatible-permutation DFS. It has no
assignment relaxation, partial-cost pruning, symmetry pruning or incumbent
branch pruning. Each leaf is scored independently in native code. Minimum
queries exhaust one optimization pass and then one pass emitting every minimum
mapping; a shared deadline spans both. Numeric shells traverse all compatible
permutations and filter only the actual leaf cost. The independent Python
multiset/permutation oracle remains in the test suite.

`edit_support.cpp` is a different algorithm for binary bond-presence distance.
For a distance k, it enumerates subsets of deleted source edges and inserted
source nonedges. If the source and target edge counts are eA and eB, the only
possible counts are `deletions=(k+eA-eB)/2` and `insertions=k-deletions`.
For each concrete support it constructs the edited source graph and enumerates
exact hard-type-compatible isomorphisms to the target, with exact degree and
neighbor-type invariants followed by full adjacency checks. These invariants
are necessary isomorphism conditions, not chemistry-based correspondence
restrictions. Atom class attributes do not become hard atom matching rules.

Any compatible mapping determines exactly one support: the source pairs whose
binary adjacency disagrees with the pulled-back target. Conversely an exact
isomorphism of an edited source graph to the target has exactly that support
and distance k. Distinct supports cannot duplicate a mapping. Exhausting supports
and their exact isomorphisms therefore enumerates the entire labeled binary
shell. Minimum search visits possible support sizes increasingly; finding the
first isomorphism proves the minimum after all smaller shells were exhausted,
then all supports in that minimum shell are completed. At-most and exact numeric
thresholds remain fixed. This is edit-support enumeration with isomorphism
reconstruction, not an assignment search renamed EDIT.

The binary native domain is `A2,B2 in {0,2}`, with `n <= 256`. Any other weighted
graph returns unsupported. Calling a Python wrapper with `binary=True` explicitly
changes the objective to bond presence; this is a separate binary track, not a
Lewis-distance result. Binary CD is integral; odd doubled exact thresholds give
a complete empty shell. Genuine weighted bond/edit support and Lewis-unary
extensions are not claimed by this implementation.

Both raw algorithms enforce soft shared deadline/cancellation, visited-state and
candidate caps. Raw class caps, native classification requests, requested symmetry
pruning and frontier/process-memory controls that are not implemented are
explicitly unsupported. They stream candidates to the independently qualified
class pipeline, and interrupted enumeration never reports total multiplicity.
