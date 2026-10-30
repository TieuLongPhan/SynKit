# Weighted best-first atom assignment

The module searches compatible bijections with a priority queue ordered by an
admissible bound in integer doubled-CD units. “Weighted” describes the Lewis
edge costs. The heuristic multiplier is exactly one; this is not the
suboptimal, inflated-heuristic variant sometimes called weighted A*.

For a partial map `p:S -> T`, let `U,V` be its residual vertices. Its committed
cost `g` contains exactly the pairs wholly in `S`. For a compatible `i -> j`,
let `c(i,j)` sum the discrepancies from `i,j` to all committed pairs. For each
hard atom type, sort the residual incident edge weights at `i` and `j`,
including zero weights and excluding the vertex itself. Let `q(i,j)` be the
sum of absolute differences between these sorted lists over all types.

Any compatible completion pairs those lists by a permutation. Sorted pairing
minimizes one-dimensional absolute discrepancy, so `q(i,j)` is no larger
than that completion's residual incident discrepancy. Summing over a complete
residual assignment counts every residual-residual edge twice. Consequently

    h = ceil(min_compatible_assignment sum(2*c(i,j) + q(i,j)) / 2)

is no greater than the integral residual cost. The bound `g+h` has no double
counting: committed pairs are in `g`, committed/residual pairs are in `c`,
and residual pairs are covered by the half incident bound. Nonbonding-electron
anchors, if present in the common encoding, are ordinary forced typed vertices;
the module adds no second unary electron term.

Each state stores the entire partial image vector. A deterministic fixed
source order gives every labeled bijection one path; there is no symmetry
pruning and no closed set keyed by a used-target set or hash. In particular,
this implementation enumerates labeled mappings and requires the shared
classification stage for all-class output.

The first complete state removed from the queue proves the minimum, because
every unexpanded completion is bounded below by a queued state's bound.
Continuing until the least queue bound exceeds that optimum enumerates every
minimum labeled mapping. Numeric exact and at-most queries retain their
original threshold even after finding a cheaper mapping. Exact queries filter
only full-state costs; their threshold is never an incumbent.

An empty exhausted frontier proves the selected shell/interval complete (and
can prove no compatible completion). Time, cancellation, candidate, state and
frontier-memory stops preserve incomplete enumeration; a minimum already
proved remains a separate fact. Frontier bytes describe allocated state and
queue storage accounted by this implementation, not whole-process RSS or an
enforced operating-system memory limit. Tests compare partial bounds against
independently scored exhaustive completions and complete tiny mapping sets.
