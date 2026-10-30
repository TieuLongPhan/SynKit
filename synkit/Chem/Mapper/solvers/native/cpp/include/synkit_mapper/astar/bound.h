#ifndef SYNKIT_MAPPER_ASTAR_BOUND_H
#define SYNKIT_MAPPER_ASTAR_BOUND_H

#include <cstdint>
#include <functional>
#include <vector>

namespace synkit_mapper {
namespace astar {

// Total (committed plus residual) doubled-CD lower bound for a compatible
// partial map. A -1 image denotes an unassigned source. The complete vector is
// required; a used-target set alone does not define this bound. Throws on an
// invalid/incompatible partial map or unsupported integer arithmetic range.
// This C++ diagnostic uses the same bound as search and is independently
// checked against exhaustive completions in the conformance suite.
int64_t partial_lower_bound2(int n, const int64_t *a2, const int64_t *b2,
                             const int32_t *types_r, const int32_t *types_p,
                             const std::vector<int32_t> &mapping,
                             const std::function<void()> &poll = {});

// Feasible typed assignment minimizing the same sorted-neighbor profile
// relaxation used at an empty partial map. This is an input-only incumbent
// seed, not an optimum certificate for the pairwise mapping objective.
std::vector<int32_t> profile_assignment_seed(
    int n, const int64_t *a2, const int64_t *b2,
    const int32_t *types_r, const int32_t *types_p,
    const std::function<void()> &poll = {});

} // namespace astar
} // namespace synkit_mapper

#endif
