#ifndef SYNKIT_MAPPER_PABS_SEED_H
#define SYNKIT_MAPPER_PABS_SEED_H

#include "synkit_mapper/core/runtime.h"

namespace synkit_mapper {
namespace pabs {

// Shared input-only feasible incumbent policy. The Problem must satisfy the
// common validated weighted/type-compatible graph contract. Returns false,
// without changing outputs, outside the legacy PABS seed domain: n<=512,
// doubled edge weights<=1024, at most16 distinct edge values, combined endpoint
// undirected edge mass<=500000. A caller with a broader domain must retain its
// own supported seed policy rather than declaring its exact backend unsupported.
//
// On true, mapping is a full compatible bijection, incumbent2 is its separately
// rescored pairwise cost, and lower_bound2 is ceil(profile-LAP minimum/2).
// The LAP uses the unchanged legacy Hungarian tie rules. polish=false retains
// just this profile assignment; true applies the existing eight swap passes
// and bounded native-SLAP/mixed-start policy. No reference, hard domains or
// optimality claim is introduced. All work shares the caller's Budget.
//
// Stop can interrupt preparation. Once a feasible initial map is available,
// mapping/incumbent2/lower_bound2 are published and remain valid on Stop; before
// that the outputs remain unchanged. The wall-limited policy does not promise
// identical seed maps or costs across independent executions.
bool prepare_seed(const Problem &problem, Budget &budget, bool polish,
                  std::vector<int32_t> &mapping, int64_t &incumbent2,
                  int64_t &lower_bound2, bool *original_profile_zero=nullptr);

} // namespace pabs
} // namespace synkit_mapper

#endif
