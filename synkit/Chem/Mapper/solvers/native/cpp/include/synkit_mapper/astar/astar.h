#ifndef SYNKIT_MAPPER_ASTAR_H
#define SYNKIT_MAPPER_ASTAR_H

#include "synkit_mapper/mapper.h"

#ifdef __cplusplus
extern "C" {
#endif
// Diagnostic for independent admissibility tests. partial has problem->n
// entries, -1 for unassigned. Returns 0 on success, -1 on invalid input.
SYNKIT_MAPPER_API int
synkit_astar_partial_bound2(const SynkitMapperProblem *problem,
                            const int32_t *partial, int64_t *bound2);
#ifdef __cplusplus
}
#endif

#endif
