#pragma once
#include "synkit_mapper/mapper.h"

#ifdef __cplusplus
extern "C" {
#endif
SYNKIT_MAPPER_API const char *synkit_mapper_assignment_error(void);
// Construct a restricted uncertainty kernel from supplied heuristic candidates.
// Consensus rows are fixed, other rows retain observed images. Small uncertain
// same-type blocks may be closed when their size <= close_block_size. No
// assertion is made that these domains contain all full-space optima.
SYNKIT_MAPPER_API int synkit_mapper_kernel_domains(
    int32_t n, int64_t count, const int32_t *mappings,
    const int32_t *types_r, const int32_t *types_p, int32_t close_block_size,
    uint8_t *allowed);
#ifdef __cplusplus
}
#endif
