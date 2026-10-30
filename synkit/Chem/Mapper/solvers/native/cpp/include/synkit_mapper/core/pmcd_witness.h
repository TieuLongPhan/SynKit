#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Per-thread witness for the most recent shared-core call. Each solve clears
// it first; a zero length means that this call retained no PMCD witness.
// Query with mapping == NULL and capacity == 0 to obtain the required length.
void synkit_mapper_clear_pmcd_witness(void);
void synkit_mapper_store_pmcd_witness(int n, const int32_t *mapping,
                                      int64_t cost2, int proven);
int synkit_mapper_last_pmcd_witness(int32_t *mapping, int capacity,
                                    int64_t *cost2, int *proven);
#ifdef __cplusplus
}
#endif
