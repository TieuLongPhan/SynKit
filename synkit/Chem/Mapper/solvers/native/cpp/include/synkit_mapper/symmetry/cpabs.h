#ifndef SYNKIT_MAPPER_SYMMETRY_CPABS_H
#define SYNKIT_MAPPER_SYMMETRY_CPABS_H
#include "synkit_mapper/mapper.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct SynkitCPABSProfile {
  int64_t discovery_nodes;
  int64_t subgroup_order;
  int64_t canonical_calls;
  int64_t augmentations;
  int64_t canonical_rejections;
  int64_t bound_rejections;
  int32_t identity_fallback_r;
  int32_t identity_fallback_p;
  double discovery_seconds;
  double canonical_seconds;
} SynkitCPABSProfile;
/* Read the current thread's most recent run. Profile is diagnostic, never a
 * claim of complete endpoint symmetry discovery. Legacy fallback flags now
 * indicate an incomplete quotient group; verified twin factors remain active. */
SYNKIT_MAPPER_API void synkit_mapper_cpabs_profile(SynkitCPABSProfile *);
typedef struct SynkitCPABSProfileV2 {
  SynkitCPABSProfile base;
  int64_t twin_blocks_r,twin_blocks_p,quotient_order_r,quotient_order_p;
  int64_t bound_calls,seed_cost2;
  double bound_seconds,seed_seconds;
} SynkitCPABSProfileV2;
SYNKIT_MAPPER_API void synkit_mapper_cpabs_profile_v2(SynkitCPABSProfileV2 *);
#ifdef __cplusplus
}
#endif
#endif
