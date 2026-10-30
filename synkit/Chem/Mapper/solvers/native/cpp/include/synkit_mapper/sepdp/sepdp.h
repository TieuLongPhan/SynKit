#ifndef SYNKIT_MAPPER_SEPDP_H
#define SYNKIT_MAPPER_SEPDP_H

#include "synkit_mapper/mapper.h"

typedef struct SynkitSepDPProfile {
  int64_t separator_width;
  int64_t states;
  int64_t predecessor_arcs;
  int64_t transitions;
  int64_t peak_accounted_table_bytes;
} SynkitSepDPProfile;

#ifdef __cplusplus
extern "C" {
#endif
SYNKIT_MAPPER_API void synkit_mapper_sepdp_profile(SynkitSepDPProfile *);
#ifdef __cplusplus
}
#endif
#endif
