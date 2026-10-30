#pragma once
#include "synkit_mapper/mapper.h"

typedef struct SynkitReactionOptions {
  int32_t abi_version;
  int32_t explicit_hydrogens;
  int32_t lewis_objective;
  int32_t dummy_balance;
  int32_t binary;
  int32_t hydrogen_quotient;
  int32_t empty_numeric_shell;
  int64_t max_canonical_nodes;
  int64_t max_hydrogen_plans;
} SynkitReactionOptions;

#ifdef __cplusplus
extern "C" {
#endif
SYNKIT_MAPPER_API int synkit_mapper_reaction_abi(void);
SYNKIT_MAPPER_API const char *synkit_mapper_reaction_error(void);
SYNKIT_MAPPER_API const char *synkit_mapper_rdkit_version(void);
// Entire parse -> objective -> native search/classification -> mapped-SMILES
// pipeline. Returned UTF-8 JSON is owned by the caller and freed below.
SYNKIT_MAPPER_API int synkit_mapper_reaction_solve(
    const char *, const SynkitReactionOptions *, const SynkitMapperRequest *,
    char **json_output);
SYNKIT_MAPPER_API void synkit_mapper_reaction_free(char *);
#ifdef __cplusplus
}
#endif
