#ifndef SYNKIT_MAPPER_CORE_H
#define SYNKIT_MAPPER_CORE_H

#include "synkit_mapper/slap.h"
#include <stdint.h>

// Additive core ABI; the original SLAP ABI1 and PABS ABI3 remain separate.
#define SYNKIT_MAPPER_CORE_ABI 1

enum SynkitBackend {
  SYNKIT_SLAP = 0,
  SYNKIT_PABS = 1,
  SYNKIT_CP_SAT = 2,
  SYNKIT_SCIP = 3,
  SYNKIT_MAXSAT = 4,
  SYNKIT_ASTAR = 5,
  SYNKIT_CPABS = 6,
  SYNKIT_ABB = 7,
  SYNKIT_KBB = 8,
  SYNKIT_KMILP = 9,
  SYNKIT_EDIT = 10,
  SYNKIT_DFS = 11,
  SYNKIT_MCES = 12,
  SYNKIT_SEPDP = 13
};
enum SynkitQuery {
  SYNKIT_MINIMUM_CLASSES = 0,
  SYNKIT_EXACT_DISTANCE = 1,
  SYNKIT_AT_MOST_DISTANCE = 2,
  SYNKIT_MINIMUM_PROOF = 3
};
enum SynkitRunStatus {
  SYNKIT_FINISHED = 0,
  SYNKIT_LIMIT = 1,
  SYNKIT_UNSUPPORTED = 2,
  SYNKIT_ERROR = 3
};
enum SynkitStopReason {
  SYNKIT_STOP_NONE = 0,
  SYNKIT_STOP_TIME = 1,
  SYNKIT_STOP_STATES = 2,
  SYNKIT_STOP_CANDIDATES = 3,
  SYNKIT_STOP_CLASSES = 4,
  SYNKIT_STOP_CALLBACK = 5,
  SYNKIT_STOP_FRONTIER = 6,
  SYNKIT_STOP_MEMORY = 7,
  SYNKIT_STOP_UNSUPPORTED = 8,
  SYNKIT_STOP_HEURISTIC = 9,
  SYNKIT_STOP_CANCELLED = 10
};

typedef struct SynkitMapperProblem {
  int32_t abi_version;
  int32_t n;
  // Actual objective matrices in doubled chemical-distance units. Unary
  // electron/H terms, if encoded as forced anchors, are already in these.
  const int64_t *a2;
  const int64_t *b2;
  const int32_t *types_r;
  const int32_t *types_p;
  // Optional extra endpoint attributes for symmetry/class identity. They are
  // NOT hard matching restrictions. Palettes must be shared across endpoints.
  const int32_t *node_colors_r;
  const int32_t *node_colors_p;
  const int32_t *edge_colors_r;
  const int32_t *edge_colors_p;
} SynkitMapperProblem;

typedef struct SynkitMapperRequest {
  int32_t abi_version;
  int32_t backend;
  int32_t query;
  int32_t
      symmetry; // 0 disabled, 1 verified subgroup; cPABS is its own backend.
  int32_t
      classify; // Reserved: 0 raw candidates, 1 requires native class support.
  int32_t workers; // Primary common track requires one worker.
  int64_t target2; // Numeric query threshold in doubled CD; otherwise ignored.
  double seconds;  // -1 unlimited; shared deadline includes callback wall time.
  int64_t max_states;
  int64_t max_candidates;
  int64_t max_classes;
  int64_t max_frontier;
  int64_t memory_limit_bytes;
  int32_t slap_max_fingerprints;
  const int32_t
      *slap_stop_flags; // Optional original-label policy; not type IDs.
  void *cancel_token;   // Created/freed through the core cancellation API.
} SynkitMapperRequest;

typedef struct SynkitMapperStatistics {
  int32_t abi_version;
  int32_t backend;
  int32_t status;
  int32_t reason;
  int32_t minimum_proven;
  int32_t search_complete;
  int32_t classification_complete;
  int32_t hydrogen_complete;
  int32_t heuristic_finished;
  int64_t minimum2;
  int64_t incumbent2;
  int64_t lower_bound2;
  int64_t nodes;
  int64_t leaves;
  int64_t candidate_count;
  int64_t class_count;
  int64_t labeled_count;
  int64_t peak_frontier;
  int64_t peak_frontier_bytes;
  int64_t peak_memory_bytes; // -1 unavailable, not a fabricated process RSS.
  double model_seconds;
  double optimization_seconds;
  double
      enumeration_seconds; // Includes callbacks; callback time also recorded.
  double classification_seconds;
  double callback_seconds;
  double elapsed_seconds;
} SynkitMapperStatistics;

typedef int (*SynkitMapperCallback)(const int32_t *mapping, int64_t cost2,
                                    void *userdata);

#ifdef __cplusplus
extern "C" {
#endif
SYNKIT_MAPPER_API int synkit_mapper_core_abi(void);
SYNKIT_MAPPER_API uint64_t synkit_mapper_capabilities(void);
SYNKIT_MAPPER_API const char *synkit_mapper_core_error(void);
SYNKIT_MAPPER_API const char *synkit_mapper_backend_version(int backend);
SYNKIT_MAPPER_API const char *synkit_mapper_backend_metadata(int backend);
SYNKIT_MAPPER_API int synkit_mapper_solve(const SynkitMapperProblem *,
                                          const SynkitMapperRequest *,
                                          SynkitMapperCallback, void *,
                                          SynkitMapperStatistics *);
// Explicit row-major allowed-correspondence mask for restricted kernel
// methods. Scope is the supplied domains, never a full-space proof by default.
SYNKIT_MAPPER_API int synkit_mapper_solve_domains(
    const SynkitMapperProblem *, const SynkitMapperRequest *, const uint8_t *,
    SynkitMapperCallback, void *, SynkitMapperStatistics *);
SYNKIT_MAPPER_API void *synkit_mapper_cancel_create(void);
SYNKIT_MAPPER_API void synkit_mapper_cancel_request(void *);
SYNKIT_MAPPER_API void synkit_mapper_cancel_free(void *);
#ifdef __cplusplus
}

// Backend entry points are C++ functions inside the one library. The public C
// dispatcher validates input, catches exceptions and translates error state.
namespace synkit_mapper {
using Problem = SynkitMapperProblem;
using Request = SynkitMapperRequest;
using Statistics = SynkitMapperStatistics;
using Callback = SynkitMapperCallback;
int solve_pabs(const Problem &, const Request &, Callback, void *,
               Statistics &);
int solve_ortools(const Problem &, const Request &, Callback, void *,
                  Statistics &);
int solve_maxsat(const Problem &, const Request &, Callback, void *,
                 Statistics &);
int solve_astar(const Problem &, const Request &, Callback, void *,
                Statistics &);
int solve_cpabs(const Problem &, const Request &, Callback, void *,
                Statistics &);
int solve_abb(const Problem &, const Request &, const uint8_t *, Callback,
              void *, Statistics &);
int solve_kernel_milp(const Problem &, const Request &, const uint8_t *,
                      Callback, void *, Statistics &);
int solve_edit_support(const Problem &, const Request &, Callback, void *,
                       Statistics &);
int solve_dfs(const Problem &, const Request &, Callback, void *, Statistics &);
int solve_mces(const Problem &, const Request &, Callback, void *, Statistics &);
int solve_sepdp(const Problem &, const Request &, Callback, void *, Statistics &);
} // namespace synkit_mapper
#endif
#endif
