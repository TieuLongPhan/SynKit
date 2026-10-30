#ifndef SYNKIT_MAPPER_SLAP_H
#define SYNKIT_MAPPER_SLAP_H

#include <stdint.h>

#if defined(_WIN32)
#define SYNKIT_MAPPER_API __declspec(dllexport)
#else
#define SYNKIT_MAPPER_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SlapStatistics {
  int64_t states;
  int64_t lap_calls;
  int64_t fingerprints;
  int64_t candidates;
  // Sum of incident-neighborhood costs in doubled-edge units. For a finished
  // singleton mapping this is twice its cost2. -1 means no finished candidate.
  int64_t best_proxy2;
  // Finished heuristic exploration is not a global optimum/completeness proof.
  int32_t completed;
  // 0 finished, 1 deadline, 2 state cap, 3 mapping cap, 4 callback stop.
  int32_t reason;
  double elapsed_seconds;
  double callback_seconds;
} SlapStatistics;

typedef int (*SlapMappingCallback)(const int32_t *mapping, int64_t cost2,
                                   void *userdata);

SYNKIT_MAPPER_API int synkit_mapper_abi(void);
SYNKIT_MAPPER_API const char *synkit_mapper_error(void);

// Dependency-free deterministic weighted SLAP. A2/B2 are dense, symmetric,
// loop-free nonnegative matrices in twice chemical-distance units. Labels
// enforce original atom compatibility. The caller explicitly supplies the
// original algorithm's stop-on-multiple policy per source atom; label IDs have
// no special chemical meaning. n is 1..512. Negative seconds/caps must be -1,
// meaning unlimited. Output is the best completed mappings reached by the
// heuristic, never a claim to globally optimal/all-class enumeration.
// The inherited stop-on-multiple policy can leave no completed bijection on
// some graph inputs; this is reported explicitly rather than silently using
// another heuristic or claiming infeasibility.
//
// All neighborhood refinement, structural-fingerprint exploration, component
// filtering, LAP solving and noninteractive all-atom splitting run in C++.
// Stable collision-free signature interning, a deterministic Hungarian tie
// rule and native stable-WL split representatives distinguish this weighted
// variant from Python's hash/tie-dependent implementation.
//
// Return 0 on a finished/interrupted run; -1 on invalid input/heuristic failure
// or an internal error.
// The thread-local error string remains valid until the next call in that
// thread. Callbacks occur only for completed, independently rescored
// bijections.
SYNKIT_MAPPER_API int
synkit_slap_solve(int n, const int64_t *a2, const int64_t *b2,
                  const int32_t *labels_r, const int32_t *labels_p,
                  const int32_t *stop_on_multiple_per_source_node,
                  int max_fingerprints, double seconds, int64_t max_states,
                  int64_t max_mappings, SlapMappingCallback callback,
                  void *userdata, SlapStatistics *statistics);

#ifdef __cplusplus
}
#endif

#endif
