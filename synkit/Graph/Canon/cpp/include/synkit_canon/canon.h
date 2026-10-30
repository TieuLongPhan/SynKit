#ifndef SYNKIT_CANON_H
#define SYNKIT_CANON_H

#include <stdint.h>
#if defined(_WIN32)
#if defined(SYNKIT_CANON_BUILD)
#define SYNKIT_CANON_API __declspec(dllexport)
#else
#define SYNKIT_CANON_API __declspec(dllimport)
#endif
#else
#define SYNKIT_CANON_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif

/* ABI 1: simple undirected, loop-free, vertex-colored CSR graph. Rows must
 * be sorted and symmetric. Buffers are borrowed for the duration of the call.
 * Limits are positive or -1; seconds also accepts zero (immediate deadline).
 * workspace_bytes bounds tracked native heap allocations, not process RSS.
 * No automorphism group enumeration is requested or returned by this ABI.
 */
typedef struct SynkitCanonRequest {
  int32_t abi_version;
  int32_t struct_size;
  int64_t n;
  int64_t arcs;
  const int64_t *offsets; /* n+1 entries */
  const int32_t *neighbors; /* arcs entries */
  const int32_t *colors; /* n entries; nonnegative exact palette ranks */
  double seconds;
  int64_t max_nodes;
  int64_t workspace_bytes;
  int64_t output_bytes; /* packed graph bytes plus canonical order */
  int32_t max_depth;
  void *cancel_token;
} SynkitCanonRequest;

typedef struct SynkitCanonStatistics {
  int32_t complete;
  /* 0 complete, 1 deadline, 2 nodes, 3 workspace, 4 output, 5 cancel, 6 depth */
  int32_t reason;
  int64_t nodes;
  int64_t leaves;
  int64_t refinement_rounds;
  int64_t peak_workspace_bytes;
  double elapsed_seconds;
  double validation_seconds;
  double search_seconds;
  double serialization_seconds;
} SynkitCanonStatistics;

/* Additive diagnostics; retained subgroup data, never a full group order. */
typedef struct SynkitCanonPruningStatistics {
  int64_t implicit_twin_factors;
  int64_t twin_vertices;
  int64_t automorphism_generators;
  int64_t orbit_rebuilds;
  int64_t symmetry_pruned_branches;
  int64_t stabilizer_calls;
  int64_t stabilizer_fallbacks;
} SynkitCanonPruningStatistics;

typedef struct SynkitCanonResult SynkitCanonResult;
SYNKIT_CANON_API int synkit_canon_abi(void);
SYNKIT_CANON_API const char *synkit_canon_error(void);
SYNKIT_CANON_API void *synkit_canon_cancel_create(void);
SYNKIT_CANON_API void synkit_canon_cancel_request(void *);
SYNKIT_CANON_API void synkit_canon_cancel_free(void *);
/* Returns 0 for complete or explicitly stopped work, -1 for malformed input
 * or runtime error. On -1, *result is NULL. Always free a non-NULL result.
 * Incomplete results expose neither a certificate nor an order.
 */
SYNKIT_CANON_API int synkit_canon_search(const SynkitCanonRequest *, SynkitCanonResult **result);
SYNKIT_CANON_API const SynkitCanonStatistics *synkit_canon_statistics(const SynkitCanonResult *);
SYNKIT_CANON_API const SynkitCanonPruningStatistics *synkit_canon_pruning_statistics(const SynkitCanonResult *);
SYNKIT_CANON_API const uint8_t *synkit_canon_bytes(const SynkitCanonResult *, int64_t *size);
SYNKIT_CANON_API const int32_t *synkit_canon_order(const SynkitCanonResult *, int64_t *size);
SYNKIT_CANON_API void synkit_canon_free(SynkitCanonResult *);

#ifdef __cplusplus
}
#endif
#endif
