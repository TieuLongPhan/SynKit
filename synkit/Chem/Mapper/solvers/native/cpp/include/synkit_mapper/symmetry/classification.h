#ifndef SYNKIT_MAPPER_SYMMETRY_CLASSIFICATION_H
#define SYNKIT_MAPPER_SYMMETRY_CLASSIFICATION_H

#include <stdint.h>
#include "synkit_mapper/slap.h"
#include "synkit_mapper/mapper.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Exact palette ranks are ordered by Python's _encode_color tokens. Node
 * colors are ranks, edge colors rank+1 (zero is absent). Endpoint edge states
 * are independent dense indices; edge_pair_colors pairs their full attributes,
 * including optional E/Z. Thus no chemistry attribute is discarded natively.
 * Palette entries are canonical ASCII JSON tokens, without surrounding list.
 * Only complete canonical searches publish keys. Equality uses full SKC1 bytes,
 * never their SHA256 identifiers. The identifiers preserve the legacy dense
 * certificate SHA256 encoding. This additive interface has version 1.
 */
typedef struct SynkitClassProblem {
  int32_t n;
  const int32_t *node_pair_colors; /* n*n */
  const int32_t *edges_r;         /* n*n endpoint state IDs */
  const int32_t *edges_p;
  int32_t edge_states_r;
  int32_t edge_states_p;
  const int32_t *edge_pair_colors;
  int32_t palette_size;
  const char *const *palette_json;
} SynkitClassProblem;

typedef struct SynkitClassStatistics {
  int64_t candidates;
  int64_t classes;
  int64_t canonical_nodes;
  int32_t complete;
  /* 0 complete, 1 deadline, 2 canonical node cap, 3 class cap, 4 callback,
   * 5 cancellation requested. */
  int32_t reason;
  double elapsed_seconds;
} SynkitClassStatistics;

typedef int (*SynkitClassCallback)(const int32_t *representative,
                                  int64_t cost2, int64_t observed_count,
                                  const unsigned char *packed_key,
                                  int64_t packed_size,
                                  const char *legacy_identifier,
                                  void *userdata);

SYNKIT_MAPPER_API int synkit_mapper_classification_abi(void);
/* Additive capacity discovery; libraries without this export support256. */
SYNKIT_MAPPER_API int synkit_mapper_classification_max_vertices(void);
SYNKIT_MAPPER_API const char *synkit_mapper_classification_error(void);
/* mappings is count*n, costs2 is count. seconds=-1 means unlimited. A shared
 * remaining deadline should be passed by the pipeline. All caps are -1 or
 * positive; max_nodes applies across the entire batch. No exceptions cross
 * callbacks. Return 0 for complete/interrupted, -1 for invalid input/error. */
SYNKIT_MAPPER_API int synkit_mapper_classify_batch(
    const SynkitClassProblem *problem, int64_t count,
    const int32_t *mappings, const int64_t *costs2,
    double seconds, int64_t max_nodes, int64_t max_classes,
    SynkitClassCallback callback, void *userdata,
    SynkitClassStatistics *statistics);
SYNKIT_MAPPER_API int synkit_mapper_classify_batch_with_cancel(
    const SynkitClassProblem *problem, int64_t count,
    const int32_t *mappings, const int64_t *costs2,
    double seconds, int64_t max_nodes, int64_t max_classes, void *cancel_token,
    SynkitClassCallback callback, void *userdata,
    SynkitClassStatistics *statistics);

/* Additive delivery mode: first discovery publishes a complete exact key
 * immediately, then a final cumulative observed-count update when needed.
 * Merge callbacks by the FULL packed key; updates are not new mappings or
 * additional classes. Complete output has final counts. Interrupted output
 * counts are nondecreasing lower bounds and incomplete flags stay explicit.
 * The legacy batch entry points and their one-callback contract are unchanged.
 * Publication and canonical work share the supplied deadline. */
SYNKIT_MAPPER_API int synkit_mapper_classify_batch_updates_with_cancel(
    const SynkitClassProblem *problem, int64_t count,
    const int32_t *mappings, const int64_t *costs2,
    double seconds, int64_t max_nodes, int64_t max_classes, void *cancel_token,
    SynkitClassCallback callback, void *userdata,
    SynkitClassStatistics *statistics);

/* Additive explicit options; both flags must be 0 or 1. updates uses the
 * cumulative-count contract above. packed_identifier=1 returns SHA256 of the
 * exact SKC1 bytes as the presentation identifier; 0 preserves legacy IDs.
 * Keys, canonicalization, deduplication, budgets and statistics are unchanged.
 * No process-global mode is set; existing entry points always use legacy IDs. */
SYNKIT_MAPPER_API int synkit_mapper_classify_batch_ex(
    const SynkitClassProblem *problem, int64_t count,
    const int32_t *mappings, const int64_t *costs2,
    double seconds, int64_t max_nodes, int64_t max_classes, void *cancel_token,
    SynkitClassCallback callback, void *userdata,
    SynkitClassStatistics *statistics, int32_t updates, int32_t packed_identifier);

/* Entire prepared search -> class pipeline. Intermediate mapping callbacks
 * stay in C++; only class records cross the output callback. Search and class
 * processing share the request deadline and core cancellation token. The
 * supplied request must have classify=1; class cap applies to distinct keys.
 * Classification node budget is separate and explicit. Search candidate cap
 * retains its candidate units; exact class equality is never a hash test.
 * Results distinguish search completion from classification completion. */
SYNKIT_MAPPER_API int synkit_mapper_solve_classified(
    const SynkitMapperProblem *, const SynkitMapperRequest *,
    const SynkitClassProblem *, int64_t max_canonical_nodes,
    SynkitClassCallback, void *, SynkitMapperStatistics *, SynkitClassStatistics *);
/* Restricted-domain variant. The row-major n*n zero/one mask is passed intact
 * to synkit_mapper_solve_domains; neither classification nor batching expands
 * it. Minimum proof/completeness scope remains exactly the supplied domains. */
SYNKIT_MAPPER_API int synkit_mapper_solve_domains_classified(
    const SynkitMapperProblem *, const SynkitMapperRequest *, const uint8_t *allowed,
    const SynkitClassProblem *, int64_t max_canonical_nodes,
    SynkitClassCallback, void *, SynkitMapperStatistics *, SynkitClassStatistics *);

/* Search followed by incremental class delivery, with cumulative-count
 * callbacks as documented above. Existing classified entry points remain
 * batch interfaces. Raw search and classification still share one deadline. */
SYNKIT_MAPPER_API int synkit_mapper_solve_classified_updates(
    const SynkitMapperProblem *, const SynkitMapperRequest *,
    const SynkitClassProblem *, int64_t max_canonical_nodes,
    SynkitClassCallback, void *, SynkitMapperStatistics *, SynkitClassStatistics *);
SYNKIT_MAPPER_API int synkit_mapper_solve_domains_classified_updates(
    const SynkitMapperProblem *, const SynkitMapperRequest *, const uint8_t *allowed,
    const SynkitClassProblem *, int64_t max_canonical_nodes,
    SynkitClassCallback, void *, SynkitMapperStatistics *, SynkitClassStatistics *);

/* Explicit options as above; allowed may be NULL for unrestricted search or
 * the unchanged n*n zero/one kernel mask. ABI1 structs are not extended. */
SYNKIT_MAPPER_API int synkit_mapper_solve_classified_ex(
    const SynkitMapperProblem *, const SynkitMapperRequest *, const uint8_t *allowed,
    const SynkitClassProblem *, int64_t max_canonical_nodes,
    SynkitClassCallback, void *, SynkitMapperStatistics *, SynkitClassStatistics *,
    int32_t updates, int32_t packed_identifier);

#ifdef __cplusplus
}
#endif
#endif
