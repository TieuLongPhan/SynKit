#ifndef SYNKIT_STEREO_H
#define SYNKIT_STEREO_H

#include "canon.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Prepared, fixed configurations with explicit, distinct vertex references.
 * Atom frames: center, then ligands. Bond frames: left references (2),
 * left owner, right owner, right references (2). See README for slot order.
 * Colors must use the SAME exact palette in both inputs. No CIP assignment,
 * chemical perception, virtual-reference resolution, or normalization occurs.
 */
enum SynkitStereoGeometry {
  SYNKIT_TETRAHEDRAL = 1, SYNKIT_SQUARE_PLANAR = 2,
  SYNKIT_TRIGONAL_BIPYRAMIDAL = 3, SYNKIT_OCTAHEDRAL = 4,
  SYNKIT_PLANAR_BOND = 5, SYNKIT_ATROP_BOND = 6
};
typedef struct SynkitStereoFrame {
  int32_t geometry;
  int32_t parity; /* +/-1; square-planar and planar-bond require 0 */
  int32_t vertices[7]; /* unused trailing slots ignored */
} SynkitStereoFrame;

typedef struct SynkitStereoGraph {
  int64_t n, arcs;
  const int64_t *offsets; /* n+1, sorted symmetric simple undirected CSR */
  const int32_t *neighbors; /* arcs */
  const int32_t *vertex_colors; /* n, nonnegative shared palette ranks */
  const int32_t *edge_colors; /* arcs, nonnegative shared palette ranks */
  int64_t frame_count;
  const SynkitStereoFrame *frames;
} SynkitStereoGraph;

enum SynkitStereoFlags {
  /* Diagnostic ablation: check stereo only at complete structural mappings. */
  SYNKIT_STEREO_DEFER_CHECKS = 1
};
typedef struct SynkitStereoRequest {
  int32_t abi_version, struct_size;
  SynkitStereoGraph source, target;
  double seconds; /* -1 unlimited, 0 immediate deadline */
  int64_t max_nodes, workspace_bytes, output_bytes; /* positive or -1 */
  int32_t max_depth; /* 0..1024; explicit assignment depth */
  int32_t flags;
  void *cancel_token; /* synkit_canon_cancel_create(), borrowed until return */
} SynkitStereoRequest;

typedef struct SynkitStereoStatistics {
  int32_t complete, reason; /* reasons match SynkitCanonStatistics */
  int32_t isomorphic; /* -1 incomplete, 0 different, 1 same */
  int64_t nodes, leaves, structural_pruned, stereo_checks, stereo_pruned;
  int64_t refinement_rounds, peak_workspace_bytes;
  double elapsed_seconds;
} SynkitStereoStatistics;

typedef struct SynkitStereoResult SynkitStereoResult;
SYNKIT_CANON_API int synkit_stereo_abi(void);
SYNKIT_CANON_API const char *synkit_stereo_error(void);
/* 0: complete or explicitly stopped; -1: invalid input/runtime error, *out NULL.
 * Stopped work is never reported as inequality and exposes no mapping. */
SYNKIT_CANON_API int synkit_stereo_match(const SynkitStereoRequest *, SynkitStereoResult **out);
SYNKIT_CANON_API const SynkitStereoStatistics *synkit_stereo_statistics(const SynkitStereoResult *);
/* mapping[v] is the target vertex for source vertex v, only on complete match. */
SYNKIT_CANON_API const int32_t *synkit_stereo_mapping(const SynkitStereoResult *, int64_t *size);
SYNKIT_CANON_API void synkit_stereo_free(SynkitStereoResult *);

#ifdef __cplusplus
}
#endif
#endif
