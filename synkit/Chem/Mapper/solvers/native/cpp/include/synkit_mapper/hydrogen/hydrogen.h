#ifndef SYNKIT_MAPPER_HYDROGEN_H
#define SYNKIT_MAPPER_HYDROGEN_H
#include "synkit_mapper/mapper.h"
#include "synkit_mapper/symmetry/classification.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Additive hydrogen ABI1. Metadata uses original full endpoint indices.
 * Objective matrices already contain forced electron anchors; neutral terminal
 * H has exactly one weight2=2 edge and no additional electron cost. */
typedef struct SynkitHydrogenAttributes {
  const int32_t *atomic_numbers_r,*atomic_numbers_p;
  const int32_t *isotopes_r,*isotopes_p;
  const int32_t *charges_r,*charges_p;
  const int32_t *radicals_r,*radicals_p;
  const int32_t *dummies_r,*dummies_p;
  const int32_t *stereo_r,*stereo_p;
} SynkitHydrogenAttributes;
enum SynkitHydrogenEligibility {
  SYNKIT_H_SUPPORTED=0,SYNKIT_H_NO_HYDROGEN=1,SYNKIT_H_ISOTOPE=2,
  SYNKIT_H_STEREO=3,SYNKIT_H_DUMMY_BOND=4,SYNKIT_H_ELECTRON_STATE=5,
  SYNKIT_H_NONTERMINAL=6,SYNKIT_H_BOND_STATE=7,SYNKIT_H_CLASS_ATTRIBUTES=8,
  SYNKIT_H_REDUCED_BACKEND=9
};
typedef struct SynkitHydrogenStatistics {
  int32_t supported,eligibility_reason,transport_complete,reason;
  int64_t full_nodes,quotient_nodes,heavy_candidates,transport_plans,transport_nodes;
  double reduction_seconds,heavy_seconds,transport_seconds;
} SynkitHydrogenStatistics;
typedef struct SynkitHydrogenTransportStatistics {
  int64_t minimum2,plans,nodes;
  int32_t complete,reason;
  double elapsed_seconds;
} SynkitHydrogenTransportStatistics;
/* Flow cells are groups*groups, row-major. Their exact labeled multiplicity
 * is decimal text, not a saturated int64. Costs are doubled CD. */
typedef int (*SynkitHydrogenTransportCallback)(const int32_t *cells,int64_t cost2,
                                             const char *multiplicity,void *);
SYNKIT_MAPPER_API int synkit_mapper_hydrogen_abi(void);
SYNKIT_MAPPER_API const char *synkit_mapper_hydrogen_error(void);
/* reservoir=-1 means all groups are real parents; otherwise it is the single
 * isolated dummy-H group. Queries use the shared minimum/exact/at-most enum.
 * Caps are -1 unlimited or positive. minimum2 ignores dummy-reservoir edges. */
SYNKIT_MAPPER_API int synkit_mapper_hydrogen_transport(
    int groups,const int32_t *rows,const int32_t *columns,int reservoir,int query,
    int64_t target2,double seconds,int64_t max_plans,int64_t max_nodes,void *cancel_token,
    SynkitHydrogenTransportCallback,void *,SynkitHydrogenTransportStatistics *);
/* Request.backend chooses the reduced heavy solver. Unsupported H states
 * return UNSUPPORTED + eligibility_reason; no implicit full-search fallback.
 * Numeric queries use a heavy minimum-H lower envelope and enumerate all
 * bounded H allocations. max_candidates counts lifted flow representatives.
 * memory_limit_bytes retains the selected heavy backend's accounted units;
 * H/class intermediate buffers and RSS are outside those units. */
SYNKIT_MAPPER_API int synkit_mapper_hydrogen_solve(
    const SynkitMapperProblem *,const SynkitHydrogenAttributes *,const SynkitMapperRequest *,
    int64_t max_transport_plans,SynkitMapperCallback,void *,SynkitMapperStatistics *,
    SynkitHydrogenStatistics *);
SYNKIT_MAPPER_API int synkit_mapper_hydrogen_solve_classified(
    const SynkitMapperProblem *,const SynkitHydrogenAttributes *,const SynkitMapperRequest *,
    const SynkitClassProblem *,int64_t max_transport_plans,int64_t max_canonical_nodes,
    SynkitClassCallback,void *,SynkitMapperStatistics *,SynkitHydrogenStatistics *,
    SynkitClassStatistics *);
/* Additive first-discovery/cumulative-count delivery. See the classification
 * update API; legacy hydrogen classified callbacks remain final batch counts. */
SYNKIT_MAPPER_API int synkit_mapper_hydrogen_solve_classified_updates(
    const SynkitMapperProblem *,const SynkitHydrogenAttributes *,const SynkitMapperRequest *,
    const SynkitClassProblem *,int64_t max_transport_plans,int64_t max_canonical_nodes,
    SynkitClassCallback,void *,SynkitMapperStatistics *,SynkitHydrogenStatistics *,
    SynkitClassStatistics *);
#ifdef __cplusplus
}
#endif
#endif
