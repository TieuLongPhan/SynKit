#ifndef SYNKIT_PARTIAL_H
#define SYNKIT_PARTIAL_H
#include <stdint.h>
#if defined(_WIN32)
# if defined(SYNKIT_PARTIAL_BUILD)
#  define SKP_API __declspec(dllexport)
# else
#  define SKP_API __declspec(dllimport)
# endif
#else
# define SKP_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif

/* ABI 1. All indices are dense array indices, never Python object addresses.
 * Bond coordinates are doubled integers (order 1.5 is encoded as 3).
 * flags: present=1, aromatic=2, valence-electron field present=4.
 * match_mask: element=1, charge=2, hcount>=4, radical=8, LP>=16, aromatic=32.
 * role: concrete=-1, query=0, attachment=1, radical=2, hydrogen=3, stereo=4.
 * Stereo ports are reserved and refused by the current compiled profile.
 * Concrete bindings preserve element and valence-electron labels regardless
 * of match_mask. Attachment/radical ports need one declared adjacent owner.
 * Source populations and restored LP baselines are bounded by 10^9.
 * Domain count=-1 means unrestricted; offset indexes the shared domain array.
 * Normal aromatic bonds have sigma=1 and pi in {0,1}. Product pi assignments
 * satisfy the local Lewis equations at fixed charges/resources. At most one
 * million propagation/branch states are searched per application; exceeding
 * this bound returns status 2 and no application batch.
 * candidate_limit also bounds the embeddings retained for any one source
 * component, before their injective joins are enumerated.
 * Structural embedding recursion has a separate one-million-state bound;
 * exceeding it also returns status 2 without partial applications.
 * In observation mode, wholly unreported input components match complete
 * host components. Their internal automorphisms and permutations among
 * components with equal image populations are quotiented exactly for target
 * observations. Mapping counts then count those representatives.
 * fixed_charge bit 0 preserves a supplied unsupported product charge;
 * observation-only bit 1 marks an unsupported known endpoint Lewis model.
 */
typedef struct SkpState { int64_t element, charge, hcount, radical, lp, ve, flags; } SkpState;
typedef struct SkpBond { int64_t order2, sigma2, pi2; } SkpBond;
typedef struct SkpDomain { int64_t offset, count; } SkpDomain;
typedef struct SkpAtom { SkpState state; int64_t atom_map; } SkpAtom;
typedef struct SkpEdge { int64_t u, v; SkpBond state; } SkpEdge;
typedef struct SkpRuleAtom {
  SkpState left, right;
  int64_t atom_map, baseline_lp, match_mask, role, owner, capacity, budget;
  int64_t side, identity, materialization, virtual_kind, stereo_slot, fixed_charge;
  SkpDomain elements, charges, radicals, bond_orders;
} SkpRuleAtom;
typedef struct SkpRuleEdge {
  int64_t u, v;
  SkpBond left, right;
  int64_t min_pi2;
} SkpRuleEdge;
typedef struct SkpRequest {
  const SkpAtom *atoms; int64_t n_atoms;
  const SkpEdge *edges; int64_t n_edges;
  const SkpRuleAtom *rule_atoms; int64_t n_rule_atoms;
  const SkpRuleEdge *rule_edges; int64_t n_rule_edges;
  const int64_t *domains; int64_t n_domains;
  int64_t partial, candidate_limit, radical_policy;
} SkpRequest;
typedef struct SkpProductAtom {
  SkpState left, right;
  int64_t atom_map, origin, introduced, owner, port;
} SkpProductAtom;
typedef struct SkpProductEdge { int64_t u, v; SkpBond left, right; } SkpProductEdge;
typedef struct SkpStats { int64_t candidates, mappings, applications, elapsed_ns; } SkpStats;
typedef struct SkpResult SkpResult;
SKP_API int skp_abi(void);
SKP_API SkpResult *skp_execute(const SkpRequest *);
/* Observation policy: presence flags select reported endpoint vertices.
 * Absence is not a material zero-electron state. This entry point transforms
 * observations and does not certify a closed material reaction or DPO diagram.
 * Invalid proper-subset completions are counted and rejected individually;
 * limits, failed full bindings and an all-invalid population remain errors.
 */
SKP_API SkpResult *skp_execute_observed(const SkpRequest *);
/* Return the first constructible full-source observation application. Target
 * labels are never an argument. A found witness does not exhaust the search;
 * callers must inspect skp_search_complete before asserting absence/completeness.
 */
SKP_API SkpResult *skp_execute_observed_first(const SkpRequest *);
/* True only after a successful exhaustive search. Errors and early witnesses
 * return false; inspect skp_status as well before using any result.
 */
SKP_API int skp_search_complete(const SkpResult *);
SKP_API int64_t skp_rejected_completion_count(const SkpResult *);
SKP_API int skp_status(const SkpResult *); /* 0=success, 1=input, 2=limit, 3=completion */
SKP_API const char *skp_error(const SkpResult *);
SKP_API const SkpStats *skp_stats(const SkpResult *);
SKP_API int64_t skp_application_count(const SkpResult *);
SKP_API const int64_t *skp_mapping(const SkpResult *, int64_t);
SKP_API const SkpProductAtom *skp_atoms(const SkpResult *, int64_t, int64_t *);
SKP_API const SkpProductEdge *skp_edges(const SkpResult *, int64_t, int64_t *);
SKP_API void skp_free(SkpResult *);
#ifdef __cplusplus
}
#endif
#endif
