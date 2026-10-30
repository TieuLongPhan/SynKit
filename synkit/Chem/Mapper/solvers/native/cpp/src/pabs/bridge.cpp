// Compile the common exact kernel once, preserving its ABI3 exports.
// The shared-core bridge owns preparation, optional bounds and phase
// accounting.
#include "../../../../exact/native/native_distance.cpp"
#include "exact_helpers.h"
#include "sparse_root_lp.h"
#include "synkit_mapper/core/pmcd_witness.h"
#include "synkit_mapper/core/runtime.h"
#include "synkit_mapper/pabs/seed.h"
#include "synkit_mapper/slap.h"
#include <set>
static thread_local int64_t bridge_embedding_statistics[14] = {};
static thread_local int64_t bridge_deck_statistics[14] = {};
#ifdef SYNKIT_MAPPER_TEST_HOOKS
static thread_local int64_t lp_assignment_statistics[6] = {};
#endif // SYNKIT_MAPPER_TEST_HOOKS

namespace synkit_mapper {
namespace {

#include "chemical_floor.inc"
#include "endpoint_symmetry.inc"
#include "seed_heuristics.inc"

struct Receiver {
  const Problem &p;
  const Request &r;
  Callback callback;
  void *userdata;
  Statistics &s;
  Budget &budget;
  static int receive(const int *map, int cost4, void *pointer) {
    auto &self = *static_cast<Receiver *>(pointer);
    std::vector<int32_t> mapping(map, map + self.p.n);
    int64_t cost = mapping_cost2(self.p, mapping);
    if (cost4 != 2 * cost)
      throw std::runtime_error(
          "PABS quarter-CD witness does not match rescoring");
    const auto start = Clock::now();
    int stop =
        self.callback ? self.callback(mapping.data(), cost, self.userdata) : 0;
    self.s.callback_seconds += elapsed(start);
    ++self.s.candidate_count;
    if (stop) {
      self.s.reason = SYNKIT_STOP_CALLBACK;
      return 1;
    }
    if (self.budget.reason()) {
      self.s.reason = self.budget.reason();
      return 1;
    }
    return 0;
  }
};
} // namespace

#include "overlap_dual.inc"

#include "profile_seed.inc"

int solve_pabs_original(const Problem &p, const Request &r, Callback callback,
                        void *userdata, Statistics &s) {
  synkit_mapper_clear_pmcd_witness();
  sparse_root_lp::clear();
  std::fill(bridge_embedding_statistics, bridge_embedding_statistics + 14, 0);
  std::fill(coherent_seed_statistics, coherent_seed_statistics + 20, 0);
  coherent_seed_mapping.clear();
  std::fill(conditioned_factor_cache_statistics,
            conditioned_factor_cache_statistics + 8, 0);

  std::fill(bridge_deck_statistics, bridge_deck_statistics + 14, 0);
  bridge_deck_statistics[13] = -1;
  Budget budget(r);
  budget.check();
  if (p.n > SYNKIT_PABS_MAX_VERTICES || r.classify || r.max_states != -1 ||
      r.max_classes != -1 || r.max_frontier != -1 ||
      r.memory_limit_bytes != -1 || r.cancel_token)
    return unsupported(s);
  const int n = p.n;
  auto stage = Clock::now();
  std::vector<int> a(n * n), b(n * n), cr(n), cp(n), order(n), pred(n * n, 0);
  std::map<int32_t, int> types;
  std::set<int> levels;
  int64_t maximum2 = 0;
  for (int i = 0; i < n; ++i) {
    cr[i] = types.emplace(p.types_r[i], types.size()).first->second;
    cp[i] = types.emplace(p.types_p[i], types.size()).first->second;
    order[i] = i;
    for (int j = 0; j < n; ++j) {
      if (p.a2[i * n + j] > 1024 || p.b2[i * n + j] > 1024)
        return unsupported(s);
      a[i * n + j] = static_cast<int>(p.a2[i * n + j]);
      b[i * n + j] = static_cast<int>(p.b2[i * n + j]);
      levels.insert(a[i * n + j]);
      levels.insert(b[i * n + j]);
      if (j > i)
        maximum2 += a[i * n + j] + b[i * n + j];
    }
  }
  if (levels.size() > 16 || maximum2 > 500000)
    return unsupported(s);
  std::vector<int> rg, pg;
  if (r.symmetry) {
    rg = endpoint_generators(p, false, budget);
    pg = endpoint_generators(p, true, budget);
  }
  std::stable_sort(order.begin(), order.end(), [&](int i, int j) {
    int di = 0, dj = 0;
    for (int k = 0; k < n; ++k) {
      di += a[i * n + k] != 0;
      dj += a[j * n + k] != 0;
    }
    return di > dj;
  });
  Perm seed;
  RootDualBranchCertificate root_certificate;
  sparse_root_lp::AssignmentCertificate assignment_certificate;
#ifdef SYNKIT_MAPPER_TEST_HOOKS
  std::fill(lp_assignment_statistics, lp_assignment_statistics + 6, 0);
  struct SaveAssignmentStatistics {
    sparse_root_lp::AssignmentCertificate &certificate;
    ~SaveAssignmentStatistics() {
      lp_assignment_statistics[0] = certificate.ready;
      lp_assignment_statistics[1] = certificate.m;
      lp_assignment_statistics[2] = certificate.root_bound2;
      lp_assignment_statistics[3] = certificate.calls;
      lp_assignment_statistics[4] = certificate.prunes;
      lp_assignment_statistics[5] = certificate.ready && certificate.m > 64;
    }
  } save_assignment_statistics{assignment_certificate};
#endif // SYNKIT_MAPPER_TEST_HOOKS

  std::fill(conditioned_dual_statistics, conditioned_dual_statistics + 6, 0);
  bool original_profile_zero = false;
  try {
    const bool polish =
        r.query == SYNKIT_MINIMUM_CLASSES || r.query == SYNKIT_MINIMUM_PROOF;
    if (!pabs::prepare_seed(p, budget, polish, seed, s.incumbent2,
                            s.lower_bound2, &original_profile_zero)) {
      s.model_seconds += elapsed(stage);
      return unsupported(s);
    }

    if (r.query == SYNKIT_MINIMUM_PROOF &&
        (!original_profile_zero || s.incumbent2 > 8) &&
        s.incumbent2 > s.lower_bound2 && budget.remaining() > 0.2) {
      const int64_t dual = conditioned_overlap_dual_floor(
          p, budget, std::min(0.03, budget.remaining() * 0.05),
          root_certificate);
      if (dual > s.incumbent2)
        throw std::runtime_error("Invalid overlap dual bound");
      s.lower_bound2 = std::max(s.lower_bound2, dual);
    }
    if (r.query == SYNKIT_MINIMUM_PROOF && n <= 256 &&
        s.incumbent2 > s.lower_bound2 && budget.remaining() > 0.05) {
      const int64_t lp_floor = sparse_root_lp::run(
          p, budget, std::min(20.0, budget.remaining() * 0.7), s.incumbent2,
          &seed, &s.incumbent2, &assignment_certificate);
      budget.check();
      if (lp_floor > s.incumbent2)
        throw std::runtime_error("Invalid exact sparse LP bound");
      s.lower_bound2 = std::max(s.lower_bound2, lp_floor);
    }

    if (r.query == SYNKIT_MINIMUM_PROOF && s.incumbent2 == s.lower_bound2) {
      budget.check();
      synkit_mapper_store_pmcd_witness(n, seed.data(), s.incumbent2, 1);
      s.minimum_proven = 1;
      s.minimum2 = s.incumbent2;
      s.search_complete = 1;
      s.model_seconds += elapsed(stage);
      return 0;
    }
  } catch (...) {
    s.model_seconds += elapsed(stage);
    throw;
  }
  s.model_seconds += elapsed(stage);

  // Eligibility uses the initial profile relaxation, before subsequent exact
  // bounds strengthen it. This schedules an additional edge-deck proof only.
  if ((r.query == SYNKIT_MINIMUM_CLASSES || r.query == SYNKIT_MINIMUM_PROOF) &&
      original_profile_zero && s.incumbent2 > s.lower_bound2 && n <= 512) {
    stage = Clock::now();
    int64_t mass_a = 0, mass_b = 0, g = 0, edge_a = 0, edge_b = 0;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < i; ++j) {
        mass_a += p.a2[i * n + j];
        mass_b += p.b2[i * n + j];
        g = std::gcd(g, p.a2[i * n + j]);
        g = std::gcd(g, p.b2[i * n + j]);
        edge_a += p.a2[i * n + j] > 0;
        edge_b += p.b2[i * n + j] > 0;
      }
    const int units = g == 1 ? 3 : 2;
    int64_t families[6] = {};
    const bool small_families =
        g > 0 && mass_a == mass_b &&
        pabs::edge_deck_sizes(n, p.a2, p.b2, p.types_r, p.types_p, units,
                              families) == 0 &&
        families[units - 1] <= 60000 && families[units + 2] <= 60000;
    budget.check();
    if (g > 0 && mass_a == mass_b && s.incumbent2 <= 16 * g &&
        s.lower_bound2 < 2LL * (units + 1) * g && small_families) {
      const double allowance = std::min(0.65, budget.remaining() * 0.70);
      Perm candidate(n);
      int64_t lower2 = 0, witness2 = -1;
      const int result =
          pabs::edge_deck(n, p.a2, p.b2, p.types_r, p.types_p, units, allowance,
                          2000000, s.incumbent2, candidate.data(), &lower2,
                          &witness2, bridge_deck_statistics);
      bridge_deck_statistics[12] = result;
      budget.check();
      if (result >= 0 && lower2 >= 0) {
        if (lower2 > s.incumbent2)
          throw std::runtime_error("Overlap bound exceeds feasible incumbent");
        if (result == 1) {
          const int64_t score = mapping_cost2(
              p, std::vector<int32_t>(candidate.begin(), candidate.end()));
          if (score != witness2 || score != lower2 || score < s.lower_bound2)
            throw std::runtime_error("Invalid overlap minimum witness");
          seed = std::move(candidate);
          s.incumbent2 = score;
        }
        s.lower_bound2 = std::max(s.lower_bound2, lower2);
        bridge_deck_statistics[13] = s.lower_bound2;
        if (s.incumbent2 == s.lower_bound2) {
          s.minimum_proven = 1;
          s.minimum2 = s.incumbent2;
          synkit_mapper_store_pmcd_witness(n, seed.data(), s.incumbent2, 1);
        }
      }
    }
    s.optimization_seconds += elapsed(stage);
  }

  int64_t raw[6] = {};
  if (r.query == SYNKIT_MINIMUM_CLASSES || r.query == SYNKIT_MINIMUM_PROOF) {
    if (!s.minimum_proven) {
      int best4 = 0;
      Perm best(n);
      stage = Clock::now();
      int status = native_minimum_impl(
          n, types.size(), levels.size() - 1, budget.remaining(), a.data(),
          b.data(), cr.data(), cp.data(), order.data(), pred.data(), pg.data(),
          pg.size() / n, rg.data(), rg.size() / n, seed.data(), best.data(),
          &best4, raw, 0, 0, false, static_cast<int>(2 * s.lower_bound2),
          root_certificate.ready ? &root_certificate : nullptr,
          conditioned_dual_statistics, nullptr,
          assignment_certificate.ready
              ? +[](const int *mapping, const int *used, [[maybe_unused]] int target, void *data) {
                  auto &certificate = *static_cast<
                      sparse_root_lp::AssignmentCertificate *>(data);
                  const int64_t bound = certificate.bound(mapping, used);
#ifdef SYNKIT_MAPPER_TEST_HOOKS
                  ++certificate.calls;
                  certificate.prunes += 2 * bound > target;
#endif // SYNKIT_MAPPER_TEST_HOOKS
                  return bound;
                }
              : nullptr,
          &assignment_certificate);
      s.optimization_seconds += elapsed(stage);
      s.nodes += raw[0];
      s.leaves += raw[1];
      if (status < 0)
        throw std::runtime_error("Legacy PABS minimum kernel error");
      s.incumbent2 =
          mapping_cost2(p, std::vector<int32_t>(best.begin(), best.end()));
      if (best4 != 2 * s.incumbent2)
        throw std::runtime_error("Invalid minimum witness units");
      if (status) {
        stopped(s, SYNKIT_STOP_TIME);
        return 0;
      }
      synkit_mapper_store_pmcd_witness(n, best.data(), s.incumbent2, 1);
      s.minimum_proven = 1;
      s.minimum2 = s.incumbent2;
      s.lower_bound2 = s.minimum2;
    }
    if (r.query == SYNKIT_MINIMUM_PROOF) {
      s.search_complete = 1;
      return 0;
    }
  }
  budget.check();
  int64_t target = (r.query == SYNKIT_MINIMUM_CLASSES ? s.minimum2 : r.target2);
  if (r.query != SYNKIT_AT_MOST_DISTANCE && target > maximum2) {
    s.search_complete = 1;
    s.labeled_count = 0;
    return 0;
  }
  target = std::min(target, maximum2);
  Receiver receiver{p, r, callback, userdata, s, budget};
  stage = Clock::now();
  int status = synkit_distance_range(
      n, types.size(), levels.size() - 1,
      r.query == SYNKIT_AT_MOST_DISTANCE ? 0 : static_cast<int>(2 * target),
      static_cast<int>(2 * target), budget.remaining(), r.max_candidates,
      a.data(), b.data(), cr.data(), cp.data(), order.data(), pred.data(),
      pg.data(), pg.size() / n, rg.data(), rg.size() / n, Receiver::receive,
      &receiver, raw);
  s.enumeration_seconds += elapsed(stage);
  s.nodes += raw[0];
  s.leaves += raw[1];
  if (status < 0)
    throw std::runtime_error("Legacy PABS range kernel error");
  if (status) {
    stopped(s, s.reason      ? s.reason
               : status == 2 ? SYNKIT_STOP_CANDIDATES
               : status == 3 ? SYNKIT_STOP_CALLBACK
                             : SYNKIT_STOP_TIME);
  } else
    s.search_complete = 1;
  s.labeled_count = rg.empty() && pg.empty() ? s.candidate_count : -1;
  return 0;
}
} // namespace synkit_mapper

#include "objective_reduction.inc"

#ifdef SYNKIT_MAPPER_TEST_HOOKS
#include "test_hooks.inc"
#endif

#ifdef SYNKIT_MAPPER_TEST_HOOKS
extern "C" int synkit_private_lp_assignment_statistics(int64_t *out,
                                                       int capacity) {
  if (!out || capacity < 6)
    return -1;
  std::copy(lp_assignment_statistics, lp_assignment_statistics + 6, out);
  return 6;
}
#endif // SYNKIT_MAPPER_TEST_HOOKS
