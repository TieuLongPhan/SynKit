#include "synkit_mapper/core/runtime.h"

#include <functional>
#include <limits>
#include <numeric>

namespace synkit_mapper {

int solve_dfs(const Problem &p, const Request &r, Callback callback,
              void *userdata, Statistics &s) {
  initialize(s, r);
  if (r.classify || r.max_classes != -1 || r.symmetry || r.max_frontier != -1 ||
      r.memory_limit_bytes != -1)
    return unsupported(s);
  Budget budget(r);
  std::vector<int32_t> mapping(p.n, -1);
  std::vector<bool> used(p.n, false);
  std::vector<int> order(p.n);
  std::iota(order.begin(), order.end(), 0);
  // Ordering is deterministic; no assignment bound or incumbent pruning is
  // used. This deliberately enumerates every compatible permutation.
  std::map<int32_t, int> counts;
  for (int i = 0; i < p.n; ++i)
    ++counts[p.types_p[i]];
  std::stable_sort(order.begin(), order.end(), [&](int i, int k) {
    return counts[p.types_r[i]] < counts[p.types_r[k]];
  });
  int64_t best = std::numeric_limits<int64_t>::max();
  bool emitting = false;
  const bool minimum =
      r.query == SYNKIT_MINIMUM_CLASSES || r.query == SYNKIT_MINIMUM_PROOF;
  double proof_at = -1;
  auto score = [&]() {
    int64_t value = 0;
    // Independent native oracle scoring; no shared production cost/bound call.
    for (int i = 0; i < p.n; ++i)
      for (int k = 0; k < i; ++k) {
        int64_t a = p.a2[i * p.n + k];
        int64_t b = p.b2[mapping[i] * p.n + mapping[k]];
        value += a > b ? a - b : b - a;
      }
    return value;
  };
  std::function<void(int)> visit = [&](int depth) {
    budget.check();
    if (s.nodes == std::numeric_limits<int64_t>::max() ||
        (r.max_states >= 0 && s.nodes >= r.max_states))
      throw Stop{SYNKIT_STOP_STATES};
    ++s.nodes;
    if (depth == p.n) {
      ++s.leaves;
      const auto value = score();
      if (value < best) {
        best = value;
        s.incumbent2 = value;
      }
      if (!emitting)
        return;
      const bool selected = minimum ? value == s.minimum2
                            : r.query == SYNKIT_EXACT_DISTANCE
                                ? value == r.target2
                                : value <= r.target2;
      if (!selected)
        return;
      ++s.candidate_count;
      if (callback) {
        auto before = Clock::now();
        int stop = callback(mapping.data(), value, userdata);
        s.callback_seconds += elapsed(before);
        if (stop)
          throw Stop{SYNKIT_STOP_CALLBACK};
      }
      budget.check();
      if (r.max_candidates >= 0 && s.candidate_count >= r.max_candidates)
        throw Stop{SYNKIT_STOP_CANDIDATES};
      return;
    }
    int i = order[depth];
    for (int j = 0; j < p.n; ++j)
      if (!used[j] && p.types_r[i] == p.types_p[j]) {
        mapping[i] = j;
        used[j] = true;
        visit(depth + 1);
        used[j] = false;
        mapping[i] = -1;
      }
  };
  try {
    budget.check();
    s.model_seconds = budget.seconds();
    if (minimum) {
      visit(0);
      s.minimum2 = s.lower_bound2 = best;
      s.minimum_proven = 1;
      proof_at = budget.seconds();
      if (r.query != SYNKIT_MINIMUM_PROOF) {
        emitting = true;
        visit(0);
      }
    } else {
      emitting = true;
      visit(0);
      // Numeric queries traverse the full compatible space too; only after
      // exhausting it may the observed least cost become a proved minimum.
      s.minimum2 = s.lower_bound2 = best;
      s.minimum_proven = 1;
    }
    s.search_complete = 1;
    if (r.query != SYNKIT_MINIMUM_PROOF)
      s.labeled_count = s.candidate_count;
  } catch (const Stop &stop) {
    stopped(s, stop.reason);
  }
  s.elapsed_seconds = budget.seconds();
  s.optimization_seconds =
      minimum ? (proof_at < 0 ? s.elapsed_seconds : proof_at) - s.model_seconds
              : 0;
  s.enumeration_seconds =
      minimum ? (proof_at < 0 ? 0 : s.elapsed_seconds - proof_at)
              : s.elapsed_seconds - s.model_seconds;
  return 0;
}
} // namespace synkit_mapper
