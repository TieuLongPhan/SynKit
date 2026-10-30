#ifndef SYNKIT_MAPPER_RUNTIME_H
#define SYNKIT_MAPPER_RUNTIME_H

#include "synkit_mapper/mapper.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

namespace synkit_mapper {
using Clock = std::chrono::steady_clock;
struct Cancellation {
  std::atomic<bool> requested{false};
};
struct Stop {
  int reason;
};
inline double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
inline bool cancelled(const Request &r) {
  return r.cancel_token && static_cast<Cancellation *>(r.cancel_token)
                               ->requested.load(std::memory_order_relaxed);
}
class Budget {
public:
  explicit Budget(const Request &request)
      : request_(request), start_(Clock::now()) {}
  double remaining() const {
    return request_.seconds < 0
               ? 1e100
               : std::max(0.0, request_.seconds - elapsed(start_));
  }
  double seconds() const { return elapsed(start_); }
  int reason() const {
    if (cancelled(request_))
      return SYNKIT_STOP_CANCELLED;
    return remaining() == 0 ? SYNKIT_STOP_TIME : SYNKIT_STOP_NONE;
  }
  void check() const {
    int r = reason();
    if (r)
      throw Stop{r};
  }

private:
  const Request &request_;
  Clock::time_point start_;
};
inline void initialize(Statistics &s, const Request &r) {
  s = {};
  s.abi_version = 1;
  s.backend = r.backend;
  s.minimum2 = s.incumbent2 = s.lower_bound2 = -1;
  s.class_count = s.labeled_count = s.peak_memory_bytes = -1;
  s.classification_complete = r.classify == 0;
  s.hydrogen_complete = 1;
}
inline void stopped(Statistics &s, int reason) {
  s.status = SYNKIT_LIMIT;
  s.reason = reason;
  s.search_complete = 0;
}
inline int unsupported(Statistics &s) {
  s.status = SYNKIT_UNSUPPORTED;
  s.reason = SYNKIT_STOP_UNSUPPORTED;
  s.search_complete = 0;
  return 0;
}
inline int64_t mapping_cost2(const Problem &p,
                             const std::vector<int32_t> &map) {
  if (static_cast<int>(map.size()) != p.n)
    throw std::invalid_argument("Wrong mapping size");
  std::vector<bool> used(p.n, false);
  for (int i = 0; i < p.n; ++i) {
    int j = map[i];
    if (j < 0 || j >= p.n || used[j] || p.types_r[i] != p.types_p[j])
      throw std::invalid_argument("Incompatible atom bijection");
    used[j] = true;
  }
  int64_t cost = 0;
  for (int i = 0; i < p.n; ++i)
    for (int k = i + 1; k < p.n; ++k)
      cost += std::abs(p.a2[i * p.n + k] - p.b2[map[i] * p.n + map[k]]);
  return cost;
}
inline void validate(const Problem &p, const Request &r) {
  if (p.abi_version != 1 || r.abi_version != 1 || p.n < 1 || p.n > 1024 ||
      !p.a2 || !p.b2 || !p.types_r || !p.types_p)
    throw std::invalid_argument("Invalid core problem ABI or dimensions");
  if (r.backend < 0 || r.backend > SYNKIT_SEPDP || r.query < 0 || r.query > 3 ||
      r.workers != 1 || (r.symmetry != 0 && r.symmetry != 1) ||
      (r.classify != 0 && r.classify != 1) || !std::isfinite(r.seconds) ||
      (r.seconds < 0 && r.seconds != -1) ||
      ((r.query == SYNKIT_EXACT_DISTANCE ||
        r.query == SYNKIT_AT_MOST_DISTANCE) &&
       r.target2 < 0))
    throw std::invalid_argument("Invalid core request");
  for (int64_t cap : {r.max_states, r.max_candidates, r.max_classes,
                      r.max_frontier, r.memory_limit_bytes})
    if (cap == 0 || cap < -1)
      throw std::invalid_argument("Core limits must be positive or -1");
  std::map<int32_t, int> counts;
  for (int i = 0; i < p.n; ++i) {
    ++counts[p.types_r[i]];
    --counts[p.types_p[i]];
    if (p.a2[i * p.n + i] || p.b2[i * p.n + i])
      throw std::invalid_argument("Core matrices must be loop free");
    for (int j = 0; j < p.n; ++j) {
      auto a = p.a2[i * p.n + j], b = p.b2[i * p.n + j];
      if (a < 0 || b < 0 || a > 1000000 || b > 1000000 ||
          a != p.a2[j * p.n + i] || b != p.b2[j * p.n + i])
        throw std::invalid_argument(
            "Core weights must be bounded nonnegative symmetric integers");
      if ((p.edge_colors_r &&
           p.edge_colors_r[i * p.n + j] != p.edge_colors_r[j * p.n + i]) ||
          (p.edge_colors_p &&
           p.edge_colors_p[i * p.n + j] != p.edge_colors_p[j * p.n + i]))
        throw std::invalid_argument("Class edge attributes must be symmetric");
    }
  }
  for (auto [type, count] : counts)
    if (count)
      throw std::invalid_argument("Unequal compatible atom inventories");
}
} // namespace synkit_mapper
#endif
