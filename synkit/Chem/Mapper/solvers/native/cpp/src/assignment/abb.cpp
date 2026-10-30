#include "synkit_mapper/core/runtime.h"
#include <limits>
#include <numeric>
#include <set>
#include <string>

namespace synkit_mapper {
namespace {
constexpr int64_t INF = std::numeric_limits<int64_t>::max() / 8;

// Integral Hungarian assignment. Forbidden domain entries have cost INF.
int64_t lap(const std::vector<std::vector<int64_t>> &cost,
            const Budget &budget, std::vector<int> *assignment = nullptr) {
  const int n = static_cast<int>(cost.size());
  std::vector<int64_t> u(n + 1), v(n + 1);
  std::vector<int> p(n + 1), way(n + 1);
  for (int i = 1; i <= n; ++i) {
    budget.check();
    p[0] = i;
    int j0 = 0;
    std::vector<int64_t> slack(n + 1, INF);
    std::vector<bool> used(n + 1, false);
    do {
      budget.check();
      used[j0] = true;
      int i0 = p[j0], j1 = 0;
      int64_t delta = INF;
      for (int j = 1; j <= n; ++j)
        if (!used[j]) {
          int64_t cur = cost[i0 - 1][j - 1] >= INF
                            ? INF
                            : cost[i0 - 1][j - 1] - u[i0] - v[j];
          if (cur < slack[j]) {
            slack[j] = cur;
            way[j] = j0;
          }
          if (slack[j] < delta) {
            delta = slack[j];
            j1 = j;
          }
        }
      if (delta >= INF)
        return INF;
      for (int j = 0; j <= n; ++j)
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else if (slack[j] < INF)
          slack[j] -= delta;
      j0 = j1;
    } while (p[j0] != 0);
    do {
      int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }
  if (assignment) {
    assignment->assign(n, -1);
    for (int j = 1; j <= n; ++j)
      (*assignment)[p[j] - 1] = j - 1;
  }
  return -v[0];
}

class AssignmentSearch {
  const Problem &p;
  const Request &r;
  const uint8_t *allowed;
  Callback callback;
  void *userdata;
  Statistics &s;
  Budget budget;
  std::vector<int> order;
  std::vector<int32_t> mapping;
  std::vector<bool> used;
  int64_t best = INF;
  int64_t threshold = 0;
  bool optimize = false;

  bool compatible(int i, int j) const {
    return p.types_r[i] == p.types_p[j] &&
           (!allowed || allowed[i * p.n + j]);
  }
  void poll() {
    budget.check();
    if (r.max_states >= 0 && s.nodes >= r.max_states)
      throw Stop{SYNKIT_STOP_STATES};
    ++s.nodes;
  }
  int64_t cross(int i, int j, int depth) const {
    int64_t value = 0;
    for (int k = 0; k < depth; ++k) {
      const int a = order[k];
      value += std::abs(p.a2[i * p.n + a] - p.b2[j * p.n + mapping[a]]);
    }
    return value;
  }
  // Every unresolved-to-committed pair is counted exactly once. Ignoring
  // unresolved-to-unresolved pairs can only decrease the nonnegative cost.
  int64_t bound(int depth, int64_t committed) {
    int count = p.n - depth;
    std::vector<int> products;
    for (int j = 0; j < p.n; ++j)
      if (!used[j])
        products.push_back(j);
    std::vector<std::vector<int64_t>> costs(
        count, std::vector<int64_t>(count, INF));
    for (int a = 0; a < count; ++a) {
      budget.check();
      const int i = order[depth + a];
      for (int b = 0; b < count; ++b)
        if (compatible(i, products[b]))
          costs[a][b] = cross(i, products[b], depth);
    }
    int64_t rest = lap(costs, budget);
    return rest >= INF ? INF : committed + rest;
  }
  void emit(int64_t cost) {
    ++s.candidate_count;
    s.labeled_count = s.candidate_count;
    const auto start = Clock::now();
    const int stop = callback ? callback(mapping.data(), cost, userdata) : 0;
    s.callback_seconds += elapsed(start);
    if (stop)
      throw Stop{SYNKIT_STOP_CALLBACK};
    budget.check();
    if (r.max_candidates >= 0 && s.candidate_count >= r.max_candidates)
      throw Stop{SYNKIT_STOP_CANDIDATES};
  }
  void visit(int depth, int64_t committed) {
    poll();
    const int64_t cutoff = optimize ? best : threshold;
    if ((optimize && committed >= cutoff) || (!optimize && committed > cutoff))
      return;
    if (depth == p.n) {
      ++s.leaves;
      if (optimize) {
        best = committed;
        s.incumbent2 = best;
      } else if (r.query != SYNKIT_EXACT_DISTANCE || committed == threshold)
        emit(committed);
      return;
    }
    int64_t lower = bound(depth, committed);
    if ((optimize && lower >= cutoff) || (!optimize && lower > cutoff))
      return;
    const int i = order[depth];
    std::vector<std::pair<int64_t, int>> choices;
    for (int j = 0; j < p.n; ++j)
      if (!used[j] && compatible(i, j))
        choices.emplace_back(cross(i, j, depth), j);
    std::sort(choices.begin(), choices.end());
    for (auto [delta, j] : choices) {
      mapping[i] = j;
      used[j] = true;
      visit(depth + 1, committed + delta);
      used[j] = false;
      mapping[i] = -1;
    }
  }

public:
  AssignmentSearch(const Problem &problem, const Request &request,
                   const uint8_t *domains, Callback cb, void *data,
                   Statistics &statistics)
      : p(problem), r(request), allowed(domains), callback(cb), userdata(data),
        s(statistics), budget(r), order(p.n), mapping(p.n, -1),
        used(p.n, false) {}
  int run() {
    if (r.classify || r.max_classes != -1 || r.max_frontier != -1 ||
        r.memory_limit_bytes != -1 || r.symmetry)
      return unsupported(s);
    double phase = 0;
    try {
      budget.check();
      std::iota(order.begin(), order.end(), 0);
      std::vector<int> sizes(p.n), degrees(p.n);
      for (int i = 0; i < p.n; ++i)
        for (int j = 0; j < p.n; ++j) {
          sizes[i] += compatible(i, j);
          degrees[i] += p.a2[i * p.n + j] != 0;
        }
      std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return sizes[a] == sizes[b] ? degrees[a] > degrees[b]
                                    : sizes[a] < sizes[b];
      });
      std::vector<std::vector<int64_t>> seed_cost(
          p.n, std::vector<int64_t>(p.n, INF));
      for (int i = 0; i < p.n; ++i) {
        budget.check();
        for (int j = 0; j < p.n; ++j)
          if (compatible(i, j))
            seed_cost[i][j] = 0;
      }
      std::vector<int> seed;
      if (lap(seed_cost, budget, &seed) < INF) {
        std::vector<int32_t> candidate(seed.begin(), seed.end());
        best = mapping_cost2(p, candidate);
        s.incumbent2 = best;
      }
      s.model_seconds = budget.seconds();
      phase = budget.seconds();
      if (r.query == SYNKIT_MINIMUM_CLASSES || r.query == SYNKIT_MINIMUM_PROOF) {
        optimize = true;
        if (best > 0 && best < INF)
          visit(0, 0);
        budget.check();
        s.optimization_seconds = budget.seconds() - phase;
        if (best < INF) {
          s.minimum_proven = 1;
          s.minimum2 = s.lower_bound2 = best;
        }
        if (r.query == SYNKIT_MINIMUM_PROOF || best == INF) {
          s.search_complete = 1;
          s.elapsed_seconds = budget.seconds();
          return 0;
        }
        threshold = best;
      } else
        threshold = r.target2;
      optimize = false;
      phase = budget.seconds();
      visit(0, 0);
      budget.check();
      s.enumeration_seconds = budget.seconds() - phase;
      s.search_complete = 1;
      if (s.labeled_count < 0)
        s.labeled_count = 0;
    } catch (const Stop &stop) {
      if (optimize)
        s.optimization_seconds = budget.seconds() - phase;
      else
        s.enumeration_seconds = budget.seconds() - phase;
      stopped(s, stop.reason);
    }
    s.elapsed_seconds = budget.seconds();
    return 0;
  }
};
} // namespace

int solve_abb(const Problem &p, const Request &r, const uint8_t *domains,
              Callback cb, void *data, Statistics &s) {
  return AssignmentSearch(p, r, domains, cb, data, s).run();
}
} // namespace synkit_mapper

namespace {
thread_local std::string assignment_error;
}
extern "C" const char *synkit_mapper_assignment_error(void) {
  return assignment_error.c_str();
}
extern "C" int synkit_mapper_kernel_domains(
    int32_t n, int64_t count, const int32_t *mappings,
    const int32_t *types_r, const int32_t *types_p, int32_t close_block_size,
    uint8_t *allowed) {
  assignment_error.clear();
  try {
    if (n < 1 || n > 1024 || count < 1 || count > 1000000 || !mappings ||
        !types_r || !types_p || !allowed || close_block_size < 0)
      throw std::invalid_argument("Invalid native uncertainty-kernel arguments");
    std::fill(allowed, allowed + n * n, uint8_t{0});
    for (int64_t k = 0; k < count; ++k) {
      std::vector<bool> seen(n, false);
      for (int i = 0; i < n; ++i) {
        const int j = mappings[k * n + i];
        if (j < 0 || j >= n || seen[j] || types_r[i] != types_p[j])
          throw std::invalid_argument("Kernel candidate is not a compatible bijection");
        seen[j] = true;
        allowed[i * n + j] = 1;
      }
    }
    std::map<int32_t, std::vector<int>> rows;
    for (int i = 0; i < n; ++i) {
      int size = 0;
      for (int j = 0; j < n; ++j)
        size += allowed[i * n + j];
      if (size > 1)
        rows[types_r[i]].push_back(i);
    }
    for (const auto &[type, block] : rows) {
      (void)type;
      if (static_cast<int>(block.size()) > close_block_size)
        continue;
      std::set<int> products;
      for (int i : block)
        for (int j = 0; j < n; ++j)
          if (allowed[i * n + j])
            products.insert(j);
      if (products.size() == block.size())
        for (int i : block)
          for (int j : products)
            allowed[i * n + j] = 1;
    }
    return 0;
  } catch (const std::exception &e) {
    assignment_error = e.what();
    return -1;
  }
}
