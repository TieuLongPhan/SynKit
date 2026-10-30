#include "synkit_mapper/core/runtime.h"

#include <functional>
#include <limits>
#include <numeric>

namespace synkit_mapper {
namespace {
using Pair = std::pair<int, int>;
struct ProofDone {};

// Enumerate k-subsets themselves, not vertex assignments with a renamed cost.
void subsets(int n, int k,
             const std::function<void(const std::vector<int> &)> &f,
             const Budget &budget) {
  if (k < 0 || k > n)
    return;
  std::vector<int> indices(k);
  std::iota(indices.begin(), indices.end(), 0);
  while (true) {
    budget.check();
    f(indices);
    int position = k - 1;
    while (position >= 0 && indices[position] == n - k + position)
      --position;
    if (position < 0)
      break;
    ++indices[position];
    for (int i = position + 1; i < k; ++i)
      indices[i] = indices[i - 1] + 1;
  }
}
} // namespace

int solve_edit_support(const Problem &p, const Request &r, Callback callback,
                       void *userdata, Statistics &s) {
  initialize(s, r);
  if (p.n > 256 || r.classify || r.max_classes != -1 || r.symmetry ||
      r.max_frontier != -1 || r.memory_limit_bytes != -1)
    return unsupported(s);
  for (int i = 0; i < p.n * p.n; ++i)
    if ((p.a2[i] != 0 && p.a2[i] != 2) || (p.b2[i] != 0 && p.b2[i] != 2))
      return unsupported(s);
  Budget budget(r);
  std::vector<Pair> present, absent;
  std::vector<uint8_t> original(p.n * p.n), product(p.n * p.n);
  std::vector<int> product_degree(p.n);
  std::vector<std::map<int32_t, int>> product_neighbours(p.n);
  for (int i = 0; i < p.n; ++i) {
    budget.check();
    for (int k = 0; k < p.n; ++k) {
      original[i * p.n + k] = p.a2[i * p.n + k] != 0;
      product[i * p.n + k] = p.b2[i * p.n + k] != 0;
      if (product[i * p.n + k]) {
        ++product_degree[i];
        ++product_neighbours[i][p.types_p[k]];
      }
      if (k < i)
        (original[i * p.n + k] ? present : absent).emplace_back(i, k);
    }
  }
  int target_edges = 0;
  for (int value : product_degree)
    target_edges += value;
  target_edges /= 2;
  const int source_edges = static_cast<int>(present.size());
  const int lower = std::abs(source_edges - target_edges);
  const int upper =
      std::min(source_edges + target_edges,
               2 * static_cast<int>(present.size() + absent.size()) -
                   source_edges - target_edges);
  const bool minimum =
      r.query == SYNKIT_MINIMUM_CLASSES || r.query == SYNKIT_MINIMUM_PROOF;
  double proof_at = -1;
  auto node = [&]() {
    budget.check();
    if (s.nodes == std::numeric_limits<int64_t>::max() ||
        (r.max_states >= 0 && s.nodes >= r.max_states))
      throw Stop{SYNKIT_STOP_STATES};
    ++s.nodes;
  };
  try {
    s.model_seconds = budget.seconds();
    const int first =
        r.query == SYNKIT_EXACT_DISTANCE
            ? static_cast<int>(std::min<int64_t>(r.target2 / 2, upper + 1))
            : lower;
    const int last =
        minimum ? upper
                : static_cast<int>(std::min<int64_t>(r.target2 / 2, upper));
    if (r.query == SYNKIT_EXACT_DISTANCE && r.target2 % 2) {
      s.search_complete = 1;
      s.labeled_count = 0;
    } else {
      for (int shell = first; shell <= last; ++shell) {
        if (shell < lower || (shell + source_edges - target_edges) % 2)
          continue;
        const int deletions = (shell + source_edges - target_edges) / 2;
        const int insertions = shell - deletions;
        bool shell_has_mapping = false;
        subsets(
            static_cast<int>(present.size()), deletions,
            [&](const std::vector<int> &removed) {
              subsets(
                  static_cast<int>(absent.size()), insertions,
                  [&](const std::vector<int> &added) {
                    node();
                    auto edited = original;
                    for (int index : removed) {
                      auto [i, k] = present[index];
                      edited[i * p.n + k] = edited[k * p.n + i] = 0;
                    }
                    for (int index : added) {
                      auto [i, k] = absent[index];
                      edited[i * p.n + k] = edited[k * p.n + i] = 1;
                    }
                    std::vector<std::vector<int>> domains(p.n);
                    for (int i = 0; i < p.n; ++i) {
                      budget.check();
                      int degree = 0;
                      std::map<int32_t, int> neighbours;
                      for (int k = 0; k < p.n; ++k)
                        if (edited[i * p.n + k]) {
                          ++degree;
                          ++neighbours[p.types_r[k]];
                        }
                      for (int j = 0; j < p.n; ++j)
                        if (p.types_r[i] == p.types_p[j] &&
                            degree == product_degree[j] &&
                            neighbours == product_neighbours[j])
                          domains[i].push_back(j);
                      if (domains[i].empty())
                        return;
                    }
                    std::vector<int> order(p.n);
                    std::iota(order.begin(), order.end(), 0);
                    std::stable_sort(
                        order.begin(), order.end(), [&](int i, int k) {
                          return domains[i].size() < domains[k].size();
                        });
                    std::vector<int32_t> mapping(p.n, -1);
                    std::vector<bool> used(p.n, false);
                    std::function<void(int)> isomorphisms = [&](int depth) {
                      node();
                      if (depth == p.n) {
                        ++s.leaves;
                        shell_has_mapping = true;
                        s.incumbent2 =
                            s.incumbent2 < 0
                                ? 2 * shell
                                : std::min<int64_t>(s.incumbent2, 2 * shell);
                        if (minimum && !s.minimum_proven) {
                          s.minimum_proven = 1;
                          s.minimum2 = s.lower_bound2 = 2 * shell;
                          proof_at = budget.seconds();
                        }
                        if (r.query == SYNKIT_MINIMUM_PROOF)
                          throw ProofDone{};
                        ++s.candidate_count;
                        if (callback) {
                          auto before = Clock::now();
                          int stop =
                              callback(mapping.data(), 2 * shell, userdata);
                          s.callback_seconds += elapsed(before);
                          if (stop)
                            throw Stop{SYNKIT_STOP_CALLBACK};
                        }
                        budget.check();
                        if (r.max_candidates >= 0 &&
                            s.candidate_count >= r.max_candidates)
                          throw Stop{SYNKIT_STOP_CANDIDATES};
                        return;
                      }
                      int i = order[depth];
                      for (int j : domains[i]) {
                        if (used[j])
                          continue;
                        bool compatible = true;
                        for (int d = 0; d < depth; ++d) {
                          int k = order[d];
                          if (edited[i * p.n + k] !=
                              product[j * p.n + mapping[k]]) {
                            compatible = false;
                            break;
                          }
                        }
                        if (!compatible)
                          continue;
                        mapping[i] = j;
                        used[j] = true;
                        isomorphisms(depth + 1);
                        used[j] = false;
                        mapping[i] = -1;
                        if (r.query == SYNKIT_MINIMUM_PROOF && s.minimum_proven)
                          return;
                      }
                    };
                    isomorphisms(0);
                  },
                  budget);
            },
            budget);
        if (minimum && shell_has_mapping)
          break;
      }
      s.search_complete = 1;
      if (r.query != SYNKIT_MINIMUM_PROOF)
        s.labeled_count = s.candidate_count;
    }
  } catch (const ProofDone &) {
    s.search_complete = 1;
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
