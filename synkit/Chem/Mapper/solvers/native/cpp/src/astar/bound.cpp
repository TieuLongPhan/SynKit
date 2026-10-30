#include "synkit_mapper/astar/bound.h"

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace synkit_mapper {
namespace astar {
namespace {

constexpr int64_t infinity = std::numeric_limits<int64_t>::max() / 16;

int64_t add(int64_t a, int64_t b) {
  if (a < 0 || b < 0 || a >= infinity || b >= infinity || a >= infinity - b)
    throw std::overflow_error("A* bound exceeds its checked integer range");
  return a + b;
}

int64_t difference(int64_t a, int64_t b) { return a >= b ? a - b : b - a; }

// Dense primal-dual assignment. Forbidden entries are infinity, not a large
// finite edit penalty that could silently admit an incompatible atom pairing.
int64_t assignment_cost(const std::vector<std::vector<int64_t>> &costs,
                        const std::function<void()> &poll,
                        std::vector<int32_t> *mapping = nullptr) {
  const int n = static_cast<int>(costs.size());
  std::vector<int64_t> u(n + 1), v(n + 1);
  std::vector<int> p(n + 1), way(n + 1);
  for (int i = 1; i <= n; ++i) {
    if (poll)
      poll();
    p[0] = i;
    int j0 = 0;
    std::vector<int64_t> best(n + 1, infinity);
    std::vector<bool> used(n + 1, false);
    do {
      used[j0] = true;
      const int i0 = p[j0];
      int j1 = 0;
      int64_t delta = infinity;
      for (int j = 1; j <= n; ++j) {
        if (used[j])
          continue;
        if (costs[i0 - 1][j - 1] != infinity) {
          const int64_t reduced = costs[i0 - 1][j - 1] - u[i0] - v[j];
          if (reduced < best[j]) {
            best[j] = reduced;
            way[j] = j0;
          }
        }
        if (best[j] < delta) {
          delta = best[j];
          j1 = j;
        }
      }
      if (delta == infinity)
        throw std::invalid_argument("No compatible residual bijection");
      for (int j = 0; j <= n; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else if (best[j] != infinity) {
          best[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);
    do {
      const int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 != 0);
  }
  int64_t total = 0;
  if (mapping)
    mapping->assign(n, -1);
  for (int j = 1; j <= n; ++j)
  {
    total = add(total, costs[p[j] - 1][j - 1]);
    if (mapping)
      (*mapping)[p[j] - 1] = j - 1;
  }
  return total;
}

using Spectrum = std::map<int32_t, std::vector<int64_t>>;

std::vector<Spectrum> spectra(int n, const int64_t *matrix,
                              const int32_t *types,
                              const std::vector<int> &remaining,
                              const std::function<void()> &poll) {
  std::vector<Spectrum> result(remaining.size());
  for (std::size_t i = 0; i < remaining.size(); ++i) {
    if (poll)
      poll();
    for (int k : remaining)
      if (remaining[i] != k)
        result[i][types[k]].push_back(matrix[remaining[i] * n + k]);
    for (auto &group : result[i])
      std::sort(group.second.begin(), group.second.end());
  }
  return result;
}

} // namespace

std::vector<int32_t> profile_assignment_seed(
    int n, const int64_t *a2, const int64_t *b2,
    const int32_t *types_r, const int32_t *types_p,
    const std::function<void()> &poll) {
  if (n < 0 || (n && (!a2 || !b2 || !types_r || !types_p)))
    throw std::invalid_argument("Invalid profile assignment input");
  std::vector<int> vertices;
  for (int i = 0; i < n; ++i)
    vertices.push_back(i);
  const auto source = spectra(n, a2, types_r, vertices, poll);
  const auto target = spectra(n, b2, types_p, vertices, poll);
  std::vector<std::vector<int64_t>> costs(n, std::vector<int64_t>(n, infinity));
  for (int i = 0; i < n; ++i) {
    if (poll)
      poll();
    for (int j = 0; j < n; ++j) {
      if (types_r[i] != types_p[j])
        continue;
      if (source[i].size() != target[j].size())
        throw std::invalid_argument("Incompatible profile type inventories");
      int64_t cost = 0;
      for (const auto &group : source[i]) {
        const auto other = target[j].find(group.first);
        if (other == target[j].end() || other->second.size() != group.second.size())
          throw std::invalid_argument("Incompatible profile type inventories");
        for (std::size_t k = 0; k < group.second.size(); ++k)
          cost = add(cost, difference(group.second[k], other->second[k]));
      }
      costs[i][j] = cost;
    }
  }
  std::vector<int32_t> mapping;
  assignment_cost(costs, poll, &mapping);
  return mapping;
}

int64_t partial_lower_bound2(int n, const int64_t *a2, const int64_t *b2,
                             const int32_t *types_r, const int32_t *types_p,
                             const std::vector<int32_t> &mapping,
                             const std::function<void()> &poll) {
  if (n < 0 || mapping.size() != static_cast<std::size_t>(n) ||
      (n && (!a2 || !b2 || !types_r || !types_p)))
    throw std::invalid_argument("Invalid A* partial-bound input");
  std::vector<int> assigned, remaining, targets;
  std::vector<bool> used(n, false);
  for (int i = 0; i < n; ++i) {
    const int j = mapping[i];
    if (j == -1) {
      remaining.push_back(i);
    } else {
      if (j < 0 || j >= n || used[j] || types_r[i] != types_p[j])
        throw std::invalid_argument("Invalid A* partial bijection");
      used[j] = true;
      assigned.push_back(i);
    }
  }
  for (int j = 0; j < n; ++j)
    if (!used[j])
      targets.push_back(j);
  int64_t committed = 0;
  for (std::size_t i = 0; i < assigned.size(); ++i)
    for (std::size_t k = 0; k < i; ++k)
      committed =
          add(committed,
              difference(a2[assigned[i] * n + assigned[k]],
                         b2[mapping[assigned[i]] * n + mapping[assigned[k]]]));
  if (remaining.empty())
    return committed;
  const auto source_spectra = spectra(n, a2, types_r, remaining, poll);
  const auto target_spectra = spectra(n, b2, types_p, targets, poll);
  std::vector<std::vector<int64_t>> costs(
      remaining.size(), std::vector<int64_t>(targets.size(), infinity));
  for (std::size_t u = 0; u < remaining.size(); ++u) {
    const int i = remaining[u];
    for (std::size_t v = 0; v < targets.size(); ++v) {
      if (poll && v % 16 == 0)
        poll();
      const int j = targets[v];
      if (types_r[i] != types_p[j])
        continue;
      int64_t cross = 0, residual = 0;
      for (int k : assigned)
        cross = add(cross, difference(a2[i * n + k], b2[j * n + mapping[k]]));
      if (source_spectra[u].size() != target_spectra[v].size())
        throw std::invalid_argument("Incompatible residual type inventories");
      for (const auto &group : source_spectra[u]) {
        const auto other = target_spectra[v].find(group.first);
        if (other == target_spectra[v].end() ||
            other->second.size() != group.second.size())
          throw std::invalid_argument("Incompatible residual type inventories");
        for (std::size_t k = 0; k < group.second.size(); ++k)
          residual =
              add(residual, difference(group.second[k], other->second[k]));
      }
      costs[u][v] = add(add(cross, cross), residual);
    }
  }
  const int64_t doubled_residual_bound = assignment_cost(costs, poll);
  return add(committed,
             doubled_residual_bound / 2 + doubled_residual_bound % 2);
}

} // namespace astar
} // namespace synkit_mapper
