// Bounded feasible-seed search only. No failed search supplies a lower bound.
#include "exact_helpers.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Vec = std::vector<int>;
struct Stop {};
struct Search {
  int n, side, lost;
  const int64_t *a, *b, *host;
  const int32_t *tr, *tp, *pt, *ht;
  int64_t target;
  double seconds;
  int64_t node_limit, *stats;
  Clock::time_point start;
  std::vector<int64_t> pattern, original;
  struct Edge {
    int u, v;
    int64_t weight;
    Vec drops;
    bool bridge = false;
  };
  std::vector<Edge> edges;
  Vec types, typep, typeh, counts, host_degrees, mapping, used, active,
      domain_count, removed, mandatory;
  std::vector<Vec> adjacency;
  std::vector<std::vector<std::pair<int, int64_t>>> host_neighbors;
  std::vector<std::vector<std::vector<int64_t>>> host_profiles;
  std::vector<char> domains;
  bool found = false;
  Vec output;
  int64_t output_cost = -1;
  void poll() {
    if (std::chrono::duration<double>(Clock::now() - start).count() >=
        seconds) {
      stats[4] = 1;
      throw Stop{};
    }
    if (stats[1] >= node_limit) {
      stats[4] = 2;
      throw Stop{};
    }
  }
  Search(int size, const int64_t *aa, const int64_t *bb, const int32_t *ta,
         const int32_t *tb, int64_t goal, double limit, int64_t nodes,
         int64_t *statistics)
      : n(size), a(aa), b(bb), tr(ta), tp(tb), target(goal), seconds(limit),
        node_limit(nodes), stats(statistics), start(Clock::now()) {}
  bool prepare() {
    int64_t ma = 0, mb = 0;
    std::map<int32_t, int> registry, balance;
    for (int i = 0; i < n; ++i) {
      if (a[i * n + i] || b[i * n + i])
        return false;
      registry.emplace(tr[i], 0);
      registry.emplace(tp[i], 0);
      ++balance[tr[i]];
      --balance[tp[i]];
      for (int j = 0; j < i; ++j) {
        if (a[i * n + j] < 0 || b[i * n + j] < 0 || a[i * n + j] > 1024 ||
            b[i * n + j] > 1024 || a[i * n + j] != a[j * n + i] ||
            b[i * n + j] != b[j * n + i])
          return false;
        ma += a[i * n + j];
        mb += b[i * n + j];
      }
    }
    for (auto entry : balance)
      if (entry.second)
        return false;
    const int64_t gap = std::abs(ma - mb), excess = target - gap;
    if (!gap || excess < 0 || excess % 2 || excess > 8)
      return false;
    lost = static_cast<int>(excess / 2);
    side = ma < mb ? 0 : 1;
    const auto *small = side ? b : a;
    host = side ? a : b;
    pt = side ? tp : tr;
    ht = side ? tr : tp;
    original.assign(small, small + n * n);
    pattern = original;
    int next = 0;
    for (auto &entry : registry)
      entry.second = next++;
    typep.resize(n);
    typeh.resize(n);
    counts.assign(next, 0);
    for (int i = 0; i < n; ++i) {
      typep[i] = registry.at(pt[i]);
      typeh[i] = registry.at(ht[i]);
      ++counts[typep[i]];
    }
    stats[6] = side;
    stats[7] = lost;
    host_neighbors.resize(n);
    host_degrees.assign(n, 0);
    host_profiles.resize(n, std::vector<std::vector<int64_t>>(next));
    for (int i = 0; i < n; ++i) {
      poll();
      for (int j = 0; j < n; ++j)
        if (host[i * n + j]) {
          host_neighbors[i].push_back({j, host[i * n + j]});
          host_profiles[i][typeh[j]].push_back(host[i * n + j]);
          ++host_degrees[i];
        }
    }
    for (auto &row : host_profiles)
      for (auto &cell : row)
        std::sort(cell.begin(), cell.end(), std::greater<int64_t>());
    std::map<std::pair<int, int>, std::set<int64_t>> palette;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < i; ++j) {
        auto p = std::minmax(typeh[i], typeh[j]);
        palette[{p.first, p.second}].insert(host[i * n + j]);
      }
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < i; ++j)
        if (small[i * n + j]) {
          Edge edge{i, j, small[i * n + j], {}, false};
          auto p = std::minmax(typep[i], typep[j]);
          for (int k = 1; k <= lost && k <= edge.weight; ++k)
            if (palette[{p.first, p.second}].count(edge.weight - k))
              edge.drops.push_back(k);
          edges.push_back(std::move(edge));
        }
    if (edges.size() > 64)
      return false;
    // Soft ordering: removing a bridge among non-singleton vertices is often
    // the newly formed bond. This affects only proposal order, never domains.
    std::vector<std::vector<std::pair<int, int>>> graph(n);
    for (int k = 0; k < static_cast<int>(edges.size()); ++k) {
      auto &e = edges[k];
      if (counts[typep[e.u]] > 1 && counts[typep[e.v]] > 1) {
        graph[e.u].push_back({e.v, k});
        graph[e.v].push_back({e.u, k});
      }
    }
    Vec tin(n, -1), low(n);
    int timer = 0;
    std::function<void(int, int)> dfs = [&](int u, int parent) {
      tin[u] = low[u] = timer++;
      for (auto e : graph[u])
        if (e.second != parent) {
          if (tin[e.first] < 0) {
            dfs(e.first, e.second);
            low[u] = std::min(low[u], low[e.first]);
            if (low[e.first] > tin[u])
              edges[e.second].bridge = true;
          } else
            low[u] = std::min(low[u], tin[e.first]);
        }
    };
    for (int i = 0; i < n; ++i)
      if (tin[i] < 0)
        dfs(i, -1);
    std::stable_sort(
        edges.begin(), edges.end(), [&](const Edge &x, const Edge &y) {
          if (x.bridge != y.bridge)
            return x.bridge > y.bridge;
          const bool xf = counts[typep[x.u]] == 1 || counts[typep[x.v]] == 1,
                     yf = counts[typep[y.u]] == 1 || counts[typep[y.v]] == 1;
          if (xf != yf)
            return xf < yf;
          return x.weight < y.weight;
        });
    return true;
  }
  bool remove(int x, int y) {
    int at = x * n + y;
    if (!domains[at])
      return true;
    domains[at] = 0;
    removed.push_back(at);
    ++stats[2];
    return --domain_count[x] > 0;
  }
  void restore(size_t checkpoint) {
    while (removed.size() > checkpoint) {
      int at = removed.back();
      removed.pop_back();
      domains[at] = 1;
      ++domain_count[at / n];
    }
  }
  bool propagate() {
    std::deque<std::pair<int, int>> queue;
    for (int x : active)
      for (int z : adjacency[x])
        queue.push_back({x, z});
    int iterations = 0;
    while (!queue.empty()) {
      if ((++iterations & 63) == 0)
        poll();
      auto edge = queue.front();
      queue.pop_front();
      int x = edge.first, z = edge.second;
      bool changed = false;
      for (int y = 0; y < n; ++y)
        if (domains[x * n + y]) {
          bool supported = false;
          for (auto neighbor : host_neighbors[y])
            if (neighbor.first != y && neighbor.second >= pattern[x * n + z] &&
                domains[z * n + neighbor.first]) {
              supported = true;
              break;
            }
          if (!supported) {
            if (!remove(x, y))
              return false;
            changed = true;
          }
        }
      if (changed)
        for (int neighbor : adjacency[x])
          queue.push_back({neighbor, x});
    }
    return true;
  }
  bool initialize() {
    mapping.assign(n, -1);
    used.assign(n, 0);
    active.clear();
    adjacency.assign(n, {});
    for (int i = 0; i < n; ++i) {
      if (counts[typep[i]] == 1)
        for (int j = 0; j < n; ++j)
          if (typep[i] == typeh[j]) {
            mapping[i] = j;
            used[j] = 1;
            break;
          }
      for (int j = 0; j < n; ++j)
        if (pattern[i * n + j])
          adjacency[i].push_back(j);
      if (!adjacency[i].empty())
        active.push_back(i);
    }
    stats[8] = std::max<int64_t>(stats[8], active.size());
    domains.assign(n * n, 0);
    domain_count.assign(n, 0);
    removed.clear();
    for (int x : active) {
      poll();
      std::vector<std::vector<int64_t>> profile(counts.size());
      for (int z : adjacency[x])
        profile[typep[z]].push_back(pattern[x * n + z]);
      for (auto &cell : profile)
        std::sort(cell.begin(), cell.end(), std::greater<int64_t>());
      for (int y = 0; y < n; ++y)
        if (typep[x] == typeh[y] && (mapping[x] < 0 || mapping[x] == y) &&
            (!used[y] || mapping[x] == y)) {
          bool legal = true;
          for (size_t t = 0; t < profile.size() && legal; ++t) {
            const auto &need = profile[t];
            const auto &available = host_profiles[y][t];
            if (need.size() > available.size()) {
              legal = false;
              break;
            }
            for (size_t k = 0; k < need.size(); ++k)
              if (need[k] > available[k]) {
                legal = false;
                break;
              }
          }
          for (int z : adjacency[x])
            if (mapping[z] >= 0 &&
                pattern[x * n + z] > host[y * n + mapping[z]]) {
              legal = false;
              break;
            }
          if (legal) {
            domains[x * n + y] = 1;
            ++domain_count[x];
          }
        }
      if (!domain_count[x])
        return false;
    }
    return propagate();
  }
  bool complete() {
    Vec full = mapping;
    std::vector<char> taken(used.begin(), used.end());
    for (int i = 0; i < n; ++i)
      if (full[i] < 0) {
        int y = 0;
        while (y < n && (taken[y] || typep[i] != typeh[y]))
          ++y;
        if (y == n)
          return false;
        full[i] = y;
        taken[y] = 1;
      }
    Vec map(n, -1);
    if (side == 0)
      map = full;
    else
      for (int i = 0; i < n; ++i)
        map[full[i]] = i;
    Vec seen(n, 0);
    for (int i = 0; i < n; ++i)
      if (map[i] < 0 || map[i] >= n || seen[map[i]]++ || tr[i] != tp[map[i]])
        return false;
    int64_t cost = 0;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < i; ++j)
        cost += std::abs(a[i * n + j] - b[map[i] * n + map[j]]);
    ++stats[3];
    if (cost > target)
      return false;
    output = std::move(map);
    output_cost = cost;
    found = true;
    return true;
  }
  bool visit() {
    poll();
    ++stats[1];
    int x = -1, best = n + 1, tie = -1;
    for (int i : active)
      if (mapping[i] < 0) {
        int assigned = 0;
        for (int j : adjacency[i])
          assigned += mapping[j] >= 0;
        if (domain_count[i] < best ||
            (domain_count[i] == best && assigned > tie)) {
          x = i;
          best = domain_count[i];
          tie = assigned;
        }
      }
    if (x < 0)
      return complete();
    Vec candidates;
    for (int y = 0; y < n; ++y)
      if (domains[x * n + y] && !used[y])
        candidates.push_back(y);
    std::stable_sort(candidates.begin(), candidates.end(), [&](int u, int v) {
      return host_degrees[u] < host_degrees[v];
    });
    for (int y : candidates) {
      poll();
      size_t checkpoint = removed.size();
      mapping[x] = y;
      used[y] = 1;
      bool legal = true;
      for (int z = 0; z < n && legal; ++z)
        if (z != y)
          legal = remove(x, z);
      for (int z : active)
        if (z != x && mapping[z] < 0 && legal)
          legal = remove(z, y);
      if (legal && propagate() && visit())
        return true;
      restore(checkpoint);
      used[y] = 0;
      mapping[x] = -1;
    }
    return false;
  }
  bool configurations(int first, int remaining) {
    poll();
    if (!remaining) {
      ++stats[0];
      if (initialize() && visit())
        return true;
      return false;
    }
    for (int k = first; k < static_cast<int>(edges.size()); ++k) {
      auto &e = edges[k];
      for (int drop : e.drops)
        if (drop <= remaining) {
          pattern[e.u * n + e.v] = pattern[e.v * n + e.u] = e.weight - drop;
          if (configurations(k + 1, remaining - drop))
            return true;
          pattern[e.u * n + e.v] = pattern[e.v * n + e.u] = e.weight;
        }
    }
    return false;
  }
};
} // namespace
int synkit_mapper::pabs::sparse_embedding_seed(
    int n, const int64_t *a, const int64_t *b, const int32_t *tr,
    const int32_t *tp, int64_t target, double seconds, int64_t nodes,
    int32_t *out, int64_t *cost, int64_t *statistics) {
  if (statistics)
    std::fill(statistics, statistics + 10, 0);
  if (cost)
    *cost = -1;
  if (n < 1 || n > 512 || !a || !b || !tr || !tp || !out || !cost ||
      !statistics || target < 0 || !std::isfinite(seconds) || seconds < 0 ||
      nodes < 1)
    return -1;
  Search search(n, a, b, tr, tp, target, seconds, nodes, statistics);
  auto finish = [&](int status) {
    statistics[5] = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        Clock::now() - search.start)
                        .count();
    return status;
  };
  try {
    if (!search.prepare())
      return finish(-1);
    if (search.configurations(0, search.lost)) {
      std::copy(search.output.begin(), search.output.end(), out);
      *cost = search.output_cost;
      return finish(1);
    }
    statistics[9] = 1;
    return finish(0);
  } catch (const Stop &) {
    return finish(0);
  } catch (...) {
    statistics[4] = 3;
    return finish(-1);
  }
}
