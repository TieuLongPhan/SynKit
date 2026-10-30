#include "synkit_mapper/mces/mces.h"
#include "../solvers/seed.h"
#include "synkit_mapper/core/runtime.h"
#include <limits>
#include <numeric>
#include <sstream>
#include <string>

namespace synkit_mapper {
namespace {
struct Vertex {
  int32_t source, target;
};
struct Metadata {
  int64_t vertices = 0, edges = 0, graph_bytes = 0;
};
thread_local Metadata metadata;
struct PhaseTimer {
  double &total;
  Clock::time_point start = Clock::now();
  ~PhaseTimer() { total += elapsed(start); }
};
class CliqueSearch {
  const Problem &p;
  const Request &r;
  Callback callback;
  void *userdata;
  Statistics &s;
  Budget budget;
  std::vector<Vertex> vertices;
  std::vector<std::vector<uint64_t>> adjacency;
  std::vector<int64_t> pair_upper;
  std::vector<int> clique;
  int64_t constant = 0, best_overlap = -1, target_overlap = 0, base_bytes = 0,
          live_bytes = 0;
  bool optimize = false, at_most = false;
  bool adjacent(int a, int b) const {
    return (adjacency[a][b / 64] >> (b % 64)) & 1ULL;
  }
  int64_t weight(int a, int b) const {
    const auto &u = vertices[a], &v = vertices[b];
    return std::min(p.a2[u.source * p.n + v.source],
                    p.b2[u.target * p.n + v.target]);
  }
  void memory(int64_t extra) {
    int64_t proposed = live_bytes + extra;
    if (r.memory_limit_bytes >= 0 &&
        base_bytes + proposed > r.memory_limit_bytes)
      throw Stop{SYNKIT_STOP_MEMORY};
    live_bytes = proposed;
    s.peak_memory_bytes =
        std::max(s.peak_memory_bytes, base_bytes + live_bytes);
  }
  void visit(const std::vector<int> &candidates, int64_t overlap) {
    budget.check();
    if (r.max_states >= 0 && s.nodes >= r.max_states)
      throw Stop{SYNKIT_STOP_STATES};
    ++s.nodes;
    if (static_cast<int>(clique.size()) == p.n) {
      ++s.leaves;
      std::vector<int32_t> mapping(p.n, -1);
      for (int v : clique)
        mapping[vertices[v].source] = vertices[v].target;
      int64_t cost = mapping_cost2(p, mapping);
      if (cost != constant - 2 * overlap)
        throw std::runtime_error(
            "Weighted MCES overlap disagrees with chemical rescoring");
      if (s.incumbent2 < 0 || cost < s.incumbent2)
        s.incumbent2 = cost;
      if (optimize) {
        best_overlap = std::max(best_overlap, overlap);
        return;
      }
      if ((at_most && overlap < target_overlap) ||
          (!at_most && overlap != target_overlap))
        return;
      ++s.candidate_count;
      s.labeled_count = s.candidate_count;
      int stop = 0;
      {
        PhaseTimer timer{s.callback_seconds};
        if (callback)
          stop = callback(mapping.data(), cost, userdata);
      }
      if (stop)
        throw Stop{SYNKIT_STOP_CALLBACK};
      budget.check();
      if (r.max_candidates >= 0 && s.candidate_count >= r.max_candidates)
        throw Stop{SYNKIT_STOP_CANDIDATES};
      return;
    }
    // Source rows form a proper coloring of the association graph: no clique
    // contains two vertices with the same source. A full clique needs every
    // remaining color, giving both a size bound and safe weighted bounds.
    std::vector<uint8_t> selected(p.n, 0);
    for (int v : clique)
      selected[vertices[v].source] = 1;
    std::vector<int64_t> maximum(p.n, -1),
        minimum(p.n, std::numeric_limits<int64_t>::max());
    size_t checked = 0;
    for (int v : candidates) {
      if (!(++checked % 1024))
        budget.check();
      int row = vertices[v].source;
      int64_t increment = 0;
      for (int q : clique)
        increment += weight(v, q);
      maximum[row] = std::max(maximum[row], increment);
      minimum[row] = std::min(minimum[row], increment);
    }
    int64_t upper = overlap, lower = overlap;
    for (int i = 0; i < p.n; ++i)
      if (!selected[i]) {
        if (maximum[i] < 0)
          return;
        upper += maximum[i];
        lower += minimum[i];
        for (int k = i + 1; k < p.n; ++k)
          if (!selected[k])
            upper += pair_upper[i * p.n + k];
      }
    if (optimize
            ? upper <= best_overlap
            : upper < target_overlap || (!at_most && lower > target_overlap))
      return;
    for (size_t i = 0; i < candidates.size(); ++i) {
      budget.check();
      int v = candidates[i];
      std::vector<int> next;
      next.reserve(candidates.size() - i - 1);
      for (size_t j = i + 1; j < candidates.size(); ++j)
        if (adjacent(v, candidates[j]))
          next.push_back(candidates[j]);
      if (next.size() + clique.size() + 1 < static_cast<size_t>(p.n))
        continue;
      int64_t increment = 0;
      for (int q : clique)
        increment += weight(v, q);
      clique.push_back(v);
      int64_t bytes = next.capacity() * sizeof(int);
      memory(bytes);
      try {
        visit(next, overlap + increment);
      } catch (...) {
        live_bytes -= bytes;
        clique.pop_back();
        throw;
      }
      live_bytes -= bytes;
      clique.pop_back();
    }
  }

public:
  CliqueSearch(const Problem &problem, const Request &request, Callback cb,
               void *data, Statistics &stats)
      : p(problem), r(request), callback(cb), userdata(data), s(stats),
        budget(request) {}
  void solve() {
    struct Finish {
      Statistics &s;
      Budget &b;
      ~Finish() { s.elapsed_seconds = b.seconds(); }
    } finish{s, budget};
    try {
      budget.check();
      std::vector<int32_t> seed;
      {
        PhaseTimer timer{s.model_seconds};
        for (int i = 0; i < p.n; ++i) {
          budget.check();
          for (int j = 0; j < p.n; ++j)
            if (p.types_r[i] == p.types_p[j])
              vertices.push_back({i, j});
          for (int k = i + 1; k < p.n; ++k)
            constant += p.a2[i * p.n + k] + p.b2[i * p.n + k];
        }
        if (vertices.size() > 8192) {
          unsupported(s);
          return;
        }
        metadata.vertices = vertices.size();
        int64_t words = (vertices.size() + 63) / 64;
        base_bytes = sizeof(*this) + vertices.capacity() * sizeof(Vertex) +
                     vertices.size() * (sizeof(std::vector<uint64_t>) +
                                        words * sizeof(uint64_t)) +
                     static_cast<int64_t>(p.n) * p.n * sizeof(int64_t) +
                     p.n * sizeof(int);
        metadata.graph_bytes = base_bytes;
        memory(0);
        adjacency.resize(vertices.size(), std::vector<uint64_t>(words, 0));
        pair_upper.assign(p.n * p.n, 0);
        size_t checks = 0;
        for (size_t i = 0; i < vertices.size(); ++i) {
          budget.check();
          for (size_t j = i + 1; j < vertices.size(); ++j) {
            if (!(++checks % 1024))
              budget.check();
            const auto &u = vertices[i], &v = vertices[j];
            if (u.source == v.source || u.target == v.target)
              continue;
            adjacency[i][j / 64] |= uint64_t{1} << (j % 64);
            adjacency[j][i / 64] |= uint64_t{1} << (i % 64);
            ++metadata.edges;
            int64_t w = weight(i, j);
            auto &bound = pair_upper[u.source * p.n + v.source];
            bound = std::max(bound, w);
            pair_upper[v.source * p.n + u.source] = bound;
          }
        }
        seed = solvers::detail::feasible_seed(p, budget);
        if (!seed.empty())
          s.incumbent2 = mapping_cost2(p, seed);
      }
      std::vector<int> root(vertices.size());
      std::iota(root.begin(), root.end(), 0);
      int64_t root_bytes = root.capacity() * sizeof(int);
      memory(root_bytes);
      if (r.query == SYNKIT_MINIMUM_CLASSES ||
          r.query == SYNKIT_MINIMUM_PROOF) {
        optimize = true;
        if (s.incumbent2 >= 0)
          best_overlap = (constant - s.incumbent2) / 2;
        {
          PhaseTimer timer{s.optimization_seconds};
          visit(root, 0);
        }
        if (best_overlap < 0) {
          s.search_complete = 1;
          return;
        }
        s.minimum_proven = 1;
        s.minimum2 = s.lower_bound2 = constant - 2 * best_overlap;
        if (r.query == SYNKIT_MINIMUM_PROOF) {
          s.search_complete = 1;
          return;
        }
        target_overlap = best_overlap;
      } else {
        int64_t threshold = constant - r.target2;
        if (r.query == SYNKIT_EXACT_DISTANCE &&
            (threshold < 0 || threshold % 2)) {
          s.search_complete = 1;
          return;
        }
        at_most = r.query == SYNKIT_AT_MOST_DISTANCE;
        target_overlap = threshold <= 0 ? 0 : (threshold + 1) / 2;
      }
      optimize = false;
      {
        PhaseTimer timer{s.enumeration_seconds};
        visit(root, 0);
      }
      s.search_complete = 1;
    } catch (const Stop &stop) {
      stopped(s, stop.reason);
    }
    s.elapsed_seconds = budget.seconds();
  }
};
} // namespace
const char *mces_run_metadata() {
  static thread_local std::string value;
  std::ostringstream out;
  out << "{\"architecture\":\"weighted_association_graph_size_n_clique\","
         "\"association_vertices\":"
      << metadata.vertices << ",\"association_edges\":" << metadata.edges
      << ",\"accounted_graph_bytes\":" << metadata.graph_bytes
      << ",\"memory_scope\":\"accounted_native_graph_and_clique_candidates\","
         "\"objective_equivalence\":\"D2=C-2*weighted_common_edge_overlap\"}";
  value = out.str();
  return value.c_str();
}
int solve_mces(const Problem &p, const Request &r, Callback callback,
               void *userdata, Statistics &s) {
  initialize(s, r);
  metadata = {};
  s.labeled_count = 0;
  s.peak_frontier = s.peak_frontier_bytes = -1;
  if (r.symmetry || r.classify || r.max_classes != -1 || r.max_frontier != -1 ||
      p.n > 256)
    return unsupported(s);
  CliqueSearch search(p, r, callback, userdata, s);
  search.solve();
  return 0;
}
} // namespace synkit_mapper
