#include "synkit_canon/stereo.h"
#include "stereo_groups.h"
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <memory_resource>
#include <string>
#include <tuple>
#include <vector>

namespace {
using namespace synkit_stereo_detail;
using Clock = std::chrono::steady_clock;
template<class T> using Vector = std::pmr::vector<T>;
using Vec = Vector<int32_t>;
struct Stop { int reason; };
thread_local std::string error;

class Workspace : public std::pmr::memory_resource {
  int64_t cap, used = 0;
  void *do_allocate(size_t bytes, size_t alignment) override {
    if (bytes > static_cast<size_t>(INT64_MAX - used) ||
        (cap >= 0 && bytes > static_cast<size_t>(cap - used))) throw Stop{3};
    void *p = std::pmr::new_delete_resource()->allocate(bytes, alignment);
    used += static_cast<int64_t>(bytes);
    peak = std::max(peak, used);
    return p;
  }
  void do_deallocate(void *p, size_t bytes, size_t alignment) override {
    used -= static_cast<int64_t>(bytes);
    std::pmr::new_delete_resource()->deallocate(p, bytes, alignment);
  }
  bool do_is_equal(const std::pmr::memory_resource &other) const noexcept override {
    return this == &other;
  }
public:
  int64_t peak = 0;
  explicit Workspace(int64_t limit) : cap(limit) {}
};
} // namespace

struct SynkitStereoResult {
  Workspace workspace;
  Vec mapping;
  SynkitStereoStatistics statistics{};
  explicit SynkitStereoResult(int64_t cap) : workspace(cap), mapping(&workspace) {
    statistics.isomorphic = -1;
  }
};

namespace {
int edge_color(const SynkitStereoGraph &g, int a, int b) {
  // Avoid pointer arithmetic on a null neighbors buffer for edgeless graphs.
  int64_t lo = g.offsets[a], hi = g.offsets[a + 1];
  while (lo < hi) {
    const int64_t mid = lo + (hi - lo) / 2;
    if (g.neighbors[mid] < b) lo = mid + 1; else hi = mid;
  }
  return lo < g.offsets[a + 1] && g.neighbors[lo] == b ? g.edge_colors[lo] : -1;
}
struct Frame { int geometry; Image vertices; };
struct Pair { int a, b; uint32_t mask; };
struct Change { size_t pair; uint32_t mask; };

class Search {
  const SynkitStereoRequest &r;
  SynkitStereoResult &out;
  SynkitStereoStatistics &s;
  std::pmr::memory_resource *mem;
  Clock::time_point start;
  Vec ca, cb, forward, reverse, source_count, target_count, available;
  Vector<Frame> fa, fb;
  Vector<Pair> pairs;
  Vector<Vec> affected_a, affected_b;
  Vector<Change> trail;
  int n;
  bool deferred;

  void check() const {
    if (r.cancel_token && synkit_canon_cancelled(r.cancel_token)) throw Stop{5};
    if (r.seconds >= 0 && std::chrono::duration<double>(Clock::now() - start).count() >= r.seconds)
      throw Stop{1};
  }
  void validate_graph(const SynkitStereoGraph &g) {
    if (g.n < 0 || g.n > INT32_MAX || g.arcs < 0 || (g.arcs & 1) ||
        g.frame_count < 0 || g.frame_count > INT32_MAX || !g.offsets ||
        (g.n && !g.vertex_colors) || (g.arcs && (!g.neighbors || !g.edge_colors)) ||
        (g.frame_count && !g.frames)) throw std::invalid_argument("Invalid stereo graph dimensions/buffers");
    if (g.offsets[0] || g.offsets[g.n] != g.arcs)
      throw std::invalid_argument("Invalid CSR endpoints");
    for (int v = 0; v < g.n; ++v) {
      check();
      if (g.vertex_colors[v] < 0 || g.offsets[v] < 0 || g.offsets[v] > g.offsets[v + 1] ||
          g.offsets[v + 1] > g.arcs) throw std::invalid_argument("Invalid CSR row/color");
    }
    for (int v = 0; v < g.n; ++v) for (int64_t k = g.offsets[v]; k < g.offsets[v + 1]; ++k) {
      check();
      int w = g.neighbors[k];
      if (w < 0 || w >= g.n || w == v || g.edge_colors[k] < 0 ||
          (k > g.offsets[v] && g.neighbors[k - 1] >= w))
        throw std::invalid_argument("CSR must be simple, sorted, loop-free with nonnegative colors");
    }
    for (int v = 0; v < g.n; ++v) for (int64_t k = g.offsets[v]; k < g.offsets[v + 1]; ++k) {
      check();
      if (edge_color(g, g.neighbors[k], v) != g.edge_colors[k])
        throw std::invalid_argument("CSR edges and their colors must be symmetric");
    }
    std::pmr::map<std::pair<int, int>, int> loci(mem);
    for (int i = 0; i < g.frame_count; ++i) {
      check();
      const auto &f = g.frames[i];
      const int arity = group(f.geometry).arity;
      bool zero = f.geometry == SYNKIT_SQUARE_PLANAR || f.geometry == SYNKIT_PLANAR_BOND;
      if (zero ? f.parity != 0 : (f.parity != 1 && f.parity != -1))
        throw std::invalid_argument("Only fixed configurations with geometry-valid parity are supported");
      for (int k = 0; k < arity; ++k) {
        if (f.vertices[k] < 0 || f.vertices[k] >= g.n)
          throw std::invalid_argument("Frame reference is not a vertex");
        for (int j = 0; j < k; ++j) if (f.vertices[j] == f.vertices[k])
          throw std::invalid_argument("Frame references must be distinct explicit vertices");
      }
      std::pair<int, int> locus;
      if (bond(f.geometry)) {
        locus = std::minmax(f.vertices[2], f.vertices[3]);
        if (edge_color(g, f.vertices[2], f.vertices[3]) < 0)
          throw std::invalid_argument("Bond frame owners must be adjacent");
        for (int k : {0, 1, 4, 5})
          if (edge_color(g, f.vertices[k], f.vertices[k < 2 ? 2 : 3]) < 0)
            throw std::invalid_argument("Bond frame ligand must be adjacent to its owner");
      } else {
        locus = {f.vertices[0], -1};
        for (int k = 1; k < arity; ++k) if (edge_color(g, f.vertices[0], f.vertices[k]) < 0)
          throw std::invalid_argument("Atom frame ligand must be adjacent to its center");
      }
      if (!loci.emplace(locus, i).second)
        throw std::invalid_argument("Duplicate stereo locus");
    }
  }
  bool equal_histograms(const Vec &a, const Vec &b, int count) {
    Vec histogram(count, 0, mem);
    for (int x : a) ++histogram[x];
    for (int x : b) --histogram[x];
    return std::all_of(histogram.begin(), histogram.end(), [](int x) { return x == 0; });
  }
  bool refine() {
    Vector<Vec> roles_a(mem), roles_b(mem);
    roles_a.resize(n); roles_b.resize(n);
    auto roles = [&](const SynkitStereoGraph &g, Vector<Vec> &rs) {
      for (int i = 0; i < g.frame_count; ++i) {
        check();
        const auto &f = g.frames[i];
        for (int j = 0; j < group(f.geometry).arity; ++j)
          rs[f.vertices[j]].push_back(2 * f.geometry + (owner(f.geometry, j) ? 1 : 0));
      }
      for (auto &row : rs) std::sort(row.begin(), row.end());
    };
    roles(r.source, roles_a); roles(r.target, roles_b);
    std::pmr::map<Vec, int> initial(mem);
    auto initialize = [&](const SynkitStereoGraph &g, const Vector<Vec> &rs, Vec &colors) {
      for (int v = 0; v < n; ++v) {
        check();
        Vec key(mem); key.push_back(g.vertex_colors[v]);
        key.insert(key.end(), rs[v].begin(), rs[v].end());
        auto entry = initial.emplace(std::move(key), static_cast<int>(initial.size()));
        colors.push_back(entry.first->second);
      }
    };
    initialize(r.source, roles_a, ca); initialize(r.target, roles_b, cb);
    int count = static_cast<int>(initial.size());
    if (!equal_histograms(ca, cb, count)) return false;
    while (true) {
      check(); ++s.refinement_rounds;
      std::pmr::map<Vec, int> palette(mem);
      Vec na(mem), nb(mem);
      auto step = [&](const SynkitStereoGraph &g, const Vec &colors, Vec &next) {
        for (int v = 0; v < n; ++v) {
          check();
          Vector<std::pair<int, int>> neighbors(mem);
          for (int64_t k = g.offsets[v]; k < g.offsets[v + 1]; ++k) {
            if ((k & 1023) == 0) check();
            neighbors.emplace_back(g.edge_colors[k], colors[g.neighbors[k]]);
          }
          std::sort(neighbors.begin(), neighbors.end());
          Vec key(mem); key.push_back(colors[v]);
          for (const auto &e : neighbors) { key.push_back(e.first); key.push_back(e.second); }
          auto entry = palette.emplace(std::move(key), static_cast<int>(palette.size()));
          next.push_back(entry.first->second);
        }
      };
      step(r.source, ca, na); step(r.target, cb, nb);
      const int next_count = static_cast<int>(palette.size());
      if (!equal_histograms(na, nb, next_count)) return false;
      ca.swap(na); cb.swap(nb);
      if (count == next_count) return true; // signatures retain old colors, so no merges
      count = next_count;
    }
  }
  uint32_t filter(const Pair &p, uint32_t mask) {
    ++s.stereo_checks;
    const auto &a = fa[p.a]; const auto &b = fb[p.b];
    const auto g = group(a.geometry);
    for (int k = 0; k < g.size; ++k) if (mask & (1U << k)) {
      bool possible = true;
      for (int i = 0; i < g.arity; ++i) {
        const int v = a.vertices[i], w = b.vertices[g.images[k][i]];
        if (ca[v] != cb[w] || (forward[v] >= 0 && forward[v] != w) ||
            (reverse[w] >= 0 && reverse[w] != v)) { possible = false; break; }
      }
      if (!possible) mask &= ~(1U << k);
    }
    return mask;
  }
  bool prepare_pairs() {
    affected_a.resize(n); affected_b.resize(n);
    source_count.resize(fa.size(), 0); target_count.resize(fb.size(), 0);
    for (int a = 0; a < static_cast<int>(fa.size()); ++a) {
      for (int b = 0; b < static_cast<int>(fb.size()); ++b) {
        check();
        if (fa[a].geometry != fb[b].geometry) continue;
        Pair p{a, b, (1U << group(fa[a].geometry).size) - 1U};
        p.mask = filter(p, p.mask);
        if (!p.mask) continue;
        if (pairs.size() >= static_cast<size_t>(INT32_MAX)) throw Stop{3};
        const int id = static_cast<int>(pairs.size());
        pairs.push_back(p); ++source_count[a]; ++target_count[b];
        for (int k = 0; k < group(fa[a].geometry).arity; ++k) {
          affected_a[fa[a].vertices[k]].push_back(id);
          affected_b[fb[b].vertices[k]].push_back(id);
        }
      }
    }
    return deferred ||
      (std::none_of(source_count.begin(), source_count.end(), [](int x) { return x == 0; }) &&
       std::none_of(target_count.begin(), target_count.end(), [](int x) { return x == 0; }));
  }
  bool constrain(int a, int b) {
    auto update = [&](int id) {
      check();
      auto &p = pairs[id];
      if (!p.mask) return true;
      const uint32_t next = filter(p, p.mask);
      if (next == p.mask) return true;
      trail.push_back({static_cast<size_t>(id), p.mask});
      p.mask = next;
      if (!next) {
        --source_count[p.a]; --target_count[p.b];
        if (!source_count[p.a] || !target_count[p.b]) return false;
      }
      return true;
    };
    for (int id : affected_a[a]) if (!update(id)) return false;
    for (int id : affected_b[b]) if (!update(id)) return false;
    return true;
  }
  void restore(size_t mark) {
    while (trail.size() > mark) {
      const auto change = trail.back(); trail.pop_back();
      auto &p = pairs[change.pair];
      if (!p.mask) { ++source_count[p.a]; ++target_count[p.b]; }
      p.mask = change.mask;
    }
  }
  bool final_stereo() {
    // Replay from full proper groups rather than trusting cached masks.
    // Distinct loci and a vertex bijection make these descriptor matches
    // injective; equal inventory size then makes them bijective.
    for (int a = 0; a < static_cast<int>(fa.size()); ++a) {
      bool found = false;
      for (int b = 0; b < static_cast<int>(fb.size()); ++b) {
        check();
        if (fa[a].geometry != fb[b].geometry) continue;
        Pair p{a, b, (1U << group(fa[a].geometry).size) - 1U};
        if (filter(p, p.mask)) { found = true; break; }
      }
      if (!found) return false;
    }
    return true;
  }
  bool structurally_possible(int a, int b) const {
    for (int v = 0; v < n; ++v) if (forward[v] >= 0 &&
        edge_color(r.source, a, v) != edge_color(r.target, b, forward[v])) return false;
    return true;
  }
  bool visit(int depth) {
    check();
    if (r.max_nodes >= 0 && s.nodes >= r.max_nodes) throw Stop{2};
    ++s.nodes;
    if (depth == n) {
      ++s.leaves;
      if (!final_stereo()) { ++s.stereo_pruned; return false; }
      if (r.output_bytes >= 0 && static_cast<uint64_t>(n) * sizeof(int32_t) >
          static_cast<uint64_t>(r.output_bytes)) throw Stop{4};
      out.mapping = forward;
      return true;
    }
    if (depth >= r.max_depth) throw Stop{6};
    // Smallest remaining color domain; prefer the mapped frontier on ties.
    int chosen = -1, best_count = INT32_MAX, best_frontier = -1;
    for (int a = 0; a < n; ++a) if (forward[a] < 0) {
      check();
      const int count = available[ca[a]];
      int frontier = 0;
      for (int64_t k = r.source.offsets[a]; k < r.source.offsets[a + 1]; ++k)
        frontier += forward[r.source.neighbors[k]] >= 0;
      if (!count) return false;
      if (count < best_count || (count == best_count && frontier > best_frontier)) {
        chosen = a; best_count = count; best_frontier = frontier;
      }
    }
    for (int b = 0; b < n; ++b) if (reverse[b] < 0 && ca[chosen] == cb[b]) {
      check();
      if (!structurally_possible(chosen, b)) { ++s.structural_pruned; continue; }
      const auto mark = trail.size();
      forward[chosen] = b; reverse[b] = chosen;
      --available[cb[b]];
      const bool viable = deferred || constrain(chosen, b);
      if (viable && visit(depth + 1)) return true;
      if (!viable) ++s.stereo_pruned;
      restore(mark);
      ++available[cb[b]];
      forward[chosen] = -1; reverse[b] = -1;
    }
    return false;
  }
public:
  Search(const SynkitStereoRequest &request, SynkitStereoResult &result, Clock::time_point began)
    : r(request), out(result), s(result.statistics), mem(&result.workspace), start(began),
      ca(mem), cb(mem), forward(mem), reverse(mem), source_count(mem), target_count(mem), available(mem),
      fa(mem), fb(mem), pairs(mem), affected_a(mem), affected_b(mem), trail(mem),
      n(0), deferred((request.flags & SYNKIT_STEREO_DEFER_CHECKS) != 0) {}
  void run() {
    check();
    validate_graph(r.source); validate_graph(r.target);
    bool same = false;
    if (r.source.n == r.target.n && r.source.arcs == r.target.arcs &&
        r.source.frame_count == r.target.frame_count) {
      n = static_cast<int>(r.source.n);
      forward.resize(n, -1); reverse.resize(n, -1);
      for (int i = 0; i < r.source.frame_count; ++i) {
        check(); fa.push_back({r.source.frames[i].geometry, normalized(r.source.frames[i])});
        fb.push_back({r.target.frames[i].geometry, normalized(r.target.frames[i])});
      }
      if (refine()) {
        available.resize(static_cast<size_t>(n) * 2, 0);
        for (int color : cb) ++available[color];
        if (prepare_pairs()) same = visit(0);
        else ++s.stereo_pruned;
      }
    }
    check();
    s.complete = 1; s.isomorphic = same ? 1 : 0;
  }
};

void validate_request(const SynkitStereoRequest &r) {
  if (r.abi_version != 1 || r.struct_size != sizeof(SynkitStereoRequest) ||
      !std::isfinite(r.seconds) || (r.seconds < 0 && r.seconds != -1) ||
      r.max_depth < 0 || r.max_depth > 1024 || (r.flags & ~SYNKIT_STEREO_DEFER_CHECKS))
    throw std::invalid_argument("Invalid stereo ABI, limits or flags");
  for (auto cap : {r.max_nodes, r.workspace_bytes, r.output_bytes})
    if (cap == 0 || cap < -1) throw std::invalid_argument("Limits must be positive or -1");
}
} // namespace

extern "C" int synkit_stereo_abi() { return 1; }
extern "C" const char *synkit_stereo_error() { return error.c_str(); }
extern "C" int synkit_stereo_match(const SynkitStereoRequest *r, SynkitStereoResult **result) {
  if (!result) return -1;
  *result = nullptr; error.clear();
  try {
    const auto start = Clock::now();
    if (!r) throw std::invalid_argument("Missing stereo request");
    validate_request(*r);
    auto out = std::make_unique<SynkitStereoResult>(r->workspace_bytes);
    try { Search(*r, *out, start).run(); }
    catch (const Stop &stop) {
      out->statistics.complete = 0; out->statistics.isomorphic = -1;
      out->statistics.reason = stop.reason; out->mapping.clear();
    }
    catch (const std::bad_alloc &) {
      out->statistics.complete = 0; out->statistics.isomorphic = -1;
      out->statistics.reason = 3; out->mapping.clear();
    }
    out->statistics.peak_workspace_bytes = out->workspace.peak;
    out->statistics.elapsed_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    *result = out.release(); return 0;
  } catch (const std::exception &exception) { error = exception.what(); return -1; }
  catch (...) { error = "Unknown native stereo matching error"; return -1; }
}
extern "C" const SynkitStereoStatistics *synkit_stereo_statistics(const SynkitStereoResult *r) {
  return r ? &r->statistics : nullptr;
}
extern "C" const int32_t *synkit_stereo_mapping(const SynkitStereoResult *r, int64_t *size) {
  const bool usable = r && r->statistics.complete && r->statistics.isomorphic == 1;
  if (size) *size = usable ? static_cast<int64_t>(r->mapping.size()) : 0;
  return usable ? r->mapping.data() : nullptr;
}
extern "C" void synkit_stereo_free(SynkitStereoResult *r) { delete r; }
