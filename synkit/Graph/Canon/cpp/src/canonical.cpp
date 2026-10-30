#include "synkit_canon/canon.h"
#include "symmetry.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <memory_resource>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Vec = std::pmr::vector<int32_t>;
using Partition = std::pmr::vector<Vec>;
struct Stop { int reason; };
struct Cancel { std::atomic<bool> requested{false}; };
thread_local std::string error;

class Workspace : public std::pmr::memory_resource {
  int64_t cap_, used_ = 0;
  void *do_allocate(size_t bytes, size_t alignment) override {
    if (bytes > static_cast<size_t>(std::numeric_limits<int64_t>::max() - used_) ||
        (cap_ >= 0 && bytes > static_cast<size_t>(cap_ - used_))) throw Stop{3};
    void *p = std::pmr::new_delete_resource()->allocate(bytes, alignment);
    used_ += static_cast<int64_t>(bytes);
    peak = std::max(peak, used_);
    return p;
  }
  void do_deallocate(void *p, size_t bytes, size_t alignment) override {
    used_ -= static_cast<int64_t>(bytes);
    std::pmr::new_delete_resource()->deallocate(p, bytes, alignment);
  }
  bool do_is_equal(const std::pmr::memory_resource &other) const noexcept override {
    return this == &other;
  }
public:
  int64_t peak = 0;
  explicit Workspace(int64_t cap) : cap_(cap) {}
};
} // namespace

struct SynkitCanonResult {
  Workspace workspace;
  std::pmr::vector<uint8_t> bytes;
  Vec order;
  SynkitCanonStatistics statistics{};
  SynkitCanonPruningStatistics pruning{};
  explicit SynkitCanonResult(int64_t cap) : workspace(cap), bytes(&workspace), order(&workspace) {}
};

namespace {
class Search {
  const SynkitCanonRequest &r;
  SynkitCanonResult &out;
  std::pmr::memory_resource *mem;
  Clock::time_point start;
  Vec best_key;
  Partition generators;
  Vec twin_class, path;
  int n;

  void check() const {
    if (r.cancel_token && static_cast<Cancel *>(r.cancel_token)->requested.load(std::memory_order_relaxed))
      throw Stop{5};
    if (r.seconds >= 0 && std::chrono::duration<double>(Clock::now() - start).count() >= r.seconds)
      throw Stop{1};
  }
  bool edge(int a, int b) const {
    return std::binary_search(r.neighbors + r.offsets[a], r.neighbors + r.offsets[a + 1], b);
  }
  static int root(Vec &parent, int v) {
    while (parent[v] != v) { parent[v] = parent[parent[v]]; v = parent[v]; }
    return v;
  }
  static void unite(Vec &parent, int a, int b) {
    a = root(parent, a); b = root(parent, b);
    if (a != b) parent[b] = a;
  }
  void prepare_twins() {
    twin_class.resize(n);
    for (int v = 0; v < n; ++v) twin_class[v] = v;
    std::pmr::map<Vec, int> open(mem), closed(mem);
    for (int v = 0; v < n; ++v) {
      check();
      Vec neighbors(mem);
      neighbors.push_back(r.colors[v]);
      for (int64_t k = r.offsets[v]; k < r.offsets[v + 1]; ++k)
        neighbors.push_back(r.neighbors[k]);
      auto same_open = open.emplace(neighbors, v);
      if (!same_open.second) unite(twin_class, v, same_open.first->second);
      // Equal open neighborhoods characterize false twins; equal closed
      // neighborhoods characterize true twins. Colors are part of both keys.
      neighbors.insert(std::lower_bound(neighbors.begin() + 1, neighbors.end(), v), v);
      auto same_closed = closed.emplace(std::move(neighbors), v);
      if (!same_closed.second) unite(twin_class, v, same_closed.first->second);
    }
    Vec sizes(n, 0, mem);
    for (int v = 0; v < n; ++v) { twin_class[v] = root(twin_class, v); ++sizes[twin_class[v]]; }
    for (int size : sizes) if (size > 1) {
      ++out.pruning.implicit_twin_factors;
      out.pruning.twin_vertices += size;
    }
  }
  Partition refine(Partition current) {
    while (true) {
      check();
      ++out.statistics.refinement_rounds;
      Vec cells(n, 0, mem);
      for (size_t c = 0; c < current.size(); ++c)
        for (int v : current[c]) cells[v] = static_cast<int>(c);
      Partition next(mem);
      for (const auto &cell : current) {
        std::pmr::map<Vec, Vec> buckets(mem);
        for (int v : cell) {
          check();
          Vec signature(mem);
          for (int64_t i = r.offsets[v]; i < r.offsets[v + 1]; ++i)
            signature.push_back(cells[r.neighbors[i]]);
          std::sort(signature.begin(), signature.end());
          auto entry = buckets.try_emplace(std::move(signature));
          entry.first->second.push_back(v);
        }
        for (auto &entry : buckets) next.push_back(std::move(entry.second));
      }
      if (next.size() == current.size()) return next;
      current = std::move(next);
    }
  }
  Vec key(const Vec &order) {
    Vec positions(n, 0, mem), value(mem);
    for (int i = 0; i < n; ++i) positions[order[i]] = i;
    value.reserve(static_cast<size_t>(n) + static_cast<size_t>(r.arcs));
    for (int v : order) value.push_back(r.colors[v]);
    for (int i = 0; i < n; ++i) {
      check();
      Vec row(mem);
      for (int64_t k = r.offsets[order[i]]; k < r.offsets[order[i] + 1]; ++k)
        if (positions[r.neighbors[k]] > i) row.push_back(positions[r.neighbors[k]]);
      std::sort(row.begin(), row.end());
      for (int j : row) { value.push_back(i); value.push_back(j); }
    }
    return value;
  }
  void leaf(const Partition &p) {
    ++out.statistics.leaves;
    Vec order(mem);
    for (const auto &cell : p) order.insert(order.end(), cell.begin(), cell.end());
    Vec candidate = key(order);
    if (out.order.empty() || candidate < best_key) {
      best_key = std::move(candidate);
      out.order = order;
    } else if (candidate == best_key && generators.size() < 256) {
      Vec map(n, 0, mem);
      for (int i = 0; i < n; ++i) map[out.order[i]] = order[i];
      // A full equal colored-graph record implies a permutation automorphism.
      // Verify the projected generator before using it to prune another branch.
      for (int v = 0; v < n; ++v) {
        check();
        if (r.colors[v] != r.colors[map[v]]) throw std::logic_error("Invalid color automorphism");
        for (int64_t k = r.offsets[v]; k < r.offsets[v + 1]; ++k)
          if (!edge(map[v], map[r.neighbors[k]])) throw std::logic_error("Invalid edge automorphism");
      }
      bool identity = true;
      for (int v = 0; v < n; ++v) identity = identity && map[v] == v;
      if (!identity && std::find(generators.begin(), generators.end(), map) == generators.end()) {
        generators.push_back(std::move(map));
        out.pruning.automorphism_generators = static_cast<int64_t>(generators.size());
      }
    }
  }
  bool stabilizes(const Vec &g, const Vec &cells) const {
    for (int v = 0; v < n; ++v) if (cells[v] != cells[g[v]]) return false;
    return true;
  }
  Partition filtered_generators(const Vec &cells) {
    Partition result(mem);
    for (const auto &g : generators) {
      check();
      if (stabilizes(g, cells)) result.push_back(g);
    }
    return result;
  }
  Partition stable_generators(const Vec &cells) {
    // Filtering generators individually can miss stabilizing compositions.
    // Retain this safe subgroup as a fallback for bounded Schreier discovery.
    Partition fallback = filtered_generators(cells);
    if (path.empty() || fallback.size() == generators.size()) return fallback;
    ++out.pruning.stabilizer_calls;
    synkit_canon::StabilizerWork work;
    Partition group(generators, mem);
    for (int point : path) {
      check();
      Partition next(mem);
      if (!synkit_canon::point_stabilizer(group, point, n, next, work, [&]() { check(); })) {
        ++out.pruning.stabilizer_fallbacks;
        return fallback;
      }
      group = std::move(next);
      if (group.empty()) break;
    }
    Partition usable(mem);
    for (auto &g : group) if (stabilizes(g, cells)) usable.push_back(std::move(g));
    return usable;
  }
  Vec orbits(const Partition &p, const Vec &cells) {
    ++out.pruning.orbit_rebuilds;
    Vec parent(n, 0, mem), owner(n, -1, mem), first(n, -1, mem);
    for (int v = 0; v < n; ++v) parent[v] = v;
    // Handle whole symmetric twin factors without generating their factorial
    // groups. Only same-cell transpositions preserve this search partition.
    for (size_t c = 0; c < p.size(); ++c) for (int v : p[c]) {
      const int block = twin_class[v];
      if (owner[block] != static_cast<int>(c)) { owner[block] = static_cast<int>(c); first[block] = v; }
      else unite(parent, first[block], v);
    }
    for (const auto &g : filtered_generators(cells)) {
      check();
      for (int v = 0; v < n; ++v) unite(parent, v, g[v]);
    }
    for (int v = 0; v < n; ++v) parent[v] = root(parent, v);
    return parent;
  }
  void visit(Partition p, int depth) {
    check();
    if (r.max_nodes >= 0 && out.statistics.nodes >= r.max_nodes) throw Stop{2};
    if (depth > r.max_depth) throw Stop{6};
    ++out.statistics.nodes;
    p = refine(std::move(p));
    size_t target = p.size();
    for (size_t i = 0; i < p.size(); ++i) if (p[i].size() > 1) {
      if (target == p.size() || p[i].size() < p[target].size()) target = i;
    }
    // A discrete root has no nontrivial automorphisms and needs no twin or
    // subgroup setup. This is common for fully distinguished stereo inputs.
    if (target == p.size()) { leaf(p); return; }
    if (twin_class.empty()) prepare_twins();
    bool all_twins = true;
    for (size_t i = 0; i < p.size(); ++i) if (p[i].size() > 1) {
      for (int v : p[i]) { check(); if (twin_class[p[i][0]] != twin_class[v]) all_twins = false; }
    }
    // Within a twin cell every ordering gives the same graph; no branching is needed.
    if (all_twins) { leaf(p); return; }
    Vec explored(mem), cells(n, 0, mem);
    for (size_t i = 0; i < p.size(); ++i)
      for (int v : p[i]) cells[v] = static_cast<int>(i);
    Vec orbit(mem);
    size_t generation = std::numeric_limits<size_t>::max();
    bool strengthened = false;
    for (int v : p[target]) {
      check();
      if (generation != generators.size()) {
        orbit = orbits(p, cells); generation = generators.size(); strengthened = false;
      }
      auto already_explored = [&]() {
        for (int previous : explored) if (orbit[v] == orbit[previous]) return true;
        return false;
      };
      bool seen = already_explored();
      // First try inexpensive twin/direct-generator orbits. Deriving a point
      // stabilizer cannot skip the first branch or improve an already skipped
      // branch, so perform it only on demand, once per generator generation.
      if (!seen && !explored.empty() && !strengthened) {
        for (const auto &g : stable_generators(cells)) {
          check();
          for (int w = 0; w < n; ++w) unite(orbit, w, g[w]);
        }
        for (int w = 0; w < n; ++w) orbit[w] = root(orbit, w);
        strengthened = true;
        seen = already_explored();
      }
      if (seen) { ++out.pruning.symmetry_pruned_branches; continue; }
      Partition child(mem);
      for (size_t i = 0; i < p.size(); ++i) {
        if (i != target) child.push_back(p[i]);
        else {
          child.emplace_back(); child.back().push_back(v);
          child.emplace_back();
          for (int w : p[i]) if (w != v) child.back().push_back(w);
        }
      }
      path.push_back(v);
      visit(std::move(child), depth + 1);
      path.pop_back();
      explored.push_back(v);
    }
  }
  void append32(uint32_t value) {
    for (int i = 0; i < 4; ++i) out.bytes.push_back(static_cast<uint8_t>(value >> (8 * i)));
  }
public:
  Search(const SynkitCanonRequest &request, SynkitCanonResult &result, Clock::time_point began)
      : r(request), out(result), mem(&out.workspace), start(began), best_key(mem), generators(mem),
        twin_class(mem), path(mem),
        n(static_cast<int>(request.n)) {}
  void run() {
    check();
    if (r.offsets[0] != 0 || r.offsets[n] != r.arcs) throw std::invalid_argument("Invalid CSR endpoints");
    for (int v = 0; v < n; ++v) {
      check();
      if (r.colors[v] < 0 || r.offsets[v] < 0 || r.offsets[v] > r.offsets[v + 1] || r.offsets[v + 1] > r.arcs)
        throw std::invalid_argument("Invalid CSR row or color");
    }
    for (int v = 0; v < n; ++v) for (int64_t k = r.offsets[v]; k < r.offsets[v + 1]; ++k) {
      check();
      int w = r.neighbors[k];
      if (w < 0 || w >= n || w == v || (k > r.offsets[v] && r.neighbors[k - 1] >= w) || !edge(w, v))
        throw std::invalid_argument("CSR must be simple, sorted, loop-free and symmetric");
    }
    const uint64_t output = 12ULL + 8ULL * static_cast<uint64_t>(n) + 4ULL * static_cast<uint64_t>(r.arcs);
    if (r.output_bytes >= 0 && output > static_cast<uint64_t>(r.output_bytes)) throw Stop{4};
    const auto search_start = Clock::now();
    out.statistics.validation_seconds = std::chrono::duration<double>(search_start - start).count();
    std::pmr::map<int32_t, Vec> buckets(mem);
    for (int i = 0; i < n; ++i) { check(); buckets[r.colors[i]].push_back(i); }
    Partition initial(mem);
    for (auto &entry : buckets) initial.push_back(std::move(entry.second));
    visit(std::move(initial), 0);
    const auto serialization_start = Clock::now();
    out.statistics.search_seconds = std::chrono::duration<double>(serialization_start - search_start).count();
    // SG01, vertex count, undirected edge count, colors, sorted endpoint pairs.
    out.bytes.reserve(static_cast<size_t>(output - 4ULL * n));
    append32(0x31304753U); append32(static_cast<uint32_t>(n)); append32(static_cast<uint32_t>(r.arcs / 2));
    for (size_t i = 0; i < best_key.size(); ++i) {
      if ((i & 1023) == 0) check();
      append32(static_cast<uint32_t>(best_key[i]));
    }
    check();
    out.statistics.serialization_seconds = std::chrono::duration<double>(Clock::now() - serialization_start).count();
    out.statistics.complete = 1;
  }
};

void validate(const SynkitCanonRequest &r) {
  if (r.abi_version != 1 || r.struct_size != sizeof(SynkitCanonRequest) ||
      r.n < 0 || r.n > INT32_MAX || r.arcs < 0 || r.arcs > UINT32_MAX || (r.arcs & 1) ||
      !r.offsets || (r.n && !r.colors) || (r.arcs && !r.neighbors) ||
      !std::isfinite(r.seconds) || (r.seconds < 0 && r.seconds != -1) ||
      r.max_depth < 0 || r.max_depth > 1024)
    throw std::invalid_argument("Invalid canonical request ABI, graph dimensions or deadline");
  for (auto cap : {r.max_nodes, r.workspace_bytes, r.output_bytes})
    if (cap == 0 || cap < -1) throw std::invalid_argument("Limits must be positive or -1");
}
} // namespace

extern "C" int synkit_canon_abi() { return 1; }
extern "C" const char *synkit_canon_error() { return error.c_str(); }
extern "C" void *synkit_canon_cancel_create() {
  try { return new Cancel; } catch (...) { return nullptr; }
}
extern "C" void synkit_canon_cancel_request(void *p) {
  if (p) static_cast<Cancel *>(p)->requested.store(true, std::memory_order_relaxed);
}
extern "C" void synkit_canon_cancel_free(void *p) { delete static_cast<Cancel *>(p); }
extern "C" int synkit_canon_search(const SynkitCanonRequest *r, SynkitCanonResult **result) {
  if (!result) return -1;
  *result = nullptr;
  error.clear();
  try {
    const auto start = Clock::now();
    if (!r) throw std::invalid_argument("Missing request");
    validate(*r);
    auto out = std::make_unique<SynkitCanonResult>(r->workspace_bytes);
    try { Search(*r, *out, start).run(); }
    catch (const Stop &stop) { out->statistics.reason = stop.reason; out->bytes.clear(); out->order.clear(); }
    catch (const std::bad_alloc &) { out->statistics.reason = 3; out->bytes.clear(); out->order.clear(); }
    out->statistics.peak_workspace_bytes = out->workspace.peak;
    out->statistics.elapsed_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    *result = out.release();
    return 0;
  } catch (const std::exception &exception) { error = exception.what(); return -1; }
  catch (...) { error = "Unknown native canonicalization error"; return -1; }
}
extern "C" const SynkitCanonStatistics *synkit_canon_statistics(const SynkitCanonResult *r) {
  return r ? &r->statistics : nullptr;
}
extern "C" const SynkitCanonPruningStatistics *synkit_canon_pruning_statistics(const SynkitCanonResult *r) {
  return r ? &r->pruning : nullptr;
}
extern "C" const uint8_t *synkit_canon_bytes(const SynkitCanonResult *r, int64_t *size) {
  if (size) *size = r && r->statistics.complete ? static_cast<int64_t>(r->bytes.size()) : 0;
  return r && r->statistics.complete ? r->bytes.data() : nullptr;
}
extern "C" const int32_t *synkit_canon_order(const SynkitCanonResult *r, int64_t *size) {
  if (size) *size = r && r->statistics.complete ? static_cast<int64_t>(r->order.size()) : 0;
  return r && r->statistics.complete ? r->order.data() : nullptr;
}
extern "C" void synkit_canon_free(SynkitCanonResult *r) { delete r; }
