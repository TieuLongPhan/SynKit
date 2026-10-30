#ifndef SYNKIT_CANON_SYMMETRY_H
#define SYNKIT_CANON_SYMMETRY_H

#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <set>
#include <utility>
#include <vector>

namespace synkit_canon {
using Permutation = std::pmr::vector<int32_t>;
using Generators = std::pmr::vector<Permutation>;

struct StabilizerWork {
  size_t remaining = 250000;
  size_t max_generators = 256;
  bool spend(size_t amount) {
    if (amount > remaining) return false;
    remaining -= amount;
    return true;
  }
};

// Schreier generators for the point stabilizer of <group>. Permutations map
// source to image. A transporter t_x sends point to x; each generated element
// is t_{g(x)}^{-1} g t_x, hence fixes point. A false return means that the
// caller must discard this attempt and use a known safe subgroup instead.
template<class Check>
bool point_stabilizer(const Generators &group, int point, int n,
                      Generators &result, StabilizerWork &work, Check check) {
  auto *mem = result.get_allocator().resource();
  result.clear();
  check();
  if (group.empty()) return true;
  bool fixed = true;
  for (const auto &g : group) if (g[point] != point) { fixed = false; break; }
  if (fixed) {
    if (!work.spend(static_cast<size_t>(n) * group.size())) return false;
    result = group;
    return true;
  }
  Permutation orbit(mem);
  Generators transport(n, mem), inverse(n, mem);
  if (!work.spend(n)) return false;
  transport[point].resize(n);
  for (int v = 0; v < n; ++v) transport[point][v] = v;
  orbit.push_back(point);
  for (size_t k = 0; k < orbit.size(); ++k) for (const auto &g : group) {
    check();
    if (!work.spend(1)) return false;
    const int x = orbit[k], y = g[x];
    if (!transport[y].empty()) continue;
    if (!work.spend(n)) return false;
    transport[y].resize(n);
    for (int v = 0; v < n; ++v) transport[y][v] = g[transport[x][v]];
    orbit.push_back(y);
  }
  for (int x : orbit) {
    check();
    if (!work.spend(n)) return false;
    inverse[x].resize(n);
    for (int v = 0; v < n; ++v) inverse[x][transport[x][v]] = v;
  }
  std::pmr::set<Permutation> known(mem);
  for (int x : orbit) for (const auto &g : group) {
    check();
    if (!work.spend(n)) return false;
    Permutation candidate(n, 0, mem);
    bool identity = true;
    for (int v = 0; v < n; ++v) {
      candidate[v] = inverse[g[x]][g[transport[x][v]]];
      identity = identity && candidate[v] == v;
    }
    if (!identity && known.insert(candidate).second) {
      if (result.size() >= work.max_generators) return false;
      result.push_back(std::move(candidate));
    }
  }
  return true;
}
} // namespace synkit_canon
#endif
