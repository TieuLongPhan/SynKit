// Private exact small-distance certificate through weighted edge-deletion
// decks. nauty 2.9.3: Apache-2.0; see vendor/nauty/COPYRIGHT and
// LICENSE-2.0.txt.
#include "../pabs/exact_helpers.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>
#ifdef SYNKIT_MAPPER_WITH_NAUTY
extern "C" {
#include "nausparse.h"
}
namespace {
using Clock = std::chrono::steady_clock;
using Vector = std::vector<int32_t>;
using Key = std::array<uint64_t, 4>;
struct Interrupted {};
std::mutex nauty_mutex;
struct Work {
  Clock::time_point started = Clock::now();
  double seconds;
  int64_t node_limit;
  int64_t statistics[12] = {};
  void check() {
    if (std::chrono::duration<double>(Clock::now() - started).count() >=
        seconds) {
      statistics[8] = 1;
      throw Interrupted{};
    }
    if (statistics[1] >= node_limit) {
      statistics[8] = 2;
      throw Interrupted{};
    }
  }
};
thread_local Work *active = nullptr;
void node_hook(graph *, int *, int *, int, int, int, int, int, int) {
  if (!active)
    return;
  ++active->statistics[1];
  try {
    active->check();
  } catch (const Interrupted &) {
    nauty_kill_request = 1;
  }
}
uint64_t mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
void add(Key &key, uint64_t color, int sign) {
  for (int k = 0; k < 4; ++k) {
    uint64_t value = mix(color + 0x632be59bd9b4e019ULL * k);
    if (sign > 0)
      key[k] += value;
    else
      key[k] -= value;
  }
}
struct Deletion {
  std::array<int, 3> edge{{-1, -1, -1}};
  int size = 0;
};
struct Endpoint {
  int n, g;
  Vector matrix, types, weights;
  std::vector<std::pair<int, int>> edges;
  std::vector<std::vector<std::pair<int, int>>> neighbors;
  std::vector<uint64_t> base_colors;
  Key base_key{};
  std::vector<char> fixed_unique;
  std::array<std::vector<uint64_t>, 3> base_refined;
  Key refined_base_key{};
  std::vector<Vector> drop_options;
  mutable std::vector<Vector> edge_generators;
  // Full-state orbit closure, not partial augmentation lex ordering. Every
  // permitted deletion multiset has a representative under this exact subgroup.
  bool orbit_leader(const Deletion &state, Work &work,
                    std::map<std::array<int, 3>, bool> &cache) const {
    if (edge_generators.empty())
      return true;
    auto found = cache.find(state.edge);
    if (found != cache.end())
      return found->second;
    std::vector<std::array<int, 3>> queue{state.edge};
    std::set<std::array<int, 3>> seen{state.edge};
    auto best = state.edge;
    for (size_t position = 0; position < queue.size(); ++position) {
      if ((position & 63) == 0)
        work.check();
      for (const auto &generator : edge_generators) {
        auto next = queue[position];
        for (int i = 0; i < state.size; ++i)
          next[i] = generator[next[i]];
        // A deletion has at most three units. Use fixed array bounds rather
        // than a generic sort over a dynamic endpoint.
        if (state.size >= 2 && next[0] > next[1])
          std::swap(next[0], next[1]);
        if (state.size == 3) {
          if (next[1] > next[2])
            std::swap(next[1], next[2]);
          if (next[0] > next[1])
            std::swap(next[0], next[1]);
        }
        if (seen.insert(next).second) {
          queue.push_back(next);
          best = std::min(best, next);
        }
      }
      if (queue.size() > 500000) {
        work.statistics[8] = 3;
        throw Interrupted{};
      }
    }
    for (const auto &value : queue)
      cache.emplace(value, value == best);
    return state.edge == best;
  }
  void accept_generator(const int *permutation) const {
    if (edge_generators.size() >= 32)
      return;
    Vector seen(n, 0);
    bool moved = false;
    for (int i = 0; i < n; ++i) {
      int j = permutation[i];
      if (j < 0 || j >= n || seen[j]++ || types[i] != types[j])
        return;
      moved = moved || i != j;
    }
    if (!moved)
      return;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < i; ++j)
        if (matrix[i * n + j] != matrix[permutation[i] * n + permutation[j]])
          return;
    Vector index(n * n, -1), action(edges.size());
    for (size_t k = 0; k < edges.size(); ++k) {
      auto e = edges[k];
      index[e.first * n + e.second] = index[e.second * n + e.first] = k;
    }
    for (size_t k = 0; k < edges.size(); ++k) {
      auto e = edges[k];
      action[k] = index[permutation[e.first] * n + permutation[e.second]];
      if (action[k] < 0)
        return;
    }
    if (std::find(edge_generators.begin(), edge_generators.end(), action) ==
        edge_generators.end())
      edge_generators.push_back(std::move(action));
  }
  Endpoint(int size, const int64_t *input, const int32_t *colors, int unit)
      : n(size), g(unit), matrix(n * n), types(colors, colors + n),
        neighbors(n) {
    for (int i = 0; i < n * n; ++i)
      matrix[i] = static_cast<int32_t>(input[i]);
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j)
        if (matrix[i * n + j]) {
          int id = edges.size();
          edges.push_back({i, j});
          weights.push_back(matrix[i * n + j]);
          neighbors[i].push_back({j, id});
          neighbors[j].push_back({i, id});
        }
    base_colors.resize(n);
    for (int i = 0; i < n; ++i) {
      base_colors[i] = profile(i, weights);
      add(base_key, base_colors[i], 1);
    }
    std::map<int, int> multiplicity;
    for (int t : types)
      ++multiplicity[t];
    fixed_unique.resize(n);
    base_refined[0] = base_colors;
    for (int i = 0; i < n; ++i) {
      fixed_unique[i] = multiplicity[types[i]] == 1;
      if (fixed_unique[i])
        base_refined[0][i] = fixed_color(i);
    }
    for (int round = 1; round <= 2; ++round) {
      base_refined[round].resize(n);
      for (int i = 0; i < n; ++i)
        base_refined[round][i] =
            refined_color(i, weights, base_refined[round - 1]);
    }
    for (uint64_t color : base_refined[2])
      add(refined_base_key, color, 1);
  }
  void constrain_overlap(const Endpoint &opposite, int max_units) {
    std::map<std::pair<int, int>, Vector> palette;
    // Zero is present only when an actual compatible nonedge exists. In
    // particular complete constant port blocks cannot lose arbitrary weight.
    for (int i = 0; i < opposite.n; ++i)
      for (int j = i + 1; j < opposite.n; ++j) {
        auto pair = std::minmax(opposite.types[i], opposite.types[j]);
        palette[{pair.first, pair.second}].push_back(
            opposite.matrix[i * opposite.n + j]);
      }
    for (auto &entry : palette) {
      auto &v = entry.second;
      std::sort(v.begin(), v.end());
      v.erase(std::unique(v.begin(), v.end()), v.end());
    }
    drop_options.assign(edges.size(), Vector{});
    for (size_t i = 0; i < edges.size(); ++i) {
      auto e = edges[i];
      auto pair = std::minmax(types[e.first], types[e.second]);
      auto found = palette.find({pair.first, pair.second});
      for (int units = 1; units <= max_units; ++units) {
        int64_t residual =
            static_cast<int64_t>(weights[i]) - static_cast<int64_t>(units) * g;
        if (residual < 0)
          break;
        if (found != palette.end() &&
            std::binary_search(found->second.begin(), found->second.end(),
                               static_cast<int>(residual)))
          drop_options[i].push_back(units);
      }
    }
  }
  std::array<int64_t, 4> level_counts() const {
    std::array<int64_t, 4> counts{{1, 0, 0, 0}};
    for (const auto &options : drop_options) {
      auto previous = counts;
      for (int level = 1; level <= 3; ++level)
        for (int units : options)
          if (units <= level)
            counts[level] = std::min<int64_t>(
                500001, counts[level] + previous[level - units]);
    }
    return counts;
  }
  uint64_t profile(int atom, const Vector &values) const {
    std::vector<std::pair<int, int>> signature;
    for (auto entry : neighbors[atom])
      if (values[entry.second])
        signature.push_back({types[entry.first], values[entry.second]});
    std::sort(signature.begin(), signature.end());
    uint64_t hash = mix(static_cast<uint32_t>(types[atom]));
    hash = mix(hash ^ signature.size());
    for (auto entry : signature) {
      hash = mix(hash ^ static_cast<uint32_t>(entry.first));
      hash = mix(hash ^ static_cast<uint32_t>(entry.second));
    }
    return hash;
  }
  Vector changed_weights(const Deletion &state) const {
    Vector values = weights;
    for (int i = 0; i < state.size; ++i)
      values[state.edge[i]] -= g;
    return values;
  }
  Vector affected(const Deletion &state) const {
    Vector atoms;
    for (int i = 0; i < state.size; ++i) {
      auto e = edges[state.edge[i]];
      atoms.push_back(e.first);
      atoms.push_back(e.second);
    }
    std::sort(atoms.begin(), atoms.end());
    atoms.erase(std::unique(atoms.begin(), atoms.end()), atoms.end());
    return atoms;
  }
  uint64_t fixed_color(int atom) const {
    return mix(0xd6e8feb86659fd93ULL ^ static_cast<uint32_t>(types[atom]));
  }
  uint64_t refined_color(int atom, const Vector &values,
                         const std::vector<uint64_t> &colors) const {
    if (fixed_unique[atom])
      return fixed_color(atom);
    std::vector<std::pair<uint64_t, int>> signature;
    for (auto entry : neighbors[atom])
      if (values[entry.second])
        signature.push_back({colors[entry.first], values[entry.second]});
    std::sort(signature.begin(), signature.end());
    uint64_t hash = mix(colors[atom] ^ signature.size());
    for (auto entry : signature) {
      hash = mix(hash ^ entry.first);
      hash = mix(hash ^ static_cast<uint32_t>(entry.second));
    }
    return hash;
  }
  Vector radius(const Vector &seeds) const {
    Vector result;
    std::vector<char> seen(n, 0);
    auto append = [&](int atom) {
      if (!fixed_unique[atom] && !seen[atom]) {
        seen[atom] = 1;
        result.push_back(atom);
      }
    };
    for (int atom : seeds) {
      append(atom);
      for (auto neighbor : neighbors[atom])
        append(neighbor.first);
    }
    return result;
  }
  Key invariant(const Deletion &state, bool refine) const {
    Vector values = changed_weights(state);
    if (!refine) {
      Key key = base_key;
      for (int atom : affected(state)) {
        add(key, base_colors[atom], -1);
        add(key, profile(atom, values), 1);
      }
      return key;
    }
    // Fixed hard types have unique legal images. Holding their colors fixed
    // defines a valid isomorphism invariant and prevents dense score ports from
    // propagating a local edit throughout the graph. Remaining colors are the
    // EXACT two-round WL colors, updated only on their dependency closure.
    Vector zero;
    auto colors = base_refined[0];
    for (int atom : affected(state))
      if (!fixed_unique[atom]) {
        zero.push_back(atom);
        colors[atom] = profile(atom, values);
      }
    auto one = radius(zero), two = radius(one);
    auto next = base_refined[1];
    for (int atom : one)
      next[atom] = refined_color(atom, values, colors);
    Key key = refined_base_key;
    for (int atom : two) {
      add(key, base_refined[2][atom], -1);
      add(key, refined_color(atom, values, next), 1);
    }
    return key;
  }
  Vector modified(const Deletion &state) const {
    Vector result = matrix;
    for (int i = 0; i < state.size; ++i) {
      auto e = edges[state.edge[i]];
      result[e.first * n + e.second] -= g;
      result[e.second * n + e.first] -= g;
    }
    return result;
  }
};
using Deck = std::map<Key, std::vector<Deletion>>;
Deck enumerate(const Endpoint &endpoint, int units, Work &work, int side) {
  Deck result;
  Deletion state;
  state.size = units;
  int64_t count = 0;
  std::map<std::array<int, 3>, bool> orbit_cache;
  if (endpoint.level_counts()[units] > 500000) {
    work.statistics[8] = 3;
    throw Interrupted{};
  }
  auto extend = [&](auto &&self, int position, int first) -> void {
    work.check();
    if (position == units) {
      if (++count > 500000) {
        work.statistics[8] = 3;
        throw Interrupted{};
      }
      if (!endpoint.orbit_leader(state, work, orbit_cache)) {
        ++work.statistics[10];
        return;
      }
      result[endpoint.invariant(state, false)].push_back(state);
      return;
    }
    if (position < 0 || position >= 3) {
      work.statistics[8] = 4;
      throw Interrupted{};
    }
    for (int edge = first; edge < static_cast<int>(endpoint.edges.size());
         ++edge)
      for (int drop : endpoint.drop_options[edge]) {
        if (position + drop > units)
          continue;
        for (int j = 0; j < drop; ++j) {
          if (position + j >= 3)
            throw std::runtime_error("deletion state overflow");
          state.edge[position + j] = edge;
        }
        self(self, position + drop, edge + 1);
      }
  };
  extend(extend, 0, 0);
  work.statistics[2 + side] += count;
  return result;
}

thread_local const Endpoint *collecting_endpoint = nullptr;
void automorphism_hook(int, int *permutation, int *, int, int, int) {
  if (!collecting_endpoint)
    return;
  try {
    collecting_endpoint->accept_generator(permutation);
  } catch (...) {
    nauty_kill_request = 1;
  }
}
std::pair<Vector, Vector> canonical(const Endpoint &endpoint,
                                    const Deletion &state, Work &work) {
  work.check();
  ++work.statistics[0];
  int n = endpoint.n;
  Vector matrix = endpoint.modified(state);
  std::map<std::pair<int, int>, Vector> cells;
  for (int i = 0; i < n; ++i)
    cells[{0, endpoint.types[i]}].push_back(i);
  std::vector<std::pair<int, int>> links;
  int vertices = n;
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j)
      if (matrix[i * n + j]) {
        int gadget = vertices++;
        cells[{1, matrix[i * n + j]}].push_back(gadget);
        links.push_back({i, gadget});
        links.push_back({j, gadget});
      }
  Vector degree(vertices, 0), neighbors(2 * links.size()), lab, partition,
      orbits(vertices);
  std::vector<size_t> offsets(vertices), cursor;
  for (auto edge : links) {
    ++degree[edge.first];
    ++degree[edge.second];
  }
  for (int i = 1; i < vertices; ++i)
    offsets[i] = offsets[i - 1] + degree[i - 1];
  cursor = offsets;
  for (auto edge : links) {
    neighbors[cursor[edge.first]++] = edge.second;
    neighbors[cursor[edge.second]++] = edge.first;
  }
  for (const auto &cell : cells) {
    for (int vertex : cell.second) {
      lab.push_back(vertex);
      partition.push_back(1);
    }
    partition.back() = 0;
  }
  sparsegraph input{};
  input.nv = vertices;
  input.nde = neighbors.size();
  input.v = offsets.data();
  input.d = degree.data();
  input.e = neighbors.data();
  input.vlen = input.dlen = vertices;
  input.elen = neighbors.size();
  SG_DECL(output);
  struct Cleanup {
    sparsegraph &value;
    ~Cleanup() { SG_FREE(value); }
  } cleanup{output};
  DEFAULTOPTIONS_SPARSEGRAPH(options);
  options.getcanon = TRUE;
  options.defaultptn = FALSE;
  options.schreier = TRUE;
  options.usernodeproc = node_hook;
  if (state.size == 0)
    options.userautomproc = automorphism_hook;
  statsblk statistics{};
  {
    std::lock_guard<std::mutex> lock(nauty_mutex);
    work.check();
    active = &work;
    collecting_endpoint = state.size == 0 ? &endpoint : nullptr;
    nauty_kill_request = 0;
    sparsenauty(&input, lab.data(), partition.data(), orbits.data(), &options,
                &statistics, &output);
    active = nullptr;
    collecting_endpoint = nullptr;
    nauty_kill_request = 0;
  }
  work.check();
  if (statistics.errstatus) {
    work.statistics[8] = 4;
    throw Interrupted{};
  }
  Vector order;
  for (int vertex : lab)
    if (vertex < n)
      order.push_back(vertex);
  if (static_cast<int>(order.size()) != n)
    throw std::runtime_error("lost an original vertex");
  Vector key;
  key.reserve(n + 3 * endpoint.edges.size() + 1);
  key.push_back(n);
  for (int vertex : order)
    key.push_back(endpoint.types[vertex]);
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j)
      if (matrix[order[i] * n + order[j]]) {
        key.push_back(i);
        key.push_back(j);
        key.push_back(matrix[order[i] * n + order[j]]);
      }
  return {std::move(key), std::move(order)};
}
int64_t witness(const Endpoint &a, const Endpoint &b, const Vector &left,
                const Vector &right, int32_t *output) {
  Vector mapping(a.n), seen(a.n, 0);
  for (int i = 0; i < a.n; ++i)
    mapping[left[i]] = right[i];
  for (int i = 0; i < a.n; ++i)
    if (mapping[i] < 0 || mapping[i] >= a.n || seen[mapping[i]]++ ||
        a.types[i] != b.types[mapping[i]])
      return -1;
  int64_t score = 0;
  for (int i = 0; i < a.n; ++i)
    for (int j = 0; j < i; ++j)
      score += std::abs(static_cast<int64_t>(a.matrix[i * a.n + j]) -
                        b.matrix[mapping[i] * a.n + mapping[j]]);
  std::copy(mapping.begin(), mapping.end(), output);
  return score;
}
} // namespace
int synkit_mapper::pabs::edge_deck(int n, const int64_t *a, const int64_t *b,
                                   const int32_t *ca, const int32_t *cb,
                                   int units, double seconds,
                                   int64_t node_limit, int64_t upper2,
                                   int32_t *mapping, int64_t *lower2,
                                   int64_t *witness2, int64_t *statistics) {
  if (statistics)
    std::fill(statistics, statistics + 12, 0);
  if (lower2)
    *lower2 = 0;
  if (witness2)
    *witness2 = -1;
  if (n < 1 || n > 512 || !a || !b || !ca || !cb || !mapping || !lower2 ||
      !witness2 || !statistics || units < 0 || units > 3 ||
      !std::isfinite(seconds) || seconds < 0 || node_limit < 1 || upper2 < 0)
    return -1;
  Work work{Clock::now(), seconds, node_limit};
  auto finish = [&](int status) {
    work.statistics[9] = static_cast<int64_t>(
        std::chrono::duration<double>(Clock::now() - work.started).count() *
        1e9);
    std::copy(work.statistics, work.statistics + 12, statistics);
    return status;
  };
  try {
    int64_t mass_a = 0, mass_b = 0, g = 0;
    std::map<int32_t, int> inventory;
    for (int i = 0; i < n; ++i) {
      ++inventory[ca[i]];
      --inventory[cb[i]];
      if (a[i * n + i] || b[i * n + i])
        return finish(-1);
      for (int j = 0; j < i; ++j) {
        if (a[i * n + j] < 0 || b[i * n + j] < 0 ||
            a[i * n + j] > std::numeric_limits<int32_t>::max() ||
            b[i * n + j] > std::numeric_limits<int32_t>::max() ||
            a[i * n + j] != a[j * n + i] || b[i * n + j] != b[j * n + i])
          return finish(-1);
        mass_a += a[i * n + j];
        mass_b += b[i * n + j];
        g = std::gcd(g, a[i * n + j]);
        g = std::gcd(g, b[i * n + j]);
      }
    }
    for (auto entry : inventory)
      if (entry.second)
        return finish(-1);
    if (mass_a != mass_b)
      return finish(-1);
    work.statistics[11] = g;
    Endpoint left(n, a, ca, static_cast<int>(g)),
        right(n, b, cb, static_cast<int>(g));
    Deletion empty;
    left.constrain_overlap(right, units);
    right.constrain_overlap(left, units);
    auto lc = canonical(left, empty, work), rc = canonical(right, empty, work);
    if (lc.first == rc.first) {
      int64_t cost = witness(left, right, lc.second, rc.second, mapping);
      if (cost != 0)
        return finish(-1);
      *lower2 = *witness2 = 0;
      work.statistics[7] = 1;
      return finish(1);
    }
    if (!g)
      return finish(-1);
    *lower2 = 2 * g;
    if (*lower2 >= upper2)
      return finish(2);
    for (int level = 1; level <= units; ++level) {
      Deck source = enumerate(left, level, work, 0),
           product = enumerate(right, level, work, 1);
      for (const auto &bucket : source) {
        auto found = product.find(bucket.first);
        if (found == product.end())
          continue;
        std::map<Key, std::vector<Deletion>> ls, rs;
        for (const auto &state : bucket.second) {
          work.check();
          ls[left.invariant(state, true)].push_back(state);
        }
        for (const auto &state : found->second) {
          work.check();
          rs[right.invariant(state, true)].push_back(state);
        }
        for (const auto &refined : ls) {
          auto compatible = rs.find(refined.first);
          if (compatible == rs.end())
            continue;
          work.statistics[4] += refined.second.size();
          work.statistics[5] += compatible->second.size();
          std::map<Vector, Vector> known;
          for (const auto &state : refined.second) {
            auto value = canonical(left, state, work);
            known.emplace(std::move(value.first), std::move(value.second));
          }
          for (const auto &state : compatible->second) {
            auto value = canonical(right, state, work);
            auto match = known.find(value.first);
            if (match != known.end()) {
              int64_t cost =
                  witness(left, right, match->second, value.second, mapping);
              if (cost != 2LL * level * g)
                return finish(-1);
              work.check();
              *lower2 = *witness2 = cost;
              work.statistics[7] = 1;
              return finish(1);
            }
          }
        }
      }
      work.check();
      work.statistics[6] = level;
      *lower2 = 2LL * (level + 1) * g;
      if (*lower2 >= upper2)
        return finish(2);
    }
    return finish(2);
  } catch (const Interrupted &) {
    *witness2 = -1;
    return finish(0);
  } catch (...) {
    *lower2 = 0;
    *witness2 = -1;
    work.statistics[8] = 5;
    return finish(-1);
  }
}

int synkit_mapper::pabs::edge_deck_sizes(int n, const int64_t *a,
                                         const int64_t *b, const int32_t *ca,
                                         const int32_t *cb, int units,
                                         int64_t *out) {
  if (!out)
    return -1;
  std::fill(out, out + 6, 0);
  if (n < 1 || n > 512 || !a || !b || !ca || !cb || units < 0 || units > 3)
    return -1;
  int64_t g = 0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      if (a[i * n + j] < 0 || b[i * n + j] < 0 || a[i * n + j] > INT32_MAX ||
          b[i * n + j] > INT32_MAX)
        return -1;
      g = std::gcd(g, a[i * n + j]);
      g = std::gcd(g, b[i * n + j]);
    }
  if (!g)
    return 0;
  Endpoint left(n, a, ca, g), right(n, b, cb, g);
  left.constrain_overlap(right, units);
  right.constrain_overlap(left, units);
  auto lc = left.level_counts(), rc = right.level_counts();
  for (int k = 0; k < 3; ++k) {
    out[k] = lc[k + 1];
    out[k + 3] = rc[k + 1];
  }
  return 0;
}

#ifdef SYNKIT_MAPPER_TEST_HOOKS
extern "C" int synkit_private_deck_fingerprint(int n, const int64_t *a,
                                               const int32_t *types, int g,
                                               int size,
                                               const int32_t *deleted_edges,
                                               uint64_t *out) {
  if (n < 1 || n > 512 || !a || !types || g < 1 || size < 0 || size > 3 ||
      !out || (size && !deleted_edges))
    return -1;
  Endpoint endpoint(n, a, types, g);
  Deletion state;
  state.size = size;
  for (int k = 0; k < size; ++k) {
    int u = deleted_edges[2 * k], v = deleted_edges[2 * k + 1];
    auto found = std::find_if(endpoint.edges.begin(), endpoint.edges.end(),
                              [&](const auto &e) {
                                return (e.first == u && e.second == v) ||
                                       (e.first == v && e.second == u);
                              });
    if (found == endpoint.edges.end())
      return -1;
    state.edge[k] = found - endpoint.edges.begin();
  }
  auto values = endpoint.changed_weights(state);
  for (int weight : values)
    if (weight < 0)
      return -1;
  auto key = endpoint.invariant(state, true);
  std::copy(key.begin(), key.end(), out);
  return 0;
}

#endif

#else
// A disabled optional certificate contributes no pruning. Full PABS remains
// exact; this also avoids using POSIX-configured third-party headers on MSVC.
int synkit_mapper::pabs::edge_deck(int, const int64_t *, const int64_t *,
                                   const int32_t *, const int32_t *, int,
                                   double, int64_t, int64_t, int32_t *,
                                   int64_t *lower2, int64_t *witness2,
                                   int64_t *statistics) {
  if (lower2)
    *lower2 = 0;
  if (witness2)
    *witness2 = -1;
  if (statistics)
    std::fill(statistics, statistics + 12, 0);
  return -1;
}
int synkit_mapper::pabs::edge_deck_sizes(int, const int64_t *, const int64_t *,
                                         const int32_t *, const int32_t *, int,
                                         int64_t *sizes) {
  if (sizes)
    std::fill(sizes, sizes + 6, 0);
  return -1;
}
#endif
