// Exact integer-distance candidate enumeration for two-sided symmetry analysis.
// ABI 3: all matrices and indices are validated by the Python wrapper.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <vector>
using Perm = std::vector<int>;
using Group = std::vector<Perm>;
using Clock = std::chrono::steady_clock;
using Callback = int (*)(const int *, void *);
using CostCallback = int (*)(const int *, int, void *);
using PrefixBoundCallback = int64_t (*)(const int *, const int *, int, void *);
constexpr int INF = 100000000;
// Full PABS search only; canonicalization and pattern-cache limits below are
// separate contracts and deliberately remain at 256 vertices.
constexpr int SYNKIT_PABS_MAX_VERTICES = 512;
// Hash only an initial branching prefix: each surviving subtree belongs to
// exactly one shard. Leaves with fewer choices are partitioned at the leaf.
constexpr int SHARD_BRANCHES = 24;
struct VectorHash {
  std::size_t operator()(const Perm &p) const noexcept {
    std::size_t h = 0;
    for (int x : p)
      h = h * 1315423911u + static_cast<unsigned>(x + 1);
    return h;
  }
};
static Perm compose(const Perm &a, const Perm &b) {
  Perm p(a.size());
  for (size_t i = 0; i < a.size(); ++i)
    p[i] = b[a[i]];
  return p;
}
static std::shared_ptr<Group> stabilize(std::shared_ptr<Group> gs, int point,
                                        int n, bool *complete = nullptr) {
  bool fixed = true;
  for (const auto &g : *gs)
    if (g[point] != point) {
      fixed = false;
      break;
    }
  if (fixed)
    return gs;
  Perm identity(n);
  std::iota(identity.begin(), identity.end(), 0);
  std::vector<Perm> trans(n), inv(n);
  std::vector<int> orbit{point};
  trans[point] = identity;
  int work = 0;
  for (size_t k = 0; k < orbit.size(); ++k)
    for (const auto &g : *gs) {
      if (++work > 100000) {
        if (complete)
          *complete = false;
        return std::make_shared<Group>();
      }
      int x = orbit[k], y = g[x];
      if (trans[y].empty()) {
        trans[y] = compose(trans[x], g);
        orbit.push_back(y);
      }
    }
  for (int x : orbit) {
    inv[x].resize(n);
    for (int i = 0; i < n; ++i)
      inv[x][trans[x][i]] = i;
  }
  auto out = std::make_shared<Group>();
  std::unordered_set<Perm, VectorHash> seen;
  for (int x : orbit)
    for (const auto &g : *gs) {
      Perm h = compose(compose(trans[x], g), inv[g[x]]);
      if (h != identity && seen.insert(h).second) {
        out->push_back(std::move(h));
        if (out->size() > 512) {
          if (complete)
            *complete = false;
          return std::make_shared<Group>();
        }
      }
    }
  return out;
}

// Experimental, bounded SR-GED-style residual minimum oracle. A separator
// removes all SOURCE edges between the two child row sets. Their remaining
// cross cost is then the absolute TARGET cut mass, independent of the child
// bijections. Unary costs already include every previously assigned atom.
// Completing this recursion proves an exact relaxed-domain residual minimum;
// a work limit, deadline or unsuitable separator returns unknown, never a
// bound. This bounded strategy does not implement the full SR-GED complexity
// theorem.
struct SeparatorResidualMinimum {
  struct Stop {};
  int n;
  const int *a, *b;
  Perm unary;
  int64_t max_states, states = 0;
  Clock::time_point started;
  double seconds;
  SeparatorResidualMinimum(int size, const int *aa, const int *bb, Perm costs,
                           int64_t limit, Clock::time_point start, double secs)
      : n(size), a(aa), b(bb), unary(std::move(costs)), max_states(limit),
        started(start), seconds(secs) {}
  void tick() {
    if (++states > max_states)
      throw Stop{};
    if ((states & 63) == 1 &&
        std::chrono::duration<double>(Clock::now() - started).count() >=
            seconds)
      throw Stop{};
  }
  static int size(unsigned mask) {
    int answer = 0;
    for (; mask; mask &= mask - 1)
      ++answer;
    return answer;
  }
  static int first(unsigned mask) {
    int bit = 0;
    while (!(mask & (1u << bit)))
      ++bit;
    return bit;
  }
  bool split(unsigned rows, unsigned &separator, unsigned &left,
             unsigned &right) {
    const int bound = 2 * size(rows) / 3;
    auto attempt = [&](unsigned removed) {
      tick();
      std::vector<unsigned> components;
      unsigned unseen = rows & ~removed;
      while (unseen) {
        unsigned component = 0, pending = unseen & (~unseen + 1u);
        unseen &= ~pending;
        while (pending) {
          int x = first(pending);
          pending &= pending - 1;
          component |= 1u << x;
          unsigned remaining = unseen;
          while (remaining) {
            int y = first(remaining);
            remaining &= remaining - 1;
            if (a[x * n + y] != 0) {
              unseen &= ~(1u << y);
              pending |= 1u << y;
            }
          }
        }
        components.push_back(component);
      }
      if (components.size() < 2)
        return false;
      for (unsigned selected = 1; selected < (1u << components.size()) - 1;
           ++selected) {
        tick();
        if (!(selected & 1u))
          continue;
        unsigned one = 0;
        for (size_t k = 0; k < components.size(); ++k)
          if (selected & (1u << k))
            one |= components[k];
        unsigned two = rows & ~(removed | one);
        if (one && two && std::max(size(one), size(two)) <= bound) {
          separator = removed;
          left = one;
          right = two;
          return true;
        }
      }
      return false;
    };
    if (attempt(0))
      return true;
    for (unsigned bits = rows; bits; bits &= bits - 1) {
      int x = first(bits);
      if (attempt(1u << x))
        return true;
    }
    for (unsigned bits = rows; bits; bits &= bits - 1) {
      int x = first(bits);
      for (unsigned other = bits & (bits - 1); other; other &= other - 1) {
        int y = first(other);
        if (attempt((1u << x) | (1u << y)))
          return true;
      }
    }
    return false;
  }
  int brute(unsigned rows, unsigned columns, const Perm &costs) {
    int best = INF;
    Perm mapping(n, -1);
    auto visit = [&](auto &&self, unsigned remaining, unsigned free,
                     unsigned assigned, int value) -> void {
      tick();
      if (value >= best)
        return;
      if (!remaining) {
        best = value;
        return;
      }
      int x = first(remaining);
      for (unsigned bits = free; bits; bits &= bits - 1) {
        int y = first(bits), delta = costs[x * n + y];
        if (delta >= INF / 2)
          continue;
        for (unsigned previous = assigned; previous; previous &= previous - 1) {
          int p = first(previous);
          delta += 2 * std::abs(a[x * n + p] - b[y * n + mapping[p]]);
        }
        mapping[x] = y;
        self(self, remaining & ~(1u << x), free & ~(1u << y),
             assigned | (1u << x), value + delta);
        mapping[x] = -1;
      }
    };
    visit(visit, rows, columns, 0, 0);
    return best;
  }
  int solve(unsigned rows, unsigned columns, const Perm &costs) {
    tick();
    int count = size(rows);
    if (!count)
      return 0;
    for (unsigned bits = rows; bits; bits &= bits - 1) {
      int x = first(bits);
      bool possible = false;
      for (unsigned choices = columns; choices; choices &= choices - 1) {
        int y = first(choices);
        if (costs[x * n + y] < INF / 2) {
          possible = true;
          break;
        }
      }
      if (!possible)
        return INF;
    }
    if (count <= 4)
      return brute(rows, columns, costs);
    unsigned separator, left, right;
    if (!split(rows, separator, left, right))
      throw Stop{};
    Perm assigned(n, -1);
    int best = INF, left_count = size(left);
    auto extend = [&](auto &&self, unsigned remaining, unsigned free,
                      int offset) -> void {
      tick();
      if (offset >= best)
        return;
      if (remaining) {
        int x = first(remaining);
        for (unsigned bits = free; bits; bits &= bits - 1) {
          int y = first(bits), delta = costs[x * n + y];
          if (delta >= INF / 2)
            continue;
          for (unsigned previous = separator & ~remaining; previous;
               previous &= previous - 1) {
            int p = first(previous);
            delta += 2 * std::abs(a[x * n + p] - b[y * n + assigned[p]]);
          }
          assigned[x] = y;
          self(self, remaining & ~(1u << x), free & ~(1u << y), offset + delta);
          assigned[x] = -1;
        }
        return;
      }
      Perm updated = costs;
      for (unsigned bits = rows & ~separator; bits; bits &= bits - 1) {
        int x = first(bits);
        for (unsigned choices = free; choices; choices &= choices - 1) {
          int y = first(choices);
          if (updated[x * n + y] >= INF / 2)
            continue;
          for (unsigned previous = separator; previous;
               previous &= previous - 1) {
            int p = first(previous);
            updated[x * n + y] +=
                2 * std::abs(a[x * n + p] - b[y * n + assigned[p]]);
          }
        }
      }
      for (unsigned one = free; one; one = (one - 1) & free) {
        tick();
        if (size(one) != left_count)
          continue;
        unsigned two = free & ~one;
        int cut = 0;
        for (unsigned l = one; l; l &= l - 1) {
          int i = first(l);
          for (unsigned r = two; r; r &= r - 1) {
            int j = first(r);
            cut += 2 * std::abs(b[i * n + j]);
          }
        }
        if (offset + cut >= best)
          continue;
        int lcost = solve(left, one, updated);
        if (lcost >= INF / 2 || offset + cut + lcost >= best)
          continue;
        int rcost = solve(right, two, updated);
        if (rcost >= INF / 2)
          continue;
        best = std::min(best, offset + cut + lcost + rcost);
      }
    };
    extend(extend, separator, columns, 0);
    return best;
  }
  bool run(int &value) {
    try {
      value = solve((1u << n) - 1, (1u << n) - 1, unary);
      return true;
    } catch (const Stop &) {
      return false;
    }
  }
};

// FULL-objective lower bound, in doubled-CD units. This is not a residual
// cost and must never be added to the traversal's committed quarter cost.

// Exact minima of -2min(w,B[j,l])-first[j]-second[l], j!=l.
// Zero target entries need only the two largest opposite potentials.
static void sparse_pair_marginals(
    int m, int64_t weight, const std::vector<int64_t> &first,
    const std::vector<int64_t> &second, const Perm &own, const Perm &other,
    const std::vector<std::vector<std::pair<int, int64_t>>> &positive,
    std::vector<int64_t> &out, int64_t forbidden = INT64_MAX) {
  out.assign(m, forbidden);
  std::vector<char> allowed(m, 0);
  int best = -1, next = -1;
  for (int l : other) {
    allowed[l] = 1;
    if (best < 0 || second[l] > second[best]) {
      next = best;
      best = l;
    } else if (next < 0 || second[l] > second[next])
      next = l;
  }
  for (int j : own) {
    const int l = best == j ? next : best;
    if (l >= 0)
      out[j] = -first[j] - second[l];
    for (auto edge : positive[j])
      if (edge.first != j && allowed[edge.first])
        out[j] = std::min(out[j], -2 * std::min(weight, edge.second) -
                                      first[j] - second[edge.first]);
  }
}

struct RootDualBranchCertificate {
  static constexpr int64_t forbidden = INT64_MAX / 16;
  struct Factor {
    int i, k;
    int64_t weight, minimum;
    std::vector<int64_t> first, second, conditional_i, conditional_k;
  };
  int n = 0, m = 0;
  int64_t constant = 0, root_value = INT64_MIN;
  bool ready = false, normalized = false;
  Perm rows, columns, column_index, types_r, types_p;
  std::vector<int64_t> reduced_unary, target;
  std::vector<Perm> row_columns;
  std::vector<Factor> factors;
  bool finalize(Clock::time_point deadline) {
    ready = false;
    if (!n)
      return false;
    std::vector<Perm> domains(m);
    std::vector<std::vector<std::pair<int, int64_t>>> positive(m);
    for (int i = 0; i < m; ++i) {
      if (Clock::now() >= deadline)
        return false;
      for (int j = 0; j < m; ++j) {
        if (reduced_unary[i * m + j] < forbidden)
          domains[i].push_back(j);
        if (target[i * m + j] > 0)
          positive[i].push_back({j, target[i * m + j]});
      }
    }
    for (auto &e : factors) {
      if (Clock::now() >= deadline)
        return false;
      sparse_pair_marginals(m, e.weight, e.first, e.second, domains[e.i],
                            domains[e.k], positive, e.conditional_i, forbidden);
      sparse_pair_marginals(m, e.weight, e.second, e.first, domains[e.k],
                            domains[e.i], positive, e.conditional_k, forbidden);
      int64_t minimum = forbidden;
      for (int j : domains[e.i])
        minimum = std::min(minimum, e.conditional_i[j]);
      if (minimum != e.minimum)
        throw std::runtime_error("Invalid sparse conditioned factor minimum");
    }
    if (Clock::now() >= deadline)
      return false;
    if (!normalized) {
      for (const auto &e : factors)
        constant += e.minimum;
      normalized = true;
    }
    ready = true;
    return true;
  }

  int64_t triplet_floor(Clock::time_point deadline, int64_t *stats) const {
    const auto begun = Clock::now();
    std::fill(stats, stats + 8, 0);
    stats[5] = root_value;
    if (!ready || m < 3 || factors.size() < 2) {
      stats[6] = root_value;
      return root_value;
    }
    struct Two {
      int first = -1, second = -1;
      int64_t a = forbidden, b = forbidden;
    };
    struct Cut {
      int center, left, right;
      int64_t gain;
    };
    std::vector<Perm> incident(m), positive(m);
    for (size_t k = 0; k < factors.size(); ++k) {
      incident[factors[k].i].push_back(k);
      incident[factors[k].k].push_back(k);
    }
    for (int j = 0; j < m; ++j)
      for (int l = 0; l < m; ++l)
        if (target[j * m + l] > 0)
          positive[j].push_back(l);
    std::vector<std::array<std::vector<Two>, 2>> table(factors.size());
    std::vector<std::array<bool, 2>> done(factors.size());
    auto prepare = [&](int index, int side) {
      if (done[index][side])
        return true;
      if (Clock::now() >= deadline)
        return false;
      const auto &e = factors[index];
      const int center = side ? e.k : e.i, outer = side ? e.i : e.k;
      const auto &own = side ? e.second : e.first;
      const auto &other = side ? e.first : e.second;
      Perm leading(3, -1);
      std::vector<char> allowed(m, 0);
      auto potential = [&](int j) {
        return other[j] - reduced_unary[outer * m + j];
      };
      for (int l : row_columns[outer]) {
        allowed[l] = 1;
        for (int p = 0; p < 3; ++p)
          if (leading[p] < 0 || potential(l) > potential(leading[p])) {
            for (int q = 2; q > p; --q)
              leading[q] = leading[q - 1];
            leading[p] = l;
            break;
          }
      }
      auto &result = table[index][side];
      result.resize(m);
      for (int j : row_columns[center]) {
        auto accept = [&](int l) {
          if (l < 0 || l == j)
            return;
          const int64_t value = -2 * std::min(e.weight, target[j * m + l]) -
                                own[j] - other[l] - e.minimum +
                                reduced_unary[outer * m + l];
          if (value < 0)
            throw std::runtime_error("Negative normalized triplet term");
          auto &t = result[j];
          if (l == t.first || l == t.second)
            return;
          if (value < t.a || (value == t.a && (t.first < 0 || l < t.first))) {
            t.second = t.first;
            t.b = t.a;
            t.first = l;
            t.a = value;
          } else if (value < t.b ||
                     (value == t.b && (t.second < 0 || l < t.second))) {
            t.second = l;
            t.b = value;
          }
        };
        // Two largest legal baseline potentials after excluding j suffice;
        // every target-positive candidate is also included at its exact cost.
        for (int l : leading)
          accept(l);
        for (int l : positive[j])
          if (allowed[l])
            accept(l);
      }
      done[index][side] = true;
      return true;
    };
    std::vector<Cut> cuts;
    bool interrupted = false;
    for (int center = 0; center < m && !interrupted; ++center) {
      const auto &edges = incident[center];
      for (size_t a = 0; a < edges.size() && !interrupted; ++a)
        for (size_t b = a + 1; b < edges.size(); ++b) {
          if (Clock::now() >= deadline) {
            interrupted = true;
            break;
          }
          const int first = edges[a], second = edges[b];
          const auto &e = factors[first];
          const auto &f = factors[second];
          const int es = e.i == center ? 0 : 1, fs = f.i == center ? 0 : 1;
          const int left = es ? e.i : e.k, right = fs ? f.i : f.k;
          if (left == right)
            continue;
          if (!prepare(first, es) || !prepare(second, fs)) {
            interrupted = true;
            break;
          }
          int64_t minimum = forbidden;
          for (int j : row_columns[center]) {
            const auto &x = table[first][es][j];
            const auto &y = table[second][fs][j];
            if (x.first < 0 || y.first < 0)
              continue;
            const int64_t pair =
                x.first == y.first ? std::min(x.b + y.a, x.a + y.b) : x.a + y.a;
            if (pair < forbidden)
              minimum = std::min(minimum, pair + reduced_unary[center * m + j]);
          }
          ++stats[0];
          if (minimum > 0 && minimum < forbidden) {
            cuts.push_back({center, left, right, minimum});
            ++stats[1];
          }
        }
    }
    std::sort(cuts.begin(), cuts.end(), [](const Cut &a, const Cut &b) {
      if (a.gain != b.gain)
        return a.gain > b.gain;
      if (a.center != b.center)
        return a.center < b.center;
      if (a.left != b.left)
        return a.left < b.left;
      return a.right < b.right;
    });
    std::vector<char> used(m, 0);
    int64_t gain = 0;
    for (const auto &cut : cuts)
      if (!used[cut.center] && !used[cut.left] && !used[cut.right]) {
        used[cut.center] = used[cut.left] = used[cut.right] = 1;
        gain += cut.gain;
        ++stats[2];
      }
    stats[3] = gain;
    stats[4] = std::chrono::duration_cast<std::chrono::nanoseconds>(
                   Clock::now() - begun)
                   .count();
    stats[6] = root_value + gain;
    stats[7] = interrupted ? 1 : 0;
    return root_value + gain;
  }

  int64_t lower_bound(const int *mapping, const int *used,
                      int quarter_target = -1,
                      int64_t cached_factor_total = -1) const {
    int64_t value =
        constant + (cached_factor_total >= 0 ? cached_factor_total : 0);
    // Every remaining contribution is nonnegative after exact factor minima
    // are absorbed. A threshold return is therefore still a FULL lower bound.
    if (quarter_target >= 0 && 2 * value > quarter_target)
      return value;
    std::array<int, 128> images;
    for (int i = 0; i < m; ++i) {
      images[i] = -1;
      const int image = mapping[rows[i]];
      if (image >= 0) {
        int j = image < n ? column_index[image] : -1;
        if (j < 0 || reduced_unary[i * m + j] >= forbidden)
          return forbidden;
        images[i] = j;
        value += reduced_unary[i * m + j];
      } else {
        int64_t minimum = forbidden;
        for (int j : row_columns[i])
          if (!used[columns[j]]) {
            minimum = std::min(minimum, reduced_unary[i * m + j]);
            if (!minimum)
              break;
          }
        if (minimum >= forbidden)
          return forbidden;
        value += minimum;
      }
      if (quarter_target >= 0 && 2 * value > quarter_target)
        return value;
    }
    if (cached_factor_total >= 0)
      return value;
    for (const auto &e : factors) {
      const int j = images[e.i], l = images[e.k];
      if (j >= 0 && l >= 0)
        value += -2 * std::min(e.weight, target[j * m + l]) - e.first[j] -
                 e.second[l] - e.minimum;
      else if (j >= 0)
        value += e.conditional_i[j] - e.minimum;
      else if (l >= 0)
        value += e.conditional_k[l] - e.minimum;
      if (quarter_target >= 0 && 2 * value > quarter_target)
        return value;
    }
    return value;
  }
};

// Performance scheduling only. Skipping a valid optional bound does not
// restrict any assignment, stabilizer, or permitted domain.
static bool conditioned_gap_allowed(int best4, int floor4, int step4) {
  if (floor4 < 0 || step4 <= 0 || best4 < floor4)
    return false;
  return static_cast<int64_t>(best4) - floor4 <= 2LL * step4;
}

// Solver-owned, exact normalized factor sum. No domain or group is changed.
struct RootDualFactorCache {
  const RootDualBranchCertificate *certificate;
  Perm row_index, images;
  std::vector<Perm> incident;
  int64_t total = 0, updates = 0, touches = 0;
  explicit RootDualFactorCache(const RootDualBranchCertificate *c,
                               const int *mapping)
      : certificate(c) {
    row_index.assign(c->n, -1);
    images.assign(c->m, -1);
    incident.resize(c->m);
    for (int i = 0; i < c->m; ++i) {
      row_index[c->rows[i]] = i;
      int image = mapping[c->rows[i]];
      if (image >= 0)
        images[i] = c->column_index[image];
    }
    for (size_t k = 0; k < c->factors.size(); ++k) {
      const auto &e = c->factors[k];
      incident[e.i].push_back(k);
      incident[e.k].push_back(k);
      total += value(e, images[e.i], images[e.k]);
    }
  }
  int64_t value(const RootDualBranchCertificate::Factor &e, int j,
                int l) const {
    if (j >= 0 && l >= 0)
      return -2 * std::min(e.weight,
                           certificate->target[j * certificate->m + l]) -
             e.first[j] - e.second[l] - e.minimum;
    if (j >= 0)
      return e.conditional_i[j] - e.minimum;
    if (l >= 0)
      return e.conditional_k[l] - e.minimum;
    return 0;
  }
  void update(int atom, int image, int sign) {
    const int i = row_index[atom];
    if (i < 0)
      return;
    const int old = images[i],
              next = sign > 0 ? certificate->column_index[image] : -1;
    for (int k : incident[i]) {
      const auto &e = certificate->factors[k];
      const int j = images[e.i], l = images[e.k];
      total +=
          value(e, e.i == i ? next : j, e.k == i ? next : l) - value(e, j, l);
      ++touches;
    }
    images[i] = next;
    ++updates;
    (void)old;
  }
};
static thread_local int64_t conditioned_factor_cache_statistics[8] = {};

struct Solver {
  int n, nt, nl, target;
  double seconds;
  int64_t cap;
  int shard_index = 0, shards = 1;
  // Legacy callers use an exact shell. Additive objective entry points widen
  // this interval or shrink its upper end as better minimum witnesses arrive.
  int lower_target;
  bool minimize = false;
  int best_cost = INF, cost_step = 1, global_floor = -1;
  Perm best_mapping;
  CostCallback cost_callback = nullptr;
  const RootDualBranchCertificate *root_certificate = nullptr;
  PrefixBoundCallback prefix_bound = nullptr;
  void *prefix_bound_data = nullptr;
  std::unique_ptr<RootDualFactorCache> factor_cache;
  int64_t root_certificate_calls = 0, root_certificate_pruned = 0;
  int64_t forced_closures = 0, leaf_full_recomputes = 0;
  int64_t *diagnostic_gaps = nullptr;
  bool separator_enabled = false;
  int separator_limit = 10;
  int64_t separator_max_states = 20000;
  int64_t separator_calls = 0, separator_known = 0, separator_pruned = 0,
          separator_states = 0;
  const int *a, *b, *ca, *cb, *order,
      *pred; // pred is reserved for ABI compatibility.
  std::vector<int> mapping, used, cross, ha, hb, levels, row_order, row_min;
  std::vector<char> permitted;
  std::vector<Perm> nonzero_b_columns;
  Callback callback;
  void *userdata;
  Clock::time_point started;
  // Resumable jobs name a subtree by its deterministic sequence of images.
  // Counting only nodes below that prefix guarantees progress even for tiny
  // slices and long chains of forced assignments.
  Perm prefix, path;
  std::vector<Perm> frontier;
  struct PrefixNode {
    std::map<int, int> children;
    bool terminal = false;
  };
  std::vector<PrefixNode> requests;
  bool batch_mode = false, balance_slice = false;
  int64_t replay_nodes = 0, new_nodes = 0;
  int64_t slice_nodes = 0, work_nodes = 0;
  Clock::time_point slice_started{};
  int status = 0;
  int64_t nodes = 0, leaves = 0, accepted = 0, pruned = 0;
  Solver(int nn, int types, int lev, int tar, double sec, int64_t limit,
         const int *aa, const int *bb, const int *cca, const int *ccb,
         const int *ord, const int *pre, Callback call, void *data)
      : n(nn), nt(types), nl(lev), target(tar), seconds(sec), cap(limit),
        lower_target(tar), a(aa), b(bb), ca(cca), cb(ccb), order(ord),
        pred(pre), mapping(n, -1), used(n, 0), cross(n * n, 0), ha(n * nt * nl),
        hb(n * nt * nl), row_order(ord, ord + n), row_min(n, -1),
        permitted(n * n, 1), callback(call), userdata(data),
        started(Clock::now()) {
    std::vector<int> values(a, a + n * n);
    values.insert(values.end(), b, b + n * n);
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    if (static_cast<int>(values.size()) != nl + 1)
      throw std::runtime_error("levels");
    levels = values;
    nonzero_b_columns.resize(n);
    for (int image = 0; image < n; ++image)
      for (int j = 0; j < n; ++j)
        if (b[j * n + image] != 0)
          nonzero_b_columns[image].push_back(j);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        for (int k = 0; k < nl; ++k) {
          if (a[i * n + j] >= levels[k + 1])
            ++ha[(i * nt + ca[j]) * nl + k];
          if (b[i * n + j] >= levels[k + 1])
            ++hb[(i * nt + cb[j]) * nl + k];
        }
  }
  bool expired() {
    return std::chrono::duration<double>(Clock::now() - started).count() >=
           seconds;
  }
  bool accepts(int cost) const {
    return cost >= lower_target && cost <= target;
  }
  void update(int atom, int image, int sign, int depth, const Perm &columns) {
    if (factor_cache)
      factor_cache->update(atom, image, sign);
    // row_order[depth+1:] and columns minus image are exactly the unassigned
    // vertices, both on entry and after the recursive row-order restoration.
    for (int pos = depth + 1; pos < n; ++pos) {
      int i = row_order[pos];
      for (int k = 0; k < nl; ++k)
        if (a[i * n + atom] >= levels[k + 1])
          ha[(i * nt + ca[atom]) * nl + k] -= sign;
      const int weight = a[i * n + atom];
      if (weight != 0) {
        for (int j : columns)
          if (j != image)
            cross[i * n + j] += sign * 2 * std::abs(weight - b[j * n + image]);
      } else {
        // Zero/zero pairs have exactly zero update. The static column support
        // is intersected with the CURRENT unassigned columns; no edge cost,
        // bound, domain or traversal decision is approximated or omitted.
        for (int j : nonzero_b_columns[image])
          if (j != image && !used[j])
            cross[i * n + j] += sign * 2 * std::abs(b[j * n + image]);
      }
    }
    for (int j : columns)
      if (j != image)
        for (int k = 0; k < nl; ++k)
          if (b[j * n + image] >= levels[k + 1])
            hb[(j * nt + cb[image]) * nl + k] -= sign;
  }
  // Dense Hungarian assignment with integer duals. Infeasibility fails closed.
  static int assignment(const std::vector<int> &cost, int m, Perm &match,
                        Perm &u, Perm &v, const Perm *seed_v = nullptr,
                        const Perm *seed_match = nullptr) {
    Perm p(m + 1), way(m + 1);
    u.assign(m + 1, 0);
    v.assign(m + 1, 0);
    // Inherited potentials are hints. Repair every row against CURRENT finite
    // edges; never assume the parent dual remains feasible after a cost change.
    const bool warm = seed_v && static_cast<int>(seed_v->size()) == m;
    if (warm)
      for (int j = 1; j <= m; ++j)
        v[j] = std::clamp((*seed_v)[j - 1], -1000000, 1000000);
    // Feasible row/column dual reductions, followed by a complementary
    // zero-reduced-cost partial matching. Augment only unmatched rows.
    for (int i = 1; i <= m; ++i) {
      int best = INF;
      if (warm) {
        for (int j = 1; j <= m; ++j)
          if (cost[(i - 1) * m + j - 1] < INF / 2)
            best = std::min(best, cost[(i - 1) * m + j - 1] - v[j]);
      } else {
        // Preserve the simple vectorizable reduction for fresh small problems.
        for (int j = 1; j <= m; ++j)
          best = std::min(best, cost[(i - 1) * m + j - 1]);
      }
      if (best >= INF / 2)
        return INF;
      u[i] = best;
    }
    for (int j = 1; j <= m; ++j) {
      int best = INF;
      for (int i = 1; i <= m; ++i)
        if (cost[(i - 1) * m + j - 1] < INF / 2)
          best = std::min(best, cost[(i - 1) * m + j - 1] - u[i]);
      if (best >= INF / 2)
        return INF;
      v[j] = best;
    }
    std::vector<char> matched(m + 1, 0);
    // Keep only complementary, finite, disjoint inherited matching edges.
    if (seed_match && static_cast<int>(seed_match->size()) == m)
      for (int i = 1; i <= m; ++i) {
        int column = (*seed_match)[i - 1];
        if (column < 0 || column >= m)
          continue;
        int j = column + 1;
        if (!p[j] && cost[(i - 1) * m + j - 1] < INF / 2 &&
            cost[(i - 1) * m + j - 1] == u[i] + v[j]) {
          p[j] = i;
          matched[i] = 1;
        }
      }
    for (int i = 1; i <= m; ++i)
      if (!matched[i])
        for (int j = 1; j <= m; ++j)
          if (!p[j] && cost[(i - 1) * m + j - 1] < INF / 2 &&
              cost[(i - 1) * m + j - 1] == u[i] + v[j]) {
            p[j] = i;
            matched[i] = 1;
            break;
          }
    for (int i = 1; i <= m; ++i) {
      if (matched[i])
        continue;
      p[0] = i;
      int j0 = 0;
      Perm minv(m + 1, INF);
      std::vector<char> seen(m + 1, 0);
      do {
        seen[j0] = 1;
        int i0 = p[j0], delta = INF, j1 = 0;
        for (int j = 1; j <= m; ++j)
          if (!seen[j]) {
            int cur = cost[(i0 - 1) * m + j - 1] - u[i0] - v[j];
            if (cur < minv[j]) {
              minv[j] = cur;
              way[j] = j0;
            }
            if (minv[j] < delta) {
              delta = minv[j];
              j1 = j;
            }
          }
        if (delta >= INF / 2)
          return INF;
        for (int j = 0; j <= m; ++j)
          if (seen[j]) {
            u[p[j]] += delta;
            v[j] -= delta;
          } else
            minv[j] -= delta;
        j0 = j1;
      } while (p[j0] != 0);
      do {
        int j1 = way[j0];
        p[j0] = p[j1];
        j0 = j1;
      } while (j0);
    }
    match.resize(m);
    int total = 0;
    for (int j = 1; j <= m; ++j) {
      match[p[j] - 1] = j - 1;
      int value = cost[(p[j] - 1) * m + j - 1];
      if (value >= INF / 2)
        return INF;
      total += value;
    }
    return total;
  }

  // Exact forced-LAP return paths for ONE source row. Unlike the ordinary
  // traversal, root message updates never consume other rows' path values.
  static void forced_to_row(const Perm &cost, int m, const Perm &match,
                            const Perm &u, const Perm &v, int chosen,
                            Perm &distance) {
    distance.assign(m, INF);
    distance[chosen] = 0;
    std::vector<char> closed(m, 0);
    for (int step = 0; step < m; ++step) {
      int next = -1;
      for (int i = 0; i < m; ++i)
        if (!closed[i] && (next < 0 || distance[i] < distance[next]))
          next = i;
      if (next < 0 || distance[next] >= INF)
        break;
      closed[next] = 1;
      const int column = match[next];
      for (int i = 0; i < m; ++i)
        if (!closed[i] && cost[i * m + column] < INF / 2) {
          const int weight = cost[i * m + column] - u[i + 1] - v[column + 1];
          if (weight < 0)
            throw std::runtime_error("Negative reduced assignment arc");
          distance[i] = std::min(distance[i], distance[next] + weight);
        }
    }
  }

  static int forced_paths(const Perm &cost, int m, const Perm &match,
                          const Perm &u, const Perm &v, Perm &dist,
                          Perm &component, bool zero_only = false) {
    dist.assign(m * m, INF);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < m; ++j)
        if (cost[i * m + match[j]] < INF / 2)
          dist[i * m + j] = cost[i * m + match[j]] - u[i + 1] - v[match[j] + 1];
    // Contract strongly connected zero-reduced-cost components. Movement
    // within each component is free, so shortest-path distances are exactly
    // the distances in this smaller weighted quotient graph.
    component.resize(m);
    std::iota(component.begin(), component.end(), 0);
    int dim = m;
    if (m > 4 || zero_only) {
      std::vector<char> seen(m, 0);
      Perm finished;
      auto forward = [&](auto &&self, int x) -> void {
        seen[x] = 1;
        for (int y = 0; y < m; ++y)
          if (dist[x * m + y] == 0 && !seen[y])
            self(self, y);
        finished.push_back(x);
      };
      for (int x = 0; x < m; ++x)
        if (!seen[x])
          forward(forward, x);
      std::fill(seen.begin(), seen.end(), 0);
      dim = 0;
      auto reverse = [&](auto &&self, int x) -> void {
        seen[x] = 1;
        component[x] = dim;
        for (int y = 0; y < m; ++y)
          if (dist[y * m + x] == 0 && !seen[y])
            self(self, y);
      };
      for (auto it = finished.rbegin(); it != finished.rend(); ++it)
        if (!seen[*it]) {
          reverse(reverse, *it);
          ++dim;
        }
      if (zero_only) {
        // At zero slack a forced edge is feasible precisely when its reduced
        // cost is zero and its endpoints share a zero-cost SCC. Nonnegative
        // reduced costs make every other alternating cycle strictly positive.
        dist.assign(dim * dim, INF);
        for (int i = 0; i < dim; ++i)
          dist[i * dim + i] = 0;
        return dim;
      }
      if (dim < m) {
        std::vector<int> quotient(dim * dim, INF);
        for (int i = 0; i < m; ++i)
          for (int j = 0; j < m; ++j) {
            int pos = component[i] * dim + component[j];
            quotient[pos] = std::min(quotient[pos], dist[i * m + j]);
          }
        dist.swap(quotient);
      } else
        std::iota(component.begin(), component.end(), 0);
    }
    for (int k = 0; k < dim; ++k)
      for (int i = 0; i < dim; ++i) {
        // A finite alternating path can exceed INF/2 in the diagnostic domain.
        int first = dist[i * dim + k];
        if (first >= INF)
          continue;
        for (int j = 0; j < dim; ++j)
          dist[i * dim + j] =
              std::min(dist[i * dim + j], first + dist[k * dim + j]);
      }
    return dim;
  }
  static uint64_t shard_mix(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
  }
  struct AssignmentSeed {
    Perm columns, images;
  };
  void visit(int depth, int committed, std::shared_ptr<Group> generators,
             std::shared_ptr<Group> reactant_generators, int branch_level = 0,
             uint64_t branch_hash = 0, int request = 0,
             const AssignmentSeed *parent_seed = nullptr) {
    if (status)
      return;
    if (minimize && global_floor >= 0 && best_cost <= global_floor)
      return;
    if (shards > 1 && branch_level == SHARD_BRANCHES &&
        shard_mix(branch_hash) % shards != static_cast<unsigned>(shard_index))
      return;
    bool inside = batch_mode ? (request < 0 || requests[request].terminal)
                             : depth >= static_cast<int>(prefix.size());
    if (batch_mode && inside)
      request = -1;
    if (slice_nodes > 0 && inside) {
      if (work_nodes == 0)
        slice_started = Clock::now();
      // Leaf classification cost varies widely; a node budget alone can strand
      // most workers behind a long final batch. Yield only at subtree
      // boundaries, preserving the same disjoint complete frontier, with
      // guaranteed progress.
      bool wall_slice =
          balance_slice && work_nodes >= 64 &&
          std::chrono::duration<double>(Clock::now() - slice_started).count() >=
              0.05;
      if (work_nodes >= slice_nodes || wall_slice) {
        frontier.push_back(path);
        return;
      }
      ++work_nodes;
    }
    ++nodes;
    if (inside)
      ++new_nodes;
    else
      ++replay_nodes;
    if (expired()) {
      status = 1;
      return;
    }
    if (diagnostic_gaps) {
      const int64_t gap = best_cost - global_floor;
      ++diagnostic_gaps[3];
      ++diagnostic_gaps[gap <= cost_step ? 4 : gap <= 2LL * cost_step ? 5 : 6];
      diagnostic_gaps[13] = std::min(diagnostic_gaps[13], gap);
      diagnostic_gaps[14] = std::max(diagnostic_gaps[14], gap);
    }
    if (committed > target) {
      ++pruned;
      return;
    }
    if (prefix_bound) {
      const int64_t full = prefix_bound(mapping.data(), used.data(), target,
                                         prefix_bound_data);
      if (expired()) {
        status = 1;
        return;
      }
      // This is a FULL-objective bound in core doubled-CD units.
      if (2 * full > target) {
        ++pruned;
        return;
      }
    }
    if (root_certificate &&
        conditioned_gap_allowed(best_cost, global_floor, cost_step)) {
      ++root_certificate_calls;
      if (diagnostic_gaps) {
        const int64_t gap = best_cost - global_floor;
        ++diagnostic_gaps[gap <= cost_step         ? 7
                          : gap <= 2LL * cost_step ? 8
                                                   : 9];
      }
      // close_forced temporarily installs full images without cache updates.
      // Recompute the original exact full bound at every leaf; leave parent
      // cache untouched.
      if (depth < n && !factor_cache)
        factor_cache.reset(
            new RootDualFactorCache(root_certificate, mapping.data()));
      if (depth == n)
        ++leaf_full_recomputes;
      const int64_t full =
          root_certificate->lower_bound(mapping.data(), used.data(), target,
                                        depth < n ? factor_cache->total : -1);
      if (expired()) {
        status = 1;
        return;
      }
      if (full >= RootDualBranchCertificate::forbidden || 2 * full > target) {
        if (diagnostic_gaps) {
          const int64_t gap = best_cost - global_floor;
          if (gap <= 2LL * cost_step) {
            ++diagnostic_gaps[10];
            ++diagnostic_gaps[gap <= cost_step ? 22 : 23];
          } else
            ++diagnostic_gaps[11];
        }
        ++root_certificate_pruned;
        ++pruned;
        return;
      }
    }
    if (depth == n) {
      if (shards > 1 && branch_level < SHARD_BRANCHES &&
          shard_mix(branch_hash) % shards != static_cast<unsigned>(shard_index))
        return;
      if (!inside)
        return;
      ++leaves;
      if (accepts(committed)) {
        ++accepted;
        if (minimize) {
          // All costs are nonnegative integers. Having a feasible incumbent
          // lets every subsequent bound exclude equality; a separate pass
          // enumerates all mappings at the proved minimum after this traversal
          // finishes.
          if (diagnostic_gaps) {
            if (!diagnostic_gaps[17]) {
              diagnostic_gaps[15] = nodes;
              diagnostic_gaps[16] = committed;
            }
            ++diagnostic_gaps[17];
          }
          best_cost = committed;
          best_mapping = mapping;
          target = committed - cost_step;
        } else {
          int stop = cost_callback
                         ? cost_callback(mapping.data(), committed, userdata)
                         : callback(mapping.data(), userdata);
          if (stop) {
            status = 3;
            return;
          }
          if (cap > 0 && accepted >= cap)
            status = 2;
        }
      }
      return;
    }
    int m = n - depth, atom = row_order[depth];
    Perm columns;
    columns.reserve(m);
    for (int j = 0; j < n; ++j)
      if (!used[j])
        columns.push_back(j);
    Perm candidates, candidate_bounds(minimize ? n : 0, INF);
    Perm forced_images(m);
    int chosen = -1;
    bool all_forced = true;
    // A singleton hard domain is forced in every feasible continuation.
    // Assigning it without a relaxation only defers pruning, never removes a
    // feasible completion. Apply the usual two-sided domain updates below.
    for (int i = 0; i < m; ++i) {
      int x = row_order[depth + i], count = 0, last = -1;
      for (int y : columns)
        if (ca[x] == cb[y] && y > row_min[x] && permitted[x * n + y] &&
            committed + cross[x * n + y] <= target) {
          ++count;
          last = y;
          // This guard only distinguishes empty, singleton, and nonsingleton
          // domains. Two witnesses settle the last case; exact LAP domains
          // below still inspect every allowed edge before any pruning or
          // branching.
          if (count == 2)
            break;
        }
      if (count == 0) {
        ++pruned;
        return;
      }
      if (count == 1) {
        forced_images[i] = last;
        if (chosen < 0) {
          chosen = i;
          candidates.push_back(last);
        }
      } else
        all_forced = false;
    }
    auto close_forced = [&]() -> bool {
      std::vector<char> images_seen(n, 0);
      bool fixed = true;
      for (int i = 0; i < m; ++i) {
        int x = row_order[depth + i], y = forced_images[i];
        if (images_seen[y]) {
          ++pruned;
          return true;
        }
        images_seen[y] = 1;
        for (const auto &g : *reactant_generators)
          if (g[x] != x) {
            fixed = false;
            break;
          }
        for (const auto &g : *generators)
          if (g[y] != y) {
            fixed = false;
            break;
          }
      }
      if (fixed) {
        // There is one bijective hard-domain completion and every future
        // stabilizer/domain update is inert. Verify its exact cost directly.
        int exact = committed;
        for (int i = 0; i < m; ++i) {
          int x = row_order[depth + i], y = forced_images[i];
          exact += cross[x * n + y];
          for (int j = 0; j < i; ++j)
            exact += 2 * std::abs(a[x * n + row_order[depth + j]] -
                                  b[y * n + forced_images[j]]);
        }
        if (!accepts(exact)) {
          ++pruned;
          return true;
        }
        int child_request = request;
        for (int i = 0; i < m; ++i) {
          int y = forced_images[i];
          if (batch_mode && child_request >= 0) {
            if (requests[child_request].terminal)
              child_request = -1;
            else {
              auto found = requests[child_request].children.find(y);
              if (found == requests[child_request].children.end())
                return true;
              child_request = found->second;
            }
          } else if (!batch_mode &&
                     depth + i < static_cast<int>(prefix.size()) &&
                     y != prefix[depth + i])
            return true;
        }
        for (int i = 0; i < m; ++i) {
          mapping[row_order[depth + i]] = forced_images[i];
          path.push_back(forced_images[i]);
        }
        ++forced_closures;
        visit(n, exact, generators, reactant_generators, branch_level,
              branch_hash, child_request);
        for (int i = 0; i < m; ++i) {
          mapping[row_order[depth + i]] = -1;
          path.pop_back();
        }
        return true;
      }
      return false;
    };
    if (all_forced && m > 1 && close_forced())
      return;
    if (separator_enabled && chosen < 0 && m >= 5 && m <= separator_limit) {
      Perm rows(row_order.begin() + depth, row_order.end()), aa(m * m),
          bb(m * m), unary(m * m, INF);
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j) {
          aa[i * m + j] = a[rows[i] * n + rows[j]];
          bb[i * m + j] = b[columns[i] * n + columns[j]];
          int x = rows[i], y = columns[j];
          if (ca[x] == cb[y] && y > row_min[x] && permitted[x * n + y])
            unary[i * m + j] = cross[x * n + y];
        }
      SeparatorResidualMinimum oracle(m, aa.data(), bb.data(), std::move(unary),
                                      separator_max_states, started, seconds);
      ++separator_calls;
      int lower = 0;
      bool known = oracle.run(lower);
      separator_states += oracle.states;
      if (expired()) {
        status = 1;
        return;
      }
      if (known) {
        ++separator_known;
        if (committed + lower > target) {
          ++pruned;
          ++separator_pruned;
          return;
        }
      }
    }
    // Forced-edge LAP minima below are necessary conditions for every
    // completion of this node. Retain their rejected domains in descendants;
    // restore exactly these removals when leaving this subtree. The values are
    // equivariant under the current two endpoint stabilizers, so the domain
    // invariant used by the existing orbit leaders is preserved.
    Perm certified_removed;
    struct RestoreCertifiedDomains {
      std::vector<char> &mask;
      Perm &removed;
      ~RestoreCertifiedDomains() {
        for (int index : removed)
          mask[index] = 1;
      }
    } restore_certified{permitted, certified_removed};
    AssignmentSeed next_seed;
    const AssignmentSeed *child_seed = parent_seed;
    if (chosen < 0) {
      Perm active_types;
      std::vector<char> type_seen(nt, 0);
      for (int i = depth; i < n; ++i)
        if (!type_seen[ca[row_order[i]]]) {
          type_seen[ca[row_order[i]]] = 1;
          active_types.push_back(ca[row_order[i]]);
        }
      std::vector<int> cost(m * m, INF);
      int64_t row_lower = 0;
      int remaining_budget = target - committed;
      for (int i = 0; i < m; ++i) {
        int x = row_order[depth + i], minimum = row_min[x], row_best = INF;
        for (int j = 0; j < m; ++j) {
          int y = columns[j];
          if (ca[x] != cb[y] || y <= minimum || !permitted[x * n + y])
            continue;
          int value = cross[x * n + y];
          if (value > remaining_budget)
            continue;
          for (int t : active_types) {
            for (int k = 0; k < nl; ++k)
              value += (levels[k + 1] - levels[k]) *
                       std::abs(ha[(x * nt + t) * nl + k] -
                                hb[(y * nt + t) * nl + k]);
            if (value > remaining_budget)
              break;
          }
          if (value > remaining_budget)
            continue;
          cost[i * m + j] = value;
          row_best = std::min(row_best, value);
        }
        // Each row contributes one nonnegative assignment cost. The partial
        // sum of row minima is already a valid lower bound for the entire LAP.
        if (row_best >= INF / 2 || row_lower + row_best > remaining_budget) {
          ++pruned;
          return;
        }
        row_lower += row_best;
      }
      Perm match, u, v, seed_v, seed_match;
#ifndef SYNKIT_COLD_ASSIGNMENT
      if (parent_seed && m >= 16) {
        Perm indices(n, -1);
        for (int j = 0; j < m; ++j)
          indices[columns[j]] = j;
        for (int j : columns)
          seed_v.push_back(parent_seed->columns[j]);
        for (int i = 0; i < m; ++i) {
          int image = parent_seed->images[row_order[depth + i]];
          seed_match.push_back(image < 0 ? -1 : indices[image]);
        }
      }
#endif
      int lower =
          assignment(cost, m, match, u, v, parent_seed ? &seed_v : nullptr,
                     parent_seed ? &seed_match : nullptr);
      if (lower >= INF / 2 || committed + lower > target) {
        ++pruned;
        return;
      }
#ifndef SYNKIT_COLD_ASSIGNMENT
      // Below this dimension bookkeeping costs more than a fresh small LAP.
      // This is an implementation selector; both branches solve the same LAP.
      if (m > 16) {
        next_seed.columns.assign(n, 0);
        next_seed.images.assign(n, -1);
        for (int j = 0; j < m; ++j)
          next_seed.columns[columns[j]] = v[j + 1];
        for (int i = 0; i < m; ++i)
          next_seed.images[row_order[depth + i]] = columns[match[i]];
        child_seed = &next_seed;
      } else
        child_seed = nullptr;
#endif
      Perm dist, component;
      int dim = forced_paths(cost, m, match, u, v, dist, component,
                             lower == remaining_budget);
      chosen = 0;
      int best_count = INF, best_neighbors = -1;
      all_forced = true;
      for (int i = 0; i < m; ++i) {
        int count = 0, neighbors = 0, x = row_order[depth + i];
        for (int j = 0; j < m; ++j) {
          int edge = cost[i * m + match[j]] - u[i + 1] - v[match[j] + 1];
          if (committed + lower + edge +
                  dist[component[j] * dim + component[i]] <=
              target) {
            ++count;
            forced_images[i] = columns[match[j]];
          } else {
            int y = columns[match[j]], index = x * n + y;
            if (ca[x] == cb[y] && y > row_min[x] && permitted[index]) {
              permitted[index] = 0;
              certified_removed.push_back(index);
            }
          }
        }
        if (count != 1)
          all_forced = false;
        for (int y = 0; y < n; ++y)
          if (mapping[y] >= 0 && a[x * n + y] != 0)
            ++neighbors;
        if (count < best_count ||
            (count == best_count && neighbors > best_neighbors)) {
          chosen = i;
          best_count = count;
          best_neighbors = neighbors;
        }
      }
      // Forced-edge relaxation certificates are necessary conditions for exact
      // completion. Singleton certified domains can close the same unique leaf.
      if (all_forced && m > 1 && close_forced())
        return;
      for (int i = 0; i < m; ++i) {
        int col = match[i], value = cost[chosen * m + col];
        if (value >= INF / 2 ||
            dist[component[i] * dim + component[chosen]] >= INF / 2)
          continue;
        int forced = lower + value - u[chosen + 1] - v[col + 1] +
                     dist[component[i] * dim + component[chosen]];
        if (committed + forced <= target) {
          candidates.push_back(columns[col]);
          if (minimize)
            candidate_bounds[columns[col]] = forced;
        }
      }
      std::sort(candidates.begin(), candidates.end());
    }
    std::swap(row_order[depth], row_order[depth + chosen]);
    struct Restore {
      std::vector<int> &v;
      int a, b;
      ~Restore() { std::swap(v[a], v[b]); }
    } restore{row_order, depth, depth + chosen};
    atom = row_order[depth];
    if (candidates.size() == 1) {
      int image = candidates[0];
      bool fixed = true;
      for (const auto &g : *reactant_generators)
        if (g[atom] != atom) {
          fixed = false;
          break;
        }
      if (fixed)
        for (const auto &g : *generators)
          if (g[image] != image) {
            fixed = false;
            break;
          }
      if (fixed) {
        // Both stabilizers already fix the selected points. The row orbit is
        // just the assigned row, whose row_min/permitted entries will never be
        // read by descendants. Thus all omitted group/domain updates are inert.
        int child_request = request;
        if (batch_mode && request >= 0) {
          auto found = requests[request].children.find(image);
          if (found == requests[request].children.end())
            return;
          child_request = found->second;
        } else if (!batch_mode && depth < static_cast<int>(prefix.size()) &&
                   image != prefix[depth])
          return;
        int delta = cross[atom * n + image];
        mapping[atom] = image;
        used[image] = 1;
        update(atom, image, 1, depth, columns);
        path.push_back(image);
        visit(depth + 1, committed + delta, generators, reactant_generators,
              branch_level, branch_hash, child_request, child_seed);
        path.pop_back();
        update(atom, image, -1, depth, columns);
        used[image] = 0;
        mapping[atom] = -1;
        return;
      }
    }
    Perm parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    auto find = [&](int x) {
      while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
      }
      return x;
    };
    for (const auto &g : *generators)
      for (int j : columns) {
        int x = find(j), y = find(g[j]);
        if (x != y)
          parent[y] = x;
      }
    Perm orbit_min(n, INF);
    for (int image : columns) {
      int root = find(image);
      orbit_min[root] = std::min(orbit_min[root], image);
    }
    std::vector<char> seen(n, 0);
    Perm row_orbit{atom};
    std::vector<char> row_seen(n, 0);
    row_seen[atom] = 1;
    for (size_t i = 0; i < row_orbit.size(); ++i)
      for (const auto &g : *reactant_generators) {
        int image = g[row_orbit[i]];
        if (!row_seen[image]) {
          row_seen[image] = 1;
          row_orbit.push_back(image);
        }
      }
    auto child_rows = stabilize(reactant_generators, atom, n);
    Perm choices;
    for (int image : candidates) {
      int orbit = find(image);
      if (!seen[orbit]) {
        seen[orbit] = 1;
        choices.push_back(image);
      }
    }
    // Keep numeric orbit leaders and their symmetry inequalities unchanged.
    // Only visit the surviving branches in a lower-bound-first order when
    // optimizing, so a good incumbent can tighten the other branches sooner.
    if (minimize)
      std::stable_sort(choices.begin(), choices.end(), [&](int x, int y) {
        return candidate_bounds[x] < candidate_bounds[y];
      });
    for (int image : choices) {
      int child_request = request;
      if (batch_mode && request >= 0) {
        auto found = requests[request].children.find(image);
        if (found == requests[request].children.end())
          continue;
        child_request = found->second;
      } else if (!batch_mode && depth < static_cast<int>(prefix.size()) &&
                 image != prefix[depth])
        continue;
      int delta = cross[atom * n + image];
      mapping[atom] = image;
      used[image] = 1;
      Perm old_min;
      for (int row : row_orbit) {
        old_min.push_back(row_min[row]);
        row_min[row] = std::max(row_min[row], image);
      }
      // Earlier two-sided choices exclude entire product orbits, so these
      // domains remain invariant under the current independent stabilizers.
      // A smaller orbit leader at any equivalent reactant row belongs to an
      // already enumerated branch, even when that row's numeric image is
      // larger.
      Perm removed;
      for (int row : row_orbit)
        for (int col : columns)
          if (orbit_min[find(col)] < image && permitted[row * n + col]) {
            permitted[row * n + col] = 0;
            removed.push_back(row * n + col);
          }
      update(atom, image, 1, depth, columns);
      int next_level = branch_level;
      uint64_t next_hash = branch_hash;
      if (choices.size() > 1 && branch_level < SHARD_BRANCHES) {
        ++next_level;
        next_hash = shard_mix(branch_hash + static_cast<unsigned>(image + 1) +
                              0x9e3779b97f4a7c15ULL);
      }
      path.push_back(image);
      visit(depth + 1, committed + delta, stabilize(generators, image, n),
            child_rows, next_level, next_hash, child_request, child_seed);
      path.pop_back();
      update(atom, image, -1, depth, columns);
      for (size_t i = 0; i < row_orbit.size(); ++i)
        row_min[row_orbit[i]] = old_min[i];
      for (int index : removed)
        permitted[index] = 1;
      used[image] = 0;
      mapping[atom] = -1;
      if (status)
        return;
    }
  }
};
extern "C" int synkit_distance_abi() { return 3; }
extern "C" int synkit_distance_candidates(
    int n, int nt, int nl, int target, double seconds, int64_t cap,
    const int *a, const int *b, const int *ca, const int *cb, const int *order,
    const int *pred, const int *gens, int count, const int *row_gens,
    int row_count, int shard_index, int shards, Callback callback,
    void *userdata, int64_t *statistics) {
  try {
    if (n < 1 || n > SYNKIT_PABS_MAX_VERTICES || nt < 1 || nt > n || nl < 0 ||
        nl > 15 || target < 0 || target > 1000000 || !callback || shards < 1 ||
        shard_index < 0 || shard_index >= shards)
      return -2;
    Solver solver(n, nt, nl, target, seconds, cap, a, b, ca, cb, order, pred,
                  callback, userdata);
    solver.shard_index = shard_index;
    solver.shards = shards;
    auto group = std::make_shared<Group>();
    for (int k = 0; k < count; ++k)
      group->emplace_back(gens + k * n, gens + (k + 1) * n);
    auto rows = std::make_shared<Group>();
    for (int k = 0; k < row_count; ++k)
      rows->emplace_back(row_gens + k * n, row_gens + (k + 1) * n);
    solver.visit(0, 0, group, rows);
    statistics[0] = solver.nodes;
    statistics[1] = solver.leaves;
    statistics[2] = solver.accepted;
    statistics[3] = solver.pruned;
    return solver.status;
  } catch (...) {
    return -1;
  }
}

static void objective_statistics(const Solver &solver, int64_t *statistics) {
  statistics[0] = solver.nodes;
  statistics[1] = solver.leaves;
  statistics[2] = solver.accepted;
  statistics[3] = solver.pruned;
  statistics[4] = solver.replay_nodes;
  statistics[5] = solver.new_nodes;
}

// Additive ABI 3 objectives. Matrices use doubled bond units; reported costs
// use quarter-CD units, exactly as the existing shell entry points do.
static int native_minimum_impl(
    int n, int nt, int nl, double seconds, const int *a, const int *b,
    const int *ca, const int *cb, const int *order, const int *pred,
    const int *gens, int count, const int *row_gens, int row_count,
    const int *seed, int *best_mapping, int *best_cost, int64_t *statistics,
    int separator_limit, int64_t separator_budget, bool report_separator,
    int global_floor = -1,
    const RootDualBranchCertificate *root_certificate = nullptr,
    int64_t *certificate_stats = nullptr, int64_t *diagnostic_gaps = nullptr,
    PrefixBoundCallback prefix_bound = nullptr, void *prefix_bound_data = nullptr) {
  try {
    if (statistics)
      std::fill(statistics, statistics + (report_separator ? 10 : 6), 0);
    if (n < 1 || n > SYNKIT_PABS_MAX_VERTICES || nt < 1 || nt > n || nl < 0 ||
        nl > 15 || !std::isfinite(seconds) || seconds < 0 || !a || !b || !ca ||
        !cb || !order || !seed || !best_mapping || !best_cost || !statistics ||
        count < 0 || row_count < 0 || (count && !gens) ||
        (row_count && !row_gens) || separator_limit < 0 ||
        separator_limit > 12 || (separator_limit && separator_budget < 1) ||
        global_floor < -1)
      return -2;
    Perm seen(n, 0), ordered(n, 0);
    for (int i = 0; i < n; ++i) {
      if (seed[i] < 0 || seed[i] >= n || seen[seed[i]]++ || order[i] < 0 ||
          order[i] >= n || ordered[order[i]]++ || ca[i] < 0 || ca[i] >= nt ||
          cb[i] < 0 || cb[i] >= nt || ca[i] != cb[seed[i]])
        return -2;
    }
    int incumbent = 0, bond_gcd = 0;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < i; ++j) {
        incumbent += 2 * std::abs(a[i * n + j] - b[seed[i] * n + seed[j]]);
        bond_gcd = std::gcd(bond_gcd, a[i * n + j]);
        bond_gcd = std::gcd(bond_gcd, b[i * n + j]);
        if (incumbent > 1000000)
          return -2;
      }
    if (global_floor > incumbent)
      return -2;
    // For doubled bonds divisible by g, |x-y| == x+y (mod 2g), including
    // signed bonds. Quarter-CD costs therefore share one residue modulo 4g.
    // Every improvement is at least this step below the feasible incumbent.
    int step = bond_gcd ? 4 * bond_gcd : 1;
    Solver solver(n, nt, nl, incumbent - step, seconds, -1, a, b, ca, cb, order,
                  pred, nullptr, nullptr);
    solver.lower_target = 0;
    solver.minimize = true;
    solver.global_floor = global_floor;
    solver.prefix_bound = prefix_bound;
    solver.prefix_bound_data = prefix_bound_data;
    solver.cost_step = step;
    solver.diagnostic_gaps = diagnostic_gaps;
    if (diagnostic_gaps) {
      std::fill(diagnostic_gaps, diagnostic_gaps + 18, 0);
      diagnostic_gaps[0] = incumbent;
      diagnostic_gaps[1] = global_floor;
      diagnostic_gaps[2] = step;
      diagnostic_gaps[13] = INT64_MAX;
      diagnostic_gaps[15] = diagnostic_gaps[16] = -1;
    }

    std::fill(conditioned_factor_cache_statistics,
              conditioned_factor_cache_statistics + 8, 0);
    solver.root_certificate = root_certificate && root_certificate->ready
                                  ? root_certificate
                                  : nullptr;
    solver.separator_enabled = separator_limit > 0;
    solver.separator_limit = separator_limit;
    solver.separator_max_states = separator_budget;
    solver.best_cost = incumbent;
    solver.best_mapping.assign(seed, seed + n);
    auto group = std::make_shared<Group>();
    for (int k = 0; k < count; ++k)
      group->emplace_back(gens + k * n, gens + (k + 1) * n);
    auto rows = std::make_shared<Group>();
    for (int k = 0; k < row_count; ++k)
      rows->emplace_back(row_gens + k * n, row_gens + (k + 1) * n);
    solver.visit(0, 0, group, rows);
    // A timeout still returns a feasible incumbent, while status 0 alone
    // certifies that every compatible improvement has been excluded.
    std::copy(solver.best_mapping.begin(), solver.best_mapping.end(),
              best_mapping);
    *best_cost = solver.best_cost;
    objective_statistics(solver, statistics);
    if (diagnostic_gaps)
      diagnostic_gaps[12] = solver.best_cost;
    conditioned_factor_cache_statistics[6] = solver.forced_closures;
    conditioned_factor_cache_statistics[7] = solver.leaf_full_recomputes;
    if (solver.factor_cache) {
      conditioned_factor_cache_statistics[0] = 1;
      conditioned_factor_cache_statistics[1] = solver.factor_cache->updates;
      conditioned_factor_cache_statistics[2] = solver.factor_cache->touches;
      conditioned_factor_cache_statistics[3] = solver.factor_cache->total;
      conditioned_factor_cache_statistics[4] = solver.root_certificate_calls;
      conditioned_factor_cache_statistics[5] = solver.root_certificate_pruned;
    }
    if (certificate_stats) {
      certificate_stats[0] = solver.root_certificate_calls;
      certificate_stats[1] = solver.root_certificate_pruned;
    }
    if (report_separator) {
      statistics[6] = solver.separator_calls;
      statistics[7] = solver.separator_known;
      statistics[8] = solver.separator_pruned;
      statistics[9] = solver.separator_states;
    }
    return solver.status;
  } catch (...) {
    return -3;
  }
}

extern "C" int synkit_distance_minimum(
    int n, int nt, int nl, double seconds, const int *a, const int *b,
    const int *ca, const int *cb, const int *order, const int *pred,
    const int *gens, int count, const int *row_gens, int row_count,
    const int *seed, int *best_mapping, int *best_cost, int64_t *statistics) {
  return native_minimum_impl(n, nt, nl, seconds, a, b, ca, cb, order, pred,
                             gens, count, row_gens, row_count, seed,
                             best_mapping, best_cost, statistics, 0, 0, false);
}

// Experimental additive ABI: stats[0:6] are unchanged, followed by oracle
// calls, completed minima, pruned nodes, and oracle states. A limit of zero
// disables the oracle. This entry point is intentionally opt-in.
extern "C" int synkit_distance_minimum_separator(
    int n, int nt, int nl, double seconds, const int *a, const int *b,
    const int *ca, const int *cb, const int *order, const int *pred,
    const int *gens, int count, const int *row_gens, int row_count,
    const int *seed, int *best_mapping, int *best_cost, int64_t *statistics,
    int residual_limit, int64_t max_states) {
  return native_minimum_impl(n, nt, nl, seconds, a, b, ca, cb, order, pred,
                             gens, count, row_gens, row_count, seed,
                             best_mapping, best_cost, statistics,
                             residual_limit, max_states, true);
}

static int native_range_impl(int n, int nt, int nl, int lower, int upper,
                             double seconds, int64_t cap, const int *a,
                             const int *b, const int *ca, const int *cb,
                             const int *order, const int *pred, const int *gens,
                             int count, const int *row_gens, int row_count,
                             CostCallback callback, void *userdata,
                             int64_t *statistics, int separator_limit,
                             int64_t separator_budget, bool report_separator) {
  try {
    if (statistics)
      std::fill(statistics, statistics + (report_separator ? 10 : 6), 0);
    if (n < 1 || n > SYNKIT_PABS_MAX_VERTICES || nt < 1 || nt > n || nl < 0 ||
        nl > 15 || lower < 0 || upper < lower || upper > 1000000 ||
        !std::isfinite(seconds) || seconds < 0 || !a || !b || !ca || !cb ||
        !order || !callback || !statistics || count < 0 || row_count < 0 ||
        (count && !gens) || (row_count && !row_gens) || separator_limit < 0 ||
        separator_limit > 12 || (separator_limit && separator_budget < 1))
      return -2;
    Solver solver(n, nt, nl, upper, seconds, cap, a, b, ca, cb, order, pred,
                  nullptr, userdata);
    solver.lower_target = lower;
    solver.cost_callback = callback;
    solver.separator_enabled = separator_limit > 0;
    solver.separator_limit = separator_limit;
    solver.separator_max_states = separator_budget;
    auto group = std::make_shared<Group>();
    for (int k = 0; k < count; ++k)
      group->emplace_back(gens + k * n, gens + (k + 1) * n);
    auto rows = std::make_shared<Group>();
    for (int k = 0; k < row_count; ++k)
      rows->emplace_back(row_gens + k * n, row_gens + (k + 1) * n);
    solver.visit(0, 0, group, rows);
    objective_statistics(solver, statistics);
    if (report_separator) {
      statistics[6] = solver.separator_calls;
      statistics[7] = solver.separator_known;
      statistics[8] = solver.separator_pruned;
      statistics[9] = solver.separator_states;
    }
    return solver.status;
  } catch (...) {
    return -3;
  }
}

extern "C" int synkit_distance_range(
    int n, int nt, int nl, int lower, int upper, double seconds, int64_t cap,
    const int *a, const int *b, const int *ca, const int *cb, const int *order,
    const int *pred, const int *gens, int count, const int *row_gens,
    int row_count, CostCallback callback, void *userdata, int64_t *statistics) {
  return native_range_impl(n, nt, nl, lower, upper, seconds, cap, a, b, ca, cb,
                           order, pred, gens, count, row_gens, row_count,
                           callback, userdata, statistics, 0, 0, false);
}

extern "C" int synkit_distance_range_separator(
    int n, int nt, int nl, int lower, int upper, double seconds, int64_t cap,
    const int *a, const int *b, const int *ca, const int *cb, const int *order,
    const int *pred, const int *gens, int count, const int *row_gens,
    int row_count, CostCallback callback, void *userdata, int64_t *statistics,
    int residual_limit, int64_t max_states) {
  return native_range_impl(n, nt, nl, lower, upper, seconds, cap, a, b, ca, cb,
                           order, pred, gens, count, row_gens, row_count,
                           callback, userdata, statistics, residual_limit,
                           max_states, true);
}

// Isolated oracle interface for independent residual-cost tests. Input bond
// matrices use doubled units, unary costs use quarter-CD units, and allowed
// is a zero/nonzero domain mask. Status 1 means unknown, not infeasibility.
extern "C" int synkit_separator_residual_minimum(
    int n, const int *a, const int *b, const int *unary, const int *allowed,
    double seconds, int64_t max_states, int *value, int64_t *states) {
  auto started = Clock::now();
  try {
    if (value)
      *value = -1;
    if (states)
      *states = 0;
    if (n < 1 || n > 12 || !a || !b || !unary || !allowed || !value ||
        !states || max_states < 1 || !std::isfinite(seconds) || seconds < 0)
      return -2;
    Perm costs(n * n, INF);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        if (a[i * n + j] != a[j * n + i] || b[i * n + j] != b[j * n + i] ||
            a[i * n + j] < -1024 || a[i * n + j] > 1024 ||
            b[i * n + j] < -1024 || b[i * n + j] > 1024 ||
            (i == j && (a[i * n + j] || b[i * n + j])) ||
            unary[i * n + j] < 0 || unary[i * n + j] > 1000000)
          return -2;
        if (allowed[i * n + j])
          costs[i * n + j] = unary[i * n + j];
      }
    SeparatorResidualMinimum oracle(n, a, b, std::move(costs), max_states,
                                    started, seconds);
    bool known = oracle.run(*value);
    *states = oracle.states;
    if (!known)
      *value = -1;
    return known ? 0 : 1;
  } catch (...) {
    return -3;
  }
}

// Additive ABI: the original entry point remains available for V8 callers.
// A normal slice emits a disjoint cover of every unvisited subtree.
using FrontierCallback = int (*)(const int *, int, void *);
extern "C" int synkit_distance_frontier(
    int n, int nt, int nl, int target, double seconds, int64_t cap,
    const int *a, const int *b, const int *ca, const int *cb, const int *order,
    const int *pred, const int *gens, int count, const int *row_gens,
    int row_count, int shard_index, int shards, Callback callback,
    void *userdata, int64_t *statistics, const int *prefix, int prefix_size,
    int64_t slice_nodes, FrontierCallback emit) {
  try {
    if (n < 1 || n > SYNKIT_PABS_MAX_VERTICES || nt < 1 || nt > n || nl < 0 ||
        nl > 15 || target < 0 || target > 1000000 || !callback || !emit ||
        shards != 1 || shard_index != 0 || prefix_size < 0 || prefix_size > n ||
        slice_nodes < 1)
      return -2;
    Solver solver(n, nt, nl, target, seconds, cap, a, b, ca, cb, order, pred,
                  callback, userdata);
    solver.prefix.assign(prefix, prefix + prefix_size);
    solver.slice_nodes = slice_nodes;
    auto group = std::make_shared<Group>();
    for (int k = 0; k < count; ++k)
      group->emplace_back(gens + k * n, gens + (k + 1) * n);
    auto rows = std::make_shared<Group>();
    for (int k = 0; k < row_count; ++k)
      rows->emplace_back(row_gens + k * n, row_gens + (k + 1) * n);
    solver.visit(0, 0, group, rows);
    statistics[0] = solver.nodes;
    statistics[1] = solver.leaves;
    statistics[2] = solver.accepted;
    statistics[3] = solver.pruned;
    for (const auto &job : solver.frontier)
      if (emit(job.data(), job.size(), userdata))
        return 3;
    return solver.status;
  } catch (...) {
    return -1;
  }
}

#include <map>

// Replay the trie of disjoint requested prefixes once. Below a terminal,
// normal DFS resumes; the emitted frontier remains a disjoint exact cover.
extern "C" int synkit_distance_frontier_batch(
    int n, int nt, int nl, int target, double seconds, int64_t cap,
    const int *a, const int *b, const int *ca, const int *cb, const int *order,
    const int *pred, const int *gens, int count, const int *row_gens,
    int row_count, int shard_index, int shards, Callback callback,
    void *userdata, int64_t *statistics, const int *values, const int *offsets,
    int prefix_count, int64_t slice_nodes, FrontierCallback emit) {
  try {
    if (n < 1 || n > SYNKIT_PABS_MAX_VERTICES || nt < 1 || nt > n || nl < 0 ||
        nl > 15 || target < 0 || target > 1000000 || !callback || !emit ||
        shards != 1 || shard_index != 0 || prefix_count < 1 ||
        prefix_count > 16384 || offsets[0] != 0 || slice_nodes < 1)
      return -2;
    Solver solver(n, nt, nl, target, seconds, cap, a, b, ca, cb, order, pred,
                  callback, userdata);
    solver.batch_mode = true;
    solver.balance_slice = prefix_count <= 4;
    solver.requests.emplace_back();
    solver.slice_nodes = slice_nodes;
    for (int k = 0; k < prefix_count; ++k) {
      int begin = offsets[k], end = offsets[k + 1], node = 0;
      if (begin < 0 || end < begin || end - begin > n)
        return -2;
      for (int j = begin; j < end; ++j) {
        if (solver.requests[node].terminal || values[j] < 0 || values[j] >= n)
          return -2;
        auto found = solver.requests[node].children.find(values[j]);
        if (found == solver.requests[node].children.end()) {
          int next = solver.requests.size();
          solver.requests[node].children[values[j]] = next;
          solver.requests.emplace_back();
          node = next;
        } else
          node = found->second;
      }
      if (solver.requests[node].terminal ||
          !solver.requests[node].children.empty())
        return -2;
      solver.requests[node].terminal = true;
    }
    auto group = std::make_shared<Group>();
    for (int k = 0; k < count; ++k)
      group->emplace_back(gens + k * n, gens + (k + 1) * n);
    auto rows = std::make_shared<Group>();
    for (int k = 0; k < row_count; ++k)
      rows->emplace_back(row_gens + k * n, row_gens + (k + 1) * n);
    solver.visit(0, 0, group, rows);
    statistics[0] = solver.nodes;
    statistics[1] = solver.leaves;
    statistics[2] = solver.accepted;
    statistics[3] = solver.pruned;
    statistics[4] = solver.replay_nodes;
    statistics[5] = solver.new_nodes;
    for (const auto &job : solver.frontier)
      if (emit(job.data(), job.size(), userdata))
        return 3;
    return solver.status;
  } catch (...) {
    return -1;
  }
}

#include <limits>
struct NativeCanon {
  int n;
  const int *colors;
  const int *edges;
  double seconds;
  int64_t max_nodes;
  Clock::time_point start = Clock::now();
  int64_t nodes = 0;
  bool incomplete = false, root_twins_only = false;
  Group generators;
  std::unordered_set<Perm, VectorHash> signatures;
  Perm best, best_key, twin_class;
  struct NeighborRows {
    Perm offsets;
    std::vector<std::pair<int, int>> entries;
    explicit NeighborRows(int n) : offsets(n + 1) { entries.reserve(4 * n); }
    struct Range {
      const std::pair<int, int> *first, *last;
      auto begin() const { return first; }
      auto end() const { return last; }
      size_t size() const { return last - first; }
    };
    Range operator[](int i) const {
      return {entries.data() + offsets[i], entries.data() + offsets[i + 1]};
    }
  };
  NeighborRows neighbors;
  NativeCanon(int nn, const int *c, const int *e, double sec, int64_t budget)
      : n(nn), colors(c), edges(e), seconds(sec), max_nodes(budget),
        twin_class(nn), neighbors(nn) {
    std::iota(twin_class.begin(), twin_class.end(), 0);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        if (edges[i * n + j] < 0)
          throw std::runtime_error("negative edge palette ID");
        if (edges[i * n + j])
          neighbors.entries.push_back({j, edges[i * n + j]});
      }
      neighbors.offsets[i + 1] = neighbors.entries.size();
    }
  }
  bool valid(const Perm &g) {
    // A bijection preserving every nonzero colored edge also preserves absent
    // edges: its injection on the finite colored edge sets is a bijection.
    std::vector<char> seen(n, 0);
    for (int i = 0; i < n; ++i) {
      if (g[i] < 0 || g[i] >= n || seen[g[i]])
        return false;
      seen[g[i]] = 1;
      if (colors[i] != colors[g[i]])
        return false;
    }
    for (int i = 0; i < n; ++i)
      for (const auto &entry : neighbors[i])
        if (entry.second != edges[g[i] * n + g[entry.first]])
          return false;
    return true;
  }
  void accept(const Perm &g) {
    bool identity = true;
    for (int i = 0; i < n; ++i)
      if (g[i] != i) {
        identity = false;
        break;
      }
    if (!identity && signatures.insert(g).second && valid(g))
      generators.push_back(g);
  }
  using Partition = std::vector<Perm>;
  // Positive edge IDs allow 0 to terminate a cell's (color,count) list.
  // This flat encoding has exactly the lexicographic order of the legacy
  // vector<pair<-cell, vector<pair<color,count>>>> signature.
  using Signature = Perm;
  Partition refine(Partition p) {
    std::vector<std::pair<int, int>> entries;
    entries.reserve(n);
    Signature sig;
    sig.reserve(4 * n);
    while (true) {
      Perm ci(n);
      for (size_t k = 0; k < p.size(); ++k)
        for (int x : p[k])
          ci[x] = k;
      Partition next;
      next.reserve(n);
      bool split = false;
      for (auto &cell : p) {
        // Singleton cells cannot split; their position and contribution to ci
        // remain unchanged, so no refinement signature is needed for them.
        if (cell.size() == 1) {
          next.push_back(std::move(cell));
          continue;
        }
        std::map<Signature, Perm> buckets;
        for (int x : cell) {
          entries.clear();
          for (const auto &entry : neighbors[x])
            entries.push_back({ci[entry.first], entry.second});
          std::sort(entries.begin(), entries.end());
          sig.clear();
          for (size_t k = 0; k < entries.size();) {
            int cell_id = entries[k].first;
            sig.push_back(-cell_id);
            while (k < entries.size() && entries[k].first == cell_id) {
              int color = entries[k].second, count = 0;
              do {
                ++count;
                ++k;
              } while (k < entries.size() &&
                       entries[k] == std::make_pair(cell_id, color));
              sig.push_back(color);
              sig.push_back(count);
            }
            sig.push_back(0);
          }
          buckets.try_emplace(sig).first->second.push_back(x);
        }
        split |= buckets.size() > 1;
        for (auto &entry : buckets)
          next.push_back(std::move(entry.second));
      }
      p = std::move(next);
      if (!split)
        return p;
    }
  }
  void visit(Partition p, bool already_refined = false) {
    if (incomplete)
      return;
    if (nodes >= max_nodes ||
        std::chrono::duration<double>(Clock::now() - start).count() >=
            seconds) {
      incomplete = true;
      return;
    }
    ++nodes;
    if (!already_refined)
      p = refine(std::move(p));
    int target = -1;
    while (true) {
      target = -1;
      for (size_t k = 0; k < p.size(); ++k)
        if (p[k].size() > 1 && (target < 0 || p[k].size() < p[target].size()))
          target = k;
      if (target < 0)
        break;
      const auto &cell = p[target];
      bool twins = true;
      for (int x : cell)
        if (twin_class[x] != twin_class[cell[0]]) {
          twins = false;
          break;
        }
      if (!twins)
        break;
      // Exact equal-row twins are freely interchangeable (all transpositions
      // are already seeded). In an equitable partition, individualizing such
      // twins cannot split any other cell. This is the same first branch as
      // repeated legacy individualization, with identical final cell ordering.
      Partition next;
      for (size_t k = 0; k < p.size(); ++k) {
        if (static_cast<int>(k) == target) {
          for (int x : p[k])
            next.push_back(Perm{x});
        } else
          next.push_back(std::move(p[k]));
      }
      p = std::move(next);
    }
    if (target < 0) {
      Perm order, key;
      order.reserve(n);
      key.reserve(n + 4 * n);
      for (const auto &cell : p)
        order.push_back(cell[0]);
      for (int x : order)
        key.push_back(colors[x]);
      // Sparse certificate has exactly dense lexicographic order: an earlier
      // nonzero entry compares larger than an implicit zero, hence -position.
      // Node colors have a fixed n-entry prefix; all present edge IDs are
      // positive. Enumerate the same upper-triangle certificate through present
      // edges. Sorting each row by canonical column preserves the exact dense
      // order, including loops, without scanning every absent edge at every
      // leaf.
      Perm inverse(n);
      for (int i = 0; i < n; ++i)
        inverse[order[i]] = i;
      std::vector<std::pair<int, int>> row;
      row.reserve(n);
      for (int i = 0; i < n; ++i) {
        row.clear();
        for (const auto &entry : neighbors[order[i]]) {
          int j = inverse[entry.first];
          if (j >= i)
            row.push_back({j, entry.second});
        }
        std::sort(row.begin(), row.end());
        for (const auto &entry : row) {
          key.push_back(-(i * n + entry.first));
          key.push_back(entry.second);
        }
      }
      if (best.empty() || key < best_key) {
        best = std::move(order);
        best_key = std::move(key);
      } else if (key == best_key) {
        Perm g(n);
        for (int i = 0; i < n; ++i)
          g[best[i]] = order[i];
        accept(g);
      }
      return;
    }
    Perm ci(n);
    for (size_t k = 0; k < p.size(); ++k)
      for (int x : p[k])
        ci[x] = k;
    Perm explored, parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    auto find = [&](int x) {
      while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
      }
      return x;
    };
    size_t processed = 0;
    for (int chosen : p[target]) {
      // Generators only append during this search. Keep each node's orbit union
      // and incorporate newly discovered witnesses after returning from a
      // child.
      if (!explored.empty())
        while (processed < generators.size()) {
          const auto &g = generators[processed++];
          bool fixed = true;
          for (int i = 0; i < n; ++i)
            if (ci[i] != ci[g[i]]) {
              fixed = false;
              break;
            }
          if (fixed)
            for (int i : p[target]) {
              int x = find(i), y = find(g[i]);
              if (x != y)
                parent[y] = x;
            }
        }
      bool seen = false;
      for (int old : explored)
        if (find(old) == find(chosen)) {
          seen = true;
          break;
        }
      if (seen)
        continue;
      Partition child = p;
      Perm rest;
      for (int x : p[target])
        if (x != chosen)
          rest.push_back(x);
      child[target] = Perm{chosen};
      child.insert(child.begin() + target + 1, rest);
      visit(std::move(child));
      if (incomplete)
        return;
      explored.push_back(chosen);
    }
  }
  bool group_order(uint64_t &answer) {
    answer = 1;
    if (root_twins_only) {
      // The invariant root partition separates all non-twin vertices. Its
      // automorphism group is exactly the direct product of the twin symmetric
      // groups; no stabilizer-chain construction is needed.
      Perm counts(n, 0);
      for (int id : twin_class)
        ++counts[id];
      for (int count : counts)
        for (int k = 2; k <= count; ++k) {
          if (answer > std::numeric_limits<uint64_t>::max() / k)
            return false;
          answer *= k;
        }
      return true;
    }
    // Every generator is supported in exactly one component of this union.
    // Thus the generated group is a direct product on disjoint supports.
    Perm parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    auto find = [&](int x) {
      while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
      }
      return x;
    };
    for (const auto &g : generators) {
      int first = -1;
      for (int i = 0; i < n; ++i)
        if (g[i] != i) {
          if (first < 0)
            first = i;
          else
            parent[find(i)] = find(first);
        }
    }
    std::map<int, Perm> components;
    for (int i = 0; i < n; ++i)
      components[find(i)].push_back(i);
    Perm index(n);
    for (const auto &entry : components) {
      const auto &points = entry.second;
      int size = points.size();
      if (size == 1)
        continue;
      // Adjacent transpositions of every exact equal-row twin class were
      // seeded before search. A support component contained in one such class
      // therefore has the full symmetric group: m!, with no Schreier work.
      // Components joining different twin classes retain the exact fallback.
      bool twins = true;
      for (int x : points)
        if (twin_class[x] != twin_class[points[0]]) {
          twins = false;
          break;
        }
      if (twins) {
        for (int k = 2; k <= size; ++k) {
          if (answer > std::numeric_limits<uint64_t>::max() / k)
            return false;
          answer *= k;
        }
        continue;
      }
      for (int i = 0; i < size; ++i)
        index[points[i]] = i;
      Group restricted;
      for (const auto &g : generators) {
        bool moved = false;
        for (int x : points)
          if (g[x] != x) {
            moved = true;
            break;
          }
        if (!moved)
          continue;
        Perm small(size);
        for (int i = 0; i < size; ++i)
          small[i] = index[g[points[i]]];
        restricted.push_back(std::move(small));
      }
      auto group = std::make_shared<Group>(std::move(restricted));
      for (int base = 0; base < size && !group->empty(); ++base) {
        bool moved = false;
        for (const auto &g : *group)
          if (g[base] != base) {
            moved = true;
            break;
          }
        if (!moved)
          continue;
        Perm orbit{base};
        std::vector<char> seen(size, 0);
        seen[base] = 1;
        for (size_t k = 0; k < orbit.size(); ++k)
          for (const auto &g : *group)
            if (!seen[g[orbit[k]]]) {
              seen[g[orbit[k]]] = 1;
              orbit.push_back(g[orbit[k]]);
            }
        if (answer > std::numeric_limits<uint64_t>::max() / orbit.size())
          return false;
        answer *= orbit.size();
        bool complete = true;
        group = stabilize(group, base, size, &complete);
        if (!complete)
          return false;
      }
    }
    return true;
  }
};
static thread_local bool profile_canonical = false;
static thread_local int64_t canonical_times[8] = {};
extern "C" void synkit_canonical_profile_enable(int enabled) {
  profile_canonical = enabled != 0;
  std::fill(canonical_times, canonical_times + 8, 0);
}
extern "C" void synkit_canonical_profile_read(int64_t *out) {
  std::copy(canonical_times, canonical_times + 8, out);
}
static int canonical_undirected_impl(int n, const int *colors, const int *edges,
                                     const int *seeds, int seed_count,
                                     double seconds, int64_t max_nodes,
                                     int *order, uint64_t *group_order,
                                     int64_t *nodes, bool require_group) {
  try {
    if (n < 1 || n > 256 || seconds < 0 || max_nodes < 1)
      return -2;
    auto profile_start = profile_canonical ? Clock::now() : Clock::time_point{};
    NativeCanon canon(n, colors, edges, seconds, max_nodes);
    // Nonadjacent twin transpositions are exact and inexpensive initial
    // witnesses.
    std::map<Perm, Perm> twins;
    for (int i = 0; i < n; ++i) {
      Perm key{colors[i]};
      key.reserve(1 + 2 * canon.neighbors[i].size());
      for (const auto &entry : canon.neighbors[i]) {
        key.push_back(entry.first);
        key.push_back(entry.second);
      }
      twins.try_emplace(std::move(key)).first->second.push_back(i);
    }
    for (const auto &entry : twins)
      for (int x : entry.second)
        canon.twin_class[x] = entry.second[0];
    NativeCanon::Partition partition;
    std::map<int, Perm> buckets;
    for (int i = 0; i < n; ++i)
      buckets[colors[i]].push_back(i);
    for (const auto &entry : buckets)
      partition.push_back(entry.second);
    partition = canon.refine(std::move(partition));
    canon.root_twins_only = true;
    for (const auto &cell : partition)
      for (int x : cell)
        if (canon.twin_class[x] != canon.twin_class[cell[0]])
          canon.root_twins_only = false;
    if (!canon.root_twins_only) {
      for (int k = 0; k < seed_count; ++k)
        canon.accept(Perm(seeds + k * n, seeds + (k + 1) * n));
      for (const auto &entry : twins)
        for (size_t k = 1; k < entry.second.size(); ++k) {
          Perm g(n);
          std::iota(g.begin(), g.end(), 0);
          std::swap(g[entry.second[k - 1]], g[entry.second[k]]);
          canon.accept(g);
        }
    }
    auto profile_ready = profile_canonical ? Clock::now() : Clock::time_point{};
    canon.visit(partition, true);
    *nodes = canon.nodes;
    auto profile_searched =
        profile_canonical ? Clock::now() : Clock::time_point{};
    if (canon.incomplete)
      return 1;
    std::copy(canon.best.begin(), canon.best.end(), order);
    *group_order = 0;
    if (require_group && !canon.group_order(*group_order))
      return 2;
    if (profile_canonical) {
      int offset = require_group ? 0 : 4;
      canonical_times[offset] +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(profile_ready -
                                                               profile_start)
              .count();
      canonical_times[offset + 1] +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              profile_searched - profile_ready)
              .count();
      canonical_times[offset + 2] +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                               profile_searched)
              .count();
      ++canonical_times[offset + 3];
    }
    return 0;
  } catch (...) {
    return -1;
  }
}

// Additive export: old callers still receive a complete group-order proof.
extern "C" int synkit_canonical_undirected(int n, const int *colors,
                                           const int *edges, const int *seeds,
                                           int seed_count, double seconds,
                                           int64_t max_nodes, int *order,
                                           uint64_t *group_order,
                                           int64_t *nodes) {
  return canonical_undirected_impl(n, colors, edges, seeds, seed_count, seconds,
                                   max_nodes, order, group_order, nodes, true);
}
extern "C" int synkit_canonical_undirected_certificate(
    int n, const int *colors, const int *edges, const int *seeds,
    int seed_count, double seconds, int64_t max_nodes, int *order,
    uint64_t *group_order, int64_t *nodes) {
  return canonical_undirected_impl(n, colors, edges, seeds, seed_count, seconds,
                                   max_nodes, order, group_order, nodes, false);
}

// Controlled ITS matrices have zero diagonal. Emit upper-triangle edges in
// canonical order directly, avoiding per-edge Python relabeling and sorting.
extern "C" int synkit_its_sparse_edges(int n, const int *order,
                                       const int *edges, int *out) {
  if (n < 1 || n > 256)
    return -1;
  std::vector<char> seen(n, 0);
  for (int i = 0; i < n; ++i) {
    if (order[i] < 0 || order[i] >= n || seen[order[i]])
      return -1;
    seen[order[i]] = 1;
  }
  int count = 0;
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) {
      int color = edges[order[i] * n + order[j]];
      if (color) {
        out[3 * count] = i;
        out[3 * count + 1] = j;
        out[3 * count + 2] = color;
        ++count;
      }
    }
  return count;
}

#include <bitset>
// A bounded cache of exact ITS patterns and explicitly witnessed R-images.
// Cache misses always fall back to complete canonicalization.
struct PatternCache {
  int n, work_limit;
  Perm base_nodes, base_edges, pending;
  Group generators;
  std::vector<std::bitset<256>> supports;
  std::unordered_set<Perm, VectorHash> known;
  size_t payload_ints = 0;
  static constexpr size_t MAX_PATTERNS = 1000000, MAX_INTS = 16 * 1024 * 1024;
  PatternCache(int nn, const int *nodes, const int *edges, const int *gens,
               int count, int work)
      : n(nn), work_limit(work), base_nodes(nodes, nodes + nn),
        base_edges(edges, edges + nn * nn) {
    for (int k = 0; k < count; ++k) {
      Perm g(gens + k * n, gens + (k + 1) * n);
      std::vector<char> seen(n, 0);
      bool valid = true;
      std::bitset<256> support;
      for (int i = 0; i < n; ++i) {
        if (g[i] < 0 || g[i] >= n || seen[g[i]] ||
            base_nodes[i] != base_nodes[g[i]]) {
          valid = false;
          break;
        }
        seen[g[i]] = 1;
        if (g[i] != i)
          support.set(i);
      }
      if (valid)
        for (int i = 0; i < n && valid; ++i)
          for (int j = 0; j < n; ++j)
            if (base_edges[i * n + j] != base_edges[g[i] * n + g[j]]) {
              valid = false;
              break;
            }
      if (valid && !support.none()) {
        generators.push_back(std::move(g));
        supports.push_back(support);
      }
    }
  }
  bool lookup(const int *nodes, const int *edges) {
    pending.clear();
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j)
        if (edges[i * n + j] != base_edges[i * n + j]) {
          pending.push_back(i * n + j);
          pending.push_back(edges[i * n + j]);
        }
    for (int i = 0; i < n; ++i)
      if (nodes[i] != base_nodes[i]) {
        pending.push_back(n * n + i);
        pending.push_back(nodes[i]);
      }
    return known.find(pending) != known.end();
  }
  void remember() {
    if (pending.size() > MAX_INTS)
      return;
    if (known.size() + work_limit + 1 > MAX_PATTERNS ||
        payload_ints + (work_limit + 1) * pending.size() > MAX_INTS) {
      known.clear();
      payload_ints = 0;
    }
    std::vector<Perm> queue{pending};
    if (known.insert(pending).second)
      payload_ints += pending.size();
    int work = 0;
    int limit =
        std::min<size_t>(work_limit, (MAX_INTS - pending.size()) /
                                         std::max<size_t>(1, pending.size()));
    for (size_t q = 0; q < queue.size() && work < limit; ++q) {
      const Perm current = queue[q];
      std::bitset<256> support;
      for (size_t j = 0; j < current.size(); j += 2) {
        int pos = current[j];
        if (pos >= n * n)
          support.set(pos - n * n);
        else {
          support.set(pos / n);
          support.set(pos % n);
        }
      }
      for (size_t gi = 0; gi < generators.size() && work < limit; ++gi) {
        if ((support & supports[gi]).none())
          continue;
        ++work;
        const auto &g = generators[gi];
        std::vector<std::pair<int, int>> entries;
        entries.reserve(current.size() / 2);
        for (size_t j = 0; j < current.size(); j += 2) {
          int pos = current[j], image;
          if (pos >= n * n)
            image = n * n + g[pos - n * n];
          else {
            int a = g[pos / n], b = g[pos % n];
            image = std::min(a, b) * n + std::max(a, b);
          }
          entries.push_back({image, current[j + 1]});
        }
        std::sort(entries.begin(), entries.end());
        Perm next;
        next.reserve(current.size());
        for (const auto &entry : entries) {
          next.push_back(entry.first);
          next.push_back(entry.second);
        }
        if (known.insert(next).second) {
          payload_ints += next.size();
          queue.push_back(std::move(next));
        }
      }
    }
  }
};
extern "C" void *synkit_pattern_cache_create(int n, const int *nodes,
                                             const int *edges, const int *gens,
                                             int count, int work) {
  try {
    if (n < 1 || n > 256 || count < 0 || work < 0 || work > 1024)
      return nullptr;
    return new PatternCache(n, nodes, edges, gens, count, work);
  } catch (...) {
    return nullptr;
  }
}
extern "C" int synkit_pattern_cache_lookup(void *handle, const int *nodes,
                                           const int *edges) {
  try {
    return static_cast<PatternCache *>(handle)->lookup(nodes, edges) ? 1 : 0;
  } catch (...) {
    return -1;
  }
}
extern "C" int synkit_pattern_cache_remember(void *handle) {
  try {
    static_cast<PatternCache *>(handle)->remember();
    return 0;
  } catch (...) {
    return -1;
  }
}
extern "C" void synkit_pattern_cache_destroy(void *handle) {
  delete static_cast<PatternCache *>(handle);
}

// Inputs are precomputed with the public Python equality/isclose semantics.
// Output: n unary flags, n context flags, then changed upper-triangle pairs.
extern "C" int synkit_its_changes(int n, const int *mapping, const int *unary,
                                  const int *paired, const int *changed,
                                  int radius, int *out) {
  if (n < 1 || n > 256 || radius < 0)
    return -1;
  std::vector<int> frontier;
  for (int i = 0; i < n; ++i) {
    out[i] = unary[i * n + mapping[i]];
    out[n + i] = out[i];
  }
  int count = 0;
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j)
      if (changed[paired[i * n + j]]) {
        out[2 * n + 2 * count] = i;
        out[2 * n + 2 * count + 1] = j;
        ++count;
        out[n + i] = out[n + j] = 1;
      }
  for (int i = 0; i < n; ++i)
    if (out[n + i])
      frontier.push_back(i);
  for (int depth = 0; depth < radius && !frontier.empty(); ++depth) {
    std::vector<int> next;
    for (int i : frontier)
      for (int j = 0; j < n; ++j)
        if (paired[i * n + j] && !out[n + j]) {
          out[n + j] = 1;
          next.push_back(j);
        }
    frontier = std::move(next);
  }
  return count;
}

// Ordered resource IDs have exactly the order of (typed element, A bond, B
// bond). Each selected vertex receives its sorted external-resource multiset.
extern "C" int synkit_its_boundary(int n, const int *selected, int count,
                                   const int *paired, int palette_size,
                                   const int *resources, int *offsets,
                                   int *out) {
  if (n < 1 || n > 256 || count < 0 || count > n || palette_size < 1)
    return -1;
  std::vector<char> inside(n, 0);
  for (int k = 0; k < count; ++k) {
    int i = selected[k];
    if (i < 0 || i >= n || inside[i])
      return -1;
    inside[i] = 1;
  }
  int size = 0;
  offsets[0] = 0;
  for (int k = 0; k < count; ++k) {
    int start = size, i = selected[k];
    for (int j = 0; j < n; ++j)
      if (!inside[j] && paired[i * n + j])
        out[size++] = resources[j * palette_size + paired[i * n + j]];
    std::sort(out + start, out + size);
    offsets[k + 1] = size;
  }
  return size;
}

// Diagnostic oracle surface: exact assignment duals and each forced-edge
// optimum. Costs are bounded nonnegative integers; INF denotes a forbidden
// edge.
extern "C" int synkit_assignment_certificate(int m, const int *input,
                                             int *match_out, int *u_out,
                                             int *v_out, int *forced_out) {
  try {
    if (m < 1 || m > 64)
      return -2;
    Perm cost(input, input + m * m), match, u, v, dist, component;
    for (int value : cost)
      if (value < 0 || (value > 1000000 && value != INF))
        return -2;
    int lower = Solver::assignment(cost, m, match, u, v);
    if (lower >= INF)
      return INF;
    int dim = Solver::forced_paths(cost, m, match, u, v, dist, component);
    std::copy(match.begin(), match.end(), match_out);
    std::copy(u.begin() + 1, u.end(), u_out);
    std::copy(v.begin() + 1, v.end(), v_out);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < m; ++j) {
        int column = match[j], path = dist[component[j] * dim + component[i]];
        forced_out[i * m + column] =
            (cost[i * m + column] >= INF / 2 || path >= INF)
                ? INF
                : lower + cost[i * m + column] - u[i + 1] - v[column + 1] +
                      path;
      }
    return lower;
  } catch (...) {
    return -1;
  }
}

// Diagnostic: arbitrary stale hints must produce independently valid duals.
extern "C" int synkit_assignment_seed_certificate(int m, const int *input,
                                                  const int *initial_v,
                                                  const int *initial_match,
                                                  int *match_out, int *u_out,
                                                  int *v_out, int *forced_out) {
  try {
    if (m < 1 || m > 64 || !initial_v || !initial_match)
      return -2;
    Perm cost(input, input + m * m), match, u, v, dist, component;
    for (int value : cost)
      if (value < 0 || (value > 1000000 && value != INF))
        return -2;
    Perm seed_v(initial_v, initial_v + m),
        seed_match(initial_match, initial_match + m);
    int lower = Solver::assignment(cost, m, match, u, v, &seed_v, &seed_match);
    if (lower >= INF)
      return INF;
    int dim = Solver::forced_paths(cost, m, match, u, v, dist, component);
    std::copy(match.begin(), match.end(), match_out);
    std::copy(u.begin() + 1, u.end(), u_out);
    std::copy(v.begin() + 1, v.end(), v_out);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < m; ++j) {
        int column = match[j], path = dist[component[j] * dim + component[i]];
        forced_out[i * m + column] =
            (cost[i * m + column] >= INF / 2 || path >= INF)
                ? INF
                : lower + cost[i * m + column] - u[i + 1] - v[column + 1] +
                      path;
      }
    return lower;
  } catch (...) {
    return -1;
  }
}

extern "C" int synkit_its_pair(int n, const int *mapping, const int *a,
                               const int *b, int levels, const int *pair_ids,
                               int *out) {
  if (n < 1 || n > 256 || levels < 1)
    return -1;
  for (int i = 0; i < n; ++i)
    if (mapping[i] < 0 || mapping[i] >= n)
      return -1;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      out[i * n + j] = (i == j) ? 0
                                : pair_ids[a[i * n + j] * levels +
                                           b[mapping[i] * n + mapping[j]]];
  return 0;
}
