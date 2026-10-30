#include "synkit_mapper/sepdp/sepdp.h"
#include "synkit_mapper/core/runtime.h"

#include <deque>
#include <functional>
#include <limits>
#include <numeric>
#include <unordered_map>

namespace {
thread_local SynkitSepDPProfile profile{};
struct Arc {
  std::size_t previous;
  int32_t target;
};
struct State {
  uint64_t used;
  int64_t overlap;
  int layer;
  std::vector<int32_t> boundary;
  std::vector<Arc> parents;
};
struct Key {
  uint64_t used;
  int64_t overlap;
  std::vector<int32_t> boundary;
  bool operator==(const Key &other) const {
    return used == other.used && overlap == other.overlap &&
           boundary == other.boundary;
  }
};
struct KeyHash {
  std::size_t operator()(const Key &key) const {
    std::size_t result = std::hash<uint64_t>{}(key.used);
    result ^= std::hash<int64_t>{}(key.overlap) + 0x9e3779b9 + (result << 6) +
              (result >> 2);
    for (auto image : key.boundary)
      result ^= std::hash<int32_t>{}(image) + 0x9e3779b9 + (result << 6) +
                (result >> 2);
    return result;
  }
};
} // namespace

namespace synkit_mapper {
int solve_sepdp(const Problem &p, const Request &r, Callback callback,
                void *userdata, Statistics &s) {
  initialize(s, r);
  profile = {};
  // The algorithm supports arbitrary weighted endpoint graphs. This finite
  // implementation domain is a target-set representation limit, not a graph
  // separator or chemical assumption silently imposed on larger inputs.
  if (p.n > 63 || r.symmetry || r.classify || r.max_classes != -1)
    return unsupported(s);
  Budget budget(r);
  std::deque<State> arena;
  std::vector<std::size_t> current;
  int64_t accounted = 0;
  double proved_at = -1;
  bool model_phase = true;
  const bool minimum =
      r.query == SYNKIT_MINIMUM_CLASSES || r.query == SYNKIT_MINIMUM_PROOF;
  auto account = [&](int64_t bytes) {
    if (bytes < 0 || accounted > std::numeric_limits<int64_t>::max() - bytes)
      throw Stop{SYNKIT_STOP_MEMORY};
    if (r.memory_limit_bytes >= 0 && accounted + bytes > r.memory_limit_bytes)
      throw Stop{SYNKIT_STOP_MEMORY};
    accounted += bytes;
    profile.peak_accounted_table_bytes =
        std::max(profile.peak_accounted_table_bytes, accounted);
    s.peak_frontier_bytes = profile.peak_accounted_table_bytes;
  };
  auto add_state = [&](uint64_t used, int64_t overlap, int layer,
                       const std::vector<int32_t> &boundary) {
    if (s.nodes == std::numeric_limits<int64_t>::max() ||
        (r.max_states >= 0 && s.nodes >= r.max_states))
      throw Stop{SYNKIT_STOP_STATES};
    // Account retained table payload conservatively, including a key for every
    // state even after its hash-table layer is released. Allocator/block/bucket
    // overhead and transient scratch are explicitly outside this payload cap.
    account(sizeof(State) + sizeof(Key) +
            2 * boundary.size() * sizeof(int32_t));
    arena.push_back({used, overlap, layer, boundary, {}});
    ++s.nodes;
    ++profile.states;
    return arena.size() - 1;
  };
  auto parent = [&](State &state, std::size_t previous, int32_t target) {
    if (state.parents.size() == state.parents.capacity()) {
      auto capacity = std::max<std::size_t>(1, state.parents.capacity() * 2);
      account((capacity - state.parents.capacity()) * sizeof(Arc));
      state.parents.reserve(capacity);
    }
    state.parents.push_back({previous, target});
    ++profile.predecessor_arcs;
  };
  try {
    budget.check();
    int64_t constant = 0;
    for (int i = 0; i < p.n; ++i)
      for (int k = 0; k < i; ++k)
        constant += p.a2[i * p.n + k] + p.b2[i * p.n + k];
    std::map<int32_t, int> count;
    for (int j = 0; j < p.n; ++j)
      ++count[p.types_p[j]];
    std::vector<int> order(p.n);
    std::iota(order.begin(), order.end(), 0);
    std::vector<int64_t> mass(p.n);
    for (int i = 0; i < p.n; ++i)
      for (int k = 0; k < p.n; ++k)
        mass[i] += p.a2[i * p.n + k];
    std::stable_sort(order.begin(), order.end(), [&](int i, int k) {
      if (count[p.types_r[i]] != count[p.types_r[k]])
        return count[p.types_r[i]] < count[p.types_r[k]];
      return mass[i] > mass[k];
    });
    std::vector<int> position(p.n);
    for (int layer = 0; layer < p.n; ++layer)
      position[order[layer]] = layer;
    std::vector<std::vector<int>> boundaries(p.n + 1);
    for (int layer = 1; layer <= p.n; ++layer) {
      budget.check();
      for (int earlier = 0; earlier < layer; ++earlier) {
        int i = order[earlier];
        for (int k = 0; k < p.n; ++k)
          if (position[k] >= layer && p.a2[i * p.n + k]) {
            boundaries[layer].push_back(i);
            break;
          }
      }
      profile.separator_width =
          std::max<int64_t>(profile.separator_width, boundaries[layer].size());
    }
    s.model_seconds = budget.seconds();
    model_phase = false;
    s.lower_bound2 = 0;
    if (!minimum && r.query == SYNKIT_EXACT_DISTANCE &&
        (constant - r.target2) % 2) {
      s.search_complete = 1;
      s.labeled_count = 0;
    } else {
      current.push_back(add_state(0, 0, 0, {}));
      s.peak_frontier = 1;
      for (int layer = 0; layer < p.n; ++layer) {
        std::unordered_map<Key, std::size_t, KeyHash> unique;
        std::vector<std::size_t> next;
        std::vector<int> old_position(p.n, -1);
        for (std::size_t k = 0; k < boundaries[layer].size(); ++k)
          old_position[boundaries[layer][k]] = static_cast<int>(k);
        const int i = order[layer];
        for (auto id : current) {
          const auto &state = arena[id];
          for (int j = 0; j < p.n; ++j) {
            budget.check();
            if ((state.used & (uint64_t{1} << j)) ||
                p.types_r[i] != p.types_p[j])
              continue;
            ++profile.transitions;
            int64_t overlap = state.overlap;
            for (std::size_t k = 0; k < boundaries[layer].size(); ++k)
              overlap += std::min(p.a2[i * p.n + boundaries[layer][k]],
                                  p.b2[j * p.n + state.boundary[k]]);
            std::vector<int32_t> boundary;
            boundary.reserve(boundaries[layer + 1].size());
            for (int source : boundaries[layer + 1])
              boundary.push_back(
                  source == i ? j : state.boundary[old_position[source]]);
            Key key{state.used | (uint64_t{1} << j), overlap, boundary};
            auto found = unique.find(key);
            std::size_t child;
            if (found == unique.end()) {
              if (r.max_frontier >= 0 &&
                  next.size() >= static_cast<std::size_t>(r.max_frontier))
                throw Stop{SYNKIT_STOP_FRONTIER};
              child = add_state(key.used, overlap, layer + 1, boundary);
              unique.emplace(std::move(key), child);
              next.push_back(child);
              s.peak_frontier = std::max<int64_t>(s.peak_frontier, next.size());
            } else {
              child = found->second;
            }
            parent(arena[child], id, j);
            if (layer + 1 == p.n) {
              const auto cost = constant - 2 * overlap;
              s.incumbent2 =
                  s.incumbent2 < 0 ? cost : std::min(s.incumbent2, cost);
            }
          }
        }
        current = std::move(next);
      }
      // Only exhausting every DP layer proves the optimum. A last-layer
      // incumbent reached just before a resource stop remains unproved.
      s.minimum2 = s.incumbent2;
      s.minimum_proven = 1;
      s.lower_bound2 = s.minimum2;
      proved_at = budget.seconds();
      if (r.query != SYNKIT_MINIMUM_PROOF) {
        std::vector<int32_t> mapping(p.n, -1);
        int64_t selected_cost = -1;
        std::function<void(std::size_t)> reconstruct = [&](std::size_t id) {
          budget.check();
          const auto &state = arena[id];
          if (state.layer == 0) {
            ++s.leaves;
            ++s.candidate_count;
            if (callback) {
              auto before = Clock::now();
              int stop = callback(mapping.data(), selected_cost, userdata);
              s.callback_seconds += elapsed(before);
              if (stop)
                throw Stop{SYNKIT_STOP_CALLBACK};
            }
            budget.check();
            if (r.max_candidates >= 0 && s.candidate_count >= r.max_candidates)
              throw Stop{SYNKIT_STOP_CANDIDATES};
            return;
          }
          for (const auto &arc : state.parents) {
            mapping[order[state.layer - 1]] = arc.target;
            reconstruct(arc.previous);
          }
        };
        for (auto id : current) {
          selected_cost = constant - 2 * arena[id].overlap;
          bool selected = minimum ? selected_cost == s.minimum2
                          : r.query == SYNKIT_EXACT_DISTANCE
                              ? selected_cost == r.target2
                              : selected_cost <= r.target2;
          if (selected)
            reconstruct(id);
        }
        s.labeled_count = s.candidate_count;
      }
      s.search_complete = 1;
    }
  } catch (const Stop &stop) {
    if (model_phase)
      s.model_seconds = budget.seconds();
    stopped(s, stop.reason);
  }
  s.elapsed_seconds = budget.seconds();
  s.optimization_seconds =
      (proved_at < 0 ? s.elapsed_seconds : proved_at) - s.model_seconds;
  s.enumeration_seconds = proved_at < 0 ? 0 : s.elapsed_seconds - proved_at;
  return 0;
}
} // namespace synkit_mapper

extern "C" void synkit_mapper_sepdp_profile(SynkitSepDPProfile *output) {
  if (output)
    *output = profile;
}
