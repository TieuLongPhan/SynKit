#include "synkit_mapper/astar/astar.h"
#include "synkit_mapper/astar/bound.h"
#include "synkit_mapper/core/runtime.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

namespace synkit_mapper {
namespace {

struct State {
  int64_t bound;
  uint64_t serial;
  int depth;
  std::vector<int32_t> mapping;
};

struct Later {
  bool operator()(const State &a, const State &b) const {
    if (a.bound != b.bound)
      return a.bound > b.bound;
    // Deeper equal-bound states reach a first solution sooner without changing
    // the best-first bound order or discarding any tied completion.
    if (a.depth != b.depth)
      return a.depth < b.depth;
    return a.serial > b.serial;
  }
};

class Frontier {
public:
  Frontier(const Request &r, Statistics &s) : request(r), statistics(s) {}
  bool empty() const { return heap.empty(); }
  int64_t bound() const { return heap.front().bound; }
  void push(State state) {
    if (request.max_frontier >= 0 &&
        heap.size() >= static_cast<std::size_t>(request.max_frontier))
      throw Stop{SYNKIT_STOP_FRONTIER};
    // Explicit reserve prevents an allocator's unspecified growth policy from
    // overshooting the declared accounted-frontier budget before it is checked.
    std::size_t new_capacity = heap.capacity();
    if (new_capacity == heap.size())
      new_capacity = std::max<std::size_t>(1, new_capacity * 2);
    const auto vector_bytes = state.mapping.capacity() * sizeof(int32_t);
    const auto proposed =
        new_capacity * sizeof(State) + mapping_bytes + vector_bytes;
    if (request.memory_limit_bytes >= 0 &&
        proposed > static_cast<uint64_t>(request.memory_limit_bytes))
      throw Stop{SYNKIT_STOP_MEMORY};
    if (new_capacity != heap.capacity())
      heap.reserve(new_capacity);
    mapping_bytes += vector_bytes;
    heap.push_back(std::move(state));
    std::push_heap(heap.begin(), heap.end(), Later{});
    statistics.peak_frontier =
        std::max<int64_t>(statistics.peak_frontier, heap.size());
    statistics.peak_frontier_bytes =
        std::max<int64_t>(statistics.peak_frontier_bytes,
                          heap.capacity() * sizeof(State) + mapping_bytes);
  }
  State pop() {
    std::pop_heap(heap.begin(), heap.end(), Later{});
    State result = std::move(heap.back());
    heap.pop_back();
    mapping_bytes -= result.mapping.capacity() * sizeof(int32_t);
    return result;
  }

private:
  const Request &request;
  Statistics &statistics;
  std::vector<State> heap;
  std::size_t mapping_bytes = 0;
};

std::vector<int> source_order(const Problem &p) {
  std::map<int32_t, int> count;
  for (int i = 0; i < p.n; ++i)
    ++count[p.types_p[i]];
  std::vector<int64_t> mass(p.n);
  std::vector<int> order(p.n);
  std::iota(order.begin(), order.end(), 0);
  for (int i = 0; i < p.n; ++i)
    for (int j = 0; j < p.n; ++j)
      mass[i] += p.a2[i * p.n + j];
  std::stable_sort(order.begin(), order.end(), [&](int i, int k) {
    if (count[p.types_r[i]] != count[p.types_r[k]])
      return count[p.types_r[i]] < count[p.types_r[k]];
    if (mass[i] != mass[k])
      return mass[i] > mass[k];
    return i < k;
  });
  return order;
}

} // namespace

int solve_astar(const Problem &p, const Request &r, Callback callback,
                void *userdata, Statistics &s) {
  initialize(s, r);
  // Raw enumeration cannot interpret a cap measured in exact ITS classes.
  // The shared class pipeline applies its own cap through callback control.
  if (r.classify || r.max_classes != -1)
    return unsupported(s);
  Budget budget(r);
  Frontier frontier(r, s);
  uint64_t serial = 0;
  const bool numeric =
      r.query == SYNKIT_EXACT_DISTANCE || r.query == SYNKIT_AT_MOST_DISTANCE;
  int64_t threshold = numeric ? r.target2 : std::numeric_limits<int64_t>::max();
  double proof_at = -1;
  bool model_phase = true;
  const std::function<void()> poll = [&]() { budget.check(); };
  try {
    budget.check();
    const auto order = source_order(p);
    std::vector<int32_t> initial(p.n, -1);
    const int64_t initial_bound = astar::partial_lower_bound2(
        p.n, p.a2, p.b2, p.types_r, p.types_p, initial, poll);
    s.lower_bound2 = initial_bound;
    s.model_seconds = budget.seconds();
    model_phase = false;
    if (initial_bound <= threshold)
      frontier.push({initial_bound, serial++, 0, std::move(initial)});
    while (!frontier.empty()) {
      budget.check();
      if (frontier.bound() > threshold)
        break;
      if (r.max_states >= 0 && s.nodes >= r.max_states)
        throw Stop{SYNKIT_STOP_STATES};
      State state = frontier.pop();
      if (!s.minimum_proven)
        s.lower_bound2 = state.bound;
      ++s.nodes;
      if (state.depth == p.n) {
        ++s.leaves;
        // A full state's bound is its exact edge cost. Its first removal from
        // a best-bound frontier proves the global minimum, including when the
        // caller requested a higher numeric shell.
        const int64_t cost = state.bound;
        if (!s.minimum_proven) {
          s.minimum_proven = 1;
          s.minimum2 = s.incumbent2 = s.lower_bound2 = cost;
          proof_at = budget.seconds();
          if (!numeric)
            threshold = cost;
        }
        if (r.query == SYNKIT_MINIMUM_PROOF)
          break;
        if (r.query == SYNKIT_EXACT_DISTANCE && cost != r.target2)
          continue;
        ++s.candidate_count;
        if (callback) {
          const auto before = Clock::now();
          const int stop = callback(state.mapping.data(), cost, userdata);
          s.callback_seconds += elapsed(before);
          if (stop)
            throw Stop{SYNKIT_STOP_CALLBACK};
        }
        budget.check();
        if (r.max_candidates >= 0 && s.candidate_count >= r.max_candidates)
          throw Stop{SYNKIT_STOP_CANDIDATES};
        continue;
      }
      std::vector<bool> used(p.n, false);
      for (int j : state.mapping)
        if (j >= 0)
          used[j] = true;
      const int i = order[state.depth];
      for (int j = 0; j < p.n; ++j) {
        budget.check();
        if (used[j] || p.types_r[i] != p.types_p[j])
          continue;
        auto mapping = state.mapping;
        mapping[i] = j;
        int64_t bound = astar::partial_lower_bound2(p.n, p.a2, p.b2, p.types_r,
                                                    p.types_p, mapping, poll);
        // Pathmax preserves admissibility even if a future stronger heuristic
        // is inconsistent. It cannot change a full state's true cost.
        bound = std::max(bound, state.bound);
        if (bound <= threshold)
          frontier.push({bound, serial++, state.depth + 1, std::move(mapping)});
      }
    }
    s.search_complete = 1;
    if (r.query != SYNKIT_MINIMUM_PROOF)
      s.labeled_count = s.candidate_count;
  } catch (const Stop &stop) {
    if (model_phase)
      s.model_seconds = budget.seconds();
    stopped(s, stop.reason);
  }
  s.elapsed_seconds = budget.seconds();
  s.optimization_seconds = proof_at < 0 ? s.elapsed_seconds - s.model_seconds
                                        : proof_at - s.model_seconds;
  s.enumeration_seconds = proof_at < 0 ? 0 : s.elapsed_seconds - proof_at;
  return 0;
}

} // namespace synkit_mapper

extern "C" int synkit_astar_partial_bound2(const SynkitMapperProblem *p,
                                           const int32_t *partial,
                                           int64_t *bound2) {
  if (!p || !partial || !bound2)
    return -1;
  try {
    SynkitMapperRequest r{};
    r.abi_version = 1;
    r.backend = SYNKIT_ASTAR;
    r.workers = 1;
    r.seconds = -1;
    r.max_states = r.max_candidates = r.max_classes = r.max_frontier = -1;
    r.memory_limit_bytes = -1;
    synkit_mapper::validate(*p, r);
    *bound2 = synkit_mapper::astar::partial_lower_bound2(
        p->n, p->a2, p->b2, p->types_r, p->types_p,
        std::vector<int32_t>(partial, partial + p->n));
    return 0;
  } catch (...) {
    return -1;
  }
}
