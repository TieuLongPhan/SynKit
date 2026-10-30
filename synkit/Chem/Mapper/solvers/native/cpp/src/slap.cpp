#include "synkit_mapper/slap.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Label = int64_t;
using Signature = std::vector<int64_t>;
constexpr int64_t INF = std::numeric_limits<int64_t>::max() / 8;
thread_local std::string last_error;

struct Stop {
  int reason;
};

class Budget {
public:
  Budget(Clock::time_point start, double seconds)
      : start_(start), seconds_(seconds) {}
  bool expired() const { return seconds_ >= 0 && elapsed() >= seconds_; }
  double elapsed() const {
    return std::chrono::duration<double>(Clock::now() - start_).count();
  }
  void check() const {
    if (expired())
      throw Stop{1};
  }

private:
  Clock::time_point start_;
  double seconds_;
};

struct Block {
  Label label;
  std::vector<int> atoms;
};

struct Side {
  std::vector<Label> labels;
  // Insertion order matters to the sequential heuristic and is kept explicit.
  std::vector<Block> blocks;
  std::shared_ptr<const std::vector<Label>> initial_wl;
};

using Neighbors = std::map<Label, std::vector<int64_t>>;
struct NeighborhoodClass {
  Label token;
  std::vector<int> atoms;
  Neighbors neighbors;
};

struct Solution {
  int64_t value;
  std::vector<int> fingerprint;
  std::vector<int> groups_r, groups_p;
  bool proper = true;
};

struct BlockData {
  std::vector<NeighborhoodClass> classes_r, classes_p;
  std::vector<Solution> solutions;
};

struct State {
  Side r, p;
  std::map<Label, std::shared_ptr<const BlockData>> irreducible;
};

struct Record {
  State state;
  int64_t value;
  std::map<Label, std::shared_ptr<const BlockData>> lap_solutions;
};

class Slap {
public:
  Slap(int n, const int64_t *a, const int64_t *b, const int32_t *lr,
       const int32_t *lp, const int32_t *stop, int fingerprints, int64_t states,
       int64_t mappings, Budget &budget, SlapStatistics &statistics)
      : n_(n), a_(a), b_(b), original_r_(lr, lr + n), original_p_(lp, lp + n),
        stop_(stop, stop + n), fingerprint_cap_(fingerprints),
        state_cap_(states), mapping_cap_(mappings), budget_(budget),
        stats_(statistics) {
    std::map<int32_t, Label> initial;
    auto labels = [&](const int32_t *source) {
      std::vector<Label> values;
      for (int i = 0; i < n_; ++i) {
        auto found = initial.find(source[i]);
        if (found == initial.end())
          found = initial.emplace(source[i], fresh()).first;
        values.push_back(found->second);
      }
      return values;
    };
    initial_.r = side(labels(lr));
    initial_.p = side(labels(lp));
  }

  void run() {
    std::vector<State> pending;
    pending.push_back(initial_);
    while (!pending.empty()) {
      budget_.check();
      std::vector<State> stack = std::move(pending);
      pending.clear();
      std::vector<Record> results;
      int64_t phase_minimum = INF;
      while (!stack.empty()) {
        budget_.check();
        if (state_cap_ >= 0 && stats_.states >= state_cap_)
          throw Stop{2};
        State current = std::move(stack.back());
        stack.pop_back();
        ++stats_.states;
        solve_path(std::move(current), stack, results, phase_minimum);
      }
      // The sequential heuristic deliberately retains only the best proxy
      // partition states reached in this round, matching the Python structure.
      results.erase(std::remove_if(results.begin(), results.end(),
                                   [&](const Record &r) {
                                     return r.value != phase_minimum;
                                   }),
                    results.end());
      while (!results.empty()) {
        budget_.check();
        Record record = std::move(results.back());
        results.pop_back();
        const auto order = ordered_labels(record.state.r);
        Label label = -1;
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
          if (block(record.state.r, *it).atoms.size() > 1) {
            label = *it;
            break;
          }
        }
        if (label < 0) {
          remember(record);
          continue;
        }
        split(record, label, pending);
      }
    }
    if (completed_.empty())
      throw std::runtime_error("SLAP produced no full bijection");
    stats_.completed = 1;
  }

  void emit(SlapMappingCallback callback, void *userdata) {
    for (const auto &entry : completed_) {
      if (budget_.expired()) {
        stats_.completed = 0;
        if (!stats_.reason)
          stats_.reason = 1;
        break;
      }
      const auto started = Clock::now();
      int stop =
          callback ? callback(entry.first.data(), entry.second, userdata) : 0;
      stats_.callback_seconds +=
          std::chrono::duration<double>(Clock::now() - started).count();
      ++stats_.candidates;
      if (stop) {
        stats_.completed = 0;
        stats_.reason = 4;
        break;
      }
    }
  }

private:
  int n_;
  const int64_t *a_, *b_;
  std::vector<int32_t> original_r_, original_p_, stop_;
  int fingerprint_cap_;
  int64_t state_cap_, mapping_cap_;
  Budget &budget_;
  SlapStatistics &stats_;
  Label next_label_ = 0;
  std::map<Signature, Label> interned_;
  State initial_;
  std::set<std::vector<int>> visited_;
  std::map<std::vector<int32_t>, int64_t> completed_;

  Label fresh() { return next_label_++; }

  Label intern(Signature signature) {
    const auto found = interned_.find(signature);
    if (found != interned_.end())
      return found->second;
    const Label label = fresh();
    interned_.emplace(std::move(signature), label);
    return label;
  }

  static Side side(std::vector<Label> labels) {
    Side result;
    result.labels = std::move(labels);
    std::map<Label, size_t> positions;
    for (size_t i = 0; i < result.labels.size(); ++i) {
      const Label label = result.labels[i];
      auto found = positions.find(label);
      if (found == positions.end()) {
        positions.emplace(label, result.blocks.size());
        result.blocks.push_back({label, {static_cast<int>(i)}});
      } else {
        result.blocks[found->second].atoms.push_back(static_cast<int>(i));
      }
    }
    return result;
  }

  static const Block &block(const Side &value, Label label) {
    const auto found =
        std::find_if(value.blocks.begin(), value.blocks.end(),
                     [&](const Block &b) { return b.label == label; });
    if (found == value.blocks.end())
      throw std::runtime_error("Unknown SLAP label block");
    return *found;
  }

  static std::vector<Label> ordered_labels(const Side &value) {
    auto blocks = value.blocks;
    std::stable_sort(blocks.begin(), blocks.end(),
                     [](const Block &x, const Block &y) {
                       return x.atoms.size() > y.atoms.size();
                     });
    std::vector<Label> result;
    for (const auto &b : blocks)
      result.push_back(b.label);
    return result;
  }

  static std::vector<int> state_key(const State &state) {
    std::map<Label, int> colors;
    std::vector<int> result;
    for (const Side *value : {&state.r, &state.p}) {
      for (Label label : value->labels) {
        auto found = colors.find(label);
        if (found == colors.end())
          found = colors.emplace(label, static_cast<int>(colors.size())).first;
        result.push_back(found->second);
      }
    }
    return result;
  }

  std::vector<NeighborhoodClass> classes(const Side &value, Label label,
                                         const int64_t *matrix) {
    std::vector<NeighborhoodClass> result;
    std::map<Label, size_t> positions;
    for (int atom : block(value, label).atoms) {
      budget_.check();
      Neighbors neighbors;
      for (int j = 0; j < n_; ++j)
        if (matrix[atom * n_ + j])
          neighbors[value.labels[j]].push_back(matrix[atom * n_ + j]);
      Signature signature{label, static_cast<int64_t>(neighbors.size())};
      for (auto &entry : neighbors) {
        std::sort(entry.second.begin(), entry.second.end(),
                  std::greater<int64_t>());
        signature.push_back(entry.first);
        signature.push_back(static_cast<int64_t>(entry.second.size()));
        signature.insert(signature.end(), entry.second.begin(),
                         entry.second.end());
      }
      const Label token = intern(std::move(signature));
      const auto found = positions.find(token);
      if (found == positions.end()) {
        positions.emplace(token, result.size());
        result.push_back({token, {atom}, std::move(neighbors)});
      } else {
        result[found->second].atoms.push_back(atom);
      }
    }
    return result;
  }

  static int64_t weight_cost(const std::vector<int64_t> &a,
                             const std::vector<int64_t> &b) {
    int64_t cost = 0;
    const size_t common = std::min(a.size(), b.size());
    for (size_t i = 0; i < common; ++i)
      cost += std::abs(a[i] - b[i]);
    for (size_t i = common; i < a.size(); ++i)
      cost += a[i];
    for (size_t i = common; i < b.size(); ++i)
      cost += b[i];
    return cost;
  }

  static int64_t neighborhood_cost(const Neighbors &a, const Neighbors &b) {
    static const std::vector<int64_t> empty;
    int64_t cost = 0;
    for (const auto &entry : a) {
      const auto found = b.find(entry.first);
      cost +=
          weight_cost(entry.second, found == b.end() ? empty : found->second);
    }
    for (const auto &entry : b)
      if (a.find(entry.first) == a.end())
        cost += weight_cost(empty, entry.second);
    return cost;
  }

  std::vector<int> lap(const std::vector<int64_t> &cost, int n) {
    budget_.check();
    ++stats_.lap_calls;
    std::vector<int64_t> u(n + 1), v(n + 1);
    std::vector<int> p(n + 1), way(n + 1);
    for (int i = 1; i <= n; ++i) {
      budget_.check();
      p[0] = i;
      int j0 = 0;
      std::vector<int64_t> minimum(n + 1, INF);
      std::vector<bool> used(n + 1, false);
      do {
        budget_.check();
        used[j0] = true;
        const int i0 = p[j0];
        int64_t delta = INF;
        int j1 = 0;
        for (int j = 1; j <= n; ++j) {
          if (used[j])
            continue;
          const int64_t current = cost[(i0 - 1) * n + j - 1] - u[i0] - v[j];
          if (current < minimum[j]) {
            minimum[j] = current;
            way[j] = j0;
          }
          if (minimum[j] < delta) {
            delta = minimum[j];
            j1 = j;
          }
        }
        if (j1 == 0)
          throw std::runtime_error("Infeasible internal LAP");
        for (int j = 0; j <= n; ++j) {
          if (used[j]) {
            u[p[j]] += delta;
            v[j] -= delta;
          } else
            minimum[j] -= delta;
        }
        j0 = j1;
      } while (p[j0] != 0);
      do {
        const int previous = way[j0];
        p[j0] = p[previous];
        j0 = previous;
      } while (j0);
    }
    std::vector<int> assignment(n);
    for (int j = 1; j <= n; ++j)
      assignment[p[j] - 1] = j - 1;
    return assignment;
  }

  static std::pair<std::vector<int>, std::vector<int>>
  components(const std::vector<int> &fingerprint, int nr, int np) {
    std::vector<int> left(nr, -1), right(np, -1);
    int group = 0;
    for (int start = 0; start < nr; ++start) {
      if (left[start] >= 0)
        continue;
      std::vector<int> pending{start};
      left[start] = group;
      while (!pending.empty()) {
        const int i = pending.back();
        pending.pop_back();
        for (int j = 0; j < np; ++j) {
          if (!fingerprint[i * np + j] || right[j] >= 0)
            continue;
          right[j] = group;
          for (int k = 0; k < nr; ++k)
            if (fingerprint[k * np + j] && left[k] < 0) {
              left[k] = group;
              pending.push_back(k);
            }
        }
      }
      ++group;
    }
    for (int &value : right)
      if (value < 0)
        value = group++;
    return {left, right};
  }

  static bool one_group(const std::vector<int> &groups) {
    return std::all_of(groups.begin(), groups.end(),
                       [&](int group) { return group == groups.front(); });
  }

  static void mark_proper(std::vector<Solution> &solutions) {
    for (size_t i = 0; i < solutions.size(); ++i) {
      for (size_t j = 0; j < solutions.size(); ++j) {
        if (i == j || !solutions[j].proper)
          continue;
        std::map<int, std::set<int>> image;
        for (size_t k = 0; k < solutions[i].groups_r.size(); ++k)
          image[solutions[i].groups_r[k]].insert(solutions[j].groups_r[k]);
        for (size_t k = 0; k < solutions[i].groups_p.size(); ++k)
          image[solutions[i].groups_p[k]].insert(solutions[j].groups_p[k]);
        if (std::all_of(image.begin(), image.end(), [](const auto &entry) {
              return entry.second.size() == 1;
            })) {
          solutions[i].proper = false;
          break;
        }
      }
    }
  }

  std::shared_ptr<const BlockData> build_block(const State &state,
                                               Label label) {
    auto data = std::make_shared<BlockData>();
    data->classes_r = classes(state.r, label, a_);
    data->classes_p = classes(state.p, label, b_);
    const int size = static_cast<int>(block(state.r, label).atoms.size());
    if (size != static_cast<int>(block(state.p, label).atoms.size()))
      throw std::runtime_error("Unbalanced refined SLAP block");
    std::vector<int> row_class, column_class;
    for (size_t i = 0; i < data->classes_r.size(); ++i)
      row_class.insert(row_class.end(), data->classes_r[i].atoms.size(),
                       static_cast<int>(i));
    for (size_t j = 0; j < data->classes_p.size(); ++j)
      column_class.insert(column_class.end(), data->classes_p[j].atoms.size(),
                          static_cast<int>(j));
    std::vector<int64_t> cost(size * size);
    for (int i = 0; i < size; ++i) {
      budget_.check();
      for (int j = 0; j < size; ++j)
        cost[i * size + j] =
            neighborhood_cost(data->classes_r[row_class[i]].neighbors,
                              data->classes_p[column_class[j]].neighbors);
    }
    const int nr = static_cast<int>(data->classes_r.size());
    const int np = static_cast<int>(data->classes_p.size());
    std::vector<int64_t> perturbed(cost);
    for (auto &value : perturbed)
      value *= 100;
    std::set<std::vector<int>> seen;
    int64_t best = INF;
    for (int iteration = 0; iteration < fingerprint_cap_; ++iteration) {
      budget_.check();
      const auto assignment = lap(perturbed, size);
      int64_t value = 0;
      for (int i = 0; i < size; ++i)
        value += cost[i * size + assignment[i]];
      if (!iteration)
        best = value;
      if (value != best)
        break;
      Solution solution;
      solution.value = value;
      if (nr == 1 || np == 1) {
        solution.groups_r.assign(nr, 0);
        solution.groups_p.assign(np, 0);
        data->solutions.push_back(std::move(solution));
        ++stats_.fingerprints;
        break;
      }
      solution.fingerprint.assign(nr * np, 0);
      for (int i = 0; i < size; ++i)
        ++solution.fingerprint[row_class[i] * np + column_class[assignment[i]]];
      if (!seen.insert(solution.fingerprint).second)
        break;
      auto groups = components(solution.fingerprint, nr, np);
      solution.groups_r = std::move(groups.first);
      solution.groups_p = std::move(groups.second);
      for (int i = 0; i < size; ++i) {
        budget_.check();
        for (int j = 0; j < size; ++j)
          if (solution.fingerprint[row_class[i] * np + column_class[j]])
            perturbed[i * size + j] += 2;
      }
      data->solutions.push_back(std::move(solution));
      ++stats_.fingerprints;
    }
    if (data->solutions.empty())
      throw std::runtime_error("SLAP found no block assignment");
    mark_proper(data->solutions);
    return data;
  }

  static void replace_block(
      Side &side, Label old_label,
      const std::vector<std::pair<Label, std::vector<int>>> &replacements) {
    side.blocks.erase(
        std::remove_if(side.blocks.begin(), side.blocks.end(),
                       [&](const Block &b) { return b.label == old_label; }),
        side.blocks.end());
    for (const auto &entry : replacements) {
      auto atoms = entry.second;
      std::sort(atoms.begin(), atoms.end());
      side.blocks.push_back({entry.first, atoms});
      for (int atom : atoms)
        side.labels[atom] = entry.first;
    }
  }

  void clear_cache(State &state, const std::vector<Label> &parents) {
    std::map<Label, std::set<Label>> neighbors;
    for (Label parent : parents) {
      for (int side_index = 0; side_index < 2; ++side_index) {
        const Side &value = side_index ? state.p : state.r;
        const auto *matrix = side_index ? b_ : a_;
        const auto found =
            std::find_if(value.blocks.begin(), value.blocks.end(),
                         [&](const Block &b) { return b.label == parent; });
        if (found == value.blocks.end())
          continue;
        for (int atom : found->atoms)
          for (int j = 0; j < n_; ++j)
            if (matrix[atom * n_ + j])
              neighbors[parent].insert(value.labels[j]);
      }
    }
    std::set<Label> dirty(parents.begin(), parents.end());
    for (auto left = neighbors.begin(); left != neighbors.end(); ++left)
      for (auto right = neighbors.begin(); right != left; ++right)
        std::set_intersection(left->second.begin(), left->second.end(),
                              right->second.begin(), right->second.end(),
                              std::inserter(dirty, dirty.end()));
    for (Label label : dirty)
      state.irreducible.erase(label);
  }

  State refine(const State &state, Label label, const Solution &solution,
               const BlockData &data) {
    State result = state;
    std::map<int, Label> group_labels;
    for (size_t j = 0; j < data.classes_p.size(); ++j)
      group_labels[solution.groups_p[j]] = data.classes_p[j].token;
    for (int group : solution.groups_r)
      if (group_labels.find(group) == group_labels.end())
        group_labels[group] = fresh();
    auto replacements = [&](const std::vector<NeighborhoodClass> &classes,
                            const std::vector<int> &groups) {
      std::vector<std::pair<Label, std::vector<int>>> merged;
      std::map<Label, size_t> positions;
      for (size_t i = 0; i < classes.size(); ++i) {
        const Label next = group_labels.at(groups[i]);
        auto found = positions.find(next);
        if (found == positions.end()) {
          positions.emplace(next, merged.size());
          merged.push_back({next, classes[i].atoms});
        } else {
          auto &atoms = merged[found->second].second;
          atoms.insert(atoms.end(), classes[i].atoms.begin(),
                       classes[i].atoms.end());
        }
      }
      return merged;
    };
    replace_block(result.r, label,
                  replacements(data.classes_r, solution.groups_r));
    replace_block(result.p, label,
                  replacements(data.classes_p, solution.groups_p));
    std::vector<Label> parents;
    for (const auto &entry : group_labels)
      parents.push_back(entry.second);
    clear_cache(result, parents);
    return result;
  }

  void solve_path(State state, std::vector<State> &stack,
                  std::vector<Record> &results, int64_t &minimum) {
    int64_t total = 0;
    std::map<Label, std::shared_ptr<const BlockData>> lap_solutions;
    for (Label label : ordered_labels(state.r)) {
      budget_.check();
      const auto cached = state.irreducible.find(label);
      auto data = cached == state.irreducible.end() ? build_block(state, label)
                                                    : cached->second;
      total += data->solutions.front().value;
      if (total > minimum)
        return;
      const int atom = block(state.r, label).atoms.front();
      const auto proper_count =
          std::count_if(data->solutions.begin(), data->solutions.end(),
                        [](const Solution &s) { return s.proper; });
      bool irreducible = cached != state.irreducible.end() ||
                         (stop_[atom] && proper_count > 1);
      if (!irreducible) {
        for (const auto &solution : data->solutions) {
          budget_.check();
          if (!solution.proper)
            continue;
          if (one_group(solution.groups_r)) {
            irreducible = true;
            break;
          }
          auto child = refine(state, label, solution, *data);
          if (visited_.insert(state_key(child)).second)
            stack.push_back(std::move(child));
        }
      }
      lap_solutions[label] = data;
      if (irreducible)
        state.irreducible[label] = data;
      else
        return;
    }
    results.push_back({std::move(state), total, std::move(lap_solutions)});
    minimum = std::min(minimum, total);
  }

  std::shared_ptr<const std::vector<Label>> wl_colors(const Side &side) {
    if (side.initial_wl)
      return side.initial_wl;
    std::vector<Label> colors = side.labels;
    for (int round = 0; round < n_; ++round) {
      budget_.check();
      std::vector<Label> next;
      for (int i = 0; i < n_; ++i) {
        std::vector<std::pair<Label, int64_t>> incident;
        for (int j = 0; j < n_; ++j)
          if (b_[i * n_ + j])
            incident.emplace_back(colors[j], b_[i * n_ + j]);
        std::sort(incident.begin(), incident.end());
        // Separate tag distinguishes these signatures from class signatures.
        Signature signature{-1, colors[i],
                            static_cast<int64_t>(incident.size())};
        for (auto [color, weight] : incident) {
          signature.push_back(color);
          signature.push_back(weight);
        }
        next.push_back(intern(std::move(signature)));
      }
      if (std::set<Label>(next.begin(), next.end()).size() ==
          std::set<Label>(colors.begin(), colors.end()).size())
        break;
      colors = std::move(next);
    }
    return std::make_shared<const std::vector<Label>>(std::move(colors));
  }

  static void assign_singleton(Side &side, Label label, int atom, Label next) {
    auto found = std::find_if(side.blocks.begin(), side.blocks.end(),
                              [&](const Block &b) { return b.label == label; });
    if (found == side.blocks.end())
      throw std::runtime_error("Cannot split missing SLAP block");
    const auto position =
        std::find(found->atoms.begin(), found->atoms.end(), atom);
    if (position == found->atoms.end())
      throw std::runtime_error("Cannot split missing SLAP atom");
    found->atoms.erase(position);
    side.labels[atom] = next;
    side.blocks.push_back({next, {atom}});
  }

  void split(Record &record, Label label, std::vector<State> &pending) {
    const Label next = fresh();
    const auto &data = *record.lap_solutions.at(label);
    record.state.p.initial_wl = wl_colors(record.state.p);
    for (const auto &solution : data.solutions) {
      budget_.check();
      if (!one_group(solution.groups_r))
        continue;
      const int source = *std::min_element(data.classes_r.front().atoms.begin(),
                                           data.classes_r.front().atoms.end());
      int target_class = 0;
      if (!solution.fingerprint.empty()) {
        while (target_class < static_cast<int>(data.classes_p.size()) &&
               solution.fingerprint[target_class] == 0)
          ++target_class;
        if (target_class == static_cast<int>(data.classes_p.size()))
          throw std::runtime_error("Empty SLAP fingerprint row");
      }
      // This is the SLAP heuristic's WL-based split representative policy,
      // not an assertion that equal WL colors prove graph automorphism.
      std::set<Label> seen_wl;
      for (int target : data.classes_p[target_class].atoms) {
        budget_.check();
        if (!seen_wl.insert((*record.state.p.initial_wl)[target]).second)
          continue;
        State child = record.state;
        assign_singleton(child.r, label, source, next);
        assign_singleton(child.p, label, target, next);
        clear_cache(child, {label, next});
        if (visited_.insert(state_key(child)).second)
          pending.push_back(std::move(child));
      }
    }
  }

  void remember(const Record &record) {
    std::vector<int32_t> mapping(n_, -1);
    std::vector<bool> used(n_, false);
    for (const auto &left : record.state.r.blocks) {
      const auto &right = block(record.state.p, left.label);
      if (left.atoms.size() != 1 || right.atoms.size() != 1)
        throw std::runtime_error("SLAP output is not fully resolved");
      const int i = left.atoms.front(), j = right.atoms.front();
      if (used[j] || original_r_[i] != original_p_[j])
        throw std::runtime_error("SLAP output violates atom compatibility");
      mapping[i] = j;
      used[j] = true;
    }
    if (std::find(mapping.begin(), mapping.end(), -1) != mapping.end())
      throw std::runtime_error("SLAP output omits atoms");
    int64_t cost = 0;
    for (int i = 0; i < n_; ++i)
      for (int j = i + 1; j < n_; ++j)
        cost += std::abs(a_[i * n_ + j] - b_[mapping[i] * n_ + mapping[j]]);
    if (record.value != 2 * cost)
      throw std::runtime_error(
          "SLAP proxy disagrees with independent full-mapping rescoring");
    if (stats_.best_proxy2 < 0 || record.value < stats_.best_proxy2) {
      stats_.best_proxy2 = record.value;
      completed_.clear();
    }
    if (record.value == stats_.best_proxy2)
      completed_.emplace(std::move(mapping), cost);
    if (mapping_cap_ >= 0 &&
        static_cast<int64_t>(completed_.size()) >= mapping_cap_)
      throw Stop{3};
  }
};

} // namespace

extern "C" {

int synkit_mapper_abi(void) { return 1; }

const char *synkit_mapper_error(void) { return last_error.c_str(); }

int synkit_slap_solve(int n, const int64_t *a, const int64_t *b,
                      const int32_t *lr, const int32_t *lp, const int32_t *stop,
                      int max_fingerprints, double seconds, int64_t max_states,
                      int64_t max_mappings, SlapMappingCallback callback,
                      void *userdata, SlapStatistics *statistics) {
  last_error.clear();
  if (!statistics) {
    last_error = "Missing SLAP statistics output";
    return -1;
  }
  *statistics = {};
  statistics->best_proxy2 = -1;
  Budget budget(Clock::now(), seconds);
  try {
    if (n < 1 || n > 512 || !a || !b || !lr || !lp || !stop ||
        max_fingerprints < 1 || !std::isfinite(seconds) || seconds < -1 ||
        (seconds < 0 && seconds != -1) || max_states == 0 || max_states < -1 ||
        max_mappings == 0 || max_mappings < -1)
      throw std::invalid_argument("Invalid SLAP ABI arguments");
    std::map<int32_t, int> inventories;
    std::map<int32_t, int32_t> policies;
    for (int i = 0; i < n; ++i) {
      budget.check();
      ++inventories[lr[i]];
      --inventories[lp[i]];
      if (stop[i] != 0 && stop[i] != 1)
        throw std::invalid_argument("SLAP stop flags must be 0 or 1");
      const auto policy = policies.emplace(lr[i], stop[i]);
      if (!policy.second && policy.first->second != stop[i])
        throw std::invalid_argument(
            "SLAP stop flags must agree within each initial label");
      if (a[i * n + i] || b[i * n + i])
        throw std::invalid_argument("SLAP matrices must be loop free");
      for (int j = 0; j < n; ++j)
        if (a[i * n + j] < 0 || b[i * n + j] < 0 || a[i * n + j] > 1000000 ||
            b[i * n + j] > 1000000 || a[i * n + j] != a[j * n + i] ||
            b[i * n + j] != b[j * n + i])
          throw std::invalid_argument(
              "SLAP matrices must be bounded nonnegative symmetric integers");
    }
    for (auto [label, count] : inventories)
      if (count)
        throw std::invalid_argument(
            "SLAP requires equal atom-label inventories");
    Slap solver(n, a, b, lr, lp, stop, max_fingerprints, max_states,
                max_mappings, budget, *statistics);
    try {
      solver.run();
    } catch (const Stop &limit) {
      statistics->completed = 0;
      statistics->reason = limit.reason;
    }
    solver.emit(callback, userdata);
  } catch (const Stop &limit) {
    statistics->completed = 0;
    statistics->reason = limit.reason;
  } catch (const std::exception &error) {
    statistics->elapsed_seconds = budget.elapsed();
    last_error = error.what();
    return -1;
  }
  statistics->elapsed_seconds = budget.elapsed();
  return 0;
}

} // extern "C"
