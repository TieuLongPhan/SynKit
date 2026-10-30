#include "symmetry.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using Plain = std::vector<int32_t>;
using Group = std::set<Plain>;
using synkit_canon::Generators;
using synkit_canon::Permutation;
using synkit_canon::StabilizerWork;

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

// Independent small-group oracle: enumerate the generated group explicitly.
Group closure(const Generators &generators, int n) {
  Plain identity(n);
  std::iota(identity.begin(), identity.end(), 0);
  Group group{identity};
  std::vector<Plain> queue{identity};
  for (size_t k = 0; k < queue.size(); ++k) for (const auto &g : generators) {
    Plain image(n);
    for (int v = 0; v < n; ++v) image[v] = g[queue[k][v]];
    if (group.insert(image).second) queue.push_back(std::move(image));
    require(queue.size() < 1000, "Small independent oracle exceeded its fixture bound");
  }
  return group;
}

Generators make_group(std::initializer_list<std::initializer_list<int32_t>> rows) {
  Generators result;
  for (const auto &row : rows) result.emplace_back(row);
  return result;
}

void check_chain(const Generators &input, int n) {
  const auto whole = closure(input, n);
  for (int first = 0; first < n; ++first) {
    Generators stable;
    StabilizerWork work;
    require(synkit_canon::point_stabilizer(input, first, n, stable, work, []() {}),
            "Unexpected stabilizer cutoff on a small fixture");
    Group expected;
    for (const auto &g : whole) if (g[first] == first) expected.insert(g);
    require(closure(stable, n) == expected, "Point stabilizer differs from independent full-group oracle");
    for (int second = 0; second < n; ++second) {
      Generators next;
      require(synkit_canon::point_stabilizer(stable, second, n, next, work, []() {}),
              "Unexpected two-point stabilizer cutoff");
      Group pair;
      for (const auto &g : expected) if (g[second] == second) pair.insert(g);
      require(closure(next, n) == pair, "Two-point stabilizer differs from independent oracle");
    }
  }
}

int main() {
  const auto symmetric = make_group({{1, 2, 3, 0}, {1, 0, 2, 3}});
  // Neither input generator fixes zero, but their group has an S3 stabilizer.
  for (const auto &g : symmetric) require(g[0] != 0, "Fixture should defeat generator filtering");
  Generators stable;
  StabilizerWork normal;
  require(synkit_canon::point_stabilizer(symmetric, 0, 4, stable, normal, []() {}), "S4 derivation failed");
  require(closure(stable, 4).size() == 6, "Stabilizing compositions were lost");
  check_chain(symmetric, 4);
  check_chain(make_group({{1, 2, 3, 4, 0}, {0, 4, 3, 2, 1}}), 5); // D5
  check_chain(make_group({{1, 0, 3, 2, 4}}), 5); // coupled swaps and a fixed point
  check_chain(make_group({{1, 0, 2, 3}, {0, 1, 3, 2}}), 4); // independent swaps
  check_chain(Generators{}, 3);
  StabilizerWork no_work{0, 256};
  require(!synkit_canon::point_stabilizer(symmetric, 0, 4, stable, no_work, []() {}),
          "Work cutoff was not explicit");
  StabilizerWork no_generators{250000, 0};
  require(!synkit_canon::point_stabilizer(symmetric, 0, 4, stable, no_generators, []() {}),
          "Generator cutoff was not explicit");
  bool checked = false;
  try {
    StabilizerWork work;
    synkit_canon::point_stabilizer(symmetric, 0, 4, stable, work, [&]() {
      checked = true;
      throw std::runtime_error("cancel");
    });
    require(false, "Cancellation was swallowed");
  } catch (const std::runtime_error &error) {
    require(checked && std::string(error.what()) == "cancel", "Unexpected cancellation result");
  }
  std::cout << "Independent stabilizer, composition, cutoff and cancellation checks passed\n";
}
