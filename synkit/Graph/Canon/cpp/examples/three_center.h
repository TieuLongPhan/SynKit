#ifndef SYNKIT_THREE_CENTER_EXAMPLE_H
#define SYNKIT_THREE_CENTER_EXAMPLE_H
#include "synkit_canon/stereo.hpp"
#include <array>
#include <string>

namespace three_center {
// Explicit H pentitol / aldopentose. The three frames follow Fischer order
// (center, up, right, down, left), with horizontal bonds toward the viewer.
// L/R describes OH position, not an independently assigned CIP label.
inline synkit::stereo::Graph molecule(const std::string &word, bool aldehyde = false,
                                     bool middle_only = false) {
  using namespace synkit::stereo;
  if (word.size() != 3 || word.find_first_not_of("LR") != std::string::npos)
    throw std::invalid_argument("Use a three-letter L/R word");
  Graph g;
  g.colors.assign(5, 0); // C: 0..4; O: 5..9; appended H vertices
  g.colors.insert(g.colors.end(), 5, 1);
  for (int c = 0; c < 4; ++c) g.edges.push_back({c, c + 1, 1});
  std::array<int, 5> hydrogen{};
  auto add_h = [&](int owner) {
    int h = static_cast<int>(g.colors.size());
    g.colors.push_back(2); g.edges.push_back({owner, h, 1});
    return h;
  };
  for (int c = 0; c < 5; ++c) {
    g.edges.push_back({c, c + 5, aldehyde && c == 0 ? 2 : 1});
    hydrogen[c] = add_h(c);
    if ((c == 0 && !aldehyde) || c == 4) add_h(c);
    if (!(aldehyde && c == 0)) add_h(c + 5);
  }
  for (int c = 1; c <= 3; ++c) {
    if (middle_only && c != 2) continue;
    int left = c + 5, right = hydrogen[c];
    if (word[c - 1] == 'R') std::swap(left, right);
    g.frames.push_back({SYNKIT_TETRAHEDRAL, 1, {c, c - 1, right, c + 1, left}});
  }
  return g;
}
inline const std::array<std::string, 8> words{
  "LLL", "LLR", "LRL", "LRR", "RLL", "RLR", "RRL", "RRR"};
inline std::string rotated(std::string word) {
  std::reverse(word.begin(), word.end());
  for (char &c : word) c = c == 'L' ? 'R' : 'L';
  return word;
}
} // namespace three_center
#endif
