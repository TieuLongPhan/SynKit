#ifndef SYNKIT_STEREO_GROUPS_H
#define SYNKIT_STEREO_GROUPS_H

#include "synkit_canon/stereo.h"
#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>

namespace synkit_stereo_detail {
using Image = std::array<int32_t, 7>;
struct Group { int arity, size; const Image *images; };

// SynKit derivation: generate finite groups from integer geometry and explicit
// frame-orientation characters. No enumerated external group table is used.
// An image acts as result[i] = frame[image[i]]. See README for provenance.
using Point = std::array<int, 3>;
using Coordinates = std::array<Point, 7>;
using Metric = std::array<Point, 3>;
struct DerivedGroup {
  int arity, size = 0;
  std::array<Image, 24> images{};
  void add(const Image &image) {
    if (size == static_cast<int>(images.size())) throw std::logic_error("Stereo group overflow");
    images[size++] = image;
  }
};
inline int determinant(const Point &a, const Point &b, const Point &c) {
  return a[0] * (b[1]*c[2] - b[2]*c[1]) -
         a[1] * (b[0]*c[2] - b[2]*c[0]) + a[2] * (b[0]*c[1] - b[1]*c[0]);
}
inline DerivedGroup rotations(int arity, const Coordinates &points, const Metric &metric) {
  std::array<std::array<int, 7>, 7> gram{};
  for (int i = 1; i < arity; ++i) for (int j = 1; j < arity; ++j)
    for (int x = 0; x < 3; ++x) for (int y = 0; y < 3; ++y)
      gram[i][j] += points[i][x] * metric[x][y] * points[j][y];
  // Any independent triple determines the orientation of the whole isometry.
  Point basis{};
  int volume = 0;
  for (int i = 1; i < arity && !volume; ++i)
    for (int j = i + 1; j < arity && !volume; ++j)
      for (int k = j + 1; k < arity && !volume; ++k) {
        volume = determinant(points[i], points[j], points[k]);
        if (volume) basis = {i, j, k};
      }
  DerivedGroup result{arity};
  Image p{};
  std::iota(p.begin(), p.begin() + arity, 0);
  do {
    bool preserves = true;
    for (int i = 1; i < arity && preserves; ++i) for (int j = i; j < arity; ++j)
      if (gram[i][j] != gram[p[i]][p[j]]) { preserves = false; break; }
    if (!preserves) continue;
    if (volume && determinant(points[p[basis[0]]], points[p[basis[1]]], points[p[basis[2]]]) != volume)
      continue;
    // For a plane, either plane isometry extends to a proper 3D rotation by
    // choosing the normal's sign. Thus the square admits all eight D4 actions.
    result.add(p);
  } while (std::next_permutation(p.begin() + 1, p.begin() + arity));
  return result;
}
inline DerivedGroup bond_frames(bool axial) {
  // Include an artificial origin, then derive actual proper spatial motions.
  // End exchange reverses both the axis and the order of the terminal vectors;
  // these signs cancel in t.(l x r). Its proper subgroup is V4, never C4.
  const Metric euclidean{{{1,0,0}, {0,1,0}, {0,0,1}}};
  Coordinates points{{{0,0,0},{1,0,-1},{-1,0,-1},{0,0,-1},{0,0,1},
    axial?Point{0,1,1}:Point{1,0,1},axial?Point{0,-1,1}:Point{-1,0,1}}};
  const auto spatial=rotations(7,points,euclidean);
  DerivedGroup result{6};
  for (int i=0;i<spatial.size;++i) {
    Image image{}; for (int j=0;j<6;++j) image[j]=spatial.images[i][j+1]-1;
    result.add(image);
  }
  return result;
}
inline Group group(int geometry) {
  if (geometry < SYNKIT_TETRAHEDRAL || geometry > SYNKIT_ATROP_BOND)
    throw std::invalid_argument("Unsupported stereo geometry");
  static const auto derived = [] {
    const Metric euclidean{{{1,0,0}, {0,1,0}, {0,0,1}}};
    const Coordinates tetra{{{0,0,0}, {1,1,1}, {1,-1,-1}, {-1,1,-1}, {-1,-1,1}}};
    const Coordinates square{{{0,0,0}, {1,0,0}, {0,1,0}, {-1,0,0}, {0,-1,0}}};
    // Equilateral triangle in an oblique basis: x.x = y.y = 2, x.y = -1.
    // Its third ray is -x-y. All calculations remain exact integer arithmetic.
    const Metric triangular{{{2,-1,0}, {-1,2,0}, {0,0,2}}};
    const Coordinates tbp{{{0,0,0}, {0,0,1}, {0,0,-1}, {1,0,0}, {0,1,0}, {-1,-1,0}}};
    const Coordinates oct{{{0,0,0}, {0,0,1}, {0,0,-1}, {1,0,0}, {0,1,0}, {-1,0,0}, {0,-1,0}}};
    std::array<DerivedGroup, 6> result{
      rotations(5, tetra, euclidean), rotations(5, square, euclidean),
      rotations(6, tbp, triangular), rotations(7, oct, euclidean),
      bond_frames(false), bond_frames(true)};
    const int orders[] = {12, 8, 6, 24, 4, 4};
    for (size_t i = 0; i < result.size(); ++i) {
      if (result[i].size != orders[i]) throw std::logic_error("Incorrect derived stereo group order");
      std::sort(result[i].images.begin(), result[i].images.begin() + result[i].size);
    }
    return result;
  }();
  const auto &g = derived[geometry - 1];
  return {g.arity, g.size, g.images.data()};
}
inline bool bond(int geometry) { return geometry >= SYNKIT_PLANAR_BOND; }
inline bool owner(int geometry, int slot) {
  return bond(geometry) ? slot == 2 || slot == 3 : slot == 0;
}
inline Image normalized(const SynkitStereoFrame &frame) {
  Image result{};
  const auto g = group(frame.geometry);
  std::copy_n(frame.vertices, g.arity, result.begin());
  if (frame.parity == -1) {
    if (frame.geometry == SYNKIT_TRIGONAL_BIPYRAMIDAL) std::swap(result[4], result[5]);
    else if (frame.geometry == SYNKIT_ATROP_BOND) std::swap(result[0], result[1]);
    else std::swap(result[1], result[2]);
  }
  return result;
}
} // namespace synkit_stereo_detail
#endif
