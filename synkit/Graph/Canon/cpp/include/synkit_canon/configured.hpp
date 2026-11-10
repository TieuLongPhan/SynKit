#ifndef SYNKIT_CONFIGURED_HPP
#define SYNKIT_CONFIGURED_HPP
#include "stereo.hpp"
#include <array>
#include <string>

namespace synkit { namespace stereo {
enum class Geometry : int {
  tetrahedral=1, square_planar=2, trigonal_bipyramidal=3, octahedral=4,
  planar_bond=5, atrop_bond=6, cumulene_axis=7, extended_cis_trans=8,
  helical=9, planar_chirality=10, framework=11
};
enum class Resource : int { atom=0, hydrogen=1, lone_pair=2 };
struct Reference {
  int32_t vertex = -1; // material atom, or owner of a virtual resource
  Resource kind = Resource::atom;
  Reference() = default;
  Reference(int32_t v) : vertex(v) {}
  Reference(int32_t v, Resource k) : vertex(v), kind(k) {}
  bool operator<(const Reference &r) const {
    return kind == r.kind ? vertex < r.vertex : kind < r.kind;
  }
  bool operator==(const Reference &r) const { return vertex == r.vertex && kind == r.kind; }
};
struct CoupledFrame {
  int32_t center = -1;
  std::array<Reference,4> references;
  int relation = 1;
};
struct Configuration {
  Geometry geometry = Geometry::tetrahedral;
  int parity = 1;
  bool fixed = true, cyclic = false;
  std::vector<Reference> frame; // six conventional frame slots; axis terminal frame
  std::vector<int32_t> path, support;
  int32_t pilot = -1;
  std::string coupling = "null"; // exact caller token, not a guessed chemical label
  std::vector<CoupledFrame> frames;
};
struct Bond { int32_t source, target; std::string color; };
struct Stereograph {
  // Exact typed attribute tokens. Callers must encode types without collisions.
  std::vector<std::string> colors;
  std::vector<Bond> bonds;
  std::vector<Configuration> configurations;
  std::string profile = "attributed_graph"; // labels already prepared attributes
};
struct ConfiguredOptions : Options {
  int64_t max_expanded_vertices = 1000000, max_expanded_edges = 10000000;
};
struct Expansion {
  std::vector<std::string> colors;
  std::vector<std::pair<int32_t,int32_t>> edges;
  int32_t atoms = 0; // original atoms are vertices 0..atoms-1
};
struct ConfiguredCanonicalResult {
  bool complete = false;
  int reason = 0;
  std::vector<uint8_t> code;
  std::vector<int32_t> atom_order;
  SynkitCanonStatistics statistics{};
  double preparation_seconds = 0, total_seconds = 0;
};
struct ConfiguredMirrorResult {
  bool complete = false, achiral = false;
  int reason = 0;
  std::vector<int32_t> mapping;
};
struct ConfiguredSymmetryResult {
  bool complete = false;
  int reason = 0;
  std::string group_order;
  std::vector<std::vector<int32_t>> generators, automorphisms;
  int64_t comparisons = 0;
};
// No perception, resonance normalization or missing-configuration inference.
// All eleven fixed families and owner-linked virtual H/LP are represented.
SYNKIT_CANON_API Expansion expand(const Stereograph &, const ConfiguredOptions & = {});
SYNKIT_CANON_API Stereograph mirror(const Stereograph &);
// Exhaustive local drawing orders; coupled frameworks expose their two
// declared shared orientations, never independent flips of coupled frames.
SYNKIT_CANON_API std::vector<Configuration> local_arrangements(const Configuration &);
SYNKIT_CANON_API ConfiguredCanonicalResult canonicalize(const Stereograph &, const ConfiguredOptions & = {});
// Exact sorted set of caller-declared drawings; no unique family atom order.
SYNKIT_CANON_API ConfiguredCanonicalResult canonicalize_family(const std::vector<Stereograph> &,
                                                              const ConfiguredOptions & = {});
// Fast direct constraints for the six local families; faithful expansion for
// extended/coupled supports. Set direct_local=false for an exact cross-check.
SYNKIT_CANON_API Result compare(const Stereograph &, const Stereograph &,
                              const ConfiguredOptions & = {}, bool direct_local = true);
SYNKIT_CANON_API ConfiguredMirrorResult classify_mirror(const Stereograph &, const ConfiguredOptions & = {});
// Exact stabilizer-chain transversals generate the complete atom automorphism
// group. Optional full closure is bounded by the same deadline/output cap.
SYNKIT_CANON_API ConfiguredSymmetryResult symmetry(const Stereograph &, bool full_group = false,
                                                  const ConfiguredOptions & = {});
}} // namespace
#endif
