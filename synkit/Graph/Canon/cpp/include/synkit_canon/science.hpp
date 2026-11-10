#ifndef SYNKIT_STEREO_SCIENCE_HPP
#define SYNKIT_STEREO_SCIENCE_HPP
#include "configured.hpp"
#include <nlohmann/json.hpp>
namespace RDKit { class ROMol; }
namespace synkit { namespace science {
using Json=nlohmann::json;
struct Options {
  double seconds=30;
  int max_depth=0;
  int64_t max_occurrences=2000000;
  bool include_extended_ring_axes=false;
};
// Native SynKit sequence rules; these functions do not read toolkit CIPCode
// or CIPRank properties and never call the RDKit CIP labeler.
Json rank_ligands(const RDKit::ROMol &,int,const std::vector<stereo::Reference> &,
                  const stereo::Stereograph &,const Options & = {});
Json assign_labels(const RDKit::ROMol &,const stereo::Stereograph &,const Options & = {});
Json perceive(const RDKit::ROMol &,const stereo::Stereograph &,bool retain_neighbors,
              const Options & = {});
Json configuration_json(const stereo::Configuration &);
stereo::Configuration configuration_from_json(const Json &);
stereo::Configuration configure_geometry(const RDKit::ROMol &,const stereo::Configuration &,
                                         int conformer=-1,double minimum=1e-5);
Json fit_octahedron(const std::vector<std::array<double,3>> &,double rmsd_limit=0.35,
                    double pose_gap=0.02);
std::vector<stereo::Configuration> coordinate_assistance(const RDKit::ROMol &,
  const stereo::Stereograph &,const RDKit::ROMol &,const std::vector<std::string> &,
  Json &audit,const Options & = {});
Json analyze_global_support(const RDKit::ROMol &,const stereo::Stereograph &,const Options & = {});
Json source_mirror_identity(const RDKit::ROMol &,const std::string &profile="chemical",const Options & = {});
Json topology_completion_identity(const RDKit::ROMol &,const stereo::Stereograph &,const Options & = {});
struct CoordinationInput { stereo::Stereograph graph,reflected; Json audit; };
CoordinationInput read_coordination(const Json &record,const std::string &root,const Options & = {});
Json exhaustive_local_orders(const RDKit::ROMol &,const stereo::Stereograph &,const Options & = {});
}} // namespace
#endif
