#include "synkit_canon/science.hpp"
#include "rdkit_preparation.hpp"
#include "native_build_info.hpp"
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/MolOps.h>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <set>
#include <queue>
using synkit::science::Json;
using namespace synkit;
namespace {
Json normalized_support(const Json &s) {
  const auto kind=s.at("kind").get<std::string>();
  if (kind=="atom") return Json::array({kind,s.at("center")});
  if (kind=="bond") { auto b=s.at("bond"); std::sort(b.begin(),b.end()); return Json::array({kind,b}); }
  auto p=s.at("path"),frames=s.at("terminal_frames"); for (auto &f:frames) std::sort(f.begin(),f.end());
  auto reversed=p; std::reverse(reversed.begin(),reversed.end());
  auto a=Json::array({kind,p,frames}),b=Json::array({kind,reversed,Json::array({frames.at(1),frames.at(0)})});
  return std::min(a,b);
}
Json observed_support(const Json &e) {
  auto d=science::configuration_from_json(e.at("support"));
  if (d.geometry==stereo::Geometry::tetrahedral) return {{"kind","atom"},{"center",d.frame.at(0).vertex}};
  if (d.geometry==stereo::Geometry::planar_bond) return {{"kind","bond"},{"bond",d.path}};
  auto frame=e.at("support").at("frame");
  return {{"kind",d.geometry==stereo::Geometry::extended_cis_trans?"path":"axis"},{"path",d.path},{"terminal_frames",Json::array({Json::array({frame.at(0),frame.at(1)}),Json::array({frame.at(4),frame.at(5)})})}};
}
Json run(const Json &c,const Json &structure,const std::string &scope,const science::Options &options) {
  Json observed=Json::array(); std::string reason; bool present=false,unsupported=false;
  const std::string family=c.at("family");
  if (scope=="sidecar_only") {
    std::map<int,std::vector<int>> adj; std::set<std::string> kinds;
    for (const auto &n:structure.at("nodes")) adj[n.at("id").get<int>()]={};
    for (const auto &e:structure.at("edges")) { int a=e.at(0),b=e.at(1); adj.at(a).push_back(b); adj.at(b).push_back(a); kinds.insert(e.at(2).get<std::string>()); }
    if (family=="helical") {
      std::vector<int> ends; for (const auto &[v,a]:adj) if (a.size()==1) ends.push_back(v);
      if (kinds==std::set<std::string>{"single"}) reason="flexible_open_chain";
      else if (ends.empty()) reason="multiple_shortest_carrier_paths";
      else if (ends.size()==2 && kinds==std::set<std::string>{"aromatic"}) {
        std::map<int,int> parent; std::queue<int> pending; pending.push(ends[0]); parent[ends[0]]=-1;
        while (!pending.empty()) { int u=pending.front(); pending.pop(); for (int v:adj.at(u)) if (!parent.count(v)) { parent[v]=u; pending.push(v); } }
        std::vector<int> path; if (parent.count(ends[1])) for (int v=ends[1];v>=0;v=parent[v]) path.push_back(v);
        std::reverse(path.begin(),path.end());
        if (path.size()==adj.size()) { present=true; observed.push_back({{"kind","path"},{"path",path}}); reason="formal_graph_carrier"; }
        else reason="unsupported_formal_helix";
      } else reason="unsupported_formal_helix";
    } else {
      std::vector<int> plane,pilots; std::set<std::string> elements;
      for (const auto &n:structure.at("nodes")) {
        if (n.value("role",std::string())=="plane") { plane.push_back(n.at("id")); elements.insert(n.at("element")); }
        if (n.value("role",std::string())=="pilot") pilots.push_back(n.at("id"));
      }
      if (pilots.size()!=1) reason="missing_pilot_reference";
      else if (elements.size()==1) reason="plane_symmetry_unresolved";
      else { present=true; reason="formal_graph_carrier"; observed.push_back({{"kind","plane"},{"plane_atoms",plane},{"pilot",pilots[0]}}); }
    }
  } else {
    std::unique_ptr<RDKit::RWMol> m(RDKit::SmilesToMol(structure.at("value").get<std::string>()));
    if (!m) throw std::invalid_argument("Conformance SMILES parse failure");
    for (auto a:m->atoms()) {
      if (a->getChiralTag()==RDKit::Atom::CHI_TETRAHEDRAL_CW || a->getChiralTag()==RDKit::Atom::CHI_TETRAHEDRAL_CCW) a->setChiralTag(RDKit::Atom::CHI_UNSPECIFIED);
      if (a->getChiralTag()==RDKit::Atom::CHI_SQUAREPLANAR || a->getChiralTag()==RDKit::Atom::CHI_TRIGONALBIPYRAMIDAL || a->getChiralTag()==RDKit::Atom::CHI_OCTAHEDRAL) a->setProp("_chiralPermutation",0);
    }
    for (auto b:m->bonds()) { b->setStereo(RDKit::Bond::STEREONONE); b->setBondDir(RDKit::Bond::NONE); }
    if (scope=="current_topology_detector") {
      auto source=prepare_rdkit_molecule(*m,true); auto inventory=science::perceive(*m,source,false,options);
      for (const auto &e:inventory.at("elements")) if (e.at("family")==c.at("family") && e.at("carrier_status")=="confirmed") observed.push_back(observed_support(e));
      present=!observed.empty();
    } else {
      try {
        auto source=prepare_rdkit_molecule(*m,true);
        int geometry=family=="square_planar"?2:(family=="trigonal_bipyramidal"?3:4);
        for (const auto &d:source.configurations) if (static_cast<int>(d.geometry)==geometry && !d.fixed) {
          auto frame=science::configuration_json(d).at("frame"); frame.erase(frame.begin());
          observed.push_back({{"kind","atom"},{"center",d.frame.at(0).vertex},{"references",frame}});
        }
        present=!observed.empty();
        if (present) reason="shape_declared_configuration_unknown";
        else {
          unsigned center=0; for (auto a:m->atoms()) if (a->getDegree()>m->getAtomWithIdx(center)->getDegree()) center=a->getIdx();
          std::vector<int> ligands; for (auto a:m->atomNeighbors(m->getAtomWithIdx(center))) ligands.push_back(a->getIdx());
          bool same=!ligands.empty(); source.configurations.clear();
          source.colors[center]=Json::array({source.colors[center],"center"}).dump();
          for (size_t i=1;i<ligands.size();++i) {
            auto a=source,b=source; a.colors[ligands[0]]=Json::array({a.colors[ligands[0]],"probe"}).dump(); b.colors[ligands[i]]=Json::array({b.colors[ligands[i]],"probe"}).dump();
            stereo::ConfiguredOptions limit; limit.seconds=options.seconds; auto result=stereo::compare(a,b,limit); same=same && result.complete() && result.isomorphic();
          }
          if (same) reason="all_ligands_symmetry_related";
        }
      } catch (const std::invalid_argument &e) { unsupported=true; reason=std::string(e.what()).find("coordination")!=std::string::npos?"wrong_coordination_number":"adapter_error"; }
    }
  }
  // Scoring starts only after all observations have been computed.
  const auto &expected=c.at("expected"); bool wanted=expected.at("outcome")=="carrier_present",passed=false;
  if (wanted) {
    for (const auto &s:observed) {
      if (scope=="current_topology_detector") passed=passed || normalized_support(s)==normalized_support(expected.at("support"));
      else if (scope=="sidecar_only") passed=passed || s==expected.at("support");
      else { auto a=s.at("references"),b=expected.at("support").at("references"); std::sort(a.begin(),a.end()); std::sort(b.begin(),b.end()); passed=passed || (s.at("center")==expected.at("support").at("center") && a==b); }
    }
  } else passed=!present;
  if (scope=="configured_adapter_only" && c.at("case_type")=="unsupported_or_ambiguous") passed=unsupported;
  if (scope!="current_topology_detector" && !expected.at("reason_code").is_null()) passed=passed && Json(reason)==expected.at("reason_code");
  return {{"id",c.at("id")},{"family",family},{"scope",scope},{"passed",passed},{"reason",reason},{"observed",observed}};
}
} // namespace
int main(int argc,char **argv) {
  try {
    std::filesystem::path input,output; science::Options options;
    for (int i=1;i<argc;++i) { if (i+1>=argc) throw std::invalid_argument("Missing value"); std::string key=argv[i],value=argv[++i]; if (key=="--input") input=value; else if (key=="--output") output=value; else if (key=="--seconds") options.seconds=std::stod(value); else if (key!="--mode") throw std::invalid_argument("Unknown option"); }
    if (output.empty() || std::filesystem::exists(output)) throw std::invalid_argument("A new output path is required");
    std::ifstream file(input); Json manifest; file>>manifest; Json records=Json::array(); int failed=0;
    for (const auto &c:manifest.at("cases")) {
      Json record;
      try { record=run(c,manifest.at("structures").at(c.at("structure").get<std::string>()),manifest.at("families").at(c.at("family").get<std::string>()).at("perception_scope"),options); }
      catch (const std::exception &e) { record={{"id",c.at("id")},{"passed",false},{"error",e.what()}}; }
      if (!record.at("passed").get<bool>()) ++failed; records.push_back(record);
    }
    Json report={{"schema","synkit.native-perception-conformance/1"},{"build",Json::parse(synkit_native_build_info)},{"records",records},{"summary",{{"cases",records.size()},{"failed",failed}}},{"python_runtime",false}};
    std::ofstream out(output); out<<report.dump(2)<<'\n'; std::cout<<report.at("summary").dump()<<'\n'; return failed?3:0;
  } catch (const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
