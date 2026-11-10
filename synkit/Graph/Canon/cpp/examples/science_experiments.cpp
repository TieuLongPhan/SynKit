// Standalone scientific execution. Python is not loaded by this process.
#include "synkit_canon/science.hpp"
#include "rdkit_preparation.hpp"
#include "native_build_info.hpp"
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/FileParsers/MolSupplier.h>
#include <GraphMol/Conformer.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/RingInfo.h>
#include <RDGeneral/versions.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <set>
#include <numeric>
#include <Eigen/Core>
using synkit::science::Json;
namespace {
using namespace synkit;
using namespace stereo;
std::vector<std::string> split(const std::string &s,char separator) {
  std::vector<std::string> out; std::stringstream stream(s); std::string value;
  while (std::getline(stream,value,separator)) out.push_back(value); return out;
}
Json input_rows(const std::filesystem::path &file) {
  std::ifstream in(file); if (!in) throw std::runtime_error("Cannot open input");
  if (file.extension()==".json") { Json input; in>>input; return input.is_array()?input:input.at(input.contains("rows")?"rows":"records"); }
  Json rows=Json::array(); std::string line;
  while (std::getline(in,line)) {
    if (line.empty()) continue; auto fields=split(line,'\t'); fields.resize(std::max<size_t>(6,fields.size()));
    std::vector<std::string> expected; std::stringstream labels(fields[2]); std::string token;
    while (labels>>token) expected.push_back(token);
    rows.push_back({{"smiles",fields[0]},{"id",fields[1]},{"expected_labels",expected},{"units",split(fields[4],',')}});
  }
  return rows;
}
Json predicted(const Stereograph &g,const Json &assignments) {
  std::set<std::string> out;
  for (size_t i=0;i<g.configurations.size();++i) {
    if (assignments[i]["label"].is_null()) continue;
    const auto &d=g.configurations[i]; std::vector<int> positions;
    if (d.geometry==Geometry::tetrahedral) positions={d.frame.at(0).vertex};
    else if (d.geometry==Geometry::helical) positions={d.path.front(),d.path.back()};
    else if (d.geometry==Geometry::planar_bond || d.geometry==Geometry::atrop_bond || d.geometry==Geometry::cumulene_axis || d.geometry==Geometry::extended_cis_trans) positions={d.frame.at(2).vertex,d.frame.at(3).vertex};
    for (int position:positions) out.insert(std::to_string(position+1)+assignments[i]["label"].get<std::string>());
  }
  return out;
}
std::set<std::string> support_signatures(const Json &inventory,const std::vector<int> &mapping={}) {
  std::set<std::string> out;
  for (const auto &element:inventory.at("elements")) {
    auto d=science::configuration_from_json(element.at("support"));
    std::vector<int> support=d.path;
    if (d.geometry==Geometry::tetrahedral) support={d.frame.at(0).vertex};
    if (!mapping.empty()) for (auto &v:support) v=mapping.at(v);
    auto reversed=support; std::reverse(reversed.begin(),reversed.end()); support=std::min(support,reversed);
    out.insert(Json::array({element.at("family"),support,element.at("carrier_status")}).dump());
  }
  return out;
}
std::set<std::string> historical_signatures(const Json &elements) {
  std::set<std::string> out;
  for (const auto &e:elements) {
    const auto &s=e.at("support"); std::vector<int> support;
    if (s.contains("center")) support={s.at("center").get<int>()};
    else support=s.at(s.contains("endpoints")?"endpoints":"path").get<std::vector<int>>();
    auto reversed=support; std::reverse(reversed.begin(),reversed.end()); support=std::min(support,reversed);
    out.insert(Json::array({e.at("family"),support,e.value("carrier_status",std::string("confirmed"))}).dump());
  }
  return out;
}
void merge(Stereograph &g,const std::vector<Configuration> &additions,const science::Options &options) {
  std::set<std::vector<uint8_t>> keys;
  std::vector<Configuration> merged;
  auto all=g.configurations; all.insert(all.end(),additions.begin(),additions.end());
  for (const auto &d:all) {
    if (!d.fixed) { merged.push_back(d); continue; }
    auto h=g; h.configurations={d};
    for (size_t i=0;i<h.colors.size();++i) h.colors[i]=Json::array({h.colors[i],i}).dump();
    ConfiguredOptions request; request.seconds=options.seconds; auto key=canonicalize(h,request);
    if (!key.complete) throw std::runtime_error("configuration_merge_resource_stop");
    if (keys.insert(key.code).second) merged.push_back(d);
  }
  g.configurations=std::move(merged);
}
Json negative_proof(const RDKit::ROMol &m,const Stereograph &g,const Json &w,const Json &observed) {
  std::vector<int> mapping(m.getNumAtoms(),-1); std::set<int> images;
  for (const auto &pair:w.at("atom_map")) {
    int a=pair.at(0),b=pair.at(1);
    if (a<0 || b<0 || static_cast<unsigned>(a)>=m.getNumAtoms() || static_cast<unsigned>(b)>=m.getNumAtoms() || mapping[a]!=-1 || !images.insert(b).second) throw std::invalid_argument("Negative witness is not injective"); mapping[a]=b;
  }
  if (images.size()!=m.getNumAtoms()) throw std::invalid_argument("Negative witness is incomplete");
  for (size_t i=0;i<mapping.size();++i) if (g.colors[i]!=g.colors[mapping[i]]) throw std::invalid_argument("Negative witness changes atom attributes");
  std::map<std::pair<int,int>,std::string> bonds;
  for (const auto &b:g.bonds) bonds[std::minmax(b.source,b.target)]=b.color;
  for (const auto &b:g.bonds) {
    auto at=bonds.find(std::minmax(mapping[b.source],mapping[b.target]));
    if (at==bonds.end() || at->second!=b.color) throw std::invalid_argument("Negative witness changes bonds");
  }
  const auto axis=w.at("axis_endpoints").get<std::vector<int>>(),ring=w.at("ring_order").get<std::vector<int>>(); int fixed=w.at("fixed_terminal");
  std::set<int> ring_set(ring.begin(),ring.end());
  if (axis.size()!=2 || axis[0]==axis[1] || ring.size()!=6 || ring_set.size()!=6 || ring[0]!=fixed || std::find(axis.begin(),axis.end(),fixed)==axis.end()) throw std::invalid_argument("Invalid negative support");
  auto bond=m.getBondBetweenAtoms(axis[0],axis[1]);
  if (!bond || bond->getBondType()!=RDKit::Bond::SINGLE || m.getRingInfo()->numBondRings(bond->getIdx())) throw std::invalid_argument("Negative axis is not a nonring single bond");
  for (int v:axis) if (mapping.at(v)!=v || !m.getAtomWithIdx(v)->getIsAromatic()) throw std::invalid_argument("Negative axis is not fixed/aromatic");
  for (size_t i=0;i<mapping.size();++i) if (!ring_set.count(i) && mapping[i]!=static_cast<int>(i)) throw std::invalid_argument("Negative witness moves outside the declared ring");
  for (int i=0;i<6;++i) {
    auto a=m.getAtomWithIdx(ring[i]);
    if (!m.getBondBetweenAtoms(ring[i],ring[(i+1)%6]) || a->getAtomicNum()!=6 || !a->getIsAromatic() || mapping.at(ring[i])!=ring[(6-i)%6] || (i && (a->getDegree()!=2 || a->getTotalNumHs()!=1))) throw std::invalid_argument("Negative ring reflection is invalid");
  }
  auto branches=w.at("swapped_terminal_references").get<std::vector<int>>();
  if (branches.size()!=2 || std::set<int>(branches.begin(),branches.end())!=std::set<int>({ring[1],ring[5]}) || mapping.at(branches[0])!=branches[1]) throw std::invalid_argument("Negative witness does not exchange branches");
  bool found=false,rejected=false;
  for (const auto &e:observed.at("elements")) if (e.at("family")=="atrop_axis") {
    auto p=e.at("support").at("path").get<std::vector<int>>();
    if (std::set<int>(p.begin(),p.end())==std::set<int>(axis.begin(),axis.end())) { found=true; rejected=e.at("carrier_status")=="symmetry_related"; }
  }
  return {{"typed_bijection_replayed",true},{"declared_ring_reflection_replayed",true},{"candidate_retained",found},{"not_promoted",rejected},{"passed",found && rejected}};
}
} // namespace
int main(int argc,char **argv) {
  try {
    std::filesystem::path input,output,sdf,reference,root=std::filesystem::current_path(); std::string mode="pipeline"; science::Options options;
    for (int i=1;i<argc;++i) {
      std::string arg=argv[i]; if (i+1>=argc) throw std::invalid_argument("Missing option value"); std::string value=argv[++i];
      if (arg=="--input") input=value; else if (arg=="--output") output=value;
      else if (arg=="--mode") mode=value; else if (arg=="--sdf") sdf=value;
      else if (arg=="--root") root=value;
      else if (arg=="--reference") reference=value;
      else if (arg=="--seconds") options.seconds=std::stod(value);
      else if (arg=="--extended-ring-axes") options.include_extended_ring_axes=std::stoi(value)!=0;
      else throw std::invalid_argument("Unknown option:"+arg);
    }
    if (input.empty() || output.empty() || std::filesystem::exists(output)) throw std::invalid_argument("Input and a NEW output path are required");
    const std::set<std::string> modes{"cip","perception","pipeline","geometry","coordination","identity","local","overlap","representation","relations"};
    if (!modes.count(mode)) throw std::invalid_argument("Unknown scientific mode");
    const auto start=std::chrono::steady_clock::now(); auto rows=input_rows(input);
    std::map<std::string,std::unique_ptr<RDKit::ROMol>> coordinates;
    if (!sdf.empty()) {
      RDKit::SDMolSupplier supplier(sdf.string(),true,true);
      while (!supplier.atEnd()) {
        std::unique_ptr<RDKit::ROMol> m(supplier.next());
        if (!m || !m->hasProp("STRUCTURE_ID")) throw std::invalid_argument("Invalid coordinate record");
        std::string id; m->getProp("STRUCTURE_ID",id);
        if (coordinates.count(id)) throw std::invalid_argument("Duplicate coordinate ID"); coordinates[id]=std::move(m);
      }
    }
    Json truth; std::map<std::string,Json> historical;
    if (!reference.empty()) {
      std::ifstream file(reference); if (!file) throw std::invalid_argument("Reference absent"); file>>truth;
      if (truth.contains("records")) for (const auto &r:truth["records"]) historical[r.at("id")]=r;
      if (truth.contains("datasets")) {
        for (auto [name,prefix]:std::vector<std::pair<std::string,std::string>>{{"acs_stereomolgraph_molecular_chirality","acs:"},{"chiralfinder_rota","rota:"},{"cip_validation_suite","cip:"}})
          for (const auto &r:truth["datasets"][name]["records"]) historical[prefix+r.at("id").get<std::string>()]=r;
      }
      if (truth.contains("coordination")) {
        for (const auto &r:truth["coordination"]["accepted"]) historical[r.at("id")]=r;
        for (const auto &r:truth["coordination"]["unresolved"]) historical[r.at("id")]=r;
      }
    }
    Json records=Json::array(); size_t exact=0,complete=0,errors=0,refused=0,regression_failures=0,reference_checks=0;
    size_t expected_count=0,predicted_count=0,true_positives=0;
    size_t local_carriers=0,local_raw_orders=0,local_failures=0,negative_checks=0,negative_failures=0;
    std::map<std::vector<uint8_t>,int> coordination_classes;
    std::map<std::string,int> reference_classes; std::map<int,std::string> observed_classes;
    std::map<std::vector<uint8_t>,int> constitutions; size_t representation_checks=0;
    for (const auto &row:rows) {
      auto row_options=options;
      row_options.include_extended_ring_axes=row.value("include_extended_ring_axes",options.include_extended_ring_axes);
      Json record={{"id",row.value("id",std::string("unknown"))}};
      auto begun=std::chrono::steady_clock::now();
      try {
        if (mode=="relations") {
          std::unique_ptr<RDKit::RWMol> left(RDKit::SmilesToMol(row.at("left").get<std::string>()));
          std::unique_ptr<RDKit::RWMol> right(RDKit::SmilesToMol(row.at("right").get<std::string>()));
          if (!left || !right) throw std::invalid_argument("SMILES parse failure");
          std::unique_ptr<RDKit::ROMol> renamed;
          if (row.value("renumber_right",false)) {
            std::vector<unsigned> order(right->getNumAtoms()); std::iota(order.rbegin(),order.rend(),0);
            renamed.reset(RDKit::MolOps::renumberAtoms(*right,order));
          }
          const auto &rhs=renamed?*renamed:*right;
          auto a=prepare_rdkit_molecule(*left,true), b=prepare_rdkit_molecule(rhs,true);
          auto ca=a,cb=b; ca.configurations.clear(); cb.configurations.clear();
          ConfiguredOptions limit; limit.seconds=options.seconds;
          auto constitutional=compare(ca,cb,limit);
          if (!constitutional.complete()) throw std::runtime_error("constitution comparison stopped");
          std::string relation;
          if (!constitutional.isomorphic()) relation="constitutionally_different";
          else {
            auto incomplete=[&](const RDKit::ROMol &m,const Stereograph &g) {
              auto inventory=science::perceive(m,g,true,row_options);
              for (const auto &e:inventory.at("elements"))
                if (e.at("carrier_status")=="confirmed" && e.at("configuration_state")=="unspecified") return true;
              return false;
            };
            if (incomplete(*left,a) || incomplete(rhs,b)) relation="incomplete";
            else {
              auto equal=compare(a,b,limit);
              if (!equal.complete()) throw std::runtime_error("stereo comparison stopped");
              if (equal.isomorphic()) relation="identical";
              else {
                auto reflection=compare(mirror(a),b,limit);
                if (!reflection.complete()) throw std::runtime_error("mirror comparison stopped");
                relation=reflection.isomorphic()?"enantiomers":"diastereomers";
              }
            }
          }
          record["expected"]=row.at("expected"); record["observed"]=relation;
          record["correct"]=record["expected"]==record["observed"];
          if (!record["correct"].get<bool>()) ++regression_failures;
          record["complete"]=true; ++complete;
        } else if (mode=="coordination") {
          auto prepared=science::read_coordination(row,root.string(),row_options);
          record["audit"]=prepared.audit;
          ConfiguredOptions request; request.seconds=options.seconds;
          auto certificate=canonicalize(prepared.graph,request);
          if (!certificate.complete) throw std::runtime_error("coordination_canonicalization_resource_stop");
          auto inserted=coordination_classes.emplace(certificate.code,static_cast<int>(coordination_classes.size()));
          record["group_id"]=inserted.first->second;
          auto reflected_certificate=canonicalize(prepared.reflected,request);
          if (!reflected_certificate.complete) throw std::runtime_error("coordinate_mirror_canonicalization_resource_stop");
          auto reflected_class=coordination_classes.emplace(reflected_certificate.code,static_cast<int>(coordination_classes.size()));
          record["coordinate_mirror_class_id"]=reflected_class.first->second;
          auto reflected=prepared.reflected,algebraic=mirror(prepared.graph);
          for (size_t i=0;i<reflected.colors.size();++i) {
            auto color=Json::array({reflected.colors[i],i}).dump(); reflected.colors[i]=color; algebraic.colors[i]=color;
          }
          auto verified=compare(reflected,algebraic,request);
          if (!verified.complete()) throw std::runtime_error("coordinate_reflection_check_resource_stop");
          record["coordinate_reflection_agrees_with_algebra"]=verified.isomorphic();
          auto relationship=classify_mirror(prepared.graph,request);
          if (!relationship.complete) throw std::runtime_error("coordination_mirror_resource_stop");
          record["achiral"]=relationship.achiral; record["mirror_mapping"]=relationship.mapping;
          Json frames=Json::array(); for (const auto &d:prepared.graph.configurations) frames.push_back(science::configuration_json(d)); record["configurations"]=frames;
          record["complete"]=true; ++complete;
        } else if (row.contains("donor_vectors")) {
          record["fit"]=science::fit_octahedron(row.at("donor_vectors").get<std::vector<std::array<double,3>>>());
          record["complete"]=true; ++complete;
        } else {
          std::unique_ptr<RDKit::RWMol> m(RDKit::SmilesToMol(row.at("smiles").get<std::string>()));
          if (!m) throw std::invalid_argument("SMILES parse failure");
          if (row.contains("coordinates")) {
            const auto &points=row.at("coordinates"); if (points.size()!=m->getNumAtoms()) throw std::invalid_argument("Coordinate atom count differs");
            auto conformer=new RDKit::Conformer(m->getNumAtoms());
            for (size_t i=0;i<points.size();++i) conformer->setAtomPos(i,RDGeom::Point3D(points[i][0].get<double>(),points[i][1].get<double>(),points[i][2].get<double>()));
            m->addConformer(conformer,true);
          }
          auto source=prepare_rdkit_molecule(*m,true);
          if (mode=="overlap") {
            auto g=source; g.configurations.clear(); stereo::ConfiguredOptions limit; limit.seconds=options.seconds;
            auto result=stereo::canonicalize(g,limit); if (!result.complete) throw std::runtime_error("constitution_canonicalization_resource_stop");
            record["constitution_class"]=constitutions.emplace(result.code,static_cast<int>(constitutions.size())).first->second;
            record["dataset"]=row.value("dataset",std::string());
          }
          if (mode=="representation") {
            if (m->getNumAtoms()>7) throw std::invalid_argument("Exhaustive representation protocol is limited to seven atoms");
            stereo::ConfiguredOptions limit; limit.seconds=options.seconds; auto original=stereo::canonicalize(source,limit);
            if (!original.complete) throw std::runtime_error("representation_baseline_resource_stop");
            std::vector<unsigned> order(m->getNumAtoms()); std::iota(order.begin(),order.end(),0); size_t checked=0;
            do {
              std::unique_ptr<RDKit::ROMol> renamed(RDKit::MolOps::renumberAtoms(*m,order));
              auto prepared=prepare_rdkit_molecule(*renamed); auto transported=stereo::canonicalize(prepared,limit); ++checked;
              if (!transported.complete || transported.code!=original.code) throw std::runtime_error("representation_transport_or_canonicalization_disagreement");
            } while (std::next_permutation(order.begin(),order.end()));
            representation_checks+=checked; record["atom_permutations_checked"]=checked; record["passed"]=true;
            std::cout<<record.at("id").dump()<<" checked "<<checked<<std::endl;
          }
          if (mode=="local") {
            auto erased=*m; RDKit::MolOps::removeStereochemistry(erased);
            record["local"]=science::exhaustive_local_orders(erased,source,row_options);
            local_carriers+=record["local"]["carriers"].size(); local_raw_orders+=record["local"]["raw_orders"].get<size_t>();
            if (!record["local"]["passed"].get<bool>()) ++local_failures;
          }
          if (mode=="identity") {
            auto restored=*m; RDKit::SmilesParserParams params; params.sanitize=false; params.removeHs=false;
            std::unique_ptr<RDKit::RWMol> raw(RDKit::SmilesToMol(row.at("smiles").get<std::string>(),params));
            if (!raw || raw->getNumAtoms()!=restored.getNumAtoms()) throw std::invalid_argument("Source tag atom transport is unavailable");
            for (auto a:raw->atoms()) if (a->getChiralTag()==RDKit::Atom::CHI_TETRAHEDRAL_CW || a->getChiralTag()==RDKit::Atom::CHI_TETRAHEDRAL_CCW) restored.getAtomWithIdx(a->getIdx())->setChiralTag(a->getChiralTag());
            record["source_declared"]=science::source_mirror_identity(restored,row.value("identity_profile",std::string("chemical")),row_options);
            record["global_topology_evidence"]=science::analyze_global_support(*m,source,row_options);
            record["topology_completion"]=science::topology_completion_identity(*m,source,row_options);
            if (record["source_declared"].value("complete",false)) record["combined_status"]=record["global_topology_evidence"].value("necessarily_chiral",false)?Json("chiral"):record["source_declared"].at("status");
          }
          if (row.contains("configurations")) { source.configurations.clear(); for (const auto &d:row["configurations"]) source.configurations.push_back(science::configuration_from_json(d)); }
          if (mode=="perception" || mode=="pipeline") {
            RDKit::SmilesParserParams params; params.sanitize=false; params.removeHs=false;
            std::unique_ptr<RDKit::RWMol> retained(RDKit::SmilesToMol(row.at("smiles").get<std::string>(),params));
            if (!retained) throw std::invalid_argument("Raw orientation SMILES parse failure");
            unsigned failed=0;
            RDKit::MolOps::sanitizeMol(*retained,failed,RDKit::MolOps::SANITIZE_ALL & ~RDKit::MolOps::SANITIZE_CLEANUPCHIRALITY);
            RDKit::MolOps::setBondStereoFromDirections(*retained);
            auto retained_source=prepare_rdkit_molecule(*retained,true);
            auto erased=*retained;
            for (auto a:erased.atoms()) if (a->getChiralTag()==RDKit::Atom::CHI_TETRAHEDRAL_CW || a->getChiralTag()==RDKit::Atom::CHI_TETRAHEDRAL_CCW) a->setChiralTag(RDKit::Atom::CHI_UNSPECIFIED);
            for (auto b:erased.bonds()) { b->setStereo(RDKit::Bond::STEREONONE); b->setBondDir(RDKit::Bond::NONE); }
            auto neutral=retained_source; neutral.configurations.clear();
            record["configuration_erased"]=science::perceive(erased,neutral,false,row_options);
            record["neighbor_orientation_retained"]=science::perceive(*retained,retained_source,true,row_options);
            std::vector<unsigned> order(retained->getNumAtoms()); std::iota(order.rbegin(),order.rend(),0);
            std::vector<int> mapping(order.begin(),order.end());
            std::unique_ptr<RDKit::ROMol> renamed(RDKit::MolOps::renumberAtoms(*retained,order));
            std::unique_ptr<RDKit::ROMol> renamed_erased(RDKit::MolOps::renumberAtoms(erased,order));
            auto named_source=prepare_rdkit_molecule(*renamed,true); auto empty=named_source; empty.configurations.clear();
            auto observed=science::perceive(*renamed,named_source,true,row_options),erased_observed=science::perceive(*renamed_erased,empty,false,row_options);
            record["neighbor_renumbering_invariant"]=support_signatures(record["neighbor_orientation_retained"],mapping)==support_signatures(observed);
            record["erased_renumbering_invariant"]=support_signatures(record["configuration_erased"],mapping)==support_signatures(erased_observed);
            if (row.contains("independent_negative_witness")) {
              record["negative_proof"]=negative_proof(*m,source,row.at("independent_negative_witness"),record["configuration_erased"]);
              ++negative_checks; if (!record["negative_proof"]["passed"].get<bool>()) ++negative_failures;
            }
          }
          if (mode=="geometry") {
            Json configurations=Json::array();
            for (const auto &d:row.at("supports")) {
              try { configurations.push_back({{"complete",true},{"configuration",science::configuration_json(science::configure_geometry(*m,science::configuration_from_json(d)))}}); }
              catch (const std::invalid_argument &e) { configurations.push_back({{"complete",false},{"reason",e.what()}}); }
            }
            record["configurations"]=configurations;
          }
          if (mode=="cip" || mode=="pipeline") {
            auto at=coordinates.find(record["id"].get<std::string>());
            if (at!=coordinates.end()) {
              Json audit; auto additions=science::coordinate_assistance(*m,source,*at->second,row.value("units",std::vector<std::string>{}),audit,options);
              merge(source,additions,options); record["coordinate_assistance"]=audit;
            }
            auto assignments=science::assign_labels(*m,source,options);
            record["assignments"]=assignments; record["predicted"]=predicted(source,assignments);
            Json configurations=Json::array(); for (const auto &d:source.configurations) configurations.push_back(science::configuration_json(d)); record["configurations"]=configurations;
            if (row.contains("expected_labels")) {
              std::set<std::string> expected=row["expected_labels"].get<std::set<std::string>>();
              record["expected"]=expected; record["exact"]=record["predicted"]==Json(expected); if (record["exact"].get<bool>()) ++exact;
              auto got=record["predicted"].get<std::set<std::string>>(); expected_count+=expected.size(); predicted_count+=got.size();
              for (const auto &label:got) if (expected.count(label)) ++true_positives;
            }
          }
          record["complete"]=true; ++complete;
        }
      } catch (const std::exception &e) {
        record["complete"]=false; record["error"]=e.what();
        if (mode=="coordination") { record["status"]="refused"; ++refused; }
        else { record["status"]="error_or_resource_stop"; ++errors; }
      }
      if (!reference.empty()) {
        const auto found=historical.find(record["id"].get<std::string>());
        bool agrees=found!=historical.end();
        if (agrees) {
          const auto &r=found->second;
          if (mode=="cip") agrees=record.value("complete",false) && record.at("predicted")==r.at("predicted");
          else if (mode=="local") {
            Json observed=Json::array();
            if (record.value("complete",false)) for (const auto &c:record.at("local").at("carriers")) observed.push_back(Json::array({c.at("geometry"),c.at("raw_orders"),c.at("observed_classes")}));
            auto expected=r.at("local_expected"); std::sort(observed.begin(),observed.end()); std::sort(expected.begin(),expected.end());
            agrees=record.value("complete",false) && observed==expected && record.at("local").at("passed").get<bool>();
            record["historical_comparison_scope"]="per-family raw-order and global-class count regression";
            if (agrees && r.contains("local_raw_partitions")) {
              for (const auto &c:record.at("local").at("carriers")) {
                const auto &expected_raw=r.at("local_raw_partitions").at(c.at("locus").get<std::string>());
                auto drawings=local_arrangements(science::configuration_from_json(c.at("seed")));
                std::map<std::string,std::string> forward,backward; std::set<std::string> raw_keys;
                agrees=agrees && drawings.size()==expected_raw.size() && drawings.size()==c.at("partition").size();
                for (size_t i=0;i<drawings.size() && agrees;++i) {
                  auto key=science::configuration_json(drawings[i]).dump(); raw_keys.insert(key);
                  if (!expected_raw.contains(key)) { agrees=false; break; }
                  auto truth_label=expected_raw.at(key).dump(),observed_label=c.at("partition").at(i).dump();
                  auto a=forward.emplace(truth_label,observed_label); auto b=backward.emplace(observed_label,truth_label);
                  agrees=agrees && a.first->second==observed_label && b.first->second==truth_label;
                }
                agrees=agrees && raw_keys.size()==drawings.size();
              }
              record["independent_raw_partition_agreement"]=agrees;
              record["historical_comparison_scope"]="complete raw partitions, independently regenerated with proper spatial axial rotations";
            }
          }
          else if (mode=="perception") {
            auto retained=r.contains("neighbor_configuration_assisted_detected_carriers")?"neighbor_configuration_assisted_detected_carriers":"local_neighbor_orientation_retained_detected_carriers";
            agrees=record.value("complete",false) && support_signatures(record.at("configuration_erased"))==historical_signatures(r.at("detected_carriers")) &&
              support_signatures(record.at("neighbor_orientation_retained"))==historical_signatures(r.at(retained)) &&
              record.at("erased_renumbering_invariant").get<bool>() && record.at("neighbor_renumbering_invariant").get<bool>();
          } else if (mode=="coordination") {
            if (r.at("status")=="unresolved") agrees=!record.value("complete",false);
            else {
              agrees=record.value("complete",false) && (record.at("achiral").get<bool>()?"achiral":"chiral")==r.at("expected_mirror_relation").get<std::string>() && record.at("coordinate_reflection_agrees_with_algebra").get<bool>();
              if (agrees) for (auto [expected_field,observed_field]:std::vector<std::pair<std::string,std::string>>{{"reference_class","group_id"},{"reference_mirror_class","coordinate_mirror_class_id"}}) {
                std::string expected=Json::array({r.at("group_id"),r.at(expected_field)}).dump(); int observed=record.at(observed_field);
                auto a=reference_classes.emplace(expected,observed); auto b=observed_classes.emplace(observed,expected);
                agrees=agrees && a.first->second==observed && b.first->second==expected;
              }
            }
          } else if (mode=="identity") {
            auto key=r.contains("global_necessarily_chiral")?"combined_status":"source_declared";
            if (r.at("status")=="chiral" || r.at("status")=="achiral") {
              agrees=record.value("complete",false) && (std::string(key)=="combined_status"?record.at(key):record.at(key).at("status"))==r.at("status");
            }
          }
        }
        record["reference_agreement"]=agrees; ++reference_checks; if (!agrees) ++regression_failures;
      }
      record["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-begun).count();
      records.push_back(std::move(record));
    }
    Json report={{"schema","synkit.native-scientific-pipeline/2"},{"geometry_model","proper_spatial_rotations; axial V4; model 2"},{"mode",mode},{"build",Json::parse(synkit_native_build_info)},
      {"rdkit",RDKit::rdkitVersion},{"input",input.string()},{"coordinate_input",sdf.string()},{"reference",reference.string()},
      {"sdks",{{"rdkit",RDKit::rdkitVersion},
        {"eigen",std::to_string(EIGEN_WORLD_VERSION)+"."+std::to_string(EIGEN_MAJOR_VERSION)+"."+std::to_string(EIGEN_MINOR_VERSION)},
        {"json",std::to_string(NLOHMANN_JSON_VERSION_MAJOR)+"."+std::to_string(NLOHMANN_JSON_VERSION_MINOR)+"."+std::to_string(NLOHMANN_JSON_VERSION_PATCH)}}},
      {"algorithm","SynKit native C++ carrier stabilizers, sequence rules and coordinate projections"},
      {"target_labels_used_for_prediction",false},{"python_runtime",false},
      {"records",records},{"summary",{{"rows",rows.size()},{"complete",complete},{"errors",errors},{"refused",refused},{"exact_cip_records",exact},
        {"reference_checks",reference_checks},{"reference_failures",regression_failures},{"coordination_source_and_mirror_classes",coordination_classes.size()},
        {"local_carriers",local_carriers},{"local_raw_orders",local_raw_orders},{"local_failures",local_failures},
        {"negative_proofs_replayed",negative_checks},{"negative_failures",negative_failures},
        {"distinct_constitutions",constitutions.size()},{"representation_permutations",representation_checks},
        {"expected_labels",expected_count},{"predicted_labels",predicted_count},{"true_positive_labels",true_positives},
        {"missing_labels",expected_count-true_positives},{"wrong_labels",predicted_count-true_positives}}},
      {"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()}};
    std::ofstream out(output); if (!out) throw std::runtime_error("Cannot create output"); out<<report.dump(2)<<'\n';
    std::cout<<report["summary"].dump()<<'\n'; return errors?2:((regression_failures || local_failures || negative_failures)?3:0);
  } catch (const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
