// JSON is an optional SDK dependency of this executable, not of the core.
#include "synkit_canon/configured.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include "native_build_info.hpp"
#ifdef SYNKIT_STEREO_WITH_RDKIT
#include "rdkit_preparation.hpp"
#include <RDGeneral/versions.h>
#endif
using json=nlohmann::json;
using namespace synkit::stereo;
using Clock=std::chrono::steady_clock;
namespace {
double elapsed(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
std::string typed(const json &v) {
  // Positive signed/unsigned integers share a domain; floats/bools do not.
  return (v.is_number_integer()?"integer":std::string(v.type_name()))+":"+v.dump();
}
std::string hex(const std::vector<uint8_t> &v) {
  const char *d="0123456789abcdef"; std::string s;
  for (auto b:v) { s+=d[b>>4]; s+=d[b&15]; } return s;
}
json selected(const json &attrs,const json &selector) {
  if (selector.is_string()) return attrs.value(selector.get<std::string>(),json(nullptr));
  if (!selector.is_array()) throw std::invalid_argument("Attribute selector must be a key or key array");
  json out=json::array();
  for (const auto &key:selector) out.push_back({key,attrs.value(key.get<std::string>(),json(nullptr))});
  return out;
}
Stereograph load(const json &row) {
  if (!row.contains("nodes")) {
#ifdef SYNKIT_STEREO_WITH_RDKIT
    return prepare_rdkit_smiles(row.at("smiles").get<std::string>(),false).front();
#else
    throw std::invalid_argument("chemical_input_requires_preparation");
#endif
  }
  Stereograph g; std::map<int64_t,int32_t> ids;
  for (const auto &n:row.at("nodes")) {
    if (!ids.emplace(n.at(0).get<int64_t>(),static_cast<int32_t>(ids.size())).second)
      throw std::invalid_argument("Duplicate atom ID");
    g.colors.push_back(typed(selected(n.at(1),row.value("atom_color",json("color")))));
  }
  auto atom=[&](const json &v) { if (!v.is_number_integer()) throw std::invalid_argument("Noninteger atom ID"); return ids.at(v.get<int64_t>()); };
  auto ref=[&](const json &v) {
    if (v.is_number_integer()) return Reference(atom(v));
    const auto s=v.get<std::string>(); Resource kind; size_t offset;
    if (s.rfind("@H:",0)==0) { kind=Resource::hydrogen; offset=3; }
    else if (s.rfind("@LP:",0)==0) { kind=Resource::lone_pair; offset=4; }
    else throw std::invalid_argument("Unknown virtual reference");
    size_t used=0; int64_t owner=std::stoll(s.substr(offset),&used);
    if (offset+used!=s.size()) throw std::invalid_argument("Invalid virtual owner");
    return Reference(ids.at(owner),kind);
  };
  for (const auto &e:row.at("edges"))
    g.bonds.push_back({atom(e.at(0)),atom(e.at(1)),typed(selected(e.at(2),row.value("bond_color",json("color"))))});
  const std::vector<std::string> families={"tetrahedral","square_planar","trigonal_bipyramidal","octahedral",
    "planar_bond","atrop_bond","cumulene_axis","extended_cis_trans","helical","planar_chirality","framework"};
  for (const auto &s:row.at("states")) {
    Configuration d; auto name=s.at("family").get<std::string>();
    auto at=std::find(families.begin(),families.end(),name);
    if (at==families.end()) throw std::invalid_argument("Unsupported geometry");
    d.geometry=static_cast<Geometry>(1+at-families.begin());
    d.fixed=!s.at("parity").is_null(); if (d.fixed) d.parity=s.at("parity").get<int>();
    for (const auto &v:s.value("atoms",json::array())) d.frame.push_back(ref(v));
    for (const auto &v:s.value("path",json::array())) d.path.push_back(atom(v));
    for (const auto &v:s.value("support",json::array())) d.support.push_back(atom(v));
    d.cyclic=s.value("cyclic",false); d.coupling=s.value("coupling_id",json(nullptr)).dump();
    if (s.contains("pilot") && !s.at("pilot").is_null()) d.pilot=atom(s.at("pilot"));
    for (const auto &f:s.value("frames",json::array())) {
      if (f.at(1).size()!=4) throw std::invalid_argument("Invalid coupled frame");
      CoupledFrame frame; frame.center=atom(f.at(0)); frame.relation=f.at(2).get<int>();
      for (size_t i=0;i<4;++i) frame.references[i]=ref(f.at(1).at(i));
      d.frames.push_back(frame);
    }
    g.configurations.push_back(d);
  }
  // Reproduce the controlled study's explicitly injected unsupported registry;
  // exercise the native boundary rather than returning its expected answer.
  if (row.value("id",std::string())=="boundary:unsupported_registry") {
    Configuration unsupported; unsupported.geometry=static_cast<Geometry>(999);
    g.configurations.push_back(unsupported);
  }
  g.profile=row.value("profile",std::string("attributed_graph")); return g;
}
json analyze(const json &rows,const ConfiguredOptions &options,bool codes) {
  json records=json::array(); int complete=0,refused=0,stopped=0;
  std::map<std::pair<std::string,std::vector<uint8_t>>,std::string> class_of;
  std::map<std::string,std::vector<uint8_t>> code_of;
  bool partition_agreement=true,mirror_agreement=true,outcome_agreement=true;
  size_t requests=0;
  auto one=[&](const json &row) {
    ++requests;
    json r={{"id",row.at("id")}};
    try {
      auto opts=options; std::vector<Stereograph> drawings;
      const bool family=row.value("kind",std::string())=="resonance";
#ifdef SYNKIT_STEREO_WITH_RDKIT
      if (family) drawings=prepare_rdkit_smiles(row.at("smiles").get<std::string>(),true);
      else drawings.push_back(load(row));
#else
      drawings.push_back(load(row));
#endif
      auto &g=drawings.front();
      if (row.contains("max_search_nodes")) opts.max_nodes=row.at("max_search_nodes");
      auto c=family?canonicalize_family(drawings,opts):canonicalize(g,opts);
      r.update({{"complete",c.complete},{"reason",c.reason},{"preparation_seconds",c.preparation_seconds},
        {"total_seconds",c.total_seconds},{"search_seconds",c.statistics.search_seconds},
        {"nodes",c.statistics.nodes},{"key_bytes",c.code.size()},{"atom_order",c.atom_order}});
      if (codes && c.complete) r["code_hex"]=hex(c.code);
      if (c.complete) {
        ConfiguredMirrorResult m;
        if (family) {
          auto reflected=drawings; for (auto &drawing:reflected) drawing=mirror(drawing);
          auto opposite=canonicalize_family(reflected,opts);
          m.complete=opposite.complete; m.reason=opposite.reason; m.achiral=m.complete && c.code==opposite.code;
        } else m=classify_mirror(g,opts);
        r["resonance_drawings"]=drawings.size();
        r["mirror_complete"]=m.complete; r["achiral"]=m.complete?json(m.achiral):json(nullptr);
        r["mirror_mapping"]=m.mapping; ++complete;
        if (row.contains("expected_group")) {
          auto label=row.at("expected_group").dump();
          auto scope=row.value("case_id",std::string());
          const auto key=std::make_pair(scope,c.code);
          auto first=class_of.emplace(key,label); auto second=code_of.emplace(scope+":"+label,c.code);
          bool agrees=first.first->second==label && second.first->second==c.code;
          r["partition_agreement"]=agrees; partition_agreement=partition_agreement && agrees;
        }
        if (row.contains("expected_achiral")) {
          bool agrees=m.complete && m.achiral==row.at("expected_achiral").get<bool>();
          r["mirror_agreement"]=agrees; mirror_agreement=mirror_agreement && agrees;
        }
      } else ++stopped;
    } catch (const std::invalid_argument &e) { r["complete"]=false; r["refusal"]=e.what(); ++refused; }
    const auto expected=row.value("expected",std::string("complete"));
    bool agrees=false;
    if (expected=="complete") agrees=r.value("complete",false) && r.value("mirror_complete",false);
    else if (expected=="bounded_search") agrees=!r.value("complete",false) && r.value("reason",0)==2;
    else if (expected=="incomplete_input") agrees=r.value("refusal",std::string())=="Unspecified configuration";
    else if (expected=="unsupported_input") agrees=r.value("refusal",std::string())=="Unsupported configured geometry";
    r["expected_outcome_agreement"]=agrees; outcome_agreement=outcome_agreement && agrees;
    records.push_back(r);
  };
  for (const auto &row:rows) {
    if (!row.contains("assignments")) one(row);
    else {
      auto base=row; base.erase("assignments");
      for (const auto &assignment:row.at("assignments")) {
        auto supplied=base; supplied.update(assignment); one(supplied);
      }
    }
  }
  return {{"records",records},{"completed",complete},{"refused",refused},{"stopped",stopped},
    {"analysis_requests",requests},{"all_inputs_completed",complete==static_cast<int>(requests)},
    {"partition_agreement",partition_agreement},{"mirror_agreement",mirror_agreement},
    {"expected_outcome_agreement",outcome_agreement}};
}
// Constitutional 1-WL invariant for candidate filtering. Palette compression is
// sorted by exact signatures, so separately processed isomorphic graphs agree.
// Stereo is deliberately omitted: exact compare remains the final authority.
std::string wl_signature(const Stereograph &g) {
  const size_t n=g.colors.size();
  std::vector<std::vector<std::pair<int,std::string>>> adj(n);
  for (const auto &b:g.bonds) {
    adj.at(b.source).push_back({b.target,b.color});
    adj.at(b.target).push_back({b.source,b.color});
  }
  std::map<std::string,int> palette;
  for (const auto &c:g.colors) palette.emplace(c,0);
  int next=0; for (auto &entry:palette) entry.second=next++;
  std::vector<int> colors; for (const auto &c:g.colors) colors.push_back(palette.at(c));
  json initial=json::array(); for (const auto &entry:palette) initial.push_back(entry.first);
  json stable=json::array(); size_t previous=palette.size();
  while (true) {
    std::vector<std::string> signatures; std::map<std::string,int> refined;
    for (size_t v=0;v<n;++v) {
      std::vector<std::pair<std::string,int>> neighbors;
      for (const auto &edge:adj[v]) neighbors.push_back({edge.second,colors.at(edge.first)});
      std::sort(neighbors.begin(),neighbors.end());
      auto key=json::array({colors[v],neighbors}).dump();
      signatures.push_back(key); refined.emplace(key,0);
    }
    next=0; for (auto &entry:refined) entry.second=next++;
    for (size_t v=0;v<n;++v) colors[v]=refined.at(signatures[v]);
    std::sort(signatures.begin(),signatures.end()); stable=signatures;
    if (refined.size()==previous) break;
    previous=refined.size();
  }
  return json::array({g.profile,n,g.bonds.size(),initial,stable}).dump();
}
json registration(const json &rows,const ConfiguredOptions &opts,int repetitions,bool use_wl_study) {
  json records=json::array(); bool passed=true;
  const std::vector<size_t> sizes=rows.size()>128?std::vector<size_t>{128,1280,12800}:std::vector<size_t>{8,16,32,64,128};
  for (size_t n:sizes) {
    if (n>rows.size()) continue;
    std::map<std::string,int> expected; std::vector<int> truth;
    for (size_t i=0;i<n;++i) truth.push_back(expected.emplace(rows.at(i).at("expected_group").dump(),static_cast<int>(expected.size())).first->second);
    for (int rep=0;rep<repetitions;++rep) for (int step=0;step<(use_wl_study?3:2);++step) {
      const int method=(rep+step)%(use_wl_study?3:2);
      bool pairwise=method!=1, prefilter=method==2; auto start=Clock::now();
      std::vector<Stereograph> graphs; for (size_t i=0;i<n;++i) graphs.push_back(load(rows.at(i)));
      double preparation=elapsed(start); auto compute=Clock::now();
      std::vector<int> labels; std::vector<size_t> representatives;
      std::map<std::vector<uint8_t>,int> index;
      int64_t calls=0,key_bytes=0,skipped=0; bool complete=true;
      std::vector<std::string> signatures;
      auto wl_started=Clock::now();
      if (prefilter) for (const auto &g:graphs) signatures.push_back(wl_signature(g));
      const double wl_seconds=prefilter?elapsed(wl_started):0;
      for (size_t i=0;i<n && complete;++i) {
        if (pairwise) {
          int label=-1;
          for (size_t j=0;j<representatives.size();++j) {
            if (prefilter && signatures[i]!=signatures[representatives[j]]) { ++skipped; continue; }
            ++calls; auto c=compare(graphs[i],graphs[representatives[j]],opts);
            if (!c.complete()) { complete=false; break; }
            if (c.isomorphic()) { label=static_cast<int>(j); break; }
          }
          if (label<0) { label=static_cast<int>(representatives.size()); representatives.push_back(i); }
          labels.push_back(label);
        } else {
          ++calls; auto c=canonicalize(graphs[i],opts);
          if (!c.complete) { complete=false; break; }
          key_bytes+=static_cast<int64_t>(c.code.size());
          labels.push_back(index.emplace(std::move(c.code),static_cast<int>(index.size())).first->second);
        }
      }
      const double computation=elapsed(compute),total=elapsed(start);
      bool correct=complete && labels==truth; passed=passed && correct;
      records.push_back({{"size",n},{"repetition",rep},{"method",prefilter?"wl_pairwise":(pairwise?"pairwise":"certificates")},
        {"complete",complete},{"partition_agreement",correct},{"preparation_seconds",preparation},
        {"group_labels",labels},
        {"computation_and_grouping_seconds",computation},{"total_seconds",total},{"calls",calls},{"wl_skipped",skipped},{"wl_seconds",wl_seconds},{"key_bytes",key_bytes}});
    }
  }
  return {{"records",records},{"passed",passed},
    {"protocol","from scratch; representative comparisons; exact in-memory key index; distinct from historical disk index"}};
}
json costs(const json &rows,const ConfiguredOptions &options,int repetitions) {
  struct Stopped { int reason; };
  json records=json::array();
  for (const auto &row:rows) for (int rep=0;rep<repetitions;++rep) for (int mode=0;mode<2;++mode) {
    auto start=Clock::now();
    json r={{"id",row.at("id")},{"repetition",rep},{"mode",mode?"full_group":"generators"},
      {"expected",row.value("expected",std::string("complete"))}};
    auto remaining=[&]() {
      auto opts=options;
      if (opts.seconds>=0) { opts.seconds-=elapsed(start); if (opts.seconds<=0) throw Stopped{1}; }
      if (row.contains("max_search_nodes")) opts.max_nodes=row.at("max_search_nodes");
      return opts;
    };
    try {
      std::vector<Stereograph> drawings;
      bool family=row.value("kind",std::string())=="resonance";
#ifdef SYNKIT_STEREO_WITH_RDKIT
      if (family) drawings=prepare_rdkit_smiles(row.at("smiles").get<std::string>(),true);
      else drawings.push_back(load(row));
#else
      drawings.push_back(load(row));
#endif
      r["preparation_seconds"]=elapsed(start); auto phase=Clock::now();
      auto c=family?canonicalize_family(drawings,remaining()):canonicalize(drawings.front(),remaining());
      r["canonicalization_seconds"]=elapsed(phase); r["certificate_bytes"]=c.code.size();
      r["search_nodes"]=c.statistics.nodes;
      if (!c.complete) throw Stopped{c.reason};
      phase=Clock::now(); int64_t generators=0,automorphisms=0; json orders=json::array();
      for (const auto &g:drawings) {
        auto group=symmetry(g,mode!=0,remaining());
        if (!group.complete) throw Stopped{group.reason};
        generators+=static_cast<int64_t>(group.generators.size());
        automorphisms+=static_cast<int64_t>(group.automorphisms.size()); orders.push_back(group.group_order);
      }
      r["symmetry_seconds"]=elapsed(phase); r["generators_returned"]=generators;
      r["automorphisms_returned"]=automorphisms; r["group_orders"]=orders; r["status"]="complete";
    } catch (const Stopped &s) { r["status"]="resource_stop"; r["reason"]=s.reason; }
    catch (const std::invalid_argument &e) { r["status"]="refused"; r["refusal"]=e.what(); }
    r["total_seconds"]=elapsed(start); records.push_back(r);
    std::cout<<json({{"id",r["id"]},{"mode",r["mode"]},{"status",r["status"]}}).dump()<<std::endl;
  }
  return {{"records",records},{"attempted_trials",records.size()},
    {"symmetry_scope","complete source-atom group per drawing; full vectors constructed then discarded; not auxiliary-graph groups"},
    {"protocol","one sequential native process; shared per-trial deadline; core workspace cap; no process RSS cap or historical pairwise diagnostics"}};
}
} // namespace
int main(int argc,char **argv) {
  try {
    std::string input,output,mode="analyze"; int reps=3; bool codes=false;
    ConfiguredOptions opts; opts.seconds=5; opts.workspace_bytes=268435456; opts.output_bytes=67108864;
    for (int i=1;i<argc;++i) {
      std::string flag=argv[i]; if (flag=="--codes") { codes=true; continue; }
      if (i+1==argc) throw std::invalid_argument("Missing option value");
      std::string value=argv[++i];
      if (flag=="--input") input=value; else if (flag=="--output") output=value;
      else if (flag=="--mode") mode=value; else if (flag=="--repetitions") reps=std::stoi(value);
      else if (flag=="--seconds") opts.seconds=std::stod(value); else throw std::invalid_argument("Unknown option");
    }
    if (input.empty() || output.empty() || reps<=0 || (mode!="analyze" && mode!="registration" && mode!="registration_wl" && mode!="costs"))
      throw std::invalid_argument("Usage: stereo_experiments --input FILE --output NEW_FILE --mode analyze|registration|costs [--codes] [--repetitions N] [--seconds N]");
    if (input==output || std::ifstream(output).good()) throw std::invalid_argument("Output must be a new file");
    std::ifstream stream(input); if (!stream) throw std::invalid_argument("Cannot open input");
    json input_data=json::parse(stream);
    const auto &rows=input_data.contains("rows")?input_data.at("rows"):input_data.at("cases");
    if (!rows.is_array() || rows.empty() || ((mode=="registration" || mode=="registration_wl") && rows.size()<8))
      throw std::invalid_argument("The selected dataset is empty or too small");
    auto start=Clock::now(); json report=(mode=="registration" || mode=="registration_wl")?registration(rows,opts,reps,mode=="registration_wl"):
      (mode=="costs"?costs(rows,opts,reps):analyze(rows,opts,codes));
    report.update({{"schema","synkit.cpp-prepared-stereo-experiments/2"},{"geometry_model","proper_spatial_rotations; axial V4; model 2"},{"mode",mode},
      {"input_path",input},{"input_records",rows.size()},{"wall_seconds",elapsed(start)},
      {"python_runtime",false},{"seconds_per_kernel",opts.seconds},{"workspace_bytes_per_kernel",opts.workspace_bytes},
      {"identity_scope","source-declared configured graphs and optional declared drawing families; no completeness/perception/CIP inference"}});
    report["build"]=json::parse(synkit_native_build_info);
    report["json_sdk_version"]=std::to_string(NLOHMANN_JSON_VERSION_MAJOR)+"."+
      std::to_string(NLOHMANN_JSON_VERSION_MINOR)+"."+std::to_string(NLOHMANN_JSON_VERSION_PATCH);
#ifdef SYNKIT_STEREO_WITH_RDKIT
    report["rdkit_version"]=RDKit::rdkitVersion;
#else
    report["rdkit_version"]=nullptr;
#endif
    std::ofstream destination(output); if (!destination) throw std::runtime_error("Cannot write output");
    destination<<report.dump(2)<<'\n'; if (!destination) throw std::runtime_error("Report write failed");
    std::cout<<"Wrote "<<output<<'\n';
    bool failed=(mode=="registration" || mode=="registration_wl")?!report.at("passed").get<bool>():
      (mode=="analyze" && (!report.at("partition_agreement").get<bool>() || !report.at("mirror_agreement").get<bool>() ||
        !report.at("expected_outcome_agreement").get<bool>()));
    return failed?2:0;
  } catch (const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
