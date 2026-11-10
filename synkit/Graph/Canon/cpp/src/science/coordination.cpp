#include "synkit_canon/science.hpp"
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/MolOps.h>
#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <set>

namespace synkit { namespace science {
namespace {
using namespace stereo;
std::unique_ptr<RDKit::RWMol> source_mol(const std::string &smiles) {
  RDKit::SmilesParserParams params; params.sanitize=false; params.removeHs=false;
  std::unique_ptr<RDKit::RWMol> m(RDKit::SmilesToMol(smiles,params)); if (!m) throw std::invalid_argument("source_smiles_parse_failure");
  unsigned failed=0; RDKit::MolOps::sanitizeMol(*m,failed,RDKit::MolOps::SANITIZE_ALL & ~RDKit::MolOps::SANITIZE_CLEANUPCHIRALITY); return m;
}
Json atom_properties(const RDKit::Atom &a) { return Json::array({a.getAtomicNum(),a.getIsotope(),a.getFormalCharge(),a.getIsAromatic(),a.getNumRadicalElectrons()}); }
std::string tag(const RDKit::Atom &a) {
  if (a.getChiralTag()==RDKit::Atom::CHI_TETRAHEDRAL_CW) return "CHI_TETRAHEDRAL_CW";
  if (a.getChiralTag()==RDKit::Atom::CHI_TETRAHEDRAL_CCW) return "CHI_TETRAHEDRAL_CCW";
  return a.getChiralTag()==RDKit::Atom::CHI_UNSPECIFIED?"CHI_UNSPECIFIED":"UNSUPPORTED_TAG";
}
} // namespace
CoordinationInput read_coordination(const Json &record,const std::string &root,const Options &) {
  using namespace stereo; using Vector=Eigen::Vector3d;
  auto path=std::filesystem::path(root)/record.at("path").get<std::string>();
  std::ifstream file(path,std::ios::binary); if (!file) throw std::invalid_argument("source_xyz_absent");
  std::string raw((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());
  std::istringstream stream(raw); std::string line; std::getline(stream,line); int count=std::stoi(line);
  if (count<1 || count>100000) throw std::invalid_argument("Invalid XYZ count");
  std::getline(stream,line); const auto metadata=Json::parse(line);
  std::vector<std::string> elements; std::vector<Vector> points;
  for (int i=0;i<count;++i) {
    if (!std::getline(stream,line)) throw std::invalid_argument("first_xyz_conformer_incomplete");
    std::istringstream entry(line); std::string symbol,extra; double x,y,z;
    if (!(entry>>symbol>>x>>y>>z) || entry>>extra) throw std::invalid_argument("Invalid XYZ row");
    Vector p(x,y,z); if (!p.allFinite()) throw std::invalid_argument("Nonfinite XYZ coordinate"); elements.push_back(symbol); points.push_back(p);
  }
  auto m=source_mol(metadata.at("smiles")),full=source_mol(metadata.at("smiles3D"));
  auto map_rows=[&](const RDKit::ROMol &mol) {
    std::map<int,int> rows;
    for (auto a:mol.atoms()) { int r=a->getAtomMapNum(); if (r<0 || r>=count || a->getSymbol()!=elements[r] || !rows.emplace(r,a->getIdx()).second) throw std::invalid_argument("source_coordinate_map_not_injective_or_element_mismatch"); }
    return rows;
  };
  auto rows=map_rows(*m),full_rows=map_rows(*full);
  if (full_rows.size()!=static_cast<size_t>(count)) throw std::invalid_argument("all_coordinate_rows_need_source_atom_witnesses");
  for (const auto &[row,index]:rows) {
    auto a=m->getAtomWithIdx(index),b=full->getAtomWithIdx(full_rows.at(row));
    if (atom_properties(*a)!=atom_properties(*b)) throw std::invalid_argument("coordinate_export_atom_attributes_disagree");
    std::set<int> compact,material,hidden;
    for (auto v:m->atomNeighbors(a)) compact.insert(v->getAtomMapNum());
    for (auto v:full->atomNeighbors(b)) { if (rows.count(v->getAtomMapNum())) material.insert(v->getAtomMapNum()); else { hidden.insert(v->getAtomMapNum()); if (v->getAtomicNum()!=1) throw std::invalid_argument("coordinate_export_hidden_hydrogen_count_disagrees"); } }
    if (compact!=material || hidden.size()!=a->getTotalNumHs()) throw std::invalid_argument("coordinate_export_connectivity_or_hydrogen_disagrees");
    for (auto bond:m->atomBonds(a)) {
      auto other=m->getAtomWithIdx(bond->getOtherAtomIdx(index))->getAtomMapNum();
      auto matching=full->getBondBetweenAtoms(full_rows.at(row),full_rows.at(other));
      if (!matching || matching->getBondType()!=bond->getBondType() || matching->getIsAromatic()!=bond->getIsAromatic()) throw std::invalid_argument("coordinate_export_bond_attributes_disagree");
    }
  }
  const auto &labels=record.at("source_configuration_labels"); std::vector<int> donors,slots; std::set<int> donor_set;
  for (const auto &label:labels) {
    int row=label.at("source_coordinate_index"),slot=label.at("source_slot_label"); donors.push_back(row); slots.push_back(slot); donor_set.insert(row);
    if (!rows.count(row)) throw std::invalid_argument("source_donor_label_mismatch");
    auto a=m->getAtomWithIdx(rows.at(row)); if (a->getSymbol()!=label.at("element").get<std::string>() || static_cast<int>(a->getIsotope())!=slot) throw std::invalid_argument("source_donor_label_mismatch");
  }
  std::sort(slots.begin(),slots.end());
  if (donors.size()!=6 || donor_set.size()!=6 || slots!=std::vector<int>({1,2,3,4,5,6})) throw std::invalid_argument("source_donor_inventory_not_six_unique_slots");
  std::vector<int> centers; for (const auto &[row,index]:rows) { auto symbol=m->getAtomWithIdx(index)->getSymbol(); if (symbol=="Mn" || symbol=="Ru") centers.push_back(row); }
  if (centers.size()!=1) throw std::invalid_argument("coordination_input_needs_one_metal_center"); int center=centers[0];
  std::set<int> metal_neighbors; for (auto a:m->atomNeighbors(m->getAtomWithIdx(rows.at(center)))) metal_neighbors.insert(a->getAtomMapNum());
  if (metal_neighbors!=donor_set) throw std::invalid_argument("supplied_donors_do_not_equal_metal_neighbors");
  std::vector<std::array<double,3>> vectors,reflected;
  for (int r:donors) { auto v=(points[r]-points[center]).eval(); vectors.push_back({v.x(),v.y(),v.z()}); reflected.push_back({-v.x(),v.y(),v.z()}); }
  auto fit=fit_octahedron(vectors),mirror_fit=fit_octahedron(reflected);
  if (!fit.at("complete").get<bool>() || !mirror_fit.at("complete").get<bool>())
    throw std::invalid_argument("oh_geometry_outside_frozen_limits:rmsd="+std::to_string(fit.at("normalized_rmsd").get<double>())+",gap="+std::to_string(fit.at("next_inequivalent_pose_gap").get<double>()));
  CoordinationInput result; auto &g=result.graph; g.profile="mace/source/chemical-with-donor-slot-isotopes-erased/1";
  for (auto a:m->atoms()) g.colors.push_back(Json::array({a->getAtomicNum(),donor_set.count(a->getAtomMapNum())?0:a->getIsotope(),a->getFormalCharge(),a->getTotalNumHs(),a->getIsAromatic(),a->getNumRadicalElectrons()}).dump());
  for (auto b:m->bonds()) {
    if (b->getStereo()!=RDKit::Bond::STEREONONE) throw std::invalid_argument("source_bond_orientation_not_in_frozen_inventory");
    auto symbol=b->getEndAtom()->getSymbol(); if (b->getBondType()==RDKit::Bond::DATIVE && symbol!="Mn" && symbol!="Ru") throw std::invalid_argument("dative_direction_not_recoverable_from_declared_types");
    g.bonds.push_back({static_cast<int>(b->getBeginAtomIdx()),static_cast<int>(b->getEndAtomIdx()),Json::array({static_cast<int>(b->getBondType()),b->getIsAromatic()}).dump()});
  }
  const std::array<int,6> basis{4,5,0,2,1,3};
  auto oh=[&](const Json &f) { Configuration d; d.geometry=Geometry::octahedral; d.frame={rows.at(center)}; for (int i:basis) d.frame.emplace_back(rows.at(donors.at(f.at("slot_to_donor").at(i).get<int>()))); return d; };
  g.configurations={oh(fit)}; result.reflected=g; result.reflected.configurations={oh(mirror_fit)};
  std::map<int,std::string> actual,expected;
  for (auto a:m->atoms()) if (a->getChiralTag()!=RDKit::Atom::CHI_UNSPECIFIED) actual[a->getAtomMapNum()]=tag(*a);
  for (const auto &annotation:record.at("source_local_orientation_annotations")) expected[annotation.at("source_coordinate_index")]=annotation.at("source_chiral_tag");
  if (actual!=expected) throw std::invalid_argument("source_local_frame_inventory_mismatch");
  Json tetra=Json::array();
  for (const auto &annotation:record.at("source_local_orientation_annotations")) {
    int row=annotation.at("source_coordinate_index"); auto a=m->getAtomWithIdx(rows.at(row));
    if (a->getSymbol()!=annotation.at("element").get<std::string>()) throw std::invalid_argument("source_annotation_element_mismatch");
    std::vector<int> coordinate_refs; Configuration d; d.frame={Reference(a->getIdx())};
    for (auto v:m->atomNeighbors(a)) { coordinate_refs.push_back(v->getAtomMapNum()); d.frame.emplace_back(v->getIdx()); }
    if (coordinate_refs.size()==3 && a->getTotalNumHs()==1) {
      std::vector<int> hydrogen;
      for (auto v:full->atomNeighbors(full->getAtomWithIdx(full_rows.at(row)))) if (v->getAtomicNum()==1 && !rows.count(v->getAtomMapNum())) hydrogen.push_back(v->getAtomMapNum());
      if (hydrogen.size()!=1) throw std::invalid_argument("owner_hydrogen_coordinate_witness_not_unique");
      coordinate_refs.push_back(hydrogen[0]); d.frame.emplace_back(a->getIdx(),Resource::hydrogen);
    }
    if (coordinate_refs.size()!=4) throw std::invalid_argument("source_tetrahedral_frame_does_not_have_four_ports");
    Vector x=points[coordinate_refs[0]]-points[coordinate_refs[3]],y=points[coordinate_refs[1]]-points[coordinate_refs[3]],z=points[coordinate_refs[2]]-points[coordinate_refs[3]];
    double volume=x.dot(y.cross(z)),scale=x.norm()*y.norm()*z.norm();
    if (scale<=1e-12 || std::abs(volume)/scale<=1e-6) throw std::invalid_argument("degenerate_tetrahedral_coordinate_witness");
    d.parity=volume>0?-1:1;
    if (d.parity!=(actual.at(row)=="CHI_TETRAHEDRAL_CW"?1:-1)) throw std::invalid_argument("source_tetrahedral_tag_disagrees_with_coordinate_volume");
    g.configurations.push_back(d); d.parity=-d.parity; result.reflected.configurations.push_back(d);
    tetra.push_back({{"center_coordinate_row",row},{"reference_coordinate_rows",coordinate_refs},{"signed_volume",volume},{"parity",-d.parity}});
  }
  result.audit={{"id",record.at("id")},{"geometry",fit},{"reflected_geometry",mirror_fit},{"tetrahedral_annotations",tetra},{"supplied_local_frames_retained",tetra.size()},{"coordinate_row_to_atom",rows}};
  return result;
}
}} // namespace
