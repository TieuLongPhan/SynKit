#include "synkit_canon/science.hpp"
#include "rdkit_preparation.hpp"
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Conformer.h>
#include <GraphMol/MolOps.h>
#include <iostream>
using namespace synkit;
int main() {
  try {
    int checks=0;
    auto require=[&](bool condition,const char *message) { if (!condition) throw std::runtime_error(message); ++checks; };
    std::unique_ptr<RDKit::RWMol> mol(RDKit::SmilesToMol("F[C@](Cl)(Br)I"));
    auto graph=prepare_rdkit_molecule(*mol); auto labels=science::assign_labels(*mol,graph);
    require(labels.size()==1 && labels[0]["label"]=="S","Native tetrahedral CIP label");
    auto mirror=stereo::mirror(graph);
    require(science::assign_labels(*mol,mirror)[0]["label"]=="R","Mirror reverses CIP");
    mol->getAtomWithIdx(1)->setProp("_CIPCode",std::string("false-label"));
    mol->getAtomWithIdx(0)->setProp("_CIPRank",999);
    require(science::assign_labels(*mol,graph)==labels,"CIP does not trust toolkit CIP properties");
    require(science::perceive(*mol,graph,false)["elements"].size()==1,"Four distinct ligands confirmed");
    std::unique_ptr<RDKit::RWMol> duplicate(RDKit::SmilesToMol("FC(F)(Br)I"));
    require(science::perceive(*duplicate,prepare_rdkit_molecule(*duplicate),false)["elements"].empty(),"Odd stabilizer rejects duplicate ligands");
    RDKit::SmilesParserParams raw_params; raw_params.sanitize=false; raw_params.removeHs=false;
    std::unique_ptr<RDKit::RWMol> cage(RDKit::SmilesToMol("C1[C@H]2C[C@H]3C[C@@H]1C[C@@H](C2)C3",raw_params));
    unsigned failed=0;
    RDKit::MolOps::sanitizeMol(*cage,failed,RDKit::MolOps::SANITIZE_ALL & ~RDKit::MolOps::SANITIZE_CLEANUPCHIRALITY);
    auto supplied=prepare_rdkit_molecule(*cage);
    require(science::perceive(*cage,supplied,false)["elements"].empty(),"Cage constitution alone has odd stabilizers");
    require(science::perceive(*cage,supplied,true)["elements"].size()==4,"Retained neighboring orientations constrain cage stabilizers");
    std::unique_ptr<RDKit::RWMol> imine(RDKit::SmilesToMol("O=C(C1=CC=NC=C1)N/N=C\\2/[C@@H]3[C@@H](C2)C=CC3"));
    require(science::source_mirror_identity(*imine).at("complete"),"Resonance drawings reuse source registry without inferring new local frames");
    bool refused=false;
    try { auto invalid=graph; invalid.configurations[0].frame.clear(); science::assign_labels(*mol,invalid); }
    catch (const std::invalid_argument &) { refused=true; }
    require(refused,"Malformed CIP frames are refused before iterator access");
    refused=false;
    try { science::Options stopped; stopped.seconds=0; science::assign_labels(*mol,graph,stopped); }
    catch (const std::runtime_error &) { refused=true; }
    require(refused,"CIP exhaustion returns no fabricated assignment");
    std::vector<std::array<double,3>> ideal{{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    auto fit=science::fit_octahedron(ideal);
    require(fit["complete"] && fit["proper_rotation_count"]==24 && fit["inequivalent_pose_count"]==30,"Octahedral independent pose fit");
    auto coplanar=ideal; coplanar[4]={1,1,0}; coplanar[5]={-1,-1,0};
    require(!science::fit_octahedron(coplanar)["complete"].get<bool>(),"Ambiguous or distorted octahedron refuses certification");
    stereo::Configuration support; support.frame={1};
    auto conf=new RDKit::Conformer(mol->getNumAtoms());
    conf->setAtomPos(1,RDGeom::Point3D(0,0,0));
    conf->setAtomPos(0,RDGeom::Point3D(1,1,1)); conf->setAtomPos(2,RDGeom::Point3D(1,-1,-1));
    conf->setAtomPos(3,RDGeom::Point3D(-1,1,-1)); conf->setAtomPos(4,RDGeom::Point3D(-1,-1,1));
    mol->addConformer(conf,true);
    auto oriented=science::configure_geometry(*mol,support);
    for (unsigned i=0;i<mol->getNumAtoms();++i) { auto p=conf->getAtomPos(i); p.x=-p.x; conf->setAtomPos(i,p); }
    require(science::configure_geometry(*mol,support).parity==-oriented.parity,"Coordinate reflection reverses tetrahedral orientation");
    for (const auto &smiles:{"FC(Cl)=C=C(F)Cl","FC(Cl)=C=C(Br)I"}) {
      std::unique_ptr<RDKit::RWMol> allene(RDKit::SmilesToMol(smiles));
      auto coordinates=new RDKit::Conformer(7);
      const double xyz[7][3]={{1,0,-1},{0,0,-1},{-1,0,-1},{0,0,0},{0,0,1},{0,1,1},{0,-1,1}};
      for (int i=0;i<7;++i) coordinates->setAtomPos(i,RDGeom::Point3D(xyz[i][0],xyz[i][1],xyz[i][2]));
      allene->addConformer(coordinates,true);
      stereo::Configuration axis; axis.geometry=stereo::Geometry::cumulene_axis; axis.path={1,3,4}; axis.frame={0,2,1,4,5,6};
      auto configured=science::configure_geometry(*allene,axis); auto base=prepare_rdkit_molecule(*allene); base.configurations={configured};
      require(!stereo::classify_mirror(base).achiral,"Allene with two distinguishable ligands at each end is chiral even when the ends are equivalent");
      require(science::assign_labels(*allene,base)[0]["label"]=="P","Symmetric allene has witnessed axial helicity without selecting a preferred end");
      for (size_t i=0;i<base.colors.size();++i) base.colors[i]=science::Json::array({base.colors[i],i}).dump();
      const auto code=stereo::canonicalize(base).code;
      for (auto drawing:stereo::local_arrangements(axis)) {
        auto projected=base; projected.configurations={science::configure_geometry(*allene,drawing)};
        require(stereo::canonicalize(projected).code==code,"Axis end/pair reordering must preserve the same coordinate configuration");
        require(science::assign_labels(*allene,projected)[0]["label"]=="P","Axial CIP label must not depend on viewing end or terminal order");
      }
    }
    std::cout<<checks<<" native science checks passed\n"; return 0;
  } catch (const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
