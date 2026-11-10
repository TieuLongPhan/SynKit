#include "rdkit_preparation.hpp"
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <GraphMol/MolOps.h>
#include <iostream>
#include <numeric>
#include <set>
using namespace synkit::stereo;
int main() {
  try {
    int checks=0;
    struct Case { const char *center,*shape,*branches; int positions; };
    const Case cases[]={{"Pt","SP","(F)(Cl)(Br)I",3},{"As","TB","(F)(Cl)(Br)(I)N",20},
                        {"Co","OH","(F)(Cl)(Br)(I)(N)P",30}};
    for (const auto &c:cases) {
      std::set<std::vector<uint8_t>> keys;
      for (int i=1;i<=c.positions;++i) {
        std::string smiles="["+std::string(c.center)+"@"+c.shape+std::to_string(i)+"]"+c.branches;
        auto drawings=prepare_rdkit_smiles(smiles,false);
        auto result=canonicalize(drawings.at(0));
        if (!result.complete) throw std::runtime_error("Native chemistry canonicalization stopped");
        keys.insert(result.code);
        std::unique_ptr<RDKit::RWMol> mol(RDKit::SmilesToMol(smiles));
        std::vector<unsigned> order(mol->getNumAtoms()); std::iota(order.rbegin(),order.rend(),0);
        std::unique_ptr<RDKit::ROMol> renamed(RDKit::MolOps::renumberAtoms(*mol,order));
        RDKit::SmilesWriteParams params; params.canonical=false;
        auto copy=prepare_rdkit_smiles(RDKit::MolToSmiles(*renamed,params),false);
        if (canonicalize(copy.at(0)).code!=result.code) throw std::runtime_error("RDKit neighbor transport changed identity");
        ++checks;
      }
      if (keys.size()!=static_cast<size_t>(c.positions)) throw std::runtime_error("Coordination positions merged");
      ++checks;
    }
    auto phosphate=prepare_rdkit_smiles("COP(=O)(N)[O-]",true);
    if (phosphate.size()!=2) throw std::runtime_error("Declared phosphate closure differs");
    auto reversed=phosphate; std::reverse(reversed.begin(),reversed.end()); reversed.push_back(phosphate.front());
    if (canonicalize_family(phosphate).code!=canonicalize_family(reversed).code)
      throw std::runtime_error("Family key depends on order/multiplicity");
    checks+=2;
    std::cout<<checks<<" native chemistry checks passed\n"; return 0;
  } catch (const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
