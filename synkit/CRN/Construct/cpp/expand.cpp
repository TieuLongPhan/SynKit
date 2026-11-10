// Native rule-based CRN expansion over RDKit reaction SMARTS.
//
// The controller follows the derivation-graph strategy of MOD (Andersen et al.,
// "Generic strategies for chemical space exploration", 2014): a universe of
// canonical molecules, a subset of new molecules that every rule application
// must touch, and incremental binding of molecules to rule reactant templates.
// Binding a molecule to a template slot is tested once per (molecule, rule);
// a mixture is only formed from molecules that bind to some assignment of the
// slots, so incompatible mixtures are never passed to the reaction engine.
//
// Semantics, ordering and deduplication reproduce synkit.CRN.Construct.CRNExpand
// with rule_backend="rdkit": the same mixtures are tried in the same order, and
// the same reaction events are emitted. The only intended difference is that the
// per-rule mixture cap counts compatible mixtures instead of all enumerated ones.
//
// Hydrogen models. "explicit" adds hydrogen atoms to every molecule and matches
// the rule's [H:n] atoms as graph vertices (the RDKit worker's model). "lewis"
// treats hydrogen as a per-atom count label of a Lewis-state graph: each mapped
// H atom of a rule is compiled into a hydrogen-count delta (-1 on its donor, +1
// on its acceptor) and a minimum hydrogen count on the donor. Molecules are
// matched as heavy-atom graphs and product hydrogen counts are set exactly from
// the reactant counts plus the deltas, never re-guessed from default valences.
// A product is accepted only when every atom whose hydrogen count changed keeps
// a non-negative, integral lone-pair count. Rules that cannot be compiled (an H
// that is unmapped, created, released, or read by degree-sensitive queries) run
// under the explicit model, and the result records why.
//
// Protocol: one JSON request on stdin, one JSON result on stdout.
#include <GraphMol/ChemReactions/Reaction.h>
#include <GraphMol/ChemReactions/ReactionParser.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/PeriodicTable.h>
#include <GraphMol/QueryAtom.h>
#include <GraphMol/QueryOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <RDGeneral/RDLog.h>
#include <RDGeneral/versions.h>

#include <boost/json.hpp>
#include <boost/json/src.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace json = boost::json;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int kProtocol = 1;

struct LimitReached {
  std::string reason;
};

struct Options {
  int repeats = 50;
  bool explicit_h = false;
  int max_components = 3;
  bool use_frontier = true;
  std::int64_t max_mixtures_per_rule_step = 50000;
  std::int64_t max_tasks_per_step = 200000;
  bool allow_self_mixtures = false;
  bool skip_no_change = true;
  bool allow_empty_side = false;
  bool strip_spectators = false;
  bool dedup_delta = true;
  bool dedup_across_rules = false;
  unsigned max_products = 10000;
  int threads = 1;
  double time_limit = -1.0;
  std::int64_t max_bindings = 50000000;
  std::string hydrogen_model = "explicit";
};

struct Rule {
  std::string smarts;
  int arity = 0;
  unsigned templates = 0;
  std::unique_ptr<RDKit::ChemicalReaction> reaction;
  // Lewis-state compilation: heavy-atom templates and hydrogen-count deltas.
  bool lewis = false;
  std::string model_note;
  std::unique_ptr<RDKit::ChemicalReaction> heavy;
  std::unordered_map<int, int> dh;

  const RDKit::ChemicalReaction &active() const { return lewis ? *heavy : *reaction; }
};

constexpr const char *kHydrogenProp = "_synkit_h";

bool degree_sensitive(const RDKit::Atom::QUERYATOM_QUERY *query) {
  if (!query) return false;
  const std::string &d = query->getDescription();
  // D counts hydrogen vertices, h/implicit-H queries read the other model, and
  // recursive SMARTS may embed hydrogen atoms.
  if (d == "AtomExplicitDegree" || d == "AtomImplicitHCount" || d == "AtomHasImplicitH" ||
      d == "RecursiveStructure")
    return true;
  for (auto it = query->beginChildren(); it != query->endChildren(); ++it)
    if (degree_sensitive(it->get())) return true;
  return false;
}

// Compile mapped hydrogen atoms into count labels; returns an empty string on
// success and the reason for falling back to explicit hydrogens otherwise.
std::string compile_lewis(Rule &rule) {
  const auto &rxn = *rule.reaction;
  std::unordered_map<int, int> donor, acceptor, need;
  auto heavy_neighbour = [](const RDKit::ROMol &mol, const RDKit::Atom *h, int &map) -> bool {
    if (h->getDegree() != 1) return false;
    for (const auto nb : mol.atomNeighbors(h)) {
      if (nb->getAtomicNum() == 1 || nb->getAtomMapNum() == 0) return false;
      map = nb->getAtomMapNum();
    }
    return true;
  };
  for (const auto &tpl : rxn.getReactants()) {
    for (const auto atom : tpl->atoms()) {
      if (atom->hasQuery() && degree_sensitive(atom->getQuery()))
        return "degree- or implicit-H-sensitive query";
      if (atom->getAtomicNum() != 1) continue;
      const int m = atom->getAtomMapNum();
      int d = 0;
      if (m == 0) return "unmapped hydrogen in reactant template";
      if (!heavy_neighbour(*tpl, atom, d)) return "hydrogen not bound to one mapped heavy atom";
      donor[m] = d;
      ++need[d];
    }
  }
  for (const auto &tpl : rxn.getProducts()) {
    for (const auto atom : tpl->atoms()) {
      if (atom->getAtomicNum() != 1) continue;
      const int m = atom->getAtomMapNum();
      int a = 0;
      if (m == 0 || !donor.count(m)) return "hydrogen created by the rule";
      if (!heavy_neighbour(*tpl, atom, a)) return "hydrogen released or bound to an unmapped atom";
      acceptor[m] = a;
    }
  }
  for (const auto &[m, d] : donor) {
    if (!acceptor.count(m)) return "hydrogen deleted by the rule";
    if (acceptor[m] != d) {
      --rule.dh[d];
      ++rule.dh[acceptor[m]];
    }
  }
  auto strip = [&](const RDKit::ROMOL_SPTR &tpl, bool reactant) -> RDKit::ROMOL_SPTR {
    auto *mol = new RDKit::RWMol(*tpl);
    RDKit::ROMOL_SPTR owner(mol);
    std::vector<unsigned> hydrogens;
    for (const auto atom : mol->atoms())
      if (atom->getAtomicNum() == 1) hydrogens.push_back(atom->getIdx());
    std::sort(hydrogens.rbegin(), hydrogens.rend());
    for (auto idx : hydrogens) mol->removeAtom(idx);
    if (reactant) {
      for (auto atom : mol->atoms()) {
        auto it = need.find(atom->getAtomMapNum());
        if (atom->getAtomMapNum() == 0 || it == need.end()) continue;
        auto query_atom = dynamic_cast<RDKit::QueryAtom *>(atom);
        if (!query_atom) throw std::runtime_error("template atom without query");
        for (int count = 0; count < it->second; ++count) {
          auto at_least = RDKit::makeAtomHCountQuery(count);
          at_least->setNegation(true);
          query_atom->expandQuery(at_least, Queries::COMPOSITE_AND);
        }
      }
    }
    return owner;
  };
  try {
    rule.heavy = std::make_unique<RDKit::ChemicalReaction>();
    for (const auto &tpl : rxn.getReactants()) rule.heavy->addReactantTemplate(strip(tpl, true));
    for (const auto &tpl : rxn.getProducts()) rule.heavy->addProductTemplate(strip(tpl, false));
    rule.heavy->initReactantMatchers(true);
  } catch (const std::exception &exc) {
    rule.heavy.reset();
    rule.dh.clear();
    return std::string("template compilation failed: ") + exc.what();
  }
  if (rule.heavy->getNumReactantTemplates() != rule.templates) {
    rule.heavy.reset();
    rule.dh.clear();
    return "template count changed";
  }
  rule.lewis = true;
  return "";
}

// Lewis admissibility of one atom: (outer electrons - charge - valence -
// radicals) must be a non-negative even number, i.e. lone pairs are integral.
bool lewis_admissible(const RDKit::Atom &atom) {
  const int z = atom.getAtomicNum();
  if (z <= 0 || z > 86) return true;
  const bool main_group = z <= 2 || (z >= 5 && z <= 10) || (z >= 13 && z <= 18) ||
                          (z >= 31 && z <= 36) || (z >= 49 && z <= 54) || (z >= 81);
  if (!main_group) return true;
  const int outer = RDKit::PeriodicTable::getTable()->getNouterElecs(z);
  const int valence = static_cast<int>(atom.getValence(RDKit::Atom::ValenceType::EXPLICIT) +
                                       atom.getValence(RDKit::Atom::ValenceType::IMPLICIT));
  const int pairs2 = outer - atom.getFormalCharge() - valence -
                     static_cast<int>(atom.getNumRadicalElectrons());
  return pairs2 >= 0 && pairs2 % 2 == 0;
}

struct Species {
  std::string key;
  RDKit::ROMOL_SPTR with_hs;
  RDKit::ROMOL_SPTR heavy;  // heavy-atom graph labelled with hydrogen counts
  // slots[rule][slot] is true when the molecule binds that reactant template.
  std::vector<std::vector<char>> slots;
};

struct Task {
  int rule = 0;
  std::vector<int> species;  // species ids, sorted by key
};

struct Event {
  int step = 0;
  int rule = 0;
  std::vector<int> reactants;  // recorded reactant side, with repeats
  std::vector<int> products;   // recorded product side, with repeats
  std::vector<int> all_products;
};

struct StepStats {
  int step = 0;
  std::size_t pool = 0, frontier = 0;
  std::int64_t bindings = 0, candidate_mixtures = 0, seen_mixtures = 0;
  std::int64_t tasks = 0, applications = 0, outcomes = 0, events = 0, lewis_rejected = 0;
  std::vector<int> capped_rules;
  bool task_budget_exhausted = false;
};

template <typename T>
T get_or(const json::object &object, const char *name, T fallback) {
  auto it = object.find(name);
  if (it == object.end() || it->value().is_null()) return fallback;
  return json::value_to<T>(it->value());
}

class Expander {
 public:
  Expander(Options options, std::vector<Rule> rules)
      : o_(options), rules_(std::move(rules)) {
    if (o_.time_limit >= 0) {
      deadline_ = Clock::now() + std::chrono::duration_cast<Clock::duration>(
                                     std::chrono::duration<double>(o_.time_limit));
    }
  }

  void check() const {
    if (deadline_ && Clock::now() >= *deadline_) throw LimitReached{"time_limit"};
  }

  // Equivalent of synkit.CRN.Construct.smiles.standardize_smiles_rdkit with
  // keep_aam=False: parse unsanitized, sanitize, strip maps, remove all Hs.
  std::optional<std::string> standardize(const std::string &text) {
    auto cached = standard_.find(text);
    if (cached != standard_.end()) return cached->second;
    std::optional<std::string> out;
    if (!text.empty()) {
      try {
        RDKit::v2::SmilesParse::SmilesParserParams params;
        params.sanitize = false;
        params.removeHs = false;
        auto mol = RDKit::v2::SmilesParse::MolFromSmiles(text, params);
        if (mol) {
          RDKit::MolOps::sanitizeMol(*mol);
          for (auto atom : mol->atoms()) atom->setAtomMapNum(0);
          RDKit::MolOps::removeAllHs(*mol, true);
          out = RDKit::MolToSmiles(*mol);
        }
      } catch (const std::exception &) {
        out.reset();
      }
    }
    standard_.emplace(text, out);
    return out;
  }

  int species_id(const std::string &key) {
    auto it = index_.find(key);
    if (it != index_.end()) return it->second;
    Species record;
    record.key = key;
    auto mol = RDKit::v2::SmilesParse::MolFromSmiles(key);
    if (!mol) throw std::runtime_error("Invalid canonical species: " + key);
    for (auto atom : mol->atoms())
      atom->setProp<int>(kHydrogenProp, static_cast<int>(atom->getTotalNumHs(false)));
    auto explicit_mol = std::make_unique<RDKit::RWMol>(*mol);
    RDKit::MolOps::addHs(*explicit_mol);
    record.heavy = RDKit::ROMOL_SPTR(mol.release());
    record.with_hs = RDKit::ROMOL_SPTR(explicit_mol.release());
    record.slots.resize(rules_.size());
    for (std::size_t r = 0; r < rules_.size(); ++r) {
      auto &slots = record.slots[r];
      slots.assign(rules_[r].templates, 0);
      if (!rules_[r].reaction) continue;
      std::vector<unsigned> which;
      const auto &target = rules_[r].lewis ? *record.heavy : *record.with_hs;
      RDKit::isMoleculeReactantOfReaction(rules_[r].active(), target, which, false);
      for (auto slot : which) slots[slot] = 1;
      ++binding_tests_;
    }
    const int id = static_cast<int>(species_.size());
    species_.push_back(std::move(record));
    index_.emplace(key, id);
    return id;
  }

  json::object run(const std::vector<std::string> &seeds) {
    std::set<std::string> pool, frontier;
    json::array seed_keys;
    for (const auto &seed : seeds) {
      auto key = standardize(seed);
      if (!key) {
        seed_keys.emplace_back(nullptr);
        continue;
      }
      species_id(*key);
      pool.insert(*key);
      frontier.insert(*key);
      seed_keys.emplace_back(json::string(*key));
    }
    std::string status = "complete";
    std::string stop = pool.empty() ? "empty_pool" : "repeats";
    int steps_run = 0;
    try {
      for (int step = 1; step <= o_.repeats && !pool.empty(); ++step) {
        if (o_.use_frontier && frontier.empty()) {
          stop = "fixed_point";
          break;
        }
        StepStats stats;
        stats.step = step;
        stats.pool = pool.size();
        stats.frontier = frontier.size();
        auto tasks = make_tasks(pool, frontier, stats);
        if (tasks.empty()) {
          steps_.push_back(stats);
          stop = "no_tasks";
          break;
        }
        auto results = run_tasks(tasks, stats);
        auto next = integrate(tasks, results, pool, step, stats);
        frontier = std::move(next);
        steps_.push_back(stats);
        steps_run = step;
        if (o_.use_frontier && frontier.empty()) {
          stop = "fixed_point";
          break;
        }
      }
    } catch (const LimitReached &limit) {
      status = "limit";
      stop = limit.reason;
    }

    json::object out;
    out["protocol"] = kProtocol;
    out["engine"] = "synkit_crn_expand";
    out["rdkit_version"] = RDKit::rdkitVersion;
    out["status"] = status;
    out["stop_reason"] = stop;
    out["steps_run"] = steps_run;
    out["seed_keys"] = std::move(seed_keys);
    json::array species;
    for (const auto &record : species_) species.emplace_back(json::string(record.key));
    out["species"] = std::move(species);
    json::array events;
    for (const auto &event : events_) {
      json::object item;
      item["step"] = event.step;
      item["rule_index"] = event.rule;
      item["reactants"] = json::value_from(event.reactants);
      item["products"] = json::value_from(event.products);
      item["all_products"] = json::value_from(event.all_products);
      events.emplace_back(std::move(item));
    }
    out["events"] = std::move(events);
    json::array pool_out, frontier_out;
    for (const auto &key : pool) pool_out.emplace_back(json::string(key));
    for (const auto &key : frontier) frontier_out.emplace_back(json::string(key));
    out["pool"] = std::move(pool_out);
    out["frontier"] = std::move(frontier_out);
    json::array steps;
    for (const auto &s : steps_) {
      json::object item;
      item["step"] = s.step;
      item["pool"] = s.pool;
      item["frontier"] = s.frontier;
      item["bindings"] = s.bindings;
      item["candidate_mixtures"] = s.candidate_mixtures;
      item["seen_mixtures"] = s.seen_mixtures;
      item["tasks"] = s.tasks;
      item["applications"] = s.applications;
      item["outcomes"] = s.outcomes;
      item["events"] = s.events;
      item["lewis_rejected"] = s.lewis_rejected;
      item["capped_rules"] = json::value_from(s.capped_rules);
      item["task_budget_exhausted"] = s.task_budget_exhausted;
      steps.emplace_back(std::move(item));
    }
    out["steps"] = std::move(steps);
    out["binding_tests"] = binding_tests_;
    out["hydrogen_model"] = o_.hydrogen_model;
    json::array rules;
    for (const auto &rule : rules_) {
      json::object item;
      item["hydrogen_model"] = rule.lewis ? "lewis" : "explicit";
      item["note"] = rule.model_note;
      json::object deltas;
      for (const auto &[map, delta] : rule.dh)
        if (delta) deltas[std::to_string(map)] = delta;
      item["hydrogen_deltas"] = std::move(deltas);
      rules.emplace_back(std::move(item));
    }
    out["rules"] = std::move(rules);
    return out;
  }

 private:
  // Candidate mixtures for one rule in the order FrontierStrategy enumerates
  // them, restricted to mixtures that admit a complete slot binding.
  std::vector<std::vector<int>> candidates(int r, const std::vector<int> &ranked,
                                           const std::vector<char> &in_frontier,
                                           StepStats &stats) {
    const Rule &rule = rules_[r];
    const int k = rule.arity;
    std::vector<std::vector<int>> found;
    if (k < 1 || k > o_.max_components || !rule.reaction ||
        rule.templates != static_cast<unsigned>(k))
      return found;
    // Per-slot candidate ranks (positions in the sorted pool).
    std::vector<std::vector<int>> slot_ranks(k);
    for (int rank = 0; rank < static_cast<int>(ranked.size()); ++rank) {
      const auto &slots = species_[ranked[rank]].slots[r];
      for (int s = 0; s < k; ++s)
        if (slots[s]) slot_ranks[s].push_back(rank);
    }
    for (const auto &ranks : slot_ranks)
      if (ranks.empty()) return found;

    std::set<std::vector<int>> multisets;
    std::vector<int> assignment(k);
    std::int64_t bindings = 0;
    // Bind slot by slot; MOD binds one graph per round with a growing offset.
    std::function<void(int)> bind = [&](int slot) {
      if (slot == k) {
        std::vector<int> mixture(assignment);
        std::sort(mixture.begin(), mixture.end());
        if (!o_.allow_self_mixtures &&
            std::adjacent_find(mixture.begin(), mixture.end()) != mixture.end())
          return;
        if (o_.use_frontier &&
            std::none_of(mixture.begin(), mixture.end(),
                         [&](int rank) { return in_frontier[rank] != 0; }))
          return;
        multisets.insert(std::move(mixture));
        return;
      }
      for (int rank : slot_ranks[slot]) {
        if ((++bindings & 4095) == 0) check();
        if (bindings > o_.max_bindings) throw LimitReached{"binding_limit"};
        if (!o_.allow_self_mixtures &&
            std::find(assignment.begin(), assignment.begin() + slot, rank) !=
                assignment.begin() + slot)
          continue;
        assignment[slot] = rank;
        bind(slot + 1);
      }
    };
    bind(0);
    stats.bindings += bindings;

    // FrontierStrategy yields (f, rest) for f in sorted(frontier) and rest in
    // lexicographic combinations; a mixture appears at its first occurrence.
    std::vector<std::pair<std::vector<int>, std::vector<int>>> keyed;
    keyed.reserve(multisets.size());
    for (const auto &mixture : multisets) {
      if (!o_.use_frontier) {
        keyed.emplace_back(mixture, mixture);
        continue;
      }
      std::vector<int> best;
      for (std::size_t i = 0; i < mixture.size(); ++i) {
        if (!in_frontier[mixture[i]] || (i && mixture[i] == mixture[i - 1])) continue;
        std::vector<int> key{mixture[i]};
        for (std::size_t j = 0; j < mixture.size(); ++j)
          if (j != i) key.push_back(mixture[j]);
        if (best.empty() || key < best) best = std::move(key);
      }
      keyed.emplace_back(std::move(best), mixture);
    }
    std::sort(keyed.begin(), keyed.end());
    found.reserve(keyed.size());
    for (auto &item : keyed) found.push_back(std::move(item.second));
    return found;
  }

  std::vector<Task> make_tasks(const std::set<std::string> &pool,
                               const std::set<std::string> &frontier, StepStats &stats) {
    std::vector<int> ranked;
    ranked.reserve(pool.size());
    for (const auto &key : pool) ranked.push_back(index_.at(key));
    std::vector<char> in_frontier(ranked.size(), 0);
    {
      std::size_t rank = 0;
      for (const auto &key : pool) {
        in_frontier[rank] = o_.use_frontier ? (frontier.count(key) ? 1 : 0) : 1;
        ++rank;
      }
    }
    std::vector<Task> tasks;
    std::int64_t budget = o_.max_tasks_per_step;
    for (int r = 0; r < static_cast<int>(rules_.size()); ++r) {
      if (budget <= 0) {
        stats.task_budget_exhausted = true;
        break;
      }
      check();
      const std::int64_t cap = std::min(o_.max_mixtures_per_rule_step, budget);
      auto mixtures = candidates(r, ranked, in_frontier, stats);
      if (static_cast<std::int64_t>(mixtures.size()) > cap) {
        stats.capped_rules.push_back(r);
        mixtures.resize(static_cast<std::size_t>(cap));
      }
      for (const auto &mixture : mixtures) {
        if (budget <= 0) {
          stats.task_budget_exhausted = true;
          break;
        }
        ++stats.candidate_mixtures;
        Task task;
        task.rule = r;
        for (int rank : mixture) task.species.push_back(ranked[rank]);
        std::string attempt = std::to_string(r);
        for (int id : task.species) {
          attempt.push_back('\x1f');
          attempt += species_[id].key;
        }
        if (!attempts_.insert(std::move(attempt)).second) {
          ++stats.seen_mixtures;
          continue;
        }
        tasks.push_back(std::move(task));
        --budget;
      }
    }
    stats.tasks = static_cast<std::int64_t>(tasks.size());
    return tasks;
  }

  // Equivalent of rdkit_worker._canonical_product.
  // Returns false when the Lewis-state hydrogen update is inadmissible.
  bool canonical_outcome(const RDKit::MOL_SPTR_VECT &outcome, const Rule &rule,
                         const RDKit::MOL_SPTR_VECT &reactants, std::string &joined) const {
    std::vector<std::string> parts;
    for (const auto &product : outcome) {
      RDKit::RWMol molecule(*product);
      std::vector<unsigned> changed;
      if (rule.lewis) {
        for (auto atom : molecule.atoms()) {
          // Hydrogen label of the reactant atom this product atom came from.
          int h = 0;
          unsigned source = 0, molecule_index = 0;
          if (!atom->getPropIfPresent<unsigned>(RDKit::common_properties::reactantAtomIdx, source) ||
              !atom->getPropIfPresent<unsigned>(RDKit::common_properties::reactantIdx, molecule_index) ||
              molecule_index >= reactants.size() ||
              !reactants[molecule_index]->getAtomWithIdx(source)->getPropIfPresent<int>(kHydrogenProp, h))
            continue;
          int map = 0;
          if (atom->getPropIfPresent<int>(RDKit::common_properties::reactionMapNum, map)) {
            auto it = rule.dh.find(map);
            if (it != rule.dh.end() && it->second) {
              h += it->second;
              changed.push_back(atom->getIdx());
            }
          }
          if (h < 0) return false;
          atom->setNumExplicitHs(static_cast<unsigned>(h));
          atom->setNoImplicit(true);
        }
      }
      RDKit::MolOps::sanitizeMol(molecule);
      for (auto idx : changed)
        if (!lewis_admissible(*molecule.getAtomWithIdx(idx))) return false;
      if (!o_.explicit_h) RDKit::MolOps::removeHs(molecule, RDKit::MolOps::RemoveHsParameters(), true);
      for (auto atom : molecule.atoms()) atom->setAtomMapNum(0);
      parts.push_back(RDKit::MolToSmiles(molecule));
    }
    std::sort(parts.begin(), parts.end());
    joined.clear();
    for (std::size_t i = 0; i < parts.size(); ++i) {
      if (i) joined.push_back('.');
      joined += parts[i];
    }
    return true;
  }

  struct TaskResult {
    std::vector<std::string> products;
    std::int64_t applications = 0, outcomes = 0, lewis_rejected = 0;
    std::string error;
  };

  TaskResult apply(const Task &task) const {
    TaskResult result;
    const Rule &rule = rules_[task.rule];
    const std::size_t n = task.species.size();
    std::vector<RDKit::ROMOL_SPTR> molecules;
    molecules.reserve(n);
    for (int id : task.species)
      molecules.emplace_back(new RDKit::ROMol(rule.lewis ? *species_[id].heavy : *species_[id].with_hs));
    std::vector<std::size_t> order(n);
    for (std::size_t i = 0; i < n; ++i) order[i] = i;
    std::unordered_set<std::string> seen;
    do {
      bool binds = true;
      for (std::size_t slot = 0; slot < n && binds; ++slot)
        binds = species_[task.species[order[slot]]].slots[task.rule][slot] != 0;
      if (!binds) continue;
      RDKit::MOL_SPTR_VECT reactants;
      for (std::size_t slot = 0; slot < n; ++slot) reactants.push_back(molecules[order[slot]]);
      auto outcomes = rule.active().runReactants(reactants, o_.max_products);
      ++result.applications;
      if (outcomes.size() >= o_.max_products) {
        result.error = "product_limit";
        return result;
      }
      for (const auto &outcome : outcomes) {
        ++result.outcomes;
        std::string product;
        try {
          if (!canonical_outcome(outcome, rule, reactants, product)) {
            ++result.lewis_rejected;
            continue;
          }
        } catch (const std::exception &) {
          if (rule.lewis) ++result.lewis_rejected;
          continue;
        }
        if (seen.insert(product).second) result.products.push_back(std::move(product));
      }
    } while (std::next_permutation(order.begin(), order.end()));
    return result;
  }

  std::vector<TaskResult> run_tasks(const std::vector<Task> &tasks, StepStats &stats) {
    std::vector<TaskResult> results(tasks.size());
    const int workers = std::max(1, std::min<int>(o_.threads, static_cast<int>(tasks.size())));
    std::atomic<std::size_t> next{0};
    std::atomic<bool> expired{false};
    auto work = [&]() {
      for (;;) {
        const std::size_t i = next.fetch_add(1);
        if (i >= tasks.size() || expired.load()) return;
        if (deadline_ && Clock::now() >= *deadline_) {
          expired = true;
          return;
        }
        try {
          results[i] = apply(tasks[i]);
        } catch (const std::exception &exc) {
          results[i].error = std::string("rule_application_error: ") + exc.what();
        }
      }
    };
    if (workers == 1) {
      work();
    } else {
      std::vector<std::thread> pool;
      for (int w = 0; w < workers; ++w) pool.emplace_back(work);
      for (auto &thread : pool) thread.join();
    }
    if (expired) throw LimitReached{"time_limit"};
    for (std::size_t i = 0; i < results.size(); ++i) {
      if (!results[i].error.empty()) {
        if (results[i].error == "product_limit") throw LimitReached{"product_limit"};
        throw std::runtime_error(results[i].error);
      }
      stats.applications += results[i].applications;
      stats.outcomes += results[i].outcomes;
      stats.lewis_rejected += results[i].lewis_rejected;
    }
    return results;
  }

  std::set<std::string> integrate(const std::vector<Task> &tasks,
                                  const std::vector<TaskResult> &results,
                                  std::set<std::string> &pool, int step, StepStats &stats) {
    std::set<std::string> next;
    for (std::size_t t = 0; t < tasks.size(); ++t) {
      check();
      const Task &task = tasks[t];
      std::unordered_set<std::string> seen;
      for (const auto &mixture : results[t].products) {
        if (mixture.empty() || !seen.insert(mixture).second) continue;
        std::vector<std::string> products;
        std::stringstream stream(mixture);
        std::string part;
        while (std::getline(stream, part, '.')) {
          if (part.empty()) continue;
          auto key = standardize(part);
          if (key) products.push_back(*key);
        }
        if (products.empty()) continue;

        std::map<std::string, int> reactant_count, product_count;
        std::vector<std::string> reactants;
        for (int id : task.species) {
          reactants.push_back(species_[id].key);
          ++reactant_count[species_[id].key];
        }
        for (const auto &key : products) ++product_count[key];
        std::vector<std::string> r_delta, p_delta;
        for (const auto &[key, count] : reactant_count) {
          auto it = product_count.find(key);
          const int extra = count - (it == product_count.end() ? 0 : it->second);
          for (int i = 0; i < extra; ++i) r_delta.push_back(key);
        }
        for (const auto &[key, count] : product_count) {
          auto it = reactant_count.find(key);
          const int extra = count - (it == reactant_count.end() ? 0 : it->second);
          for (int i = 0; i < extra; ++i) p_delta.push_back(key);
        }

        std::vector<std::string> r_keep = reactants, p_keep = products;
        if (o_.strip_spectators) {
          r_keep.clear();
          p_keep.clear();
          for (const auto &key : reactants)
            if (!product_count.count(key)) r_keep.push_back(key);
          for (const auto &key : products)
            if (!reactant_count.count(key)) p_keep.push_back(key);
        }
        if (o_.skip_no_change && r_delta.empty() && p_delta.empty()) continue;
        if (!o_.allow_empty_side && (r_keep.empty() || p_keep.empty())) continue;
        if (o_.dedup_delta) {
          std::string key = o_.dedup_across_rules ? std::string("*") : std::to_string(task.rule);
          key.push_back('\x1e');
          for (const auto &s : r_delta) key += s + '\x1f';
          key.push_back('\x1e');
          for (const auto &s : p_delta) key += s + '\x1f';
          if (!deltas_.insert(std::move(key)).second) continue;
        }
        Event event;
        event.step = step;
        event.rule = task.rule;
        for (const auto &key : r_keep) event.reactants.push_back(species_id(key));
        for (const auto &key : p_keep) event.products.push_back(species_id(key));
        for (const auto &key : products) event.all_products.push_back(species_id(key));
        events_.push_back(std::move(event));
        ++stats.events;
        for (const auto &key : products) {
          if (pool.insert(key).second) next.insert(key);
        }
      }
    }
    return next;
  }

  Options o_;
  std::vector<Rule> rules_;
  std::optional<Clock::time_point> deadline_;
  std::vector<Species> species_;
  std::unordered_map<std::string, int> index_;
  std::unordered_map<std::string, std::optional<std::string>> standard_;
  std::unordered_set<std::string> attempts_, deltas_;
  std::vector<Event> events_;
  std::vector<StepStats> steps_;
  std::int64_t binding_tests_ = 0;
};

Options parse_options(const json::object &object) {
  Options o;
  o.repeats = get_or<int>(object, "repeats", o.repeats);
  o.explicit_h = get_or<bool>(object, "explicit_h", o.explicit_h);
  o.max_components = get_or<int>(object, "max_components", o.max_components);
  o.use_frontier = get_or<bool>(object, "use_frontier", o.use_frontier);
  o.max_mixtures_per_rule_step =
      get_or<std::int64_t>(object, "max_mixtures_per_rule_step", o.max_mixtures_per_rule_step);
  o.max_tasks_per_step = get_or<std::int64_t>(object, "max_tasks_per_step", o.max_tasks_per_step);
  o.allow_self_mixtures = get_or<bool>(object, "allow_self_mixtures", o.allow_self_mixtures);
  o.skip_no_change = get_or<bool>(object, "skip_no_change", o.skip_no_change);
  o.allow_empty_side = get_or<bool>(object, "allow_empty_side", o.allow_empty_side);
  o.strip_spectators = get_or<bool>(object, "strip_spectators", o.strip_spectators);
  o.dedup_delta = get_or<bool>(object, "dedup_delta", o.dedup_delta);
  o.dedup_across_rules = get_or<bool>(object, "dedup_across_rules", o.dedup_across_rules);
  o.max_products = get_or<unsigned>(object, "max_products", o.max_products);
  o.threads = get_or<int>(object, "threads", o.threads);
  o.time_limit = get_or<double>(object, "time_limit", o.time_limit);
  o.max_bindings = get_or<std::int64_t>(object, "max_bindings", o.max_bindings);
  o.hydrogen_model = get_or<std::string>(object, "hydrogen_model", o.hydrogen_model);
  if (o.hydrogen_model != "explicit" && o.hydrogen_model != "lewis")
    throw std::invalid_argument("hydrogen_model must be explicit or lewis");
  if (o.hydrogen_model == "lewis" && o.explicit_h)
    throw std::invalid_argument("hydrogen_model=lewis writes implicit-hydrogen products");
  if (o.max_components < 1) throw std::invalid_argument("max_components must be >= 1");
  if (o.max_products < 1) throw std::invalid_argument("max_products must be >= 1");
  return o;
}

}  // namespace

json::object handle(const std::string &text) {
  try {
    const json::object request = json::parse(text).as_object();
    if (get_or<int>(request, "protocol", kProtocol) != kProtocol)
      throw std::invalid_argument("unsupported protocol");
    Options options = parse_options(request.contains("options")
                                        ? request.at("options").as_object()
                                        : json::object());
    std::vector<Rule> rules;
    for (const auto &item : request.at("rules").as_array()) {
      const auto &object = item.as_object();
      Rule rule;
      rule.smarts = json::value_to<std::string>(object.at("smarts"));
      rule.arity = json::value_to<int>(object.at("arity"));
      rule.reaction = RDKit::v2::ReactionParser::ReactionFromSmarts(rule.smarts);
      if (!rule.reaction) throw std::invalid_argument("Invalid RDKit reaction SMARTS: " + rule.smarts);
      rule.reaction->initReactantMatchers();
      rule.templates = rule.reaction->getNumReactantTemplates();
      if (options.hydrogen_model == "lewis") rule.model_note = compile_lewis(rule);
      rules.push_back(std::move(rule));
    }
    std::vector<std::string> seeds;
    for (const auto &item : request.at("seeds").as_array())
      seeds.push_back(json::value_to<std::string>(item));
    Expander expander(options, std::move(rules));
    return expander.run(seeds);
  } catch (const std::exception &exc) {
    json::object out;
    out["protocol"] = kProtocol;
    out["status"] = "error";
    out["error"] = exc.what();
    return out;
  }
}

// One request per invocation, or with --serve one request per input line so
// that a session pays the RDKit load cost once.
int main(int argc, char **argv) {
  boost::logging::disable_logs("rdApp.*");
  if (argc > 1 && std::string(argv[1]) == "--serve") {
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      std::cout << json::serialize(handle(line)) << '\n' << std::flush;
    }
    return 0;
  }
  std::string text((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
  auto out = handle(text);
  std::cout << json::serialize(out) << '\n';
  return out.at("status").as_string() == "error" ? 1 : 0;
}
