#include "synkit_mapper/reaction/reaction.h"
#include "synkit_mapper/core/runtime.h"
#include "synkit_mapper/assignment/assignment.h"
#include "synkit_mapper/hydrogen/hydrogen.h"
#include "synkit_mapper/symmetry/classification.h"
#include <GraphMol/MolOps.h>
#include <GraphMol/PeriodicTable.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <RDGeneral/versions.h>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>

namespace {
using namespace synkit_mapper;
using Identity = std::pair<int, int>;
thread_local std::string last_error;

std::string quote(const std::string &s) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : s) {
    if (c == '"' || c == '\\')
      out << '\\' << c;
    else if (c < 32)
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c)
          << std::dec;
    else
      out << c;
  }
  out << '"';
  return out.str();
}

// Exact _encode_color-compatible token trees. Every primitive needed here is
// encoded as a string (numeric attributes are wrapped by the legacy _typed).
struct Token {
  bool sequence = false;
  std::string text;
  std::vector<Token> items;
  bool operator<(const Token &other) const {
    if (sequence != other.sequence)
      return sequence < other.sequence;
    return sequence ? items < other.items : text < other.text;
  }
  std::string json() const {
    if (!sequence)
      return quote(text);
    std::string result = "[";
    for (size_t i = 0; i < items.size(); ++i)
      result += (i ? "," : "") + items[i].json();
    return result + "]";
  }
};
Token literal(const std::string &s) { return {false, s, {}}; }
Token tuple(std::vector<Token> values) {
  values.insert(values.begin(), literal("tuple"));
  return {true, "", std::move(values)};
}
Token string_token(const std::string &s) {
  return {true, "", {literal("str"), literal(s)}};
}
Token typed(const std::string &type, const std::string &representation) {
  return tuple({string_token("builtins." + type), string_token(representation)});
}
Token weight_token(int64_t doubled) {
  std::string value = "0x0.0p+0";
  if (doubled) {
    double weight = doubled / 2.0;
    int exponent;
    double fraction = std::frexp(weight, &exponent) * 2 - 1;
    auto mantissa = static_cast<uint64_t>(std::llround(std::ldexp(fraction, 52)));
    std::ostringstream stream;
    stream << "0x1." << std::hex << std::setw(13) << std::setfill('0') << mantissa
           << "p" << std::dec << (exponent - 1 >= 0 ? "+" : "") << exponent - 1;
    value = stream.str();
  }
  return {true, "", {literal("float"), literal(value)}};
}

struct Endpoint {
  std::unique_ptr<RDKit::RWMol> molecule;
  std::vector<Identity> labels;
  std::vector<int32_t> charges, radicals, hcounts, dummies, stereo_anchors;
  std::vector<std::string> stereo;
  std::vector<int64_t> weights;
  std::vector<std::string> bond_stereo;
  std::vector<int32_t> types, node_colors, edge_colors, atomic_numbers, isotopes;
};
std::string stereo_name(RDKit::Bond::BondStereo value) {
  switch (value) {
  case RDKit::Bond::STEREONONE: return "STEREONONE";
  case RDKit::Bond::STEREOANY: return "STEREOANY";
  case RDKit::Bond::STEREOZ: return "STEREOZ";
  case RDKit::Bond::STEREOE: return "STEREOE";
  case RDKit::Bond::STEREOCIS: return "STEREOCIS";
  case RDKit::Bond::STEREOTRANS: return "STEREOTRANS";
  default: throw std::invalid_argument("Unsupported RDKit stereobond state");
  }
}
int nonbonding(const RDKit::Atom &atom) {
  double bond_sum = 0;
  if (atom.getIsAromatic())
    bond_sum = atom.getTotalValence() - atom.getTotalNumHs(false);
  else
    for (auto bond : atom.getOwningMol().atomBonds(&atom))
      bond_sum += bond->getBondTypeAsDouble();
  double available = RDKit::PeriodicTable::getTable()->getNouterElecs(atom.getAtomicNum())
                     - atom.getFormalCharge() - atom.getNumRadicalElectrons()
                     - bond_sum - atom.getTotalNumHs(false);
  int lone_pairs = std::max(0, static_cast<int>(std::floor((available + 1e-8) / 2)));
  return 2 * lone_pairs + atom.getNumRadicalElectrons();
}
Endpoint parse(const std::string &smiles, bool explicit_hydrogens, const Budget &budget) {
  budget.check();
  RDKit::SmilesParserParams params;
  params.removeHs = false;
  Endpoint result;
  result.molecule.reset(RDKit::SmilesToMol(smiles, params));
  if (!result.molecule)
    throw std::invalid_argument("Could not parse reaction endpoint");
  budget.check();
  for (auto atom : result.molecule->atoms()) {
    if (!atom->getAtomicNum())
      throw std::invalid_argument("Wildcard atoms require a concrete element");
    atom->setAtomMapNum(0);
  }
  budget.check();
  if (explicit_hydrogens)
    RDKit::MolOps::addHs(*result.molecule);
  else {
    RDKit::MolOps::removeHs(*result.molecule);
    for (auto atom : result.molecule->atoms())
      if (atom->getAtomicNum() == 1)
        throw std::invalid_argument("Free or isotopic H requires explicit hydrogens");
  }
  RDKit::MolOps::assignStereochemistry(*result.molecule, true, true);
  budget.check();
  for (auto atom : result.molecule->atoms()) {
    result.labels.emplace_back(atom->getAtomicNum(), atom->getIsotope());
    result.charges.push_back(atom->getFormalCharge());
    result.radicals.push_back(atom->getNumRadicalElectrons());
    result.hcounts.push_back(atom->getTotalNumHs());
    result.dummies.push_back(0);
    result.stereo_anchors.push_back(0);
    std::string cip;
    atom->getPropIfPresent("_CIPCode", cip);
    result.stereo.push_back(cip);
  }
  return result;
}
void append_node(Endpoint &e, Identity type, bool dummy) {
  e.labels.push_back(type);
  e.charges.push_back(0); e.radicals.push_back(0); e.hcounts.push_back(0);
  e.dummies.push_back(dummy); e.stereo.push_back(""); e.stereo_anchors.push_back(0);
}
void prepare(Endpoint &left, Endpoint &right, const SynkitReactionOptions &o,
             const Budget &budget) {
  std::map<Identity, int> counts[2], universe;
  Endpoint *sides[] = {&left, &right};
  for (int side = 0; side < 2; ++side)
    for (auto type : sides[side]->labels)
      ++counts[side][type];
  if (!o.dummy_balance && counts[0] != counts[1])
    throw std::invalid_argument("Unbalanced atom inventories; enable dummy balance explicitly");
  for (auto &count : counts)
    for (auto [type, n] : count)
      universe[type] = std::max(universe[type], n);
  std::map<Identity, int> hard_ids;
  std::map<std::string, int> node_ids, edge_ids;
  for (int side = 0; side < 2; ++side) {
    budget.check();
    auto &e = *sides[side];
    for (auto [type, maximum] : universe)
      for (int i = counts[side][type]; i < maximum; ++i)
        append_node(e, type, true);
    if (o.lewis_objective)
      append_node(e, {0, -1}, true);
    int n = static_cast<int>(e.labels.size());
    if (n < 1 || n > synkit_mapper_classification_max_vertices())
      throw std::invalid_argument("Native reaction classification capacity exceeded");
    e.weights.assign(n * n, 0); e.bond_stereo.assign(n * n, "");
    e.edge_colors.assign(n * n, 0);
    for (auto bond : e.molecule->bonds()) {
      int i = bond->getBeginAtomIdx(), j = bond->getEndAtomIdx();
      double order = o.binary ? 1 : bond->getBondTypeAsDouble();
      if (!std::isfinite(order) || order < 0 || std::abs(order * 2 - std::round(order * 2)) > 1e-9)
        throw std::invalid_argument("Unsupported bond weight lattice");
      e.weights[i * n + j] = e.weights[j * n + i] = std::llround(order * 2);
      auto name = stereo_name(bond->getStereo());
      e.bond_stereo[i * n + j] = e.bond_stereo[j * n + i] = name;
      if (bond->getStereo() != RDKit::Bond::STEREONONE) {
        e.stereo_anchors[i] = i + 1; e.stereo_anchors[j] = j + 1;
      }
      int color = edge_ids.emplace(name, edge_ids.size() + 1).first->second;
      e.edge_colors[i * n + j] = e.edge_colors[j * n + i] = color;
    }
    if (o.lewis_objective)
      for (auto atom : e.molecule->atoms()) {
        int i = atom->getIdx(), electrons = nonbonding(*atom);
        e.weights[i * n + n - 1] = e.weights[(n - 1) * n + i] = electrons;
      }
    for (int i = 0; i < n; ++i) {
      auto type = e.labels[i];
      e.types.push_back(hard_ids.emplace(type, hard_ids.size()).first->second);
      e.atomic_numbers.push_back(type.first); e.isotopes.push_back(type.second);
      auto key = std::to_string(type.first) + ":" + std::to_string(type.second) + ":" +
                 std::to_string(e.charges[i]) + ":" + std::to_string(e.radicals[i]) + ":" +
                 std::to_string(e.hcounts[i]) + ":" + e.stereo[i] + ":" +
                 std::to_string(e.dummies[i]) + ":" + std::to_string(e.stereo_anchors[i]);
      e.node_colors.push_back(node_ids.emplace(key, node_ids.size()).first->second);
    }
  }
}

struct Palette {
  std::vector<int32_t> nodes, edges_r, edges_p, edge_pairs;
  std::vector<std::string> json;
  std::vector<const char *> pointers;
  SynkitClassProblem problem{};
  Palette(const Endpoint &a, const Endpoint &b, const Budget &budget) {
    int n = a.labels.size();
    std::vector<const Token *> node_tokens;
    std::set<Token> all;
    for (int i = 0; i < n; ++i) {
      budget.check();
      for (int j = 0; j < n; ++j) {
        std::vector<Token> props;
        const char *names[] = {"charge", "radical", "hcount", "stereo", "dummy"};
        for (int k = 0; k < 5; ++k) {
          Token left, right;
          if (k == 3) {
            left = typed("str", "'" + a.stereo[i] + "'");
            right = typed("str", "'" + b.stereo[j] + "'");
          } else if (k == 4) {
            left = typed("bool", a.dummies[i] ? "True" : "False");
            right = typed("bool", b.dummies[j] ? "True" : "False");
          } else {
            const std::vector<int32_t> *pa[] = {&a.charges, &a.radicals, &a.hcounts};
            const std::vector<int32_t> *pb[] = {&b.charges, &b.radicals, &b.hcounts};
            left = typed("int", std::to_string((*pa[k])[i]));
            right = typed("int", std::to_string((*pb[k])[j]));
          }
          props.push_back(tuple({string_token(names[k]), left, right}));
        }
        auto id = a.labels[i];
        auto value = tuple({typed("tuple", "(" + std::to_string(id.first) + ", " +
                                 std::to_string(id.second) + ")"), tuple(props), tuple({})});
        // Most explicit H atoms share attributes. Retain interned tokens rather
        // than n*n deep copies of their nested legacy certificate structure.
        node_tokens.push_back(&*all.insert(value).first);
      }
    }
    using State = std::pair<int64_t, std::string>;
    std::vector<State> states[2];
    std::vector<int32_t> *matrices[] = {&edges_r, &edges_p};
    const Endpoint *sides[] = {&a, &b};
    for (int side = 0; side < 2; ++side) {
      std::map<Token, State> unique;
      const auto &e = *sides[side];
      for (int k = 0; k < n * n; ++k)
        unique.emplace(tuple({weight_token(e.weights[k]), string_token(e.bond_stereo[k])}),
                       State{e.weights[k], e.bond_stereo[k]});
      std::map<State, int> ids;
      for (const auto &[token, state] : unique) {
        (void)token;
        ids[state] = states[side].size(); states[side].push_back(state);
      }
      for (int k = 0; k < n * n; ++k)
        matrices[side]->push_back(ids.at({e.weights[k], e.bond_stereo[k]}));
    }
    std::vector<Token> edge_tokens;
    std::vector<bool> present;
    for (auto left : states[0])
      for (auto right : states[1]) {
        auto token = tuple({weight_token(left.first), weight_token(right.first),
                            string_token(left.second), string_token(right.second)});
        edge_tokens.push_back(token);
        present.push_back(left.first || right.first);
        if (present.back()) all.insert(token);
      }
    if (all.size() >= 65536)
      throw std::invalid_argument("Native reaction class palette exceeds uint16");
    std::map<Token, int> ranks;
    for (const auto &token : all) {
      ranks[token] = json.size(); json.push_back(token.json());
    }
    for (const auto *token : node_tokens) nodes.push_back(ranks.at(*token));
    for (size_t k = 0; k < edge_tokens.size(); ++k)
      edge_pairs.push_back(present[k] ? ranks.at(edge_tokens[k]) + 1 : 0);
    for (const auto &value : json) pointers.push_back(value.c_str());
    problem = {n, nodes.data(), edges_r.data(), edges_p.data(),
               static_cast<int32_t>(states[0].size()), static_cast<int32_t>(states[1].size()),
               edge_pairs.data(), static_cast<int32_t>(pointers.size()), pointers.data()};
  }
};

struct OutputRecord {
  std::string before_count, after_count;
  int64_t count;
};
struct Output {
  const Endpoint &a, &b;
  const Budget &budget;
  std::vector<OutputRecord> records;
  std::map<std::string,size_t> positions;
  bool timed_out = false;
  std::exception_ptr error;
};
struct KernelCandidates {
  int n;
  std::vector<int32_t> mappings;
  std::exception_ptr error;
};
int remember_kernel_candidate(const int32_t *mapping, int64_t, void *data) {
  auto &buffer = *static_cast<KernelCandidates *>(data);
  try {
    buffer.mappings.insert(buffer.mappings.end(), mapping, mapping + buffer.n);
    return 0;
  } catch (...) { buffer.error = std::current_exception(); return 1; }
}
int receive_class(const int32_t *mapping, int64_t cost2, int64_t count,
                  const unsigned char *packed, int64_t size, const char *identifier, void *data) {
  auto &out = *static_cast<Output *>(data);
  try {
    if (out.budget.reason()) { out.timed_out = true; return 1; }
    std::string key(reinterpret_cast<const char*>(packed),static_cast<size_t>(size));
    auto existing=out.positions.find(key);
    if(existing!=out.positions.end()) {
      auto &record=out.records[existing->second];
      if(count<record.count) throw std::runtime_error("Nonmonotone native class count");
      record.count=count;
      return 0;
    }
    RDKit::RWMol left(*out.a.molecule), right(*out.b.molecule);
    int n = out.a.labels.size();
    std::vector<int> inverse(n, -1);
    for (int i = 0; i < n; ++i) inverse[mapping[i]] = i + 1;
    for (auto atom : left.atoms()) atom->setAtomMapNum(atom->getIdx() + 1);
    for (auto atom : right.atoms()) atom->setAtomMapNum(inverse[atom->getIdx()]);
    std::ostringstream record;
    record << "{\"mapping\":[";
    for (int i = 0; i < n; ++i) record << (i ? "," : "") << mapping[i];
    record << "],\"distance\":" << cost2 / 2.0
           << ",\"class_id\":" << quote(identifier)
           << ",\"observed_mapping_count\":";
    const std::string before_count=record.str();
    record.str("");record.clear();
    record << ",\"mapped_reaction\":"
           << quote(RDKit::MolToSmiles(left) + ">>" + RDKit::MolToSmiles(right));
    for (int side = 0; side < 2; ++side) {
      record << (side == 0 ? ",\"reactant_only_maps\":[" : ",\"product_only_maps\":[");
      bool first = true;
      for (int i = 0; i < n; ++i) {
        bool selected = side == 0 ? i < static_cast<int>(left.getNumAtoms()) && mapping[i] >= static_cast<int>(right.getNumAtoms())
                                 : i >= static_cast<int>(left.getNumAtoms()) && mapping[i] < static_cast<int>(right.getNumAtoms());
        if (selected) { record << (first ? "" : ",") << i + 1; first = false; }
      }
      record << ']';
    }
    record << '}';
    if (out.budget.reason()) { out.timed_out = true; return 1; }
    out.positions.emplace(std::move(key),out.records.size());
    out.records.push_back({before_count,record.str(),count});
    return 0;
  } catch (...) { out.error = std::current_exception(); return 1; }
}

std::string statistics_json(const Statistics &s) {
  std::ostringstream out;
  out << '{';
#define FIELD(name) out << "\"" #name "\":" << s.name << ',';
  FIELD(backend) FIELD(status) FIELD(reason) FIELD(minimum_proven)
  FIELD(search_complete) FIELD(classification_complete) FIELD(hydrogen_complete)
  FIELD(heuristic_finished) FIELD(minimum2) FIELD(incumbent2) FIELD(lower_bound2)
  FIELD(nodes) FIELD(leaves) FIELD(candidate_count) FIELD(class_count)
  FIELD(labeled_count) FIELD(peak_frontier) FIELD(peak_frontier_bytes)
  FIELD(peak_memory_bytes) FIELD(model_seconds) FIELD(optimization_seconds)
  FIELD(enumeration_seconds) FIELD(classification_seconds) FIELD(callback_seconds)
#undef FIELD
  out << "\"elapsed_seconds\":" << s.elapsed_seconds << '}';
  return out.str();
}
} // namespace

extern "C" {
int synkit_mapper_reaction_abi(void) { return 1; }
const char *synkit_mapper_reaction_error(void) { return last_error.c_str(); }
const char *synkit_mapper_rdkit_version(void) { return RDKit::rdkitVersion; }
void synkit_mapper_reaction_free(char *p) { delete[] p; }
int synkit_mapper_reaction_solve(const char *reaction, const SynkitReactionOptions *o,
                                 const SynkitMapperRequest *request, char **output) {
  const auto invocation_start = Clock::now();
  double preparation_time = -1;
  int augmented_nodes = 0;
  std::string agents;
  last_error.clear();
  if (output) *output = nullptr;
  try {
    if (!reaction || !o || !request || !output || o->abi_version != 1 ||
        request->abi_version != 1 || !std::isfinite(request->seconds) ||
        (request->seconds < 0 && request->seconds != -1) ||
        (o->lewis_objective && (!o->explicit_hydrogens || o->binary)) ||
        (o->max_canonical_nodes != -1 && o->max_canonical_nodes < 1) ||
        (o->max_hydrogen_plans != -1 && o->max_hydrogen_plans < 1))
      throw std::invalid_argument("Invalid standalone reaction options");
    for (int flag : {o->explicit_hydrogens, o->lewis_objective, o->dummy_balance,
                     o->binary, o->hydrogen_quotient, o->empty_numeric_shell})
      if (flag != 0 && flag != 1) throw std::invalid_argument("Reaction flags must be boolean");
    Budget budget(*request);
    std::string input(reaction);
    size_t first = input.find('>'), last = input.rfind('>');
    if (first == std::string::npos || first == last || first == 0 ||
        last + 1 == input.size() || input.find('>', first + 1) != last)
      throw std::invalid_argument("Expected reactants>agents>products");
    agents = input.substr(first + 1, last - first - 1);
    Endpoint a = parse(input.substr(0, first), o->explicit_hydrogens, budget);
    Endpoint b = parse(input.substr(last + 1), o->explicit_hydrogens, budget);
    prepare(a, b, *o, budget);
    augmented_nodes = a.labels.size();
    Problem problem{1, static_cast<int32_t>(a.labels.size()), a.weights.data(), b.weights.data(),
                    a.types.data(), b.types.data(), a.node_colors.data(), b.node_colors.data(),
                    a.edge_colors.data(), b.edge_colors.data()};
    validate(problem, *request);
    Palette palette(a, b, budget);
    const double preparation_seconds = budget.seconds();
    preparation_time = preparation_seconds;
    Request adjusted = *request;
    adjusted.seconds = request->seconds < 0 ? -1 : budget.remaining();
    adjusted.classify = request->query != SYNKIT_MINIMUM_PROOF;
    Statistics stats{};
    SynkitClassStatistics class_stats{};
    SynkitHydrogenStatistics hstats{};
    bool hydrogen_fallback = false;
    int64_t kernel_seed_candidates = 0;
    int kernel_seed_finished = 0;
    double kernel_seed_seconds = 0;
    Output records{a, b, budget, {}, {}, false, nullptr};
    int result;
    if (o->empty_numeric_shell) {
      if (request->query != SYNKIT_EXACT_DISTANCE)
        throw std::invalid_argument("Empty-shell flag requires an exact numeric query");
      initialize(stats, adjusted);
      stats.search_complete = stats.classification_complete = stats.hydrogen_complete = 1;
      stats.class_count = stats.labeled_count = 0;
    } else if (request->backend == SYNKIT_KBB || request->backend == SYNKIT_KMILP) {
      if (o->hydrogen_quotient)
        throw std::invalid_argument("Hydrogen quotient is a full-space exact option, not a kernel restriction");
      Request seed = adjusted;
      seed.backend = SYNKIT_SLAP; seed.query = SYNKIT_MINIMUM_CLASSES;
      seed.symmetry = seed.classify = 0;
      seed.max_classes = seed.max_frontier = seed.memory_limit_bytes = -1;
      seed.max_candidates = 10000;
      std::vector<int32_t> flags(problem.n, 0);
      seed.slap_stop_flags = flags.data();
      KernelCandidates candidates{problem.n, {}, nullptr};
      Statistics seed_stats{};
      result = synkit_mapper_solve(&problem, &seed, remember_kernel_candidate, &candidates, &seed_stats);
      if (candidates.error) std::rethrow_exception(candidates.error);
      if (result) throw std::runtime_error(synkit_mapper_core_error());
      kernel_seed_seconds = seed_stats.elapsed_seconds;
      kernel_seed_finished = seed_stats.heuristic_finished;
      kernel_seed_candidates = candidates.mappings.size() / problem.n;
      if (!kernel_seed_candidates) {
        stats = seed_stats; stats.backend = request->backend;
        stats.classification_complete = stats.search_complete = 0;
      } else {
        std::vector<uint8_t> allowed(problem.n * problem.n);
        result = synkit_mapper_kernel_domains(problem.n, kernel_seed_candidates,
                 candidates.mappings.data(), problem.types_r, problem.types_p, 4, allowed.data());
        if (result) throw std::runtime_error(synkit_mapper_assignment_error());
        adjusted.seconds = request->seconds < 0 ? -1 : budget.remaining();
        if (request->max_states >= 0) {
          if (seed_stats.nodes >= request->max_states)
            throw Stop{SYNKIT_STOP_STATES};
          adjusted.max_states = request->max_states - seed_stats.nodes;
        }
        if (request->query == SYNKIT_MINIMUM_PROOF)
          result = synkit_mapper_solve_domains(&problem, &adjusted, allowed.data(), nullptr, nullptr, &stats);
        else
          result = synkit_mapper_solve_domains_classified_updates(&problem, &adjusted, allowed.data(),
                   &palette.problem, o->max_canonical_nodes, receive_class, &records, &stats, &class_stats);
        if (result) {
          std::string message = synkit_mapper_core_error();
          if (message.empty()) message = synkit_mapper_classification_error();
          throw std::runtime_error(message.empty() ? "Native kernel pipeline failed" : message);
        }
        stats.nodes += seed_stats.nodes;
      }
    } else if (o->hydrogen_quotient) {
      std::vector<int32_t> sa(a.labels.size()), sb(b.labels.size());
      for (size_t i = 0; i < sa.size(); ++i) {
        sa[i] = !a.stereo[i].empty() || a.stereo_anchors[i];
        sb[i] = !b.stereo[i].empty() || b.stereo_anchors[i];
        if (input.find('@') != std::string::npos || input.find('|') != std::string::npos)
          sa[i] = sb[i] = 1;
      }
      SynkitHydrogenAttributes attrs{a.atomic_numbers.data(), b.atomic_numbers.data(),
          a.isotopes.data(), b.isotopes.data(), a.charges.data(), b.charges.data(),
          a.radicals.data(), b.radicals.data(), a.dummies.data(), b.dummies.data(),
          sa.data(), sb.data()};
      if (request->query == SYNKIT_MINIMUM_PROOF)
        result = synkit_mapper_hydrogen_solve(&problem, &attrs, &adjusted,
                 o->max_hydrogen_plans, nullptr, nullptr, &stats, &hstats);
      else
        result = synkit_mapper_hydrogen_solve_classified_updates(&problem, &attrs, &adjusted,
                 &palette.problem, o->max_hydrogen_plans, o->max_canonical_nodes,
                 receive_class, &records, &stats, &hstats, &class_stats);
      if (result) throw std::runtime_error(synkit_mapper_hydrogen_error());
      if (stats.status == SYNKIT_UNSUPPORTED) {
        hydrogen_fallback = true;
        adjusted.seconds = request->seconds < 0 ? -1 : budget.remaining();
        if (hstats.eligibility_reason == SYNKIT_H_CLASS_ATTRIBUTES)
          adjusted.symmetry = 0;
        if (request->query == SYNKIT_MINIMUM_PROOF)
          result = synkit_mapper_solve(&problem, &adjusted, nullptr, nullptr, &stats);
        else
          result = synkit_mapper_solve_classified_updates(&problem, &adjusted, &palette.problem,
                   o->max_canonical_nodes, receive_class, &records, &stats, &class_stats);
        if (result) {
          std::string message = synkit_mapper_core_error();
          if (message.empty()) message = synkit_mapper_classification_error();
          throw std::runtime_error(message.empty() ? "Native H fallback failed" : message);
        }
      }
    } else if (request->query == SYNKIT_MINIMUM_PROOF) {
      result = synkit_mapper_solve(&problem, &adjusted, nullptr, nullptr, &stats);
      if (result) throw std::runtime_error(synkit_mapper_core_error());
    } else {
      result = synkit_mapper_solve_classified_updates(&problem, &adjusted, &palette.problem,
               o->max_canonical_nodes, receive_class, &records, &stats, &class_stats);
      if (result) {
        std::string message = synkit_mapper_core_error();
        if (message.empty()) message = synkit_mapper_classification_error();
        throw std::runtime_error(message.empty() ? "Native reaction pipeline failed" : message);
      }
    }
    if (records.error) std::rethrow_exception(records.error);
    if (records.timed_out) {
      stats.status = SYNKIT_LIMIT; stats.reason = SYNKIT_STOP_TIME;
      stats.classification_complete = 0;
    }
    stats.elapsed_seconds = budget.seconds();
    bool complete = stats.search_complete && stats.classification_complete &&
                    stats.hydrogen_complete && request->query != SYNKIT_MINIMUM_PROOF &&
                    request->backend != SYNKIT_SLAP;
    std::ostringstream json;
    json << std::setprecision(17) << "{\"pipeline\":\"rdkit_cpp\",\"rdkit_version\":"
         << quote(RDKit::rdkitVersion) << ",\"complete\":" << (complete ? "true" : "false")
         << ",\"minimum_proven\":" << (stats.minimum_proven ? "true" : "false")
         << ",\"minimum_cost\":";
    if (stats.minimum_proven) json << stats.minimum2 / 2.0; else json << "null";
    json << ",\"objective\":" << quote(o->lewis_objective ? "lewis" : "bond")
         << ",\"balance\":" << quote(o->dummy_balance ? "dummy" : "strict")
         << ",\"hydrogen\":" << quote(o->explicit_hydrogens ? "explicit" : "heavy")
         << ",\"agents\":" << quote(agents)
         << ",\"hydrogen_quotient\":" << (o->hydrogen_quotient ? "true" : "false")
         << ",\"hydrogen_quotient_used\":" << (o->hydrogen_quotient && !hydrogen_fallback && hstats.supported ? "true" : "false")
         << ",\"hydrogen_quotient_fallback\":" << (hydrogen_fallback ? "true" : "false")
         << ",\"hydrogen_supported\":" << (hstats.supported ? "true" : "false")
         << ",\"hydrogen_eligibility_reason\":" << hstats.eligibility_reason
         << ",\"augmented_nodes\":" << problem.n
         << ",\"search_scope\":" << quote(request->backend == SYNKIT_KBB || request->backend == SYNKIT_KMILP ? "restricted_kernel" : request->backend == SYNKIT_SLAP ? "heuristic_candidates" : "full_space")
         << ",\"kernel_seed_candidates\":" << kernel_seed_candidates
         << ",\"kernel_seed_finished\":" << (kernel_seed_finished ? "true" : "false")
         << ",\"kernel_seed_seconds\":" << kernel_seed_seconds
         << ",\"preparation_seconds\":" << preparation_seconds
         << ",\"statistics\":" << statistics_json(stats) << ",\"classes\":[";
    for (size_t i = 0; i < records.records.size(); ++i) {
      const auto &record=records.records[i];
      json << (i ? "," : "") << record.before_count << record.count << record.after_count;
    }
    json << "]}";
    auto text = json.str();
    *output = new char[text.size() + 1];
    std::memcpy(*output, text.c_str(), text.size() + 1);
    return 0;
  } catch (const Stop &stop) {
    Statistics stats{};
    initialize(stats, *request);
    stopped(stats, stop.reason);
    stats.classification_complete = stats.hydrogen_complete = 0;
    stats.elapsed_seconds = elapsed(invocation_start);
    std::ostringstream json;
    json << std::setprecision(17) << "{\"pipeline\":\"rdkit_cpp\",\"rdkit_version\":"
         << quote(RDKit::rdkitVersion) << ",\"complete\":false,\"minimum_proven\":false,\"minimum_cost\":null"
         << ",\"objective\":" << quote(o->lewis_objective ? "lewis" : "bond")
         << ",\"balance\":" << quote(o->dummy_balance ? "dummy" : "strict")
         << ",\"hydrogen\":" << quote(o->explicit_hydrogens ? "explicit" : "heavy")
         << ",\"agents\":" << quote(agents)
         << ",\"hydrogen_quotient\":" << (o->hydrogen_quotient ? "true" : "false")
         << ",\"hydrogen_quotient_used\":false,\"hydrogen_quotient_fallback\":false,\"hydrogen_supported\":false"
         << ",\"hydrogen_eligibility_reason\":0,\"augmented_nodes\":" << augmented_nodes
         << ",\"search_scope\":" << quote(request->backend == SYNKIT_KBB || request->backend == SYNKIT_KMILP ? "restricted_kernel" : request->backend == SYNKIT_SLAP ? "heuristic_candidates" : "full_space")
         << ",\"kernel_seed_candidates\":0,\"kernel_seed_finished\":false,\"kernel_seed_seconds\":0"
         << ",\"preparation_seconds\":" << (preparation_time < 0 ? stats.elapsed_seconds : preparation_time)
         << ",\"classes\":[],\"statistics\":" << statistics_json(stats) << '}';
    std::string text = json.str();
    if (!output) return -1;
    *output = new char[text.size() + 1];
    std::memcpy(*output, text.c_str(), text.size() + 1);
    return 0;
  } catch (const std::exception &e) { last_error = e.what(); return -1; }
}
}
