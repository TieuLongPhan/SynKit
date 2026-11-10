#include "synkit_partial/partial.h"
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
int checks = 0;
void require(bool value, const char *message) {
  ++checks;
  if (!value) throw std::runtime_error(message);
}
SkpState state(int element, int charge, int hydrogen, int radical, int lp, int ve) {
  return {element, charge, hydrogen, radical, lp, ve, 5};
}
SkpRuleAtom rule_atom(SkpState left, SkpState right, int baseline, int atom_map) {
  SkpRuleAtom atom{};
  atom.left = left; atom.right = right; atom.atom_map = atom_map;
  atom.baseline_lp = baseline; atom.match_mask = 63;
  atom.role = -1; atom.owner = -1; atom.capacity = 1; atom.budget = -1;
  atom.identity = -1; atom.stereo_slot = -1;
  atom.elements = atom.charges = atom.radicals = atom.bond_orders = {0, -1};
  return atom;
}
}
int main() {
  try {
    require(skp_abi() == 1, "ABI mismatch");
    SkpAtom host[] = {{state(1, 0, 3, 0, 0, 4), 1}, {state(2, 0, 0, 0, 3, 7), 2}};
    SkpEdge edges[] = {{0, 1, {2, 2, 0}}};
    SkpRuleAtom atoms[] = {
      rule_atom(state(1, 0, 0, 0, 0, 4), state(1, 0, 0, 0, 0, 4), 0, 1),
      rule_atom(state(2, 0, 0, 0, 0, 7), state(2, -1, 0, 0, 1, 7), 3, 2),
      rule_atom(state(3, -1, 0, 0, 1, 6), state(3, 0, 0, 0, 0, 6), 2, 3),
    };
    SkpRuleEdge rule_edges[] = {
      {0, 1, {2, 2, 0}, {0, 0, 0}, -1},
      {0, 2, {0, 0, 0}, {2, 2, 0}, -1},
    };
    SkpRequest request{host, 2, edges, 1, atoms, 3, rule_edges, 2, nullptr, 0, 1, -1, 0};
    auto result = skp_execute(&request);
    require(skp_status(result) == 0, skp_error(result));
    require(skp_application_count(result) == 1, "one partial binding expected");
    int64_t n = 0;
    const auto *product = skp_atoms(result, 0, &n);
    require(n == 4, "oxygen and one unit port must be introduced");
    require(product[2].left.lp == 3 && product[2].right.lp == 2, "absolute LP baseline lost");
    require(product[2].left.charge == -1 && product[2].right.charge == 0, "oxygen charge invalid");
    require(product[3].port == 1 && product[3].owner == 2, "port owner invalid");
    require(!(product[3].left.flags & 4), "symbolic port asserted valence electrons");
    require(skp_mapping(result, 0)[2] == -1, "observed and introduced atoms conflated");
    skp_free(result);
    request.candidate_limit = 0;
    result = skp_execute(&request);
    require(skp_status(result) == 2 && skp_application_count(result) == 0, "limit returned partial results");
    skp_free(result); request.candidate_limit = -1;
    atoms[2].right.charge = 1;
    result = skp_execute(&request);
    require(skp_status(result) == 3 && skp_application_count(result) == 0, "bad completion accepted");
    skp_free(result); atoms[2].right.charge = 0;
    edges[0].state = {2, 1, 1};
    result = skp_execute(&request);
    require(skp_status(result) == 1 && skp_application_count(result) == 0, "half-pair state accepted");
    skp_free(result); edges[0].state = {2, 2, 0};
    atoms[0].elements = {0, 1};
    result = skp_execute(&request);
    require(skp_status(result) == 1 && skp_application_count(result) == 0, "out-of-range domain accepted");
    skp_free(result); atoms[0].elements = {0, -1};
    host[1].atom_map = 1;
    result = skp_execute(&request);
    require(skp_status(result) == 1, "duplicate atom maps accepted");
    skp_free(result);
    result = skp_execute(nullptr);
    require(skp_status(result) == 1 && skp_application_count(result) == 0, "null request accepted");
    skp_free(result);
    {
      SkpAtom boundary_host[] = {
        {state(1, 0, 3, 0, 0, 4), 1},
        {state(1, 0, 3, 0, 0, 4), 2},
      };
      SkpEdge boundary_host_edges[] = {{0, 1, {2, 2, 0}}};
      SkpState symbol{0, 0, 0, 0, 0, 0, 1};
      SkpRuleAtom boundary_atoms[] = {
        rule_atom(state(1, 0, 0, 0, 0, 4), state(1, 0, 0, 0, 0, 4), 0, 1),
        rule_atom(symbol, symbol, 0, 2),
      };
      boundary_atoms[1].role = 0;
      SkpRuleEdge boundary_edges[] = {{0, 1, {2, 2, 0}, {2, 2, 0}, -1}};
      SkpRequest boundary_request{
        boundary_host, 2, boundary_host_edges, 1, boundary_atoms, 2,
        boundary_edges, 1, nullptr, 0, 0, -1, 0,
      };
      result = skp_execute(&boundary_request);
      require(skp_status(result) == 0 && skp_application_count(result) == 2,
              "persistent wildcard boundary rejected");
      skp_free(result);
      // Equal total order does not make an electron-coordinate edit read-only.
      boundary_edges[0].right = {2, 0, 2};
      result = skp_execute(&boundary_request);
      require(skp_status(result) == 1 && skp_application_count(result) == 0,
              "wildcard boundary sigma/pi edit accepted");
      skp_free(result);
      boundary_edges[0].right = boundary_edges[0].left;
      int64_t SkpState::* const state_fields[] = {
        &SkpState::hcount, &SkpState::lp, &SkpState::radical, &SkpState::charge,
      };
      for (auto field : state_fields) {
        boundary_atoms[1].right = symbol;
        ++(boundary_atoms[1].right.*field);
        result = skp_execute(&boundary_request);
        require(skp_status(result) == 1 && skp_application_count(result) == 0,
                "wildcard boundary atom-state edit accepted");
        skp_free(result);
      }
      boundary_atoms[1].right = symbol;
      boundary_atoms[1].right.flags |= 2;
      result = skp_execute(&boundary_request);
      require(skp_status(result) == 1 && skp_application_count(result) == 0,
              "wildcard boundary aromaticity edit accepted");
      skp_free(result);
    }
    {
      // Aromatize cyclohexane without supplying any product double bonds.
      // A second benzene component must retain its exact source phase.
      SkpAtom cyclic_host[12];
      SkpEdge cyclic_edges[12];
      SkpRuleAtom cyclic_atoms[6];
      SkpRuleEdge cyclic_rule_edges[6];
      for (int i = 0; i < 6; ++i) {
        auto saturated = state(1, 0, 2, 0, 0, 4);
        auto aromatic = state(1, 0, 1, 0, 0, 4); aromatic.flags |= 2;
        cyclic_host[i] = {saturated, i + 1};
        cyclic_host[i + 6] = {aromatic, i + 7};
        cyclic_atoms[i] = rule_atom(saturated, aromatic, 0, i + 1);
        cyclic_edges[i] = {i, (i + 1) % 6, {2, 2, 0}};
        cyclic_edges[i + 6] = {i + 6, (i + 1) % 6 + 6, {3, 2, 2 * (i % 2)}};
        cyclic_rule_edges[i] = {i, (i + 1) % 6, {2, 2, 0}, {3, 2, 0}, -1};
      }
      SkpRequest cyclic_request{cyclic_host, 12, cyclic_edges, 12, cyclic_atoms, 6,
                                cyclic_rule_edges, 6, nullptr, 0, 0, -1, 0};
      result = skp_execute(&cyclic_request);
      require(skp_status(result) == 0 && skp_application_count(result) == 12,
              "native aromatization failed");
      const auto *aromatized_atoms = skp_atoms(result, 0, &n);
      bool flags_correct = true, spectator_atoms_unchanged = true;
      for (int i = 0; i < 6; ++i)
        flags_correct = flags_correct && (aromatized_atoms[i].right.flags & 2)
                        && aromatized_atoms[i].right.hcount == 1;
      for (int i = 6; i < 12; ++i) {
        auto a = aromatized_atoms[i].left, b = aromatized_atoms[i].right;
        spectator_atoms_unchanged = spectator_atoms_unchanged
          && a.flags == b.flags && a.hcount == b.hcount && a.charge == b.charge;
      }
      require(flags_correct, "product aromatic flags were not propagated");
      require(spectator_atoms_unchanged, "aromatic spectator atom state changed");
      const auto *aromatized_edges = skp_edges(result, 0, &n);
      int pi_valence[6] = {};
      bool spectator_phase_unchanged = true;
      for (int i = 0; i < n; ++i) {
        auto edge = aromatized_edges[i];
        if (edge.u < 6 && edge.v < 6) {
          pi_valence[edge.u] += edge.right.pi2;
          pi_valence[edge.v] += edge.right.pi2;
        } else {
          spectator_phase_unchanged = spectator_phase_unchanged
            && edge.left.order2 == edge.right.order2
            && edge.left.sigma2 == edge.right.sigma2 && edge.left.pi2 == edge.right.pi2;
        }
      }
      bool balanced = true;
      for (int value : pi_valence) balanced = balanced && value == 2;
      require(balanced, "native Kekule assignment violates local carbon valence");
      require(spectator_phase_unchanged, "unchanged aromatic spectator was rephased");
      skp_free(result);
      // Five unit pi demands cannot be paired by any common assignment.
      cyclic_atoms[0].right.hcount = 2;
      result = skp_execute(&cyclic_request);
      require(skp_status(result) == 3 && skp_application_count(result) == 0,
              "inconsistent global aromatic demands were accepted");
      skp_free(result);
      // Reverse direction: aromatic bonds and flags become ordinary singles.
      for (int i = 0; i < 6; ++i) {
        cyclic_host[i].state = cyclic_host[i + 6].state;
        cyclic_atoms[i] = rule_atom(cyclic_host[i].state, state(1, 0, 2, 0, 0, 4), 0, i + 1);
        cyclic_edges[i].state = {3, 2, 2 * (i % 2)};
        cyclic_rule_edges[i].left = cyclic_edges[i].state;
        cyclic_rule_edges[i].right = {2, 2, 0};
      }
      result = skp_execute(&cyclic_request);
      require(skp_status(result) == 0 && skp_application_count(result) > 0,
              "native dearomatization failed");
      const auto *saturated_atoms = skp_atoms(result, 0, &n);
      bool removed_flags = true;
      for (int i = 0; i < n; ++i) if (saturated_atoms[i].origin >= 0)
        removed_flags = removed_flags && !(saturated_atoms[i].right.flags & 2)
                        && saturated_atoms[i].right.hcount == 2;
      require(removed_flags, "dearomatization retained aromatic atom flags");
      skp_free(result);
    }
    {
      // Path vertices are numbered with all even positions first. A static
      // assignment order explores disconnected permutations; frontier growth
      // must enumerate exactly the 48 oriented embeddings into this cycle.
      SkpAtom path_host[24];
      SkpEdge path_host_edges[24];
      SkpRuleAtom path_atoms[12];
      SkpRuleEdge path_edges[11];
      for (int i = 0; i < 24; ++i) {
        path_host[i] = {state(1, 0, 2, 0, 0, 4), i + 1};
        path_host_edges[i] = {i, (i + 1) % 24, {2, 2, 0}};
      }
      for (int i = 0; i < 12; ++i)
        path_atoms[i] = rule_atom(state(1, 0, 0, 0, 0, 4),
                                   state(1, 0, 0, 0, 0, 4), 0, i + 1);
      auto path_index = [](int i) { return i % 2 ? 6 + i / 2 : i / 2; };
      for (int i = 0; i < 11; ++i)
        path_edges[i] = {path_index(i), path_index(i + 1), {2, 2, 0}, {2, 2, 0}, -1};
      SkpRequest path_request{path_host, 24, path_host_edges, 24, path_atoms, 12,
                              path_edges, 11, nullptr, 0, 0, 100, 0};
      result = skp_execute(&path_request);
      require(skp_status(result) == 0 && skp_application_count(result) == 48,
              "connected frontier failed to enumerate path embeddings");
      skp_free(result);
    }
    {
      // Observed substitution omits the bromide product. Observation absence
      // must never be treated as a material zero-electron atom or a query.
      SkpState absent{};
      SkpAtom observed_host[] = {{state(1, 0, 3, 0, 0, 4), 1},
                                {state(2, 0, 0, 0, 3, 7), 2},
                                {state(3, -1, 1, 0, 3, 6), 3}};
      SkpEdge observed_edges[] = {{0, 1, {2, 2, 0}}};
      SkpRuleAtom observed_atoms[] = {
        rule_atom(observed_host[0].state, observed_host[0].state, 0, 1),
        rule_atom(observed_host[1].state, absent, 0, 2),
        rule_atom(observed_host[2].state, state(3, 0, 1, 0, 2, 6), 0, 3)};
      SkpRuleEdge observed_rule_edges[] = {{0, 1, {2, 2, 0}, {}, -1},
                                          {0, 2, {}, {2, 2, 0}, -1}};
      SkpRequest observed_request{observed_host, 3, observed_edges, 1, observed_atoms, 3,
                                  observed_rule_edges, 2, nullptr, 0, 0, -1, 0};
      result = skp_execute(&observed_request);
      require(skp_status(result) == 1, "closed entry point admitted a one-sided observation");
      skp_free(result);
      result = skp_execute_observed(&observed_request);
      require(skp_status(result) == 0 && skp_application_count(result) == 1,
              "observed forward substitution failed");
      const auto *observed_product = skp_atoms(result, 0, &n);
      require(!(observed_product[1].right.flags & 1) && !(observed_product[1].right.flags & 4),
              "unreported bromide was asserted as a material atom");
      require(observed_product[2].right.charge == 0, "observed methanol charge is wrong");
      skp_free(result);
      SkpAtom reverse_host[] = {{observed_atoms[0].right, 1}, {observed_atoms[2].right, 2}};
      SkpEdge reverse_edges[] = {{0, 1, {2, 2, 0}}};
      for (auto &a : observed_atoms) { auto original = a.left; a.left = a.right; a.right = original; }
      for (auto &b : observed_rule_edges) { auto original = b.left; b.left = b.right; b.right = original; }
      observed_request.atoms = reverse_host; observed_request.n_atoms = 2;
      observed_request.edges = reverse_edges;
      result = skp_execute_observed(&observed_request);
      require(skp_status(result) == 0 && skp_application_count(result) == 1,
              "observed backward substitution failed");
      const auto *reverse_product = skp_atoms(result, 0, &n);
      const auto *reverse_map = skp_mapping(result, 0);
      require(reverse_map[1] == -1 && n == 3 && !(reverse_product[2].left.flags & 1)
              && (reverse_product[2].right.flags & 1) && reverse_product[2].right.element == 2,
              "unobserved source bromine was matched or not restored");
      skp_free(result);
    }
    {
      // A wholly unreported cyclohexane has twelve internal automorphisms.
      // All remove the same whole host component, so one target representative
      // suffices even with a one-candidate limit. Methane is retained.
      SkpState absent{};
      SkpAtom whole_host[7]; SkpRuleAtom whole_atoms[7];
      SkpEdge whole_edges[6]; SkpRuleEdge whole_rule_edges[6];
      whole_host[0] = {state(1, 0, 4, 0, 0, 4), 1};
      whole_atoms[0] = rule_atom(whole_host[0].state, whole_host[0].state, 0, 1);
      for (int i = 1; i <= 6; ++i) {
        whole_host[i] = {state(1, 0, 2, 0, 0, 4), i+1};
        whole_atoms[i] = rule_atom(whole_host[i].state, absent, 0, i+1);
        whole_edges[i-1] = {i, i % 6 + 1, {2, 2, 0}};
        whole_rule_edges[i-1] = {i, i % 6 + 1, {2, 2, 0}, {}, -1};
      }
      SkpRequest whole_request{whole_host, 7, whole_edges, 6, whole_atoms, 7,
                               whole_rule_edges, 6, nullptr, 0, 0, 1, 0};
      result = skp_execute_observed(&whole_request);
      require(skp_status(result) == 0 && skp_application_count(result) == 1,
              "wholly unreported component automorphisms exhausted the cap");
      require(skp_stats(result)->candidates == 1, "disappearing component was not quotiented");
      skp_free(result);
      result = skp_execute_observed_first(&whole_request);
      require(skp_status(result) == 0 && skp_application_count(result) == 1,
              "native first witness failed");
      require(!skp_search_complete(result), "first witness was falsely reported as exhaustive");
      skp_free(result);
      whole_request.candidate_limit = 0;
      result = skp_execute_observed_first(&whole_request);
      require(skp_status(result) == 2 && skp_application_count(result) == 0,
              "first-witness search silently ignored its candidate bound");
      skp_free(result);
      SkpRequest identity_request{whole_host, 1, nullptr, 0, nullptr, 0,
                                  nullptr, 0, nullptr, 0, 1, 1, 0};
      result = skp_execute_observed(&identity_request);
      require(skp_status(result) == 0 && skp_application_count(result) == 1,
              "empty observed rule did not produce the identity");
      skp_free(result);
      result = skp_execute(&identity_request);
      require(skp_status(result) == 1, "closed profile admitted an empty rule");
      skp_free(result);
    }
    {
      SkpAtom typed_host[] = {{state(1, 0, 4, 0, 0, 4), 1}};
      SkpRuleAtom typed_atoms[] = {rule_atom(typed_host[0].state, typed_host[0].state, 0, 1)};
      SkpRequest typed_request{typed_host, 1, nullptr, 0, typed_atoms, 1,
                               nullptr, 0, nullptr, 0, 0, -1, 0};
      auto refused = [&]() {
        auto failure = skp_execute(&typed_request);
        bool correct = skp_status(failure) == 1 && skp_application_count(failure) == 0
                       && !skp_search_complete(failure);
        skp_free(failure);
        require(correct, "invalid standalone contract was not refused atomically");
      };
      typed_atoms[0].baseline_lp = std::numeric_limits<int64_t>::max(); refused();
      typed_atoms[0].baseline_lp = 1000000000;
      typed_atoms[0].left.lp = 1; refused();
      typed_atoms[0].baseline_lp = 0; typed_atoms[0].left.lp = 0;
      typed_atoms[0].side = 3; refused(); typed_atoms[0].side = 0;
      typed_atoms[0].materialization = 3; refused(); typed_atoms[0].materialization = 0;
      typed_atoms[0].virtual_kind = 1; refused(); typed_atoms[0].virtual_kind = 0;
      typed_atoms[0].match_mask = 64; refused(); typed_atoms[0].match_mask = 63;
      typed_host[0].state.flags = 4; refused(); typed_host[0].state.flags = 5;
      typed_request.partial = 2; refused(); typed_request.partial = 0;
      typed_host[0].state.ve = 6; typed_host[0].state.lp = 1;
      result = skp_execute(&typed_request);
      require(skp_status(result) == 0 && skp_application_count(result) == 0,
              "a concrete binding changed its atomic valence label");
      require(skp_search_complete(result), "exhausted valid empty search was marked incomplete");
      skp_free(result);
      typed_host[0].state = typed_atoms[0].left;
      typed_request.candidate_limit = 0;
      result = skp_execute(&typed_request);
      require(skp_status(result) == 2 && !skp_search_complete(result),
              "limited search was marked complete");
      skp_free(result);
    }
    {
      SkpAtom port_host[] = {{state(1, 0, 3, 0, 0, 4), 1},
                             {state(1, 0, 2, 0, 0, 4), 2},
                             {state(1, 0, 3, 0, 0, 4), 3}};
      SkpEdge port_host_edges[] = {{0, 1, {2, 2, 0}}, {1, 2, {2, 2, 0}}};
      SkpState symbol{0, 0, 0, 0, 0, 0, 1};
      SkpRuleAtom port_atoms[] = {
        rule_atom(state(1, 0, 0, 0, 0, 4), state(1, 0, 0, 0, 0, 4), 0, 1),
        rule_atom(symbol, symbol, 0, 2),
        rule_atom(state(1, 0, 0, 0, 0, 4), state(1, 0, 0, 0, 0, 4), 0, 3)};
      port_atoms[1].role = 0;
      SkpRuleEdge port_edges[] = {{0, 1, {2, 2, 0}, {2, 2, 0}, -1},
                                  {1, 2, {2, 2, 0}, {2, 2, 0}, -1}};
      SkpRequest port_request{port_host, 3, port_host_edges, 2, port_atoms, 3,
                              port_edges, 2, nullptr, 0, 0, -1, 0};
      result = skp_execute(&port_request);
      require(skp_status(result) == 0 && skp_application_count(result) == 2,
              "two-incidence query atom was treated as a leaf port");
      skp_free(result);
      port_atoms[1].role = 1; port_atoms[1].owner = 0;
      result = skp_execute(&port_request);
      require(skp_status(result) == 1 && !skp_application_count(result),
              "attachment port with two owner incidences was admitted");
      skp_free(result);
      port_atoms[1].role = 0; port_atoms[1].owner = 1;
      result = skp_execute(&port_request);
      require(skp_status(result) == 1 && !skp_application_count(result),
              "nonadjacent wildcard owner was admitted");
      skp_free(result);
      port_atoms[1].owner = -1; port_atoms[1].role = 4;
      result = skp_execute(&port_request);
      require(skp_status(result) == 1 && !skp_application_count(result),
              "uncompiled stereo port was admitted");
      skp_free(result);
      port_atoms[1].role = 0;
      port_host_edges[0].state = {3, 0, 0};
      result = skp_execute(&port_request);
      require(skp_status(result) == 1 && !skp_application_count(result),
              "aromatic display order hid invalid sigma/pi occupancies");
      skp_free(result);
    }
    std::cout << checks << " standalone native checks passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
