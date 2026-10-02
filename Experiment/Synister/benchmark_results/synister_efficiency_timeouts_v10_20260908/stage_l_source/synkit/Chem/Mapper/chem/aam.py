from ..slap.sequential import GraphMatcher
from ..exact.certificate import certify_result, certify_results_exact
from .smiles import (
    HAS_RDKIT,
    smiles2lgp,
    get_numbered_rxn_smiles,
    expand_reaction_center_hydrogens,
    reaction_center_atom_maps_from_signature,
    reaction_center_signature_from_mapped_smiles,
)
from .its import (
    dedup_mapped_rxns,
    mapped_rxn_is_electron_balanced,
)


def _hydrogen_modes(add_Hs):
    if add_Hs is True:
        return True, True, False
    if add_Hs is False:
        return False, False, False
    if isinstance(add_Hs, str) and add_Hs in {"reaction_center", "center"}:
        return False, True, True
    raise ValueError("add_Hs must be True, False, or 'reaction_center'")


def _result_hydrogen_mapped_smiles(rxn_smiles, map_nums_pair, selected_maps, binary):
    """Map reaction-center hydrogens in a constrained second pass."""
    expanded_rxn = expand_reaction_center_hydrogens(
        rxn_smiles,
        map_nums_pair,
        selected_maps,
    )
    mapper = AAMapper(binary=binary)
    try:
        mapper.map_smiles(
            expanded_rxn,
            add_Hs=False,
            break_sym="all",
            unique=True,
            certify=False,
            electron_balance=False,
            enumerate_exact=False,
        )
        return [r["smiles"] for r in mapper.results] or [expanded_rxn]
    except Exception:
        return [expanded_rxn]


def _effective_hcount_weight(rxn_smiles, hcount_weight, hcount_mode):
    """Return active H-count weight for a reaction."""
    if hcount_weight <= 0:
        return 0.0
    if hcount_mode in (None, "always"):
        return hcount_weight
    if hcount_mode in {"product_acid", "product-acid"}:
        from rdkit import Chem

        product = Chem.MolFromSmiles(rxn_smiles.split(">>", 1)[1])
        acid = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
        return (
            hcount_weight
            if product is not None and product.HasSubstructMatch(acid)
            else 0.0
        )
    raise ValueError("hcount_mode must be 'always' or 'product_acid'")


class AAMapper(GraphMatcher):
    """Reaction-SMILES AAM mapper."""

    def __init__(
        self,
        binary=True,
        max_lap_fingerprints=10000,
        cache_label_blocks=False,
        deterministic_labels=False,
    ):

        super().__init__(
            binary,
            max_lap_fingerprints=max_lap_fingerprints,
            cache_label_blocks=cache_label_blocks,
            deterministic_labels=deterministic_labels,
        )
        self._valfactor = 2
        self.enumeration_result = None

    def get_maps(self, lgp, break_sym_targets=None, interactive=False, base=None):
        """Graph mappings plus chemical distance."""
        super().get_maps(lgp, break_sym_targets, interactive, base)

        for r in self.results:
            if r["val"] % self._valfactor == 0:
                r["cd"] = int(r["val"] // self._valfactor)
            else:
                r["cd"] = r["val"] / self._valfactor

    def map_smiles(  # noqa: C901
        self,
        rxn_smiles,
        add_Hs=True,
        break_sym="heavy",
        interactive=False,
        unique=True,
        certify=False,
        electron_balance=False,
        enumerate_exact=False,
        hcount_weight=0.0,
        hcount_mode="always",
        repair_depth=0,
        repair_cap=128,
        repair_slack=0.0,
        repair_min_cd=4.0,
        repair_final=False,
        CD=None,
        max_bijections=1_000_000,
        time_limit_seconds=None,
        symmetry_pruning=False,
        max_symmetry_automorphisms=256,
        symmetry_timeout_seconds=0.25,
        expand_symmetry=False,
        collect_mappings=True,
        mapping_callback=None,
        compute_minimum_cost=True,
        assignment_upper_bound=True,
        atom_profile_pruning=True,
        seed_center_order=True,
    ):
        """Map reaction SMILES; results contain mapped SMILES and exact ``cd``.

        Set ``CD='minimal'`` to enumerate every globally minimal labeled atom
        mapping, or set ``CD`` to a non-negative number (for example ``CD=8``)
        to enumerate the complete exact-distance shell. These full-space modes
        are separate from ``enumerate_exact=True``, which retains the faster
        historical uncertainty-kernel enumeration. ``max_bijections`` guards
        the factorial full-space search and raises rather than returning an
        incomplete result; set it to ``None`` only when a deadline provides the
        guard. ``symmetry_pruning=True`` returns certified lex leaders under a
        bounded set of verified product automorphisms. If bounded discovery is
        incomplete, multiple representatives of a full orbit may remain. For
        memory-bounded enumeration, set ``collect_mappings=False`` and provide
        ``mapping_callback(mapping, cost)``; exact counts remain available on
        the enumeration result without retaining every mapping in memory.
        For a numeric shell, ``compute_minimum_cost=False`` avoids a separate
        global-minimum pass when that metadata is not required.
        """

        if not HAS_RDKIT:
            raise ImportError("RDKit is required for processing SMILES")
        if CD is not None and enumerate_exact:
            raise ValueError("CD and enumerate_exact cannot be used together")

        self.enumeration_result = None

        if not self.binary:
            self._valfactor = 4

        graph_add_hs, display_hs, reaction_center_hs = _hydrogen_modes(add_Hs)
        active_hcount_weight = _effective_hcount_weight(
            rxn_smiles,
            hcount_weight,
            hcount_mode,
        )

        lgp = smiles2lgp(rxn_smiles, add_Hs=graph_add_hs)

        targets = self._get_targets(break_sym, lgp[0].props["atomic numbers"])

        if interactive:
            natoms = len(lgp[0].labels)
            idxs_1based = list(range(1, natoms + 1))
            smis = get_numbered_rxn_smiles(
                rxn_smiles,
                [idxs_1based, idxs_1based],
                explicit_hs=graph_add_hs,
            ).split(">>")
            print("Reaction SMILES with 1-based indexes")
            print(smis[0])
            print(">>")
            print(smis[1])
            print()

        if CD is not None:
            self._enumerate_smiles_at_cd(
                rxn_smiles,
                lgp,
                CD=CD,
                max_bijections=max_bijections,
                time_limit_seconds=time_limit_seconds,
                symmetry_pruning=symmetry_pruning,
                max_symmetry_automorphisms=max_symmetry_automorphisms,
                symmetry_timeout_seconds=symmetry_timeout_seconds,
                expand_symmetry=expand_symmetry,
                collect_mappings=collect_mappings,
                mapping_callback=mapping_callback,
                compute_minimum_cost=compute_minimum_cost,
                assignment_upper_bound=assignment_upper_bound,
                atom_profile_pruning=atom_profile_pruning,
                seed_center_order=seed_center_order,
                graph_add_hs=graph_add_hs,
                display_hs=display_hs,
                reaction_center_hs=reaction_center_hs,
                unique=unique,
                electron_balance=electron_balance,
                certify=certify,
            )
            return

        self.get_maps(lgp, break_sym_targets=targets, interactive=interactive, base=1)

        if enumerate_exact:
            from ..exact.kernel import extract_kernel
            from ..exact.enumerate import (
                annotate_hcount_scores,
                complete_mapping,
                enumerate_kernel_optima,
                expand_results_by_local_swaps,
                improve_results_by_pair_swaps,
                improve_results_by_hcount_permutations,
            )
            from ..exact.certificate import Certificate
            from ..slap.lap import recover_mapping

            self.results = improve_results_by_pair_swaps(
                lgp,
                self.results,
                binary=self.binary,
                valfactor=self._valfactor,
            )
            kernel_seed_results = list(self.results)
            self.results = expand_results_by_local_swaps(
                lgp,
                self.results,
                binary=self.binary,
                valfactor=self._valfactor,
                depth=repair_depth,
                cap=repair_cap,
                slack=repair_slack,
                min_cd=repair_min_cd,
            )
            repair_applied = any(r.get("repair") == "local-swap" for r in self.results)
            repair_candidates = (
                list(self.results) if repair_final and repair_applied else []
            )
            self.results = improve_results_by_hcount_permutations(
                lgp,
                self.results,
                binary=self.binary,
                valfactor=self._valfactor,
                hcount_weight=active_hcount_weight,
            )
            if active_hcount_weight:
                self._annotate_smiles_results(
                    rxn_smiles,
                    graph_add_hs,
                    display_hs,
                    reaction_center_hs,
                    lgp,
                )
                if electron_balance:
                    balanced = []
                    rejected = []
                    for r in self.results:
                        ok = mapped_rxn_is_electron_balanced(r["its_smiles"])
                        r["electron_balanced"] = ok
                        if ok is not False:
                            balanced.append(r)
                        else:
                            rejected.append(r)
                    self.results = balanced or rejected
                if unique and not reaction_center_hs and len(self.results) > 1:
                    self.results = dedup_mapped_rxns(
                        self.results, smiles_key="its_smiles"
                    )
                if certify:
                    method = "hcount-biased" if active_hcount_weight else "local-repair"
                    for r in self.results:
                        r["certificate"] = Certificate(
                            upper_bound=float(r["cd"]),
                            lower_bound=float("nan"),
                            method=method,
                        )
                return

            kernel = extract_kernel(kernel_seed_results, lgp, binary=self.binary)
            enumerated = enumerate_kernel_optima(
                kernel,
                rxn_smiles=rxn_smiles,
                unique=unique,
                electron_balance=electron_balance,
                explicit_hs=display_hs,
                reaction_center_hs=reaction_center_hs,
            )
            self.enumeration_result = enumerated
            self.results = enumerated.results
            if repair_candidates:
                seen_mappings = {
                    tuple(r.get("mapping") or recover_mapping(r["lgp"]))
                    for r in self.results
                }
                repair_append = []
                for result in repair_candidates:
                    mapping = complete_mapping(
                        lgp,
                        result.get("mapping") or recover_mapping(result["lgp"]),
                        binary=self.binary,
                    )
                    key = tuple(mapping)
                    if key in seen_mappings:
                        continue
                    seen_mappings.add(key)
                    updated = dict(result)
                    updated["mapping"] = mapping
                    repair_append.append(updated)
                if repair_append:
                    exact_results = self.results
                    self.results = repair_append
                    self._annotate_smiles_results(
                        rxn_smiles,
                        graph_add_hs,
                        display_hs,
                        reaction_center_hs,
                        lgp,
                    )
                    self.results = exact_results + self.results
                    if electron_balance:
                        balanced = []
                        rejected = []
                        for r in self.results:
                            ok = mapped_rxn_is_electron_balanced(r["its_smiles"])
                            r["electron_balanced"] = ok
                            if ok is not False:
                                balanced.append(r)
                            else:
                                rejected.append(r)
                        self.results = balanced or rejected
                    if unique and not reaction_center_hs and len(self.results) > 1:
                        if len(self.results) <= 64:
                            self.results = dedup_mapped_rxns(
                                self.results,
                                smiles_key="its_smiles",
                            )
                        else:
                            seen_rxns = set()
                            deduped = []
                            for r in self.results:
                                key = r.get("its_smiles") or r.get("smiles")
                                if key in seen_rxns:
                                    continue
                                seen_rxns.add(key)
                                deduped.append(r)
                            self.results = deduped
            annotate_hcount_scores(
                lgp,
                self.results,
                binary=self.binary,
                hcount_weight=active_hcount_weight,
            )

            if certify:
                if active_hcount_weight:
                    method = "hcount-biased"
                elif repair_final and repair_applied:
                    method = "enum+repair"
                elif enumerated.proven_optimal and enumerated.enumeration_complete:
                    method = "enum-exact"
                elif enumerated.proven_optimal:
                    method = "single-exact"
                else:
                    method = "enum"
                for r in self.results:
                    lower_bound = (
                        float("nan") if active_hcount_weight else enumerated.cost
                    )
                    r["certificate"] = Certificate(
                        upper_bound=float(r["cd"]),
                        lower_bound=lower_bound,
                        method=method,
                    )
        else:
            self._annotate_smiles_results(
                rxn_smiles,
                graph_add_hs,
                display_hs,
                reaction_center_hs,
                lgp,
            )

        if not enumerate_exact and electron_balance:
            balanced = []
            rejected = []
            for r in self.results:
                ok = mapped_rxn_is_electron_balanced(r["its_smiles"])
                r["electron_balanced"] = ok
                if ok is not False:
                    balanced.append(r)
                else:
                    rejected.append(r)
            self.results = balanced or rejected

        if (
            not enumerate_exact
            and unique
            and not reaction_center_hs
            and len(self.results) > 1
        ):
            self.results = dedup_mapped_rxns(self.results, smiles_key="its_smiles")

        if not enumerate_exact:
            if certify == "exact":
                certify_results_exact(self.results, self.binary)
            elif certify:
                for r in self.results:
                    certify_result(r, self.binary)

    def enumerate_smiles(self, rxn_smiles, CD="minimal", **kwargs):
        """Enumerate a complete exact-CD shell and return its search record.

        This is the explicit counterpart to :meth:`map_smiles`. It also
        populates :attr:`results` with mapped reaction SMILES.
        """
        if "enumerate_exact" in kwargs:
            raise TypeError("enumerate_smiles does not accept enumerate_exact")
        if "CD" in kwargs:
            raise TypeError("CD was supplied more than once")
        self.map_smiles(rxn_smiles, CD=CD, enumerate_exact=False, **kwargs)
        return self.enumeration_result

    def _enumerate_smiles_at_cd(
        self,
        rxn_smiles,
        lgp,
        *,
        CD,
        max_bijections,
        time_limit_seconds,
        symmetry_pruning,
        max_symmetry_automorphisms,
        symmetry_timeout_seconds,
        expand_symmetry,
        collect_mappings,
        mapping_callback,
        compute_minimum_cost,
        assignment_upper_bound,
        atom_profile_pruning,
        seed_center_order,
        graph_add_hs,
        display_hs,
        reaction_center_hs,
        unique,
        electron_balance,
        certify,
    ):
        from ..exact.certificate import Certificate
        from ..exact.distance import enumerate_distance_mappings
        from ..exact.enumeration_result import mapping_to_lgp
        from ..slap.lap import recover_mapping

        initial_mapping = None
        if isinstance(CD, str) and CD.lower() == "minimal":
            try:
                # A heuristic result is only an incumbent. The exact search
                # still explores the complete atom-compatible assignment space.
                self.get_maps(lgp, break_sym_targets=None, interactive=False, base=1)
                if self.results:
                    candidate = recover_mapping(self.results[0]["lgp"])
                    if sorted(candidate) == list(range(len(candidate))):
                        initial_mapping = candidate
            except Exception:
                initial_mapping = None

        enumerated = enumerate_distance_mappings(
            lgp,
            CD=CD,
            binary=self.binary,
            max_bijections=max_bijections,
            time_limit_seconds=time_limit_seconds,
            certify=bool(certify),
            symmetry_pruning=symmetry_pruning,
            max_symmetry_automorphisms=max_symmetry_automorphisms,
            symmetry_timeout_seconds=symmetry_timeout_seconds,
            expand_symmetry=expand_symmetry,
            initial_mapping=initial_mapping,
            collect_mappings=collect_mappings,
            mapping_callback=mapping_callback,
            compute_minimum_cost=compute_minimum_cost,
            assignment_upper_bound=assignment_upper_bound,
            atom_profile_pruning=atom_profile_pruning,
            seed_center_order=seed_center_order,
        )
        self.enumeration_result = enumerated
        self.results = [
            {
                "lgp": mapping_to_lgp(lgp, mapping),
                "mapping": mapping,
                "cd": distance,
            }
            for mapping, distance in zip(enumerated.mappings, enumerated.distances)
        ]
        self._annotate_smiles_results(
            rxn_smiles,
            graph_add_hs,
            display_hs,
            reaction_center_hs,
            lgp,
        )

        if electron_balance:
            balanced = []
            rejected = []
            for result in self.results:
                accepted = mapped_rxn_is_electron_balanced(result["its_smiles"])
                result["electron_balanced"] = accepted
                (rejected if accepted is False else balanced).append(result)
            self.results = balanced or rejected

        if unique and not reaction_center_hs and len(self.results) > 1:
            self.results = dedup_mapped_rxns(self.results, smiles_key="its_smiles")

        metadata = {
            "target": enumerated.target,
            "scope": enumerated.scope,
            "complete": enumerated.complete,
            "status": enumerated.status,
            "truncation_reason": enumerated.truncation_reason,
            "minimum_cd": enumerated.minimum_cost,
            "total_bijections": enumerated.total_bijections,
            "maximum_cd_upper_bound": enumerated.maximum_cost_upper_bound,
            "visited_leaves": enumerated.visited_leaves,
            "pruned_branches": enumerated.pruned_branches,
            "elapsed_seconds": enumerated.elapsed_seconds,
            "visited_nodes": enumerated.visited_nodes,
            "symmetry_pruned_branches": enumerated.symmetry_pruned_branches,
            "symmetry_automorphism_count": enumerated.symmetry_automorphism_count,
            "symmetry_search_complete": enumerated.symmetry_search_complete,
            "lower_bound_pruned_branches": enumerated.lower_bound_pruned_branches,
            "upper_bound_pruned_branches": enumerated.upper_bound_pruned_branches,
            "raw_mapping_count": len(enumerated.mappings),
            "selected_mapping_count": enumerated.selected_mapping_count,
            "certificate_sha256": (
                enumerated.certificate.certificate_sha256
                if enumerated.certificate is not None
                else None
            ),
        }
        for result in self.results:
            result["enumeration"] = dict(metadata)
            if (
                certify
                and enumerated.complete
                and enumerated.minimum_cost is not None
            ):
                result["certificate"] = Certificate(
                    upper_bound=float(result["cd"]),
                    lower_bound=float(enumerated.minimum_cost),
                    method=(
                        "full-enumeration-minimal"
                        if enumerated.target == "minimal"
                        else "full-enumeration-exact-cd"
                    ),
                )

    def _annotate_smiles_results(
        self,
        rxn_smiles,
        graph_add_hs,
        display_hs,
        reaction_center_hs,
        lgp,
    ):
        mapped_results = []
        for r in self.results:
            r["its_smiles"] = get_numbered_rxn_smiles(
                rxn_smiles,
                [r["lgp"][0].labels, r["lgp"][1].labels],
                explicit_hs=False,
            )

        if reaction_center_hs:
            seen_signatures = set()
            unique_center_results = []
            for r in self.results:
                signature = reaction_center_signature_from_mapped_smiles(
                    r["its_smiles"]
                )
                r["_reaction_center_signature"] = signature
                if not signature:
                    unique_center_results.append(r)
                    continue
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                unique_center_results.append(r)
            self.results = unique_center_results

        for r in self.results:
            if graph_add_hs:
                from ..exact.enumerate import complete_mapping
                from ..slap.lap import recover_mapping

                mapping = complete_mapping(
                    lgp,
                    recover_mapping(r["lgp"]),
                    binary=self.binary,
                )
                react_nums = list(range(1, len(mapping) + 1))
                prod_nums = [0] * len(mapping)
                for i, p in enumerate(mapping):
                    prod_nums[p] = i + 1
                map_nums_pair = [react_nums, prod_nums]
            elif reaction_center_hs:
                selected_maps = reaction_center_atom_maps_from_signature(
                    r.get("_reaction_center_signature")
                    or reaction_center_signature_from_mapped_smiles(r["its_smiles"])
                )
                map_nums_pair = [r["lgp"][0].labels, r["lgp"][1].labels]
                smiles_list = _result_hydrogen_mapped_smiles(
                    rxn_smiles,
                    map_nums_pair,
                    selected_maps,
                    binary=self.binary,
                )
                for mapped_smiles in smiles_list:
                    rr = dict(r)
                    rr["smiles"] = mapped_smiles
                    mapped_results.append(rr)
                continue
            else:
                map_nums_pair = [r["lgp"][0].labels, r["lgp"][1].labels]

            r["smiles"] = get_numbered_rxn_smiles(
                rxn_smiles,
                map_nums_pair,
                explicit_hs=display_hs,
            )
            mapped_results.append(r)

        self.results = mapped_results

    def _get_targets(self, break_sym, atomic_nums):

        if break_sym == "heavy":
            return [i for i in range(len(atomic_nums)) if atomic_nums[i] > 1]
        elif break_sym == "all":
            return list(range(len(atomic_nums)))
        else:
            return break_sym
