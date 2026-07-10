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
    ):
        """Map reaction SMILES; results contain mapped SMILES and cd."""

        if not HAS_RDKIT:
            raise ImportError("RDKit is required for processing SMILES")

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
