"""Read-only run-state views for the RBL engine."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from synkit.Graph.Fusion import (
    FUSION_PROOF_SCHEMA,
    FusionCandidate,
    graph_identity_digest,
)
from synkit.Synthesis.RBL.policy import (
    AcceptanceTask,
    SearchOutcomeStatus,
)
from synkit.Synthesis.RBL.proof import RBL_PROOF_SCHEMA

ITSLike = Any
RBL_RESULT_SCHEMA = "synkit.rbl-result/2"


class RBLStateMixin:
    """Expose defensive copies of RBL outputs and diagnostics."""

    @property
    def template_its(self) -> Optional[ITSLike]:
        """Return the standardized representation of the prepared template."""
        return self._template_its

    @property
    def forward_its(self) -> List[ITSLike]:
        """Return ITS graphs from the last forward application."""
        return list(self._forward_its)

    @property
    def backward_its(self) -> List[ITSLike]:
        """Return ITS graphs from the last backward application."""
        return list(self._backward_its)

    @property
    def fused_its(self) -> List[ITSLike]:
        """Return accepted fused ITS graphs."""
        return list(self._fused_its)

    @property
    def fused_rsmis(self) -> List[str]:
        """Return accepted fused reaction SMILES."""
        return list(self._fused_rsmis)

    @property
    def fusion_candidates(self) -> List[FusionCandidate]:
        """Return proof-bearing candidates from the fusion stage."""
        return list(self._fusion_candidates)

    @property
    def rbl_proofs(self) -> List[Any]:
        """Return independently replayable RBL certificates."""
        return list(self._rbl_proofs)

    @property
    def last_reaction(self) -> Optional[str]:
        """Return the last processed reaction, if any."""
        return self._last_reaction

    @property
    def result(self) -> Dict[str, Any]:
        """Return the stable result and termination record for the last run."""
        search = dict(self._fusion_search_stats)
        complete = bool(search.get("complete", False))
        incomplete_reasons = list(search.get("incomplete_reasons", ()))
        if self._fused_rsmis:
            status = SearchOutcomeStatus.FOUND
        elif complete:
            status = SearchOutcomeStatus.PROVED_NONE
        elif search.get("operational_failures", 0) or any(
            issue.get("code")
            in {
                "FUSION_OPERATION_FAILED",
                "FUSION_SERIALIZATION_FAILED",
            }
            for reports in self._diagnostics.values()
            for report in reports
            for issue in report.get("issues", ())
        ) and not search:
            status = SearchOutcomeStatus.ERROR
        else:
            status = SearchOutcomeStatus.INCOMPLETE

        strict = (
            self._active_search_policy.acceptance_task
            is AcceptanceTask.STRICT_RECONSTRUCTION
        )
        acceptance_policy = (
            {
                "task": "strict_reconstruction",
                "relation": "exact_component_multiset_inclusion",
                "preserve_original_sides": ["reactants", "products"],
                "conservation_boundary": self.conservation_boundary,
                "environment_delta": dict(self.environment_delta),
                "require_mapped_material": True,
                "use_chirality": True,
            }
            if strict
            else {
                "task": "compatibility",
                "preserve_original_sides": list(self.preserve_original_sides),
                "relation": "component_injective_subgraph",
                "use_chirality": True,
            }
        )
        proof_digests_by_outcome: dict[str, list[str]] = {}
        for proof in self._rbl_proofs:
            payload = proof.to_dict()
            proof_digests_by_outcome.setdefault(proof.final_digest, []).append(
                payload["document_digest"]
            )
        outcomes = []
        for index, rsmi in enumerate(self._fused_rsmis):
            signature = (
                graph_identity_digest(self._fused_its[index])
                if index < len(self._fused_its)
                else None
            )
            proof_digests = (
                sorted(proof_digests_by_outcome.get(signature, ()))
                if signature is not None
                else []
            )
            outcomes.append(
                {
                    "rsmi": rsmi,
                    "canonical_signature": signature,
                    "proof_digests": proof_digests,
                    "proof_count": len(proof_digests),
                }
            )
        return {
            "schema": RBL_RESULT_SCHEMA,
            "fused_rsmis": list(self._fused_rsmis),
            "mode": self._last_stop_mode,
            "reason": self._last_stop_reason,
            "metadata": dict(self._last_stop_metadata),
            "n_forward_its": len(self._forward_its),
            "n_backward_its": len(self._backward_its),
            "n_fused_its": len(self._fused_its),
            "fusion_proof_schema": FUSION_PROOF_SCHEMA,
            "rbl_proof_schema": RBL_PROOF_SCHEMA,
            "fusion_candidates": [
                candidate.to_dict() for candidate in self._fusion_candidates
            ],
            "rbl_proofs": [proof.to_dict() for proof in self._rbl_proofs],
            "outcomes": outcomes,
            "fusion_search": search,
            "search_status": status.value,
            "complete": complete,
            "reason_incomplete": incomplete_reasons,
            "verified_fusion_mode": self.verified_mode,
            "search_policy": self._active_search_policy.to_dict(),
            "acceptance_policy": acceptance_policy,
            "diagnostics": self.diagnostics,
        }

    @property
    def diagnostics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return diagnostics grouped by reactor stage."""
        return {stage: list(reports) for stage, reports in self._diagnostics.items()}


__all__ = ["RBL_RESULT_SCHEMA", "RBLStateMixin"]
