"""Read-only run-state views for the RBL engine."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from synkit.Graph.Fusion import FUSION_PROOF_SCHEMA, FusionCandidate

ITSLike = Any
RBL_RESULT_SCHEMA = "synkit.rbl-result/1"


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
    def last_reaction(self) -> Optional[str]:
        """Return the last processed reaction, if any."""
        return self._last_reaction

    @property
    def result(self) -> Dict[str, Any]:
        """Return the stable result and termination record for the last run."""
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
            "fusion_candidates": [
                candidate.to_dict() for candidate in self._fusion_candidates
            ],
            "fusion_search": dict(self._fusion_search_stats),
            "verified_fusion_mode": self.verified_mode,
            "search_policy": self._active_search_policy.to_dict(),
            "acceptance_policy": {
                "preserve_original_sides": list(self.preserve_original_sides),
                "relation": "component_injective_subgraph",
                "use_chirality": True,
            },
            "diagnostics": self.diagnostics,
        }

    @property
    def diagnostics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return diagnostics grouped by reactor stage."""
        return {stage: list(reports) for stage, reports in self._diagnostics.items()}


__all__ = ["RBL_RESULT_SCHEMA", "RBLStateMixin"]
