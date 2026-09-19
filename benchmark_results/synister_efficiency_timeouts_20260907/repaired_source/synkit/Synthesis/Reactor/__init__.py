"""Reaction-rule application engines and workflows.

Public classes are resolved lazily so importing a lightweight policy or error
does not initialize RDKit, NetworkX, or the batch-processing stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .output.policy import RawITSApplicationSerializationWarning
from .stereo.assignment import (
    StereoBranchLimitError,
    StereoWildcardAssignmentLimitError,
)

if TYPE_CHECKING:
    from .core.engine import SynReactor
    from .core.strategy import Strategy
    from .variants.imbalanced import ImbaEngine
    from .variants.partial import PartialEngine
    from .workflow.batch import BatchReactor
    from .workflow.benchmark import Benchmark
    from .workflow.postprocess import PostSyn
    from .workflow.rule_filter import RuleFilter

_LAZY_EXPORTS = {
    "SynReactor": ("core.engine", "SynReactor"),
    "Strategy": ("core.strategy", "Strategy"),
    "BatchReactor": ("workflow.batch", "BatchReactor"),
    "Benchmark": ("workflow.benchmark", "Benchmark"),
    "PostSyn": ("workflow.postprocess", "PostSyn"),
    "RuleFilter": ("workflow.rule_filter", "RuleFilter"),
    "ImbaEngine": ("variants.imbalanced", "ImbaEngine"),
    "PartialEngine": ("variants.partial", "PartialEngine"),
}

__all__ = [
    *_LAZY_EXPORTS,
    "StereoBranchLimitError",
    "StereoWildcardAssignmentLimitError",
    "RawITSApplicationSerializationWarning",
]


def __getattr__(name: str) -> Any:
    """Resolve one public Reactor class on first access."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public names in interactive discovery."""
    return sorted(set(globals()) | set(__all__))
