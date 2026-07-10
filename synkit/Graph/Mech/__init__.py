from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synkit.Graph.Mech.lwg_editor import LWGEditResult, LWGEditor, LWGStepReport

__all__ = ["LWGEditor", "LWGEditResult", "LWGStepReport"]


def __getattr__(name: str):
    if name in __all__:
        from synkit.Graph.Mech.lwg_editor import (
            LWGEditResult,
            LWGEditor,
            LWGStepReport,
        )

        exports = {
            "LWGEditor": LWGEditor,
            "LWGEditResult": LWGEditResult,
            "LWGStepReport": LWGStepReport,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
