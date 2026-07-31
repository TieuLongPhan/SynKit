"""Fail-closed controls for stereo assignment and product branching."""


class StereoWildcardAssignmentLimitError(RuntimeError):
    """Raised before an exhaustive typed-port search would exceed its cap."""

    def __init__(self, limit: int, discovered: int):
        self.limit = limit
        self.discovered = discovered
        self.permitted = limit
        self.requested = discovered
        super().__init__(
            "Stereo wildcard assignment search is incomplete: "
            f"permitted={limit}, requested_at_least={discovered}."
        )


class StereoBranchLimitError(RuntimeError):
    """Raised before stereo product expansion would exceed its hard cap."""

    def __init__(self, permitted: int, requested: int):
        self.limit = permitted
        self.discovered = requested
        self.permitted = permitted
        self.requested = requested
        super().__init__(
            "Stereo product branch search is incomplete: "
            f"permitted={permitted}, requested_at_least={requested}."
        )


__all__ = [
    "StereoBranchLimitError",
    "StereoWildcardAssignmentLimitError",
]
