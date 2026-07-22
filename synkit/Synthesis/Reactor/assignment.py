"""Fail-closed controls for exhaustive stereo wildcard assignment."""


class StereoWildcardAssignmentLimitError(RuntimeError):
    """Raised before an exhaustive typed-port search would exceed its cap."""

    def __init__(self, limit: int, discovered: int):
        self.limit = limit
        self.discovered = discovered
        super().__init__(
            "Stereo wildcard assignment search is incomplete: "
            f"limit={limit}, discovered_at_least={discovered}."
        )


__all__ = ["StereoWildcardAssignmentLimitError"]
