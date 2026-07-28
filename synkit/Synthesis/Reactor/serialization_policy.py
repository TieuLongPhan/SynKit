"""Public diagnostics for raw ITS serialization policies."""

from __future__ import annotations

from typing import Iterable


class RawITSApplicationSerializationWarning(UserWarning):
    """Warn that selected raw ITS applications could not be serialized."""

    def __init__(self, indices: Iterable[int]) -> None:
        self.indices = tuple(indices)
        rendered = ", ".join(map(str, self.indices))
        super().__init__(
            "Skipped raw ITS application(s) that could not be serialized: "
            + rendered
        )
