from enum import Enum
from typing import Union


class Strategy(str, Enum):
    """
    Strategy for sub-graph matching/application:

      - ALL:       classic VF2 on the whole graph
      - COMPONENT: component-aware only (no cross-CC backtracking)
      - BACKTRACK: component-aware with backtracking across CCs
    """

    ALL = "all"
    COMPONENT = "comp"
    BACKTRACK = "bt"

    @classmethod
    def from_string(cls, value: Union[str, "Strategy"]) -> "Strategy":
        """
        Convert a string or Strategy to a Strategy enum.

        Parameters
        ----------
        value : str or Strategy
            The strategy to parse.

        Returns
        -------
        Strategy
            Parsed Strategy.

        Raises
        ------
        ValueError
            If the input is not a valid Strategy.
        """
        if isinstance(value, cls):
            return value
        try:
            return cls(value.lower())  # type: ignore[arg-type]
        except ValueError as e:
            raise ValueError(f"Unknown strategy: {value!r}") from e

    def __str__(self) -> str:
        """Return the strategy’s canonical code (e.g. 'all', 'comp', 'bt')."""
        return self.value

    def __repr__(self) -> str:
        """Return the enum-style representation, e.g. 'Strategy.ALL'."""
        return f"{self.__class__.__name__}.{self.name}"
