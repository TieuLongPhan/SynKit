from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import networkx as nx

Mapping = dict[int, int]
Validator = Callable[[nx.Graph, Mapping], bool]
Recognizer = Callable[
    [nx.Graph, "FunctionalGroupPattern"], list["FunctionalGroupMatch"]
]


@dataclass(frozen=True)
class FunctionalGroupPattern:
    """Graph-native functional-group definition."""

    name: str
    graph: nx.Graph
    group_nodes: tuple[int, ...]
    parents: tuple[str, ...] = ()
    suppresses: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    anchor_node: int | None = None
    priority: int = 0
    validator: Validator | None = None
    recognizer: Recognizer | None = None
    public: bool = True


@dataclass(frozen=True)
class FunctionalGroupMatch:
    """One matched functional group in a host graph."""

    name: str
    group_nodes: tuple[int, ...]
    mapping: Mapping
    pattern: FunctionalGroupPattern


@dataclass
class FunctionalGroupRegistry:
    """Container for functional-group patterns and hierarchy metadata."""

    patterns: list[FunctionalGroupPattern] = field(default_factory=list)

    def add(self, pattern: FunctionalGroupPattern) -> None:
        self.patterns.append(pattern)

    def extend(self, patterns: Iterable[FunctionalGroupPattern]) -> None:
        self.patterns.extend(patterns)

    def by_name(self, name: str) -> FunctionalGroupPattern:
        for pattern in self.patterns:
            if pattern.name == name:
                return pattern
        raise KeyError(name)

    def is_ancestor(self, ancestor: str, child: str) -> bool:
        """Return whether ``ancestor`` is an ancestor of ``child``."""
        seen: set[str] = set()
        stack = [child]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            try:
                parents = self.by_name(current).parents
            except KeyError:
                parents = ()
            for parent in parents:
                if parent == ancestor:
                    return True
                stack.append(parent)
        return False

    def execution_order(self) -> list[FunctionalGroupPattern]:
        """Return patterns in prerequisite-respecting order."""
        by_name = {pattern.name: pattern for pattern in self.patterns}
        visited: set[str] = set()
        ordered: list[FunctionalGroupPattern] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            pattern = by_name[name]
            for required in pattern.requires:
                if required in by_name:
                    visit(required)
            ordered.append(pattern)

        for pattern in self.patterns:
            visit(pattern.name)
        return ordered
