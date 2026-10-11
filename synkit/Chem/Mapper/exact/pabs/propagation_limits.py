"""Shared cooperative deadline signal for the opt-in propagation engine."""

import time


class PropagationDeadline(TimeoutError):
    """An unfinished bound is interrupted, rather than judged infeasible."""


def check_deadline(deadline):
    """Raise before another bounded unit of search or preprocessing work."""
    if deadline is not None and time.perf_counter() >= deadline:
        raise PropagationDeadline("Synister-CP wall-time budget expired")
