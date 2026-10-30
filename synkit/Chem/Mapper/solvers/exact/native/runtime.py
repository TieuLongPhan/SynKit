"""Optional operating-system controls for native mapper workers."""

import os

WORKER_ADDRESS_SPACE_BYTES = 4 * 1024**3


def limit_worker_address_space(resource):
    """Lower supported soft limits, preserving inherited soft and hard ceilings."""
    if resource is None or not hasattr(resource, "RLIMIT_AS"):
        return None
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        ceilings = [value for value in (soft, hard) if value != resource.RLIM_INFINITY]
        limit = min([WORKER_ADDRESS_SPACE_BYTES, *ceilings])
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        return limit
    except (OSError, ValueError):
        # Some macOS kernels expose RLIMIT_AS but reject setting this limit.
        # This optional control must not prevent worker startup and barriers.
        return None


def worker_cpus(workers):
    """Return available CPU IDs to pin, or an empty list when unsupported."""
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return []
    try:
        return sorted(os.sched_getaffinity(0))[:workers]
    except OSError:
        return []


def pin_worker_cpu(cpu):
    """Apply optional CPU pinning without making an OS tuning failure fatal."""
    if cpu is not None and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {cpu})
        except OSError:
            pass
