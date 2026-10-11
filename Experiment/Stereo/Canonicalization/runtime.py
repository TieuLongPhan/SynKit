"""Portable process measurements and wall budgets for stereo experiments."""

from contextlib import contextmanager
import os
from pathlib import Path
import signal
import sys
import threading
import time

import psutil

try:
    import resource
except ImportError:
    resource = None

MIB = 1024 * 1024


def process_rss_mib(pid):
    """Read native resident memory, including the high-water mark when available."""
    try:
        info = psutil.Process(pid).memory_info()
        peak = max(info.rss, getattr(info, "peak_wset", 0))
        if sys.platform.startswith("linux"):
            try:
                lines = Path(f"/proc/{pid}/status").read_text().splitlines()
            except (FileNotFoundError, ProcessLookupError):
                return peak / MIB
            peak = max((peak, *(int(line.split()[1]) * 1024 for line in lines if line.startswith("VmHWM:"))))
        return peak / MIB
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return 0.0


def peak_rss_mib():
    """Return this process's native peak RSS in MiB on Linux, macOS and Windows."""
    if resource is not None:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (MIB if sys.platform == "darwin" else 1024)
    return process_rss_mib(os.getpid())


def process_cpu_seconds(pid):
    """Read user and system CPU time; a worker that has exited contributes zero."""
    try:
        usage = psutil.Process(pid).cpu_times()
        return usage.user + usage.system
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return 0.0


def load_average():
    """Keep unavailable load averages explicit rather than reporting zero load."""
    return os.getloadavg() if hasattr(os, "getloadavg") else (None, None, None)


@contextmanager
def wall_time_limit(seconds, error):
    """Interrupt with POSIX alarms when available; reject late results everywhere.

    On Windows, callers must also check deadlines during work or use a parent
    worker timeout when a computation requires preemptive interruption.
    """
    if seconds <= 0:
        raise error
    started = time.monotonic()
    alarms = (all(hasattr(signal, name) for name in ("SIGALRM", "ITIMER_REAL", "setitimer"))
              and threading.current_thread() is threading.main_thread())

    def expired(_signum, _frame):
        raise error

    if alarms:
        previous_handler = signal.signal(signal.SIGALRM, expired)
        previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
        if time.monotonic() - started > seconds:
            raise error
    finally:
        if alarms:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_timer[0] > 0:
                remaining = max(0.000001, previous_timer[0] - (time.monotonic() - started))
                signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])
