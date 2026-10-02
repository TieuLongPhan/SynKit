"""Ordinary or separately instrumented E4 experimental source variant."""

from contextlib import nullcontext
import json
import resource
import sys
import time
from unittest.mock import patch

from Experiment.Synister import ablation_worker
from Experiment.Synister.isolated_ablation import compile_variant
from synkit.Chem.Mapper.exact import distance


def perform(task):
    started = time.perf_counter()
    function, identity = compile_variant(task['variant'])
    compile_seconds = time.perf_counter()-started
    if task.get('instrumented', False):
        from Experiment.Synister.ablation_observer import Observer
        observer = Observer(function, identity)
    else:
        observer = None
    with patch.object(distance, 'enumerate_distance_mappings', function), patch.dict(
            ablation_worker.CONFIGURATIONS, {task['variant']: identity['options']}), (observer or nullcontext()):
        result = ablation_worker.perform(task)
    result.update(experimental_solver={k:v for k,v in identity.items() if k != 'transformed_source'},
                  compile_seconds=compile_seconds, instrumented=observer is not None,
                  instrumentation=observer.result() if observer else None)
    return result


if __name__ == '__main__':
    task = json.load(sys.stdin)
    limit = task['memory_gib']*1024**3
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    started = time.perf_counter()
    try:
        result = perform(task)
    except Exception as exc:
        result = {'complete':False,'minimum_proved':False,'termination':'worker_error',
                  'error_type':type(exc).__name__,'error':str(exc)}
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result.update(worker_seconds=time.perf_counter()-started,cpu_seconds=usage.ru_utime+usage.ru_stime,peak_rss_kib=usage.ru_maxrss)
    print(json.dumps(result,allow_nan=False))
