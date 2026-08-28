# Synister 16-CPU / 60-second campaign handoff

## Stopped campaign

The service `synister-global-shells-flower10k-v4.service` was stopped cleanly
on 2026-08-28. No `run_synister_global_shells.py` process remained afterward.

- Branch at stop: `mtg`
- Output: `benchmark_results/synister_global_shells_flower10k_v4_30s`
- Manifest: `be0e4968af7ddfc6c6f6d90aff5de441de99cc95cd1aec7ccbf9800e58f0969e`
- Recorded cases: 392 / 10,000
- Error records: 0
- `reference_cd`: 280 complete, 112 timeout
- `minimal`: 244 complete, 148 timeout

`summary.json` was regenerated after the stop and now reflects all 392 atomic
case records. Preserve this directory as the 30-second campaign; its records
cannot be mixed with a 60-second campaign because the timeout is part of the
hashed manifest.

## New branch and runner

Use branch `synister-16cpu-60s`. It adds `--workers`, uses spawn-based worker
processes, bounds the in-flight queue to twice the worker count, and leaves all
case and summary writes to the parent process. Numerical libraries remain at
one thread per worker.

The frozen dataset is tracked in the companion Synister repository on its
`dev` branch at commit `900d40e`. On a host that does not have that repository,
place it beside SynKit and verify the archive:

```bash
git clone --branch dev --single-branch \
  git@github.com:TieuLongPhan/Synister.git ../Synister

sha256sum ../Synister/data/flower_test_10000_v252.csv.gz
# e40647847169a7fc98af1aab44fa81c7a64deb70b77085ae3deb3925e39642b0
```

Activate the intended Python environment, switch SynKit to this branch, and
launch a fresh output directory:

```bash
git switch synister-16cpu-60s
scripts/start_synister_global_shells_16cpu.sh \
  ../Synister/data/flower_test_10000_v252.csv.gz \
  benchmark_results/synister_global_shells_v4_60s_w16
```

The launcher requests 16 CPU cores and configures:

- 16 worker processes;
- 60 seconds per shell (`both` can therefore take about 120 seconds per case);
- 4 GiB address-space limit per worker;
- 8 GiB aggregate memory soft limit;
- 10 GiB aggregate memory hard limit;
- no swap allocation by the service;
- stop-on-OOM behavior, preserving all atomic records already written.

The per-worker address-space limit is intentionally larger than 10 GiB / 16:
shared libraries and mapped virtual address space count toward `RLIMIT_AS`.
The systemd cgroup is the authoritative aggregate resident-memory boundary.

## Monitor, stop, and resume

```bash
tail -f benchmark_results/synister_global_shells_v4_60s_w16/campaign.log

systemctl --user status synister-global-shells-v4-60s-w16.service

systemctl --user show synister-global-shells-v4-60s-w16.service \
  -p MemoryCurrent -p MemoryPeak -p MemorySwapCurrent -p CPUUsageNSec

systemctl --user stop synister-global-shells-v4-60s-w16.service
```

To resume, rerun the identical launcher command. The manifest is checked before
work starts and valid case records are skipped. Do not alter workers, timeout,
memory ceiling, dataset, or implementation within an existing output directory;
use a new directory for any such change.
