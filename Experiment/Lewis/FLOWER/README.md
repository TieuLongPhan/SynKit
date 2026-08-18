# FLOWER bidirectional rule replay

This directory retains the portable full-reaction replay payload:

- `full-reaction-batches/`: ten gzip batches plus their SHA-256 manifest;
- `replay.py`: the streaming forward/backward replay driver;
- `run_bidirectional_replay.sh`: the single full-corpus launcher;

The batches contain 425,517 full reactions and 851,034 directional replays.
They are already compressed and must not be unpacked: `replay.py` streams gzip
directly. The original 3.4 GB elementary-step splits, intermediate combined
files, and prior run outputs are reproducible and are intentionally not kept.

Run from a SynKit checkout with RDKit and SynKit available to Python:

```bash
bash Experiment/Lewis/FLOWER/run_bidirectional_replay.sh
```

The safe default is one concurrent batch. On a server, select the number of
cores explicitly and put the output on suitable storage:

```bash
JOBS=8 \
PYTHON=/opt/conda/envs/synkit/bin/python \
bash Experiment/Lewis/FLOWER/run_bidirectional_replay.sh /data/flower-replay
```

`CASE_TIMEOUT=30` is the default diagnostic ceiling; set `CASE_TIMEOUT=0` for
unbounded enumeration. The launcher verifies every compressed batch against
the manifest before starting. Each batch immediately flushes non-passing rows
to its own `bugs.jsonl`; after all batches finish, the launcher also writes the
aggregate `bugs.jsonl` at the output root. `cases.jsonl.gz`, `summary.json`, and
`runner.log` remain available per batch. Every case and timeout has a portable
`case_id` such as `batch-07-of-10.txt.gz:321`, together with `batch_file`,
`batch_row`, `source_label`, direction, and failed expansion stage.

Use `bash Experiment/Lewis/FLOWER/run_bidirectional_replay.sh --help` for all
runtime controls. Copy the SynKit checkout and this retained batch directory
to the other machine; the FLOWER directory alone is insufficient because
`replay.py` imports the SynKit replay implementation.
