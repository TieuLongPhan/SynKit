# Repository scripts

Run these entrypoints from any working directory:

- `lint.sh`: Python file-size policy and Flake8 checks;
- `pytest.sh`: the full test suite, or supplied pytest arguments;
- `build_doc.sh`: strict Sphinx HTML documentation build.

`check_python_file_size.py` and `python_file_size_baseline.json` support the
lint entrypoint and its regression tests.

## Synister global-shell experiment

All Synister implementation and experiment code lives in SynKit. Run the
reference-blinded global experiment with an explicit frozen dataset:

```bash
python scripts/run_synister_global_shells.py \
  --dataset /path/to/flower_test_10000.csv.gz \
  --output benchmark_results/synister_global_shells_v4 \
  --mode both \
  --workers 1 \
  --time-limit-per-shell 300 \
  --memory-limit-gib 6
```

The input CSV must contain `source_line`, `reaction_id`, and `mapped_reaction`.
The runner uses one numerical thread per worker. Parallel workers only compute;
the parent process is the sole writer of immutable gzip JSON case records. It
independently blinds the endpoint atom orders, runs either the unrestricted
shell at the held-out reference CD, the unrestricted globally minimal shell,
or both, and reveals the reference only for post-search evaluation. Timeouts,
caps, and errors are never counted as complete cases.

`--memory-limit-gib` is an address-space limit for each worker, not an aggregate
limit. On a 16-worker workstation, use the cgroup-based launcher to enforce an
8 GiB soft and 10 GiB hard limit for the complete service while preventing it
from consuming swap:

```bash
scripts/start_synister_global_shells_16cpu.sh \
  /path/to/flower_test_10000.csv.gz \
  benchmark_results/synister_global_shells_v4_60s_w16
```

The launcher uses 16 workers, 60 seconds per shell, a 4 GiB per-worker
address-space ceiling, `MemoryHigh=8G`, `MemoryMax=10G`, and
`MemorySwapMax=0`. If a worker is killed by the cgroup, the pool stops instead
of silently converting the remaining campaign into error records. Existing
atomic case records remain resumable with the identical command.

Verify the case and manifest digests and regenerate censored,
manuscript-facing statistics with:

```bash
python scripts/summarize_synister_evidence.py \
  benchmark_results/synister_global_shells_v4 \
  --output derived_findings.json
```

## Exact alternative-ITS application

Generate one deterministic AAM representative for every exact ITS class that
differs from a supplied mapped reference:

```bash
python scripts/run_synister_alternative_its.py \
  --dataset /path/to/flower_test_10000.csv.gz \
  --source-line 109 \
  --output benchmark_results/alternative_its_line_109.json \
  --target minimal \
  --target reference \
  --target 10 \
  --seed-mode reference \
  --memory-limit-gib 6 \
  --time-limit 300
```

``reference`` uses the scalar CD of the mapped input. ``minimal`` proves and
enumerates the global optimum; any non-negative number requests that exact
global shell, whether or not it equals the reference CD. A reference seed is
an incumbent and ordering hint only. The output includes exact completeness,
stable ITS identifiers, representative mappings, and correspondences in the
input atom-map coordinates. These are reference-relative structural
alternatives, not claims of mechanistic invalidity.

## Synister manuscript figures

Regenerate the vector pilot and CD-spectrum panel directly from the frozen,
digest-verified evidence:

```bash
python scripts/plot_synister_figures.py
```

The plotting script validates the 100-case cohort identity, record 84:1, and
seed-invariant exact shell counts before writing
`paper/synister/figures/pilot_spectrum.pdf`. Its palette matches the TikZ
workflow and the shared `../Style` visual vocabulary.
