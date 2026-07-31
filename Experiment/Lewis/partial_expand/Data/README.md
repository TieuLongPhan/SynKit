# Minimal partial-expansion benchmark

This experiment compares minimal partial atom-mapping expansion. The general
corpus contains 39,732 records with an independent fully mapped `smart`
reference. The radical corpus contains 5,426 records without an independent
full-AAM reference, so it reports valid-completion coverage instead of mapping
accuracy.

The methods are:

- `synkit`: the fixed `ITSExpand.expand_rsmi` implementation from the current
  LWG checkout and `synkit` environment;
- `gm`: historical GranMapache expansion;
- `rb1`: historical `PartialAAMs.extend`;
- `rb2`: historical `PartialAAMs.extend_g`.

Collect all general-corpus runtime metadata with one command:

```bash
./Experiment/Lewis/partial_expand/run_runtime_metadata.sh
```

The script executes all 39,732 reactions five times for each of the four
methods. It records compact
`general-<method>-run-<NN>-timings.json.gz` files plus run-level JSON reports.
Every timing artifact contains one generation time and status per reaction.
Validation occurs outside the generation timer.

No figure is generated. Bulky generated candidates and detailed case files are
deleted after the compact metadata and validation summaries are written.
Outputs go to a fresh timestamped directory under `Experiment/Lewis/Runs`,
which is ignored by Git.

The default historical setup expects the reconstructed `aam` conda environment.
It reuses `/tmp/PartialAAMs-008` when that checkout is at the required commit,
or clones the official PartialAAMs repository there when the path is absent.
The selected historical environments must contain the compatible `gmapache`
package. GM and RB1/RB2 may use distinct environments through `--gm-env` and
`--rb-env`; `--external-env` sets both. A GranMapache checkout is optional and
used only to record its Git commit via `--gmapache`.

Historical workers explicitly exclude the active SynKit checkout from Python's
import path and verify that SynKit 0.0.6 was loaded from the selected conda
environment. This prevents a current source tree from silently changing the
historical algorithms.

If current LWG completes but the historical stage fails, resume into the same
output root without repeating LWG:

```bash
./Experiment/Lewis/partial_expand/run_runtime_metadata.sh \
  --historical-only --output-dir Experiment/Lewis/Runs/<run-directory>
```

Use `--limit 10 --repetitions 1` for a pilot. Run
`./Experiment/Lewis/partial_expand/run_runtime_metadata.sh --help` for
environment, timeout, progress, and output options.
