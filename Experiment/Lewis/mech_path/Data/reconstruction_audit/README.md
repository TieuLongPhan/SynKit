# Mechanism reconstruction failures

This directory retains only the identifiers of unresolved records from the
completed mechanism-reconstruction run:

- `polar-failures.csv`: 3,274 source-mapping failures from 95,888 polar cases;
- `radical-failures.csv`: 10 source-annotation failures from 5,426 radical
  cases.

Both files contain one column, `source_row`, giving the one-based logical row
in the corresponding source corpus. They intentionally retain no reaction,
condition, diagnostic-message, or other third-party source field. Detailed
diagnostics can be regenerated locally with `mech_path/audit.py`.

For radical data, the fixed baseline is the completed source-annotation audit.
Newer, stricter event-group grammar findings are reported separately as
`current_policy_warnings` in the generated summary and do not silently change
the manuscript denominator.

The source corpora are registered in `Experiment/Lewis/Data/README.md`. The
completed result was 98,030 reconstructions from 101,314 inputs. Excluding the
3,284 confirmed source defects, reconstruction was 98,030/98,030.
