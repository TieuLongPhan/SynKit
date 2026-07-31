# Mechanism reconstruction failures

This directory receives locally generated, identifier-only outputs from the
mechanism-reconstruction rerun. The CSV files are intentionally ignored and
are not part of the commit tree:

- `polar-failures.csv`: no unresolved records from 95,888 polar cases;
- `radical-failures.csv`: the one unresolved source-annotation conflict from
  5,426 radical cases;
- `radical-arrow-review.csv`: eleven reviewed row IDs with a compact
  recorded-arrow issue, reviewed action, and outcome.

The failure files contain one column, `source_row`, giving the one-based logical
row in the corresponding source corpus.  The review table contains no
structures, conditions, or other source fields.  Detailed diagnostics can be
regenerated locally with `mech_path/audit.py`.

Repeated positive maps in a polar endpoint are repaired deterministically:
retain the first occurrence, remove each later label, and let guarded expansion
assign fresh maps.  For symmetry-equivalent occurrences, retaining the other
member produces the same endpoint result.

The source corpora are registered in `Experiment/Lewis/Data/README.md`. The
completed result is 101,313 reconstructions from 101,314 inputs.  ID 2,207 is
retained as unresolved because no local electron-flow group matches both its
hydrogen transfer and declared product radical.
