# Mechanism reconstruction failures

This directory retains only the unresolved records from the completed
mechanism-reconstruction run:

- `polar-failures.csv`: 3,274 source-mapping failures from 95,888 polar cases;
- `radical-failures.csv`: 10 source-annotation failures from 5,426 radical
  cases.

Both files include the one-based source row, failure origin, status, issue
codes, exception details, and the original source fields. The polar file also
retains the source provenance statement.

The successful cases, full input corpora, per-case JSONL evidence, aggregate
summary, and superseded 28-row radical audit set were removed after the final
run. Its result was 98,030 reconstructions from 101,314 inputs. Excluding the
3,284 confirmed source defects, reconstruction was 98,030/98,030.
