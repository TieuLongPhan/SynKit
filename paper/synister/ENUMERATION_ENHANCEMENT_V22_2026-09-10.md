# V22: complete disk-backed reference-CD shell — 10 September 2026

Line 40548 (PC:1189), reference-CD distance 17, now has a complete exact result.
The separate disk-backed run finished enumeration, classification and exports
in 1,527.187 seconds (25.45 minutes). Both held-out reference classes are present.

| Quantity | Complete result |
|---|---:|
| Native candidate mappings | 13,187,584 |
| Exact ITS classes | 8,000,310 |
| Exact radius-one template classes | 7,997,022 |
| Weighted product-orbit representatives | 31,791,010 |
| Labeled mappings | 1,017,312,320 |
| Product automorphism group order | 32 |
| Coordinator peak RSS | 749.65 MiB |
| Maximum child peak RSS | 343.35 MiB |

These are distinct counting units. The exact reference permutation itself was
not an emitted representative, but its ITS and template classes were observed.

## Protocol and implementation

This is an explicitly different storage/budget protocol: 16 workers on CPUs
0–15, the unchanged 4 GiB address-space limit per worker, a 14,400-second wall
allowance and no cumulative retained-record cap. It is not a pass under the
original 60-second/one-million-record protocol. Historical campaign results
remain unchanged; a new full 10,000-reaction rerun has not been performed.

The optional backend is `synkit/Chem/Mapper/exact/disk_frontier.py`, with CLI
`scripts/run_synister_disk_shell.py`. The existing capped in-memory backend is
unchanged. Workers retain history only within one batch. All completed batch
records reach a SQLite store, which deduplicates by the full exact certificate.
Lossless compression reduces storage; public hashes alone never decide equality.

Duplicate class records must agree in multiplicity, stabilizer, template and
coordinate-orbit contributions. Only a new exact class changes the totals.
Counts use arbitrary-precision Python integers stored as decimal text. Template
counts, exact candidate witnesses and full class records live on disk. The
coordinator retains bounded batches and a bounded SQLite page cache. The 16 KiB
database page size avoids excessive overflow pages for these records. A free
space guard stops the run before exhausting the output filesystem.

Class tables are separate sorted gzip JSONL files containing identifier/count
pairs, with uncompressed SHA-256 hashes recorded in the result. This avoids
materializing multi-million-row arrays or one huge JSON string in RAM. The
database and completed exports occupy about 20 GiB; each compressed public
class table is about 281 MiB. Large artifacts remain under `/tmp` because the
workspace filesystem has insufficient free space.

## Verification

- 132 tests passed across native, minimum-proof and disk-backed suites.
- Six weighted-graph controls, deliberately tiny batches, match the in-memory
  counts, complete class maps, coordinate frequencies and reference membership.
- Forced public-ID collisions retain distinct full certificates. Counts above
  int64 range, inconsistent duplicate records and expired deadlines are tested.
- All 6,098,851 earlier partial ITS classes occur with identical weights.
- All 6,095,563 earlier partial template classes occur with counts no smaller
  than before. Counts may grow when later ITS classes share a template.
- Both complete exported tables sum to 31,791,010 weighted representatives;
  all table hashes and row counts match the result manifest. The audit verified
  absence of public-ID collisions in these particular exports.
- An independently scheduled native search counted 13,187,584 candidates.
  The complete disk run and SQLite exact-witness cardinality both match it.
- SQLite quick_check passes. Frozen source hashes remained unchanged during
  the full run; library SHA-256 is
  `e44527c7f89131b3ccad970f3e0e5e1aaab4caaf2c65f9214c9aa936ab5648e6`.
- The packaged CLI completed a real line-21609 reference-CD control. Final
  import-only lint cleanup does not alter the frozen algorithm; the eight
  disk tests were rerun afterward. Ruff checks for all new files and
  `git diff --check` pass.

The first disk pilot was intentionally interrupted to improve SQLite page
layout. Its artifacts are preserved and are not completion evidence. The
authoritative result uses the revised frozen source directory below.

## Reproduction

Use the intended Python environment and a fresh output directory on a filesystem
with enough space:

```bash
python scripts/run_synister_disk_shell.py \
  --dataset ../Synister/data/flower_test_10000_v252.csv.gz \
  --dataset-sha256 e40647847169a7fc98af1aab44fa81c7a64deb70b77085ae3deb3925e39642b0 \
  --source-line 40548 \
  --library /path/to/verified/libsynkit_distance.so \
  --output /tmp/synister_40548_fresh \
  --seconds 14400 --workers 16
```

Exact frozen source, library location and executed runner:
`/tmp/synister_v22_20260910/revised/`.

Authoritative result and external class tables:
`/tmp/synister_v22_20260910/revised/full/result.json`,
`its_class_counts.jsonl.gz`, `templates_class_counts.jsonl.gz`, and `exact.sqlite`
in the same directory.

Audits: `/tmp/synister_v22_20260910/final_audit.json` and
`/tmp/synister_v22_20260910/database_audit.json`.
Test logs: `tests.log` and `final_disk_tests.log` in that directory.
Compact report data are copied beside this report as
`ENUMERATION_ENHANCEMENT_V22_2026-09-10.json`.

Together with the separate V21 minimum-proof repair for 21609, both remaining
tasks now have complete exact outputs. Their different protocols must remain
explicit in any combined summary or paper.
