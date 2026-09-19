# Synister efficiency V9 — 8 September 2026

**All three previously unresolved reference-CD shells now have complete exact results**, including reaction-center frequencies, ITS/template class counts, and held-out reference-class checks. **This does not mean all three meet the original 60-second protocol.** The completed runs use 16 workers and a one-million retained-record cap, with larger time allowances for the two large shells.

| Source line | Timed allowance | Search/classification/merge | Analysis wall time | Product-orbit representatives | ITS classes | Template classes |
|---|---:|---:|---:|---:|---:|---:|
| 28163 | 300 s | 11.16 s | 14.54 s | 114,330 | 11,300 | 11,280 |
| 30474 | 600 s | 244.10 s | 314.52 s | 27,783,018 | 461,522 | 425,109 |
| 22361 | 300 s | 98.68 s | 121.53 s | 642,926 | 125,162 | 124,962 |

All three held-out reference classes are present. The labeled-mapping counts are 5,619,548,160 for 28163; 6,145,159,053,312 for 30474; and 987,534,336 for 22361.

A separate final-build run retains the 60-second timed budget and eight workers per case, with the explicitly enlarged one-million-record cap:

| Source line | Result | Search/classification/merge | Analysis wall time |
|---|---|---:|---:|
| 28163 | Complete | 11.91 s | 15.27 s |
| 30474 | Timeout | 60.37 s | 78.98 s |
| 22361 | Timeout | 60.32 s | 70.42 s |

The 60-second run still completes only 28163. Its wall time was 29.31 seconds in V8 and is approximately 15 seconds here. These are individual observations on a shared host, not an isolated speedup study. Other CPU workloads were active during verification.

## What changed

The optional native backend now uses a shared queue of resumable subtrees. Each job records its deterministic assignment prefix and returns a disjoint cover of its unvisited descendants when its node slice ends. A worker can take another job when it finishes, reducing idle time from static partitions. Jobs are batched, and each worker prepares its immutable native inputs once. Prefix replay is included in visited-node counts.

Controlled ITS colors are encoded directly from matrices using exactly the existing typed color rules, including template boundary colors. This avoids repeatedly constructing NetworkX graphs. Exact canonical keys cache their hashes for repeated dictionary lookups; equality still compares the complete certificate. Deserialization recomputes the hash under the receiving process's hash seed, and collisions never establish equality.

Exact weighted records merge into counts and frequencies while the search continues. Public class identifiers use the unchanged legacy certificate, hashed incrementally without constructing its dense string. These output optimizations preserve the existing class IDs.

The original default Python search modules remain byte-for-byte unchanged. The explicit native entry point is analyze_reference_blinded_native_shell in synkit/Chem/Mapper/native_analysis.py. Its scheduler defaults to frontier; the static scheduler remains available explicitly. The standard configuration's 100,000 cap was not silently changed.

## Budgets and completeness

The one-million cap counts retained worker-local reactant/product double-orbit records, including copies held by different workers. It does not count the millions of weighted product representatives. The original 100,000 product-representative cap cannot cover any of these full shells. Even a 100,000 distinct-class cap is insufficient for 30474 and 22361, which have 461,522 and 125,162 ITS classes respectively.

The timed phase includes search, exact classification, transfer, and merging. It excludes preparation, the reference-free seed, post-search reference reveal, public identifier formatting, and JSON writing. Completion requires an empty pending queue, every submitted task accounted for, and no exhausted shared budget. A timed-out result may take extra wall time to summarize its partial output.

Each worker has a 4 GiB address-space limit. The parent has no imposed address-space limit; measured peaks are retained in every full record. Large class spectra make both parent memory and output processing significant.

An initial final-build verification of 30474 hit its 300-second allowance under concurrent CPU load, after observing 27,736,981 product representatives. That partial result is preserved in completion_run. The complete replacement is a separate completion_retry record, run on CPUs 16–31 with a 600-second allowance. The earlier independent diagnostic had already completed this shell in about 255 seconds; neither observation is presented as a 60-second completion.

## Validation and reproducibility

**276 tests passed**, with one optional skip and the same known provenance test deselected. The 34 focused tests cover exhaustive small mapping spaces, exact group/canonical-code oracles, weighted frequencies, template boundaries, resumable partition coverage even with one-node slices, global caps and zero deadlines, cross-process key transfer, forced hash collisions, and legacy identifier escaping.

All **46 previously completed reference-CD regression cases match exactly** in counts, frequencies, entropies, group orders, reference checks, and public ITS/template class identifiers. Independently scheduled completed runs of all three difficult cases agree on candidate counts, weighted counts, and ITS/template class counts. Case 28163 also matches its complete V8 class IDs and frequency spectrum.

No full 1,200-task rerun was performed. Uniform original-protocol campaign totals must not be relabeled as fully complete using these expanded-budget results.

Build the native library explicitly with scripts/build_synister_native.py. Then use scripts/benchmark_synister_native.py with the V9 selection, the V9 verified_source directory, the built library path, and an empty output directory. The completed-run resource settings are --workers 16 --mapping-cap 1000000 with up to --seconds 600. The bounded comparison uses --workers 8 --seconds 60.

Artifacts are under benchmark_results/synister_efficiency_timeouts_v9_20260908/:

- verified_source/ and build/: frozen implementation and immutable compiled library.
- completion_run/: complete 28163/22361 results and the preserved partial 30474 attempt.
- completion_retry/: complete 30474 result with its separate manifest.
- bounded_run/: final-build 60-second comparison.
- native_regression/ and native_regression_audit.json: all 46 exact comparisons.
- completion_audit.json: agreement across completed runs.
- default_source_audit.json and tests.log: unchanged-default audit and test output.
- long_30474.json and long_22361.json: completed earlier diagnostics.

All final manifests confirm the source stayed unchanged during their runs. Selected-file Ruff and whitespace checks pass; the C++ build emitted no warnings.

Frozen Python/C++ source SHA-256: 912d0c2eb920d5392dd059afee23704fbba1c5ebe963da76235e10502cef492c

Native library SHA-256: 833a5702d144342a2b7580038b6930b329c3a56158bdff433a46c824aed3e47c
