# Exact alternative-ITS application case

This directory freezes the reviewer-facing alternative-AAM application for
FlowER record `84:1` (dataset source line 109). The mapped reaction itself is
not redistributed. Its SHA-256 identifier is
`d8992069dde70b5894943449190df35c1acf4b6c6d87adc2e501a295e03e5f36`.

`record.json` was produced by `scripts/run_synister_alternative_its.py` using
one process, one numerical thread, a hard 6 GiB address-space limit and 300 s
per query. Its payload digest is
`6706dbe4ff97e786cbe2436661219241b41277b0849611307be40ed80754c130`;
the bound implementation digest is
`d48db5c30262d1c1a327002e7f89c32ac76b54ede52f0bba19c11b6ab3389856`.
Peak resident memory was 217,404 KiB.

Each target was repeated with the held-out reference, reference-free SLAP and
no seed. All three modes returned the same status, labeled shell count, exact
ITS-class count and alternative-class count. The seed changed only search
order and node count; it did not restrict the candidate set.

| Query | Labeled AAMs | Exact ITS classes | Reference class present | Alternative ITS classes |
|---|---:|---:|:---:|---:|
| Global minimum, CD 6 | 8 | 2 | no | 2 |
| Numeric CD 4 | 0 | 0 | no | 0 |
| Numeric CD 6 | 8 | 2 | no | 2 |
| Reference CD 8 | 16 | 4 | yes | 3 |
| Numeric CD 10 | 52 | 13 | no | 13 |
| Numeric CD 12 | 180 | 45 | no | 45 |

The alternatives are exact structural counterexamples relative to the
reference ITS under the declared CD shell. They are suitable as auditable
contrastive or hard-negative candidates, but the word *negative* does not
assert mechanistic impossibility or annotation error.
