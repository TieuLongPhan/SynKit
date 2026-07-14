# SynKit mechanism test data

This directory intentionally keeps only JSON files containing executable test
cases:

- `radical_reviewed.json` — 80 chemistry-reviewed, macro-balanced radical
  mechanism cases with replay and rule-reapplication evidence embedded in each
  case.
- `small_rewrite_conformance.json` — nine compact cases for routine inspection
  of hydrogen transfer and rule-level stereochemistry.

The radical cases were selected deterministically from
`Data/Mech/radical.csv`. Four chemistry corrections, the original reactions,
review decisions, and replay results are retained inside the corresponding
canonical cases, so a separate review or release-status JSON is unnecessary.

The compact conformance data is deliberately small rather than statistically
representative. Five cases cover explicit mapped hydrogen as H-atom, hydride,
proton, or H2 transfer. Six cover E/Z or tetrahedral stereo, with the two E/Z
cases shared by both groups. It includes retention, inversion, formation,
destruction, strict wrong-isomer rejection, and racemic SN1 capture.

Raw candidate pools, audit summaries, and release-owner checklists do not
belong in this directory. They should be regenerated from their source data or
kept as project documentation rather than presented as test cases.
