# SynKit mechanism test data

This directory keeps executable, self-contained JSON test cases:

- `radical_reviewed.json` — 80 chemistry-reviewed, macro-balanced radical
  mechanism cases with replay and rule-reapplication evidence embedded in each
  case.
- `small_rewrite_conformance.json` — nine compact cases for routine inspection
  of hydrogen transfer and rule-level stereochemistry.
- `non_tetrahedral_rewrite_conformance.json` — four project-owned, balanced
  SP/TBP/OH/assigned-atrop fixtures with forward/reverse application,
  Lewis-state endpoint audits, explicit claim boundaries, and the required
  corruption controls.
- `stereo.json` — 48-case stereo conformance draft: 33 end-to-end executable
  transformations, four graph-only transformations, three isotope-deferred
  transformations, and eight specification-only negative assertions. Its
  semantics, stable catalog references, and deferred backlog are embedded in
  the JSON rather than maintained in a parallel catalog.

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

The compact pinned six-class graph-isomorphism result is therefore stored at
`Data/Conformance/stereomolgraph_graph_conformance.json`, outside this
executable mechanism-case directory.
