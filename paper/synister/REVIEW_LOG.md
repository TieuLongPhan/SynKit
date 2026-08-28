# Synister internal review log

The review rounds were performed against the implementation and frozen
schema-v4 evidence, not only against manuscript prose.

This is a chronological log. Round 7 supersedes earlier statements about
per-shell timing, validation counts and campaign digests; those older entries
are retained only to document what was believed and later corrected.

## Round 1 — theory and exact-method review

Findings:

- Earlier prose classified only threshold CD. The exact supplied-CD
  non-emptiness problem also has a direct Clique reduction and is NP-complete;
  enumeration itself is output-sensitive rather than “NP-complete”.
- “Sparse assignment” overstated a search that maintains dense incremental
  `n x n` costs, although molecular endpoints are sparse.
- A schema field called mapping entropy contained `log(ITS class count)`, not
  `log(verified-subgroup mapping representatives)`.
- Hydrogen-flow multiplicities needed an exhaustive labeled-bijection control,
  and the balanced-inventory precondition needed to be explicit.
- Molecular orbits cannot be globally sorted. Symmetry claims must be
  conditional on verified generators and complete point-stabilizer chains.

Actions:

- Added the exact-shell NP-completeness proof, separated the GI zero shell and
  stated the factorial output lower bound.
- Renamed the method incremental assignment branch-and-bound.
- Split mapping Hartley entropy from exact ITS Hartley entropy in schema v4.
- Added exhaustive hydrogen-flow versus labeled-bijection tests through four
  hydrogens on three parents.
- Retained the relabelled-C5 regression, exact witness validation and explicit
  quotient-completeness status.

Revalidation: targeted theory/method tests passed; schema-v4 evidence was
regenerated rather than relabeled.

## Round 2 — computational-science and evidence review

Findings:

- Development-era FlowER campaigns mixed reference-conditioned and global
  scopes and were unsuitable as headline evidence.
- Fractions from a one-second campaign are censored by computational
  difficulty and cannot estimate the full dataset.
- “Reference class” could mean exact transported endpoint state or exact
  canonical ITS class; the manuscript result needs the latter.
- The former manuscript contained chronology and unsupported prospective
  claims rather than a reproducible downstream consequence.

Actions:

- Froze a new independently blinded 100-case campaign: one process, one
  numerical thread, 6 GiB hard address-space cap and one second per shell.
- Stored all 100 digest-verified case records, manifest, resource and
  environment records; no raw reference mapping or reaction SMILES is stored.
- Regenerated derived findings from exact canonical ITS fields and complete
  shells only.
- Limited the scientific conclusion to the reproducible direct consequence:
  exact centre/template labels vary across globally minimal maps. Timeouts are
  reported as incomplete and excluded from denominators.

Revalidation: 46 minimal shells and 52 reference-CD shells closed; all
structure and quotient analyses used in their denominators were complete.

## Round 3 — editorial and reproducibility review

Findings:

- The inherited 39-page manuscript was a development chronicle with hidden
  blocks, excessive displays and mixed query scopes.
- The frozen evidence initially lacked a software/hardware environment record.
- Long code paths and justified table columns produced LaTeX layout warnings.
- SynKit version metadata in the bibliography was stale.

Actions:

- Rewrote the manuscript as a concise Article: unheaded introduction,
  Results, Discussion, Methods, availability and disclosure sections.
- Reduced it to 3,092 prose words reported by `texcount`, four floats, five
  displayed equations and a 125-word abstract.
- Added a separate 15-page, 4,095-word Supplementary Information manuscript
  with 21 displayed equations and the production-level formal proofs retained
  from, and corrected against, the older development manuscript.
- Added exact environment versions and resource measurements, updated SynKit
  to 1.6.2, and made evidence regeneration commands explicit.
- Corrected table/path layout; the round-three ten-page PDF built with no LaTeX,
  citation, reference, overfull or underfull warnings.

Remaining limitations are stated in the Discussion: pilot-scale censoring,
heavy-atom structural CD, exponential worst-case output and the need for a
larger preregistered cohort before population or predictive-model claims.

## Final acceptance

- 92 Mapper/validator tests passed with one optional external MILP test
  deselected; peak resident memory was 206,248 KiB and swap was zero.
- SynKit's file-size, docstring, complexity and flake8 gates passed, and all
  changed Mapper/application Python files passed Black's check.
- The strict offline Sphinx build, 11-page Article and 17-page Supplement all
  completed without warnings.
- All 100 blinded evidence records verified against manifest
  `2a10b5d9b46c6d585a319f7940d808ace3338c6ef22137d3fb23bb7615e5bbb7`;
  its implementation digest matches the final SynKit source.

## Round 4 — application and reference-assistance review

Findings:

- “Negative AAM” was not a mathematically defined output: an alternative to a
  dataset ITS is not automatically a false mechanism.
- The implementation could enumerate arbitrary global shells and canonicalize
  ITS classes, but did not expose one representative AAM per non-reference
  class in the original atom-map coordinates.
- A ground-truth map can define a scalar shell and seed search, but it is not a
  guaranteed acceleration; incumbent quality and traversal order are distinct.
- Product symmetry must preserve every unary attribute used by exact ITS
  identity, including under non-default configuration.

Actions:

- Added a fail-closed alternative-ITS API for `CD="reference"`, any numeric CD,
  and `CD="minimal"`, plus a serial 6 GiB/300 s exporter with digest-bound AAM
  correspondences.
- Defined the exact reference-relative class set and proved class coverage
  under verified product-symmetry pruning.
- Added exhaustive seed-invariance, arbitrary-shell, empty-shell,
  interruption, mapped-coordinate and unary-property symmetry regressions.
- Re-derived the frozen pilot application yield: 39 alternative ITS classes
  across 16 closed minimal-shell cases and 54 across 18 reference-CD cases.
- Froze a six-target FlowER case study repeated with reference, SLAP and no
  seed. All 18 runs agreed semantically; node counts showed that GT seeding is
  not uniformly faster.

Revalidation: 92 Mapper/validator tests passed with one optional MILP test
deselected under the 6 GiB cap; peak RSS was 206,248 KiB with zero swap. The
source-size, docstring, complexity, flake8 and Black gates passed. Strict
offline Sphinx completed without warnings. The Article is 11 pages and 3,766
`texcount` words; the formal Supplement is 17 pages, 4,986 `texcount` words and 22
displayed equations. Both LaTeX logs are warning-free. The application payload
and implementation digests replay, as does the unchanged schema-v4 pilot
implementation digest.

## Round 5 — publication-figure review

Findings:

- The original workflow was a generic box diagram and did not expose the
  distinction between candidate scope, seed ordering and completeness.
- The principal pilot result and record-84 shell ladder were readable only as
  prose and tables.

Actions:

- Adapted the colour-blind-safe palette, restrained panels, badge hierarchy
  and typography of `../Style` into a repository-local TikZ vocabulary.
- Rebuilt the workflow around global query semantics, exact backend choice,
  verified symmetry and fail-closed output status.
- Added a deterministic vector evidence plot generated directly from the
  digest-verified pilot and application JSON. It preserves exact counts,
  denominators, the empty CD-4 shell and the CD-6/CD-8 distinction.

Revalidation: the evidence plot rendered at 85,964 KiB peak RSS and the Article
built at 70,288 KiB under the 6 GiB cap. The final page-scale visual inspection
found no clipped labels or ambiguous encodings.

## Round 6 — formal narrative and visual-contract review

Findings:

- The opening motivated ambiguity but introduced labeled maps, symmetry orbits
  and ITS classes sequentially, so their distinct inferential roles were not
  visible at the point where the scientific question was posed.
- The main Methods described the assignment bounds verbally without displaying
  the prefix interaction or the complete lower/upper subtree interval.
- The workflow used four small, softly filled software stages; at journal scale
  this weakened the proof flow and diverged from the outlined-panel, serif-title
  hierarchy in `../Style`.

Actions:

- Recast the Introduction around a latent compatible bijection and three
  non-interchangeable multiplicities: labeled maps, verified subgroup orbits
  and exact ITS classes. Added explicit structural and reaction-centre
  identifiability questions and a four-part contribution statement.
- Defined the global assignment space, conditioned subspace and fail-closed
  result states in the Article Methods. Tightened the exact-shell Clique
  reduction and made the counting/output boundary explicit.
- Added the incremental prefix interaction, element-block assignment extrema
  and full certified subtree interval, then formalized the centre intersection,
  centre union and exact ITS-code spectrum.
- Rebuilt Figure 1 as a three-panel mathematical contract: set-valued query,
  exact closure and structural inference. The local style now follows the
  canonical palette, serif panel titles, black badges, hairline panels and
  restrained semantic fills from `../Style`; the evidence plot uses the same
  serif badge vocabulary.
- Kept the incomplete 10,000-reaction exploratory run out of the manuscript;
  no population claim or completed-cohort implication was added.

Revalidation: deterministic regeneration of `pilot_spectrum.pdf` produced the
same SHA-256 (`db5802ee57d005b4465043f6c6d00de43d6daf2be47452aa704763471f34945a`)
on consecutive renders. The 11-page, 4,088-word Article and unchanged 17-page,
4,986-word Supplement build without LaTeX, citation, reference, overfull or
underfull warnings. Page-scale inspection covered the revised Introduction,
workflow, evidence plot and formal Methods equations.

## Round 7 — mathematical-scope and NCS reporting audit

Findings:

- Fixed-heavy hydrogen lifting had been described too broadly. Minimizing
  heavy CD before lifting can discard the globally best additive full-atom
  map.
- Arbitrary floating-point inputs have tolerance-shell, not algebraic-equality,
  semantics. The pilot's half-integer bond lattice is a verified special case
  where the two coincide.
- Structural symmetry must preserve every unary property used by reaction-
  centre and ITS identity; coordinate-level centre instability is distinct
  from non-isomorphic ITS multiplicity.
- The historical campaign digest covers a listed source-file set and no
  longer matches current source. Pilot deadline semantics, frozen selection and
  certificate scope also required narrower wording.
- The parallel runner had a progress-logging `NameError`, and the manuscript
  summarizer and campaign summary writer needed actual payload verification.

Actions:

- Proved
  `min_H CD_full(pi) = CD_heavy(pi) + sum_i |r_i-p_{pi(i)}|`, added the
  three-path counterexample, and derived the closed labeled multiplicity
  `T! prod_i max(r_i,q_i)!/|r_i-q_i|!`.
- Added an O(n)-parent closed-form hydrogen summary, exhaustive equality tests,
  and an explicit regression showing that heavy-only and full-CD optimizers
  can differ. The integrated combined-objective branch-and-bound remains
  clearly out of scope.
- Forced all downstream unary properties into structural symmetry refinement,
  added the corresponding regression, and made shell, quotient and structure
  completion separate in the prose.
- Recomputed manifest, case payload and manifest-binding digests before
  manuscript statistics; made the campaign summary writer reject invalid
  payloads; fixed and smoke-tested the parallel progress path.
- Added Statistics and reproducibility, AI-assistance, Funding, data/code
  archive requirements and explicit remaining submission metadata.

Revalidation: the Mapper/validator acceptance command completed **102 tests**,
including the available PuLP/CBC control, under a 6 GiB address-space cap and
one numerical thread. Peak RSS was **205,508 KiB** with zero swap. Further
format and evidence checks passed. The final 13-page Article and 18-page
Supplement build with resolved citations and references and no LaTeX,
overfull or underfull warnings. An independent final audit confirmed the
hydrogen factorization, global combined objective, aggregate labeled-count
formula and path counterexample, and prompted exact wording for shared deadline
budgets, tolerance shells and tolerance-defined reaction centres.
