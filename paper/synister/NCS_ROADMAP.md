# Synister: theory–method–science roadmap

## Central claim

Synister treats atom mapping as exact, set-valued graph-transformation
inference. It proves and enumerates complete chemical-distance shells, compresses
only verified symmetries, and uses the resulting spectra to quantify when
reaction centres are identifiable from endpoint graphs.

This is stronger than a mapper-speed paper and different from a
reference-reproduction study.

## Three declared query scopes

1. **Global numeric shell:** reaction plus a supplied CD; no reference mapping
   participates in candidate removal.
2. **Global minimal shell:** prove the first feasible CD and enumerate every
   optimizer.
3. **Reference-conditioned shell:** fix a declared reference exterior and
   enumerate its variable support. This is an audit mode, not global evidence.

References may seed or order either global search but cannot restrict them.

## Theory requirements

- Binary threshold feasibility is NP-complete and optimization is NP-hard;
  exact-shell enumeration is classified as output-sensitive. Any stronger
  counting-complexity claim requires its own reduction. The zero-distance
  graph-isomorphism case is stated separately.
- For binary graphs, prove the exact broken/formed-edge budget and the disjoint
  edit-support/isomorphism decomposition.
- Prove stabilizer-chain enumeration gives one representative per verified
  subgroup orbit and that the product action on bijections is free.
- Prove the fixed-context lifting theorem with all boundary costs.
- State output and working-memory lower/upper bounds without calling the
  current `O(n^CD)` edit construction fixed-parameter tractable.

## Method requirements

- Exact edit-support backend for small numeric CD.
- Incremental element-blocked assignment branch-and-bound fallback.
- For streamed minimal-CD output, a prerequisite proof pass followed by shell
  enumeration within the declared outer budget; collected mode may filter a
  one-pass search after the optimum is proved.
- Verified automorphism generators and point stabilizers.
- Exact fixed-heavy hydrogen elimination with a closed labeled-multiplicity
  formula. Global full-atom optimization must minimize heavy CD plus the
  hydrogen unary cost; post hoc lifting of heavy-CD minimizers is not enough.
- Count, compact-orbit, and streaming outputs with explicit incomplete status.
- An auditable selector that changes backend, never scientific scope.

## Scientific result

For each globally minimal shell, report:

- labeled mapping multiplicity;
- symmetry-quotiented multiplicity;
- verified-subgroup log multiplicity (subgroup-dependent unless the full
  automorphism group is proved);
- invariant reaction-centre intersection;
- possible reaction-centre union; and
- per-bond exact change frequency across the shell.

Test how arbitrary single-map selection changes extracted templates, reaction
families, and mapping-dependent learning labels.

## Evidence policy

- Exhaustive colored graphs through the largest practical tiny size are the
  correctness oracle.
- A supplied reference CD may define a shell, but reference inclusion is then
  a search-completeness control rather than an accuracy result.
- The main independent chemical experiment hides both the reference mapping
  and its CD, proves the minimal shell, and compares afterward.
- Reference-conditioned FlowER results remain clearly labeled and cannot be
  pooled with global results.
- Every timeout, output cap, or memory limit remains an incomplete record.

## Six-display-item manuscript

1. Problem, scopes, and exact-shell definitions.
2. Edit-support theorem and hybrid algorithm.
3. Exhaustive correctness and symmetry controls.
4. Runtime/memory phase diagram and selector ablation.
5. FlowER/expert-set ambiguity atlas.
6. Consequences for reaction centres, templates, or downstream labels.

## Promotion gates

The journal-facing claim is not ready until:

1. numeric edit-support and assignment backends have exact set equality on all
   tiny controls;
2. minimal-CD mode proves the same minima and full optimizer sets as brute
   force;
3. symmetry compression expands exactly to the labeled shell;
4. a blinded, global FlowER cohort has frozen manifests and resource records;
5. ambiguity produces a reproducible downstream scientific finding; and
6. the main text contains final evidence rather than development chronology.

## Status after seven internal review rounds (28 August 2026)

Gates 1--3 and 6 are satisfied by exhaustive controls and the rewritten
Article. Gate 4 is satisfied at pilot scale by a frozen 100-case blinded
campaign with explicit censoring and resource records; a larger preregistered
cohort remains necessary before population-level claims. Gate 5 is satisfied
only at the direct structural-label level: exact reaction-centre/template
labels change across maps, and each closed shell yields a candidate-complete
panel with one AAM per non-reference exact ITS class. The frozen pilot yields
39 such classes across 16 closed minimal-shell cases and 54 across 18
reference-CD cases. A measured predictive or scientific downstream outcome
remains necessary for a strong Nature Computational Science submission and is
not claimed here.

The journal main text is 13 pages, with approximately 2,100 words outside the
abstract, Methods, availability statements and end matter, and four display
items. A separate Supplementary Information
manuscript restores the rigorous production-level definitions and proofs from
the longer development draft while excluding unimplemented proposals from the
validated method claim.

## Author-supplied items required before submission

- affiliation, corresponding-author email and ORCID;
- final Funding statement;
- tagged SynKit release, immutable Git commit and archival software DOI;
- archival DOI for generated evidence and figure source data; and
- results of the larger preregistered campaign plus matched-objective runtime,
  bound, symmetry and hydrogen ablations if targeting Nature Computational
  Science rather than a narrower methods venue.
